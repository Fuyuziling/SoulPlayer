import type { Sample, SensorFeatures } from './types';
import { clampDt, hypot3, median, pitchFromGravity, rollFromGravity } from './filters';

/**
 * 窄脉冲宽度"测不到"的哨兵值。
 *
 * 之所以不用 0 或 NaN：契约里的条件是 `narrow_pulse_width_ms: {max: 60}`，
 * 用 0 会让"测不到"被当成"一次极窄的敲击"而永远通过——正好是最危险的方向。
 * 用一个明显超限的大数，测不到就自动不通过，语义与失败一致。
 */
export const NARROW_PULSE_UNMEASURABLE = 999;


export interface FeatureConfig {
  /** 本组特征使用的时间窗口（秒）。不同动作时间尺度不同，由预设的 window_ms 决定。 */
  windowSec: number;
  /** 静止门（m/s^2）：线性加速度低于此值视为静止，用于速度归零、漂移抑制。 */
  stillGate: number;
  /** 本机静置噪声底（启动校准时测得）。 */
  noiseFloor: number;
}

/**
 * 在滑动窗口上计算一组特征。
 *
 * 三个刻意的设计：
 *
 * 1. **速度/位移每帧从窗口起点重新积分**，不做跨帧累加。二次积分的漂移是
 *    随时长平方增长的，一旦做成累加式，跑十分钟之后"静置"也会报出几十厘米
 *    的位移，判定必然失效。每次重算的代价可以接受（1s 窗口 60~120 个点）。
 *
 * 2. **静止时对速度做衰减**（×0.4/帧），而不是直接置零。直接置零会在
 *    "轻晃 → 静止 → 轻晃"的边界产生速度阶跃，位移曲线出现假的跳变台阶。
 *
 * 3. **垂直比用速度而不是加速度**。加速度的 z 分量里混着翻面、倾角变化带来的
 *    重力估计滞后，噪声很大；速度积分之后竖直平移的成分才明显，用它判断
 *    "这是横向甩还是上下抖"更准。
 */
export function computeFeatures(samples: readonly Sample[], cfg: FeatureConfig): SensorFeatures {
  const n = samples.length;
  const empty = emptyFeatures(cfg);
  if (n < 4) return empty;

  const first = samples[0];
  const last = samples[n - 1];
  const spanSec = last.t - first.t;
  // 覆盖时长不足窗口的 85% 就不算填满。按时间而不是帧数判断，
  // 这样 30Hz 和 120Hz 的设备在同一个 physical 时间点开始出判定。
  const windowFull = spanSec >= cfg.windowSec * 0.85 && n >= 10;

  let vx = 0;
  let vy = 0;
  let vz = 0;
  let px = 0;
  let py = 0;
  let pz = 0;

  let velMagPeak = 0;
  let velHPeak = 0;
  let velZPeak = 0;
  let linAccPeak = 0;
  let sumSq = 0;
  let stillCount = 0;
  /** 连续静止帧数。衰减必须等"静止成立了一段时间"，见下方注释。 */
  let stillRun = 0;

  const posX: number[] = [0];
  const posY: number[] = [0];
  const velXSeries: number[] = [0];
  const velYSeries: number[] = [0];
  const accXSeries: number[] = [];
  const accYSeries: number[] = [];
  const mags: number[] = [];

  for (let i = 0; i < n; i++) {
    const s = samples[i];
    const mag = hypot3(s.lx, s.ly, s.lz);
    mags.push(mag);
    accXSeries.push(s.lx);
    accYSeries.push(s.ly);
    if (mag > linAccPeak) linAccPeak = mag;
    sumSq += mag * mag;
    const isStillFrame = mag < cfg.stillGate;
    if (isStillFrame) stillCount++;
    stillRun = isStillFrame ? stillRun + 1 : 0;

    if (i === 0) continue;
    const prev = samples[i - 1];
    const dt = clampDt(s.t - prev.t);
    // 梯形积分：加速度 → 速度
    vx += ((s.lx + prev.lx) / 2) * dt;
    vy += ((s.ly + prev.ly) / 2) * dt;
    vz += ((s.lz + prev.lz) / 2) * dt;
    // 静止衰减：三个条件叠在一起才是对的，少一个都会出问题——
    //   ① 连续静止 ≥ 3 帧。往复摇动每半个周期都会短暂回到加速度零点，
    //      若单帧就衰减，衰减只作用在过零的一侧，积分失去对称性，
    //      一来一回会被积出一个方向明确的假位移（实测 4.2Hz 摇动报出 21cm）。
    //   ② 不设"速度必须小"的条件。曾经加过 speed<0.35，结果摇完手停下时
    //      残余的 0.5m/s 永远衰减不掉，速度带着位移一路漂到 27cm，
    //      反而把摇动喂成了甩动。
    //   ③ 系数 0.55/帧 @60Hz ≈ 24ms 时间常数。曾试过 0.82（≈100ms），
    //      残余速度衰减太慢，甩完后位移从 0.45m 虚高到 0.75m。
    if (stillRun >= 3) {
      vx *= 0.55;
      vy *= 0.55;
      vz *= 0.55;
    }
    // 速度 → 位移
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;
    posX.push(px);
    posY.push(py);
    velXSeries.push(vx);
    velYSeries.push(vy);

    const vm = hypot3(vx, vy, vz);
    if (vm > velMagPeak) velMagPeak = vm;
    const vh = Math.hypot(vx, vy);
    if (vh > velHPeak) velHPeak = vh;
    if (Math.abs(vz) > velZPeak) velZPeak = Math.abs(vz);
  }

  const spanX = range(posX);
  const spanY = range(posY);

  // 主轴取水平面内位移跨度更大的那个轴。竖直（z）位移不参与主轴竞争，
  // 因为"上下抖手机"在语义上不是横向甩动，由 verticalRatio 单独挡。
  const dominantAxis: 'x' | 'y' = spanX >= spanY ? 'x' : 'y';
  const lateralSpan = dominantAxis === 'x' ? spanX : spanY;
  const dispAlong = dominantAxis === 'x' ? px : py;
  // 净位移小于 1.5cm 判定为方向不明——往返晃动会落进这个区间，从而不触发。
  const lateralDirection: -1 | 0 | 1 =
    Math.abs(dispAlong) < 0.015 ? 0 : dispAlong > 0 ? 1 : -1;

  const verticalRatio = velZPeak / (velHPeak + 0.05);

  // 主轴速度的反向次数。死区 0.3 m/s 用来滤掉静止段和动作收尾时的噪声翻转——
  // 没有死区的话，速度衰减到 0 的过程会被数成多次反向。
  const velReversals = countReversals(dominantAxis === 'x' ? velXSeries : velYSeries, 0.3);
  // 加速度符号反转：3 m/s^2 死区远高于本机噪声底（0.2~0.6），又远低于甩动峰值（20+），
  // 既能滤噪又不会漏掉往复。
  const accSignChanges = countReversals(dominantAxis === 'x' ? accXSeries : accYSeries, 3.0);

  const pitchStart = pitchFromGravity(first.grav);
  const pitchDeg = pitchFromGravity(last.grav);
  const rollDeg = rollFromGravity(last.grav);

  // 横滚角的整窗标准差。逐样本算 roll 而不是只看首末，
  // 因为"甩完之后的收敛过程"恰恰是中间那段最抖。
  let rollSum = 0;
  let rollSumSq = 0;
  for (let i = 0; i < n; i++) {
    const r = rollFromGravity(samples[i].grav);
    rollSum += r;
    rollSumSq += r * r;
  }
  const rollMean = rollSum / n;
  const rollStd = Math.sqrt(Math.max(0, rollSumSq / n - rollMean * rollMean));

  // 角速度积分：绕设备 x 轴（横轴）的累计转角，作为翻面的第二证据。
  // 重力估计有 0.25s 时滞，短窗口内首末倾角差可能只到 90°，此时用陀螺仪补上。
  let gyroIntX = 0;
  let gyroMagPeak = 0;
  for (let i = 1; i < n; i++) {
    const s = samples[i];
    const prev = samples[i - 1];
    const dt = clampDt(s.t - prev.t);
    gyroIntX += ((s.gx + prev.gx) / 2) * dt;
    const gm = hypot3(s.gx, s.gy, s.gz);
    if (gm > gyroMagPeak) gyroMagPeak = gm;
  }
  const flipTiltDeg = Math.abs(pitchDeg - pitchStart);
  const flipGyroDeg = (Math.abs(gyroIntX) * 180) / Math.PI;
  const flipAngleDeg = Math.max(flipTiltDeg, flipGyroDeg);

  const faceState: SensorFeatures['faceState'] =
    pitchDeg < 45 ? 'up' : pitchDeg > 135 ? 'down' : 'edge';
  const faceFirst: SensorFeatures['faceState'] =
    pitchStart < 45 ? 'up' : pitchStart > 135 ? 'down' : 'edge';
  let faceTransition: SensorFeatures['faceTransition'] = 'none';
  if (faceFirst !== 'down' && faceState === 'down') faceTransition = 'to_down';
  else if (faceFirst === 'down' && faceState !== 'down') faceTransition = 'from_down';

  const peaks = findPeaks(mags, samples, Math.max(cfg.noiseFloor * 3, 2.5));
  const peakCount = peaks.length;
  const diffs: number[] = [];
  for (let i = 1; i < peaks.length; i++) diffs.push(peaks[i] - peaks[i - 1]);
  const peakIntervalMs = median(diffs) * 1000;

  const stillRatio = stillCount / n;
  const isStill = stillRatio > 0.9 && linAccPeak < cfg.stillGate * 2;

  return {
    timestamp: last.t,
    source: 'sensor',
    sampleHz: 0, // 由引擎用 SampleBuffer.sampleHz() 填，特征层不重复估计
    windowFull,
    linAccPeak,
    linAccRms: Math.sqrt(sumSq / n),
    velMagPeak,
    velHPeak,
    verticalRatio,
    dominantAxis,
    lateralSpan,
    lateralDirection,
    velReversals,
    accSignChanges,
    dispX: px,
    dispY: py,
    gyroMagPeak,
    pitchDeg,
    rollDeg,
    rollStd,
    flipAngleDeg,
    tiltDeltaDeg: flipTiltDeg,
    faceState,
    faceTransition,
    peakCount,
    peakIntervalMs,
    stillRatio,
    isStill,
    noiseFloor: cfg.noiseFloor,
    holdMs: {},
    // 节律字段由引擎用 computeRhythm 在 2s 窗口上算出来后合并进来，
    // 这里给"节律不成立"的中性值，保证即使引擎没合并也不会误通过。
    cadenceHz: 0,
    cadenceStability: 0,
    impactRms: 0,
    narrowPulseWidthMs: NARROW_PULSE_UNMEASURABLE,
    tapPairs: 0,
    rhythmBreakMs: 0,
  };
}

/**
 * 显著的加速度峰值检测。
 * 三个约束缺一不可：超噪声门、是局部极大、与上一个峰间隔 > 80ms。
 * 80ms 对应 12.5Hz，比人手能做到的最快往复（约 8Hz）还高一档，
 * 目的是不让同一个冲击的抖动被数成多个峰。
 */
function findPeaks(
  mags: readonly number[],
  samples: readonly Sample[],
  gate: number,
  minGapSec = 0.08,
): number[] {
  const out: number[] = [];
  let lastT = -Infinity;
  for (let i = 1; i < mags.length - 1; i++) {
    const m = mags[i];
    if (m < gate) continue;
    if (m >= mags[i - 1] && m > mags[i + 1]) {
      const t = samples[i].t;
      if (t - lastT > minGapSec) {
        out.push(t);
        lastT = t;
      }
    }
  }
  return out;
}

function range(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  let lo = xs[0];
  let hi = xs[0];
  for (const v of xs) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return hi - lo;
}

/**
 * 统计序列穿过零点的次数（带死区）。
 * 死区不是可选项：不加的话静置时速度在 ±0 附近抖动，一次静置能被数出几十次反向，
 * 所有依赖它的门都会失效。
 */
function countReversals(series: readonly number[], deadzone: number): number {
  let reversals = 0;
  let lastSign = 0;
  for (const v of series) {
    if (Math.abs(v) < deadzone) continue;
    const sign = v > 0 ? 1 : -1;
    if (lastSign !== 0 && sign !== lastSign) reversals++;
    lastSign = sign;
  }
  return reversals;
}

/** 节律分析结果。固定用 2s 窗口，与各预设自己的动作窗口无关。 */
export interface RhythmResult {
  cadenceHz: number;
  stability: number;
  impactRms: number;
  narrowPulseWidthMs: number;
  tapPairs: number;
  /**
   * 距最近一次冲击峰的时间（ms）。
   *
   * 这个量必须基于"瞬时的上一个峰"，而不是"2s 窗口里节律是否还成立"——
   * 后者有整个窗口的滞后：人已经停下，窗口里还留着 1.5s 的跑步数据，
   * 节律仍然成立，于是抬腕判据要等两秒才满足，抬完手都放下了。
   */
  sinceLastPeakMs: number;
  /** 步频是否可测。稳定度是否达标由引擎按契约判断，这里不下结论。 */
  periodic: boolean;
}

/** 窄脉冲的宽度上限（ms）。指尖敲击 20~50ms，脚步冲击 80ms+，70 落在空档。 */
const NARROW_PULSE_MAX_MS = 70;

/**
 * 2s 窗口上的节律与脉冲分析。
 *
 * 为什么必须独立于 computeFeatures：步频要靠 4~6 个峰才估得稳，
 * 而甩动判定只需要 0.36s。用同一个窗口同时服务两者是不可能的——
 * 0.36s 里只有 1~2 个峰，步频估不出来；2s 窗口里做甩动判定，
 * 位移跨度和速度峰值早就被别的动作稀释了。
 */
export function computeRhythm(samples: readonly Sample[], noiseFloor: number): RhythmResult {
  const n = samples.length;
  const empty: RhythmResult = {
    cadenceHz: 0,
    stability: 0,
    impactRms: 0,
    narrowPulseWidthMs: NARROW_PULSE_UNMEASURABLE,
    tapPairs: 0,
    sinceLastPeakMs: NARROW_PULSE_UNMEASURABLE,
    periodic: false,
  };
  if (n < 20) return empty;

  const first = samples[0];
  const last = samples[n - 1];
  const spanSec = last.t - first.t;
  if (spanSec < 1.2) return empty;
  const hz = (n - 1) / spanSec;

  const mags: number[] = new Array(n);
  /** 沿重力方向的冲击分量（取绝对值）。步频检测用它。 */
  const alongAbs: number[] = new Array(n);
  /** 垂直于重力方向的冲击分量。窄脉冲检测用它，理由见下方注释。 */
  const perp: number[] = new Array(n);
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const s = samples[i];
    const m = hypot3(s.lx, s.ly, s.lz);
    mags[i] = m;
    sumSq += m * m;
    // 把线性加速度分成"沿重力"与"垂直重力"两个分量。
    //
    // 为什么窄脉冲检测必须用垂直分量：脚步冲击是身体上下运动的反作用，
    // 沿重力方向传导；而"敲机身"敲的是屏幕法向，垂直于重力。这个判据
    // 与采样率无关，比脉冲宽度可靠得多——100Hz 下 30ms 的敲击只有 3 个样本，
    // 宽度测量有 ±10ms 的相位误差，80ms 的脚步最差能被读成 60ms，
    // 与敲击的 50ms 撞在一起。实测不分离分量时，190 步/分的跑步会
    // 凑出 15 次假双击。
    const gm = hypot3(s.grav[0], s.grav[1], s.grav[2]) || 1;
    const along = (s.lx * s.grav[0] + s.ly * s.grav[1] + s.lz * s.grav[2]) / gm;
    alongAbs[i] = Math.abs(along);
    const p2 = m * m - along * along;
    perp[i] = p2 > 0 ? Math.sqrt(p2) : 0;
  }
  const impactRms = Math.sqrt(sumSq / n);

  // 冲击门槛只负责"切出脉冲段"，不承担判别职责：跑步冲击和指尖敲击的峰值
  // 本来就重叠（都在 15~25 m/s^2），靠峰值区分必然顾此失彼。判别交给宽度。
  // 10 这个下限是物理量而非可调阈值：它要低于任何一次真实敲击，高于静置噪声。
  const pulseGate = Math.max(noiseFloor * 4, 10.0);

  // 步频的峰必须在**沿重力分量**上找，不能用总模长：
  // 横向摆臂（跑步时 8~15 m/s^2，与脚步冲击同量级）在总模长上也是峰，
  // 会和脚步峰交替出现，把 176 步/分测成 4.26Hz。实测不分离时稳定度 0.45，
  // 而跑步的真实稳定度应该 <0.15。
  // 去重间隔 200ms：一次触地在沿重力分量上可能出现上下两个峰（相隔半周期），
  // 200ms 比半周期（约 170ms）大、比步周期（280ms+）小，正好一步留一个。
  const peaks = findPeaks(alongAbs, samples, Math.max(noiseFloor * 3, 5.0), 0.2);
  let cadenceHz = 0;
  let stability = 0;
  if (peaks.length >= 3) {
    const diffs: number[] = [];
    for (let i = 1; i < peaks.length; i++) diffs.push(peaks[i] - peaks[i - 1]);
    const med = median(diffs);
    // 0.15s（400 步/分）~1.2s（50 步/分）之外不当作步频：
    // 更快说明同一个冲击被数成了多个峰，更慢说明是随手动作而不是节律。
    if (med > 0.15 && med < 1.2) {
      cadenceHz = 1 / med;
      const devs = diffs.map((d) => Math.abs(d - med));
      stability = median(devs) / med;
    }
  }

  // 窄脉冲：在**垂直重力分量**上找连续的超阈值段，再按段时间宽度二次筛。
  // 轴向负责排除脚步冲击，宽度负责排除摆臂这类慢变动作，两者缺一都会误判。
  const sampleMs = 1000 / hz;
  const narrow: number[] = [];
  let lastWidth = NARROW_PULSE_UNMEASURABLE;
  let i = 0;
  while (i < n) {
    if (perp[i] <= pulseGate) {
      i++;
      continue;
    }
    let j = i;
    while (j + 1 < n && perp[j + 1] > pulseGate) j++;
    // 贴窗口边界的段宽度不可靠（真实脉冲有一半在窗口外），直接跳过——
    // 否则窗口每滑过一个脉冲都会读出一个"半个宽度"的假窄脉冲，
    // 而假窄脉冲恰好会满足 tap 的宽度条件。
    if (i > 0 && j < n - 1) {
      const width = (samples[j].t - samples[i].t) * 1000 + sampleMs;
      if (width <= NARROW_PULSE_MAX_MS) {
        narrow.push(samples[i].t);
        lastWidth = width;
      }
    }
    i = j + 1;
  }

  // 成对窄脉冲：相邻两击间隔 120~320ms。贪心配对并跳过已配的，
  // 三击只算一对——宁可少报一次，也不要让一次三击产生两个切换动作。
  let tapPairs = 0;
  for (let k = 0; k + 1 < narrow.length; k++) {
    const gap = narrow[k + 1] - narrow[k];
    if (gap >= 0.12 && gap <= 0.32) {
      tapPairs++;
      k++;
    }
  }

  return {
    cadenceHz,
    stability,
    impactRms,
    narrowPulseWidthMs: lastWidth,
    tapPairs,
    sinceLastPeakMs:
      peaks.length > 0 ? (last.t - peaks[peaks.length - 1]) * 1000 : spanSec * 1000,
    periodic: cadenceHz > 0,
  };
}

export function emptyFeatures(cfg: FeatureConfig): SensorFeatures {
  return {
    timestamp: 0,
    source: 'sensor',
    sampleHz: 0,
    windowFull: false,
    linAccPeak: 0,
    linAccRms: 0,
    velMagPeak: 0,
    velHPeak: 0,
    verticalRatio: 0,
    dominantAxis: null,
    lateralSpan: 0,
    lateralDirection: 0,
    velReversals: 0,
    accSignChanges: 0,
    dispX: 0,
    dispY: 0,
    gyroMagPeak: 0,
    pitchDeg: 0,
    rollDeg: 0,
    rollStd: 0,
    flipAngleDeg: 0,
    tiltDeltaDeg: 0,
    faceState: 'unknown',
    faceTransition: 'none',
    peakCount: 0,
    peakIntervalMs: 0,
    stillRatio: 1,
    isStill: true,
    noiseFloor: cfg.noiseFloor,
    holdMs: {},
    cadenceHz: 0,
    cadenceStability: 0,
    impactRms: 0,
    narrowPulseWidthMs: NARROW_PULSE_UNMEASURABLE,
    tapPairs: 0,
    rhythmBreakMs: 0,
  };
}
