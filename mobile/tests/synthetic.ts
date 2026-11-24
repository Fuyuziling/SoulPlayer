import type { RawSample } from '../src/core/types';

/**
 * 合成 IMU 数据。测试不依赖真机，也不依赖随机：伪随机用固定种子，
 * 同一个用例每次跑出的样本完全一致，失败可以复现。
 *
 * 物理模型（与 contracts/sensor_feature_schema.json 的坐标系一致）：
 *   设备竖直握持时 pitch=90°，屏幕朝上 pitch=0°，朝下 pitch=180°；
 *   roll>0 表示向右倾斜。重力在设备坐标系下的投影由这两个角唯一确定。
 */

const D2R = Math.PI / 180;

export function gravityVector(pitchDeg: number, rollDeg: number): [number, number, number] {
  const p = pitchDeg * D2R;
  const r = rollDeg * D2R;
  return [
    9.81 * Math.sin(p) * Math.sin(r),
    -9.81 * Math.sin(p) * Math.cos(r),
    -9.81 * Math.cos(p),
  ];
}

/** 固定种子伪随机，保证用例可复现。 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Pose {
  pitch: number;
  roll: number;
}

export interface MoveOptions {
  /** 该段的线性加速度（设备坐标，m/s^2），u 为 0~1 的段内进度。 */
  linear?: (u: number) => [number, number, number];
  from?: Pose;
  to?: Pose;
  /** 姿态插值缓动，默认 smoothstep。 */
  ease?: (u: number) => number;
}

export class ImuSim {
  private out: RawSample[] = [];
  private t = 0;
  private pose: Pose;
  private readonly dt: number;
  private readonly rand: () => number;

  constructor(
    private readonly hz = 60,
    private readonly noise = 0.2,
    seed = 20261107,
    startPose: Pose = { pitch: 90, roll: 0 },
  ) {
    this.dt = 1 / hz;
    this.rand = mulberry32(seed);
    this.pose = { ...startPose };
  }

  /** 静止一段时间。 */
  still(sec: number): this {
    const n = Math.round(sec * this.hz);
    for (let i = 0; i < n; i++) this.emit([0, 0, 0], [0, 0, 0]);
    return this;
  }

  /**
   * 一段动作：姿态从 from 线性（缓动）过渡到 to，同时叠加 linear 给出的加速度。
   * 角速度由姿态差分得到——这比手工写 gyro 更不容易和重力变化脱节。
   */
  move(sec: number, opts: MoveOptions = {}): this {
    const n = Math.max(2, Math.round(sec * this.hz));
    const from = opts.from ?? { ...this.pose };
    const to = opts.to ?? { ...this.pose };
    const ease = opts.ease ?? ((u) => u * u * (3 - 2 * u));
    let prevPitch = this.pose.pitch;
    let prevRoll = this.pose.roll;

    for (let i = 1; i <= n; i++) {
      const u = i / n;
      const k = ease(u);
      const pitch = from.pitch + (to.pitch - from.pitch) * k;
      const roll = from.roll + (to.roll - from.roll) * k;
      const gyro: [number, number, number] = [
        ((pitch - prevPitch) * D2R) / this.dt,
        ((roll - prevRoll) * D2R) / this.dt,
        0,
      ];
      const lin = opts.linear ? opts.linear(u) : ([0, 0, 0] as [number, number, number]);
      this.emitRaw(lin, gyro, { pitch, roll });
      prevPitch = pitch;
      prevRoll = roll;
    }
    this.pose = to;
    return this;
  }

  /** 按「姿态」直接生成，不需要动作，用于纯姿态变化（翻面、放到桌上）。 */
  poseTo(sec: number, to: Pose, opts: { linear?: MoveOptions['linear'] } = {}): this {
    return this.move(sec, { to, linear: opts.linear });
  }

  /**
   * 跑步：周期性脚步冲击 + 同频的横向摆臂。
   *
   * 冲击宽度 100ms、峰值 20 m/s^2——这是跑步最难缠的地方：**峰值与"敲击"完全重叠**，
   * 唯一能分开两者的是宽度（敲击 20~50ms，脚步 80ms+）。
   * 横向摆臂取步频的一半（左右臂交替），峰值 8 时位移约 9.4cm，
   * 已经逼近甩动判定的 14cm 跨度门——这就是"跑步必须另设判定基线"的量化依据，
   * 也是日常六条手势在跑步模式下必须整体退出的原因。
   */
  run(sec: number, opts: { bpm?: number; amp?: number; sway?: number } = {}): this {
    const bpm = opts.bpm ?? 176;
    const amp = opts.amp ?? 20;
    const sway = opts.sway ?? 8;
    const n = Math.round(sec * this.hz);
    const period = 60 / bpm;
    const duty = 0.1;
    const swayPeriod = period * 2;
    for (let i = 0; i < n; i++) {
      const tt = (i / this.hz) % period;
      const imp = tt < duty ? amp * Math.sin((Math.PI * tt) / duty) : 0;
      const sw = sway * Math.sin((2 * Math.PI * (i / this.hz)) / swayPeriod);
      this.emit([sw, imp, 0], [0, 0, 0]);
    }
    return this;
  }

  /**
   * 双击机身：两次窄脉冲，默认宽 30ms、间隔 200ms，沿 z 轴（敲背面）。
   * 与 run() 的关键差别是宽度——30ms 在 100Hz 下只有 3 个样本，
   * 脚步冲击是 100ms（10 个样本），computeRhythm 按 70ms 分界。
   */
  doubleTap(opts: { gapMs?: number; amp?: number; widthMs?: number } = {}): this {
    const gap = opts.gapMs ?? 200;
    const amp = opts.amp ?? 40;
    const width = opts.widthMs ?? 30;
    this.pulse(amp, width);
    this.holdMs(gap - width);
    this.pulse(amp, width);
    this.holdMs(150);
    return this;
  }

  /** 抬腕：从垂握（90°）抬到 40°，保持一会儿，再放下。 */
  raisePhone(holdSec = 1.0): this {
    this.poseTo(0.35, { pitch: 40, roll: 0 });
    this.still(holdSec);
    this.poseTo(0.35, { pitch: 90, roll: 0 });
    return this;
  }

  private pulse(amp: number, widthMs: number): void {
    const n = Math.max(1, Math.round((widthMs / 1000) * this.hz));
    for (let i = 0; i < n; i++) {
      const v = amp * Math.sin((Math.PI * (i + 0.5)) / n);
      this.emit([0, 0, v], [0, 0, 0]);
    }
  }

  private holdMs(ms: number): void {
    const n = Math.max(1, Math.round((ms / 1000) * this.hz));
    for (let i = 0; i < n; i++) this.emit([0, 0, 0], [0, 0, 0]);
  }

  samples(): RawSample[] {
    return this.out;
  }

  private emit(lin: [number, number, number], gyro: [number, number, number]): void {
    this.emitRaw(lin, gyro, this.pose);
  }

  private emitRaw(lin: [number, number, number], gyro: [number, number, number], pose: Pose): void {
    const g = gravityVector(pose.pitch, pose.roll);
    const jitter = (): number => (this.rand() - 0.5) * 2 * this.noise;
    this.out.push({
      t: this.t,
      ax: g[0] + lin[0] + jitter(),
      ay: g[1] + lin[1] + jitter(),
      az: g[2] + lin[2] + jitter(),
      gx: gyro[0] + jitter() * 0.05,
      gy: gyro[1] + jitter() * 0.05,
      gz: gyro[2] + jitter() * 0.05,
      hasRotation: true,
    });
    this.t += this.dt;
    this.pose = pose;
  }
}

/** 单次甩动：半正弦加速度冲量。波形时长由 move() 的时长决定，此处只管幅值。
 *  amp=22 配 move(0.22) 时约 0.34m 行程、3.1m/s 峰值速度。 */
export function flickImpulse(dir: -1 | 1, amp = 22): (u: number) => [number, number, number] {
  return (u) => [dir * amp * Math.sin(Math.PI * u), 0, 0];
}

/** 轻晃：小幅短促，用于验证"不该触发"的一侧。 */
export function smallJiggle(amp = 5): (u: number) => [number, number, number] {
  return (u) => [amp * Math.sin(Math.PI * u), 0, 0];
}

/**
 * 摇一摇：4.2Hz 往复。
 *
 * 必须用 -cos 相位而不是 sin：从 v(0)=0 起步的 sin 加速度，其速度积分
 * v(t)=A/(2πf)(1-cos) 恒为非负，位移是单调前进的——那是"单向脉冲串"，
 * 不是往复摇动。用 -cos 相位才能得到正负交替的速度（真实摇动的形态），
 * 位移跨度也才回到物理上合理的 2A/(2πf)² ≈ 4cm。
 * 前 100ms 加渐入包络，避免 t=0 处加速度突变。
 */
export function shakeWave(amp = 15, freq = 4.2): (u: number) => [number, number, number] {
  return (u) => {
    const env = Math.min(1, u / 0.1);
    return [-amp * Math.cos(2 * Math.PI * freq * u) * env, 0, 0];
  };
}
