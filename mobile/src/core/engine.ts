import { SampleBuffer } from './ring';
import { GravityEstimator, clamp, clampDt, hypot3 } from './filters';
import { computeFeatures, computeRhythm, NARROW_PULSE_UNMEASURABLE, type RhythmResult } from './features';
import { CooldownTable } from './cooldown';
import {
  DEFAULT_COOLDOWN_RULES,
  DEFAULT_GATES,
  DEFAULT_PRESETS,
  MODES,
  modesOf,
  windowMsOf,
} from './presets';
import type {
  ActionEvent,
  Condition,
  CooldownRule,
  FrameDiagnostics,
  GatesConfig,
  ModeId,
  Placement,
  Preset,
  RawSample,
  Sample,
  SensorFeatures,
} from './types';

/** 节律窗口（秒）。100Hz 下 200 个样本，够 4~6 个跑步冲击峰估步频。 */
const RHYTHM_WINDOW_SEC = 2.0;

/**
 * 2s 窗口 + 峰值检测需要足够的量。给到 2.5s 是留出 computeRhythm
 * 要求的 1.2s 跨度下限，同时也容纳最长预设窗口（抬腕 900ms）而不被截断。
 */
const BUFFER_SEC = 2.5;
const BUFFER_CAPACITY = 400;

/** 校准被运动污染时的回退噪声底。 */
const FALLBACK_NOISE_FLOOR = 0.6;

/**
 * 校准期间 p90 超过此值，说明用户根本没静止（跑步冲击 RMS 远超这个数）。
 * 此时把噪声底写进去会造成不可逆的伤害，见 finishCalibration 的注释。
 */
const CALIBRATION_MOTION_LIMIT = 2.5;

export interface EngineOptions {
  presets?: Preset[];
  gates?: GatesConfig;
  cooldownRules?: Record<string, CooldownRule>;
  noiseFloor?: number;
  mode?: ModeId;
  placement?: Placement;
}

type PresetWithWindow = Preset & { window_ms?: number };

/** 契约字段名 → 特征读取器。手写映射而不是字符串索引，编译器能查出漏配。 */
const FEATURE_GETTERS: Record<string, (f: SensorFeatures) => number | string | null> = {
  lin_acc_peak: (f) => f.linAccPeak,
  lin_acc_rms: (f) => f.linAccRms,
  vel_mag_peak: (f) => f.velMagPeak,
  vel_h: (f) => f.velHPeak,
  vertical_ratio: (f) => f.verticalRatio,
  lateral_span: (f) => f.lateralSpan,
  lateral_direction: (f) => f.lateralDirection,
  vel_reversals: (f) => f.velReversals,
  acc_sign_changes: (f) => f.accSignChanges,
  disp_x: (f) => f.dispX,
  disp_y: (f) => f.dispY,
  gyro_mag_peak: (f) => f.gyroMagPeak,
  pitch_deg: (f) => f.pitchDeg,
  roll_deg: (f) => f.rollDeg,
  roll_std: (f) => f.rollStd,
  flip_angle_deg: (f) => f.flipAngleDeg,
  tilt_delta_deg: (f) => f.tiltDeltaDeg,
  face_state: (f) => f.faceState,
  face_transition: (f) => f.faceTransition,
  peak_count: (f) => f.peakCount,
  peak_interval_ms: (f) => f.peakIntervalMs,
  still_ratio: (f) => f.stillRatio,
  // 节律与脉冲（跑步模式用）
  cadence_hz: (f) => f.cadenceHz,
  cadence_stability: (f) => f.cadenceStability,
  impact_rms: (f) => f.impactRms,
  narrow_pulse_width_ms: (f) => f.narrowPulseWidthMs,
  tap_pairs: (f) => f.tapPairs,
  rhythm_break_ms: (f) => f.rhythmBreakMs,
};

interface GateResult {
  ok: boolean;
  gate?: string;
  detail?: string;
}

/**
 * 传感器判定引擎。
 *
 * 与 PC 端视觉链路严格同构的五层：
 *   采集 ingest() → 特征 computeFeatures() → 判定 evaluate() → 冷却 CooldownTable → 事件
 *
 * 几处与 PC 端不同的必要偏离，都是为了手机这个物理场景：
 *
 * 1. **多窗口**。PC 端所有动作共用 20 帧窗口，因为摄像头帧率稳定且动作尺度接近。
 *    手机上甩动 0.3s、摇一摇 0.5s、翻面 0.5s，用一个窗口必然顾此失彼。
 *
 * 2. **噪声底自适应**。手机型号的加速度计噪声差 3~5 倍，硬编码的幅度门
 *    在 A 机上灵敏、在 B 机上失灵。
 *
 * 3. **动作锁定按 action 而非全局**。PC 端一次只可能有一个挥手，锁死没问题；
 *    手机上"甩左"锁定期间用户可能立刻"甩右"，锁全局会吞掉合法输入。
 *
 * 4. **模式互斥**（本项目独有）。使用模式不是"多一组动作"，而是换一套判定基线。
 *    跑步时的持续节律会同时破坏日常手势的幅度门、稳定性门与校准基线，
 *    所以不属于当前模式的预设**整体移出匹配池**——连特征都不算。
 *    这是唯一可靠的互斥方式：靠调阈值让两套动作共存，必然有一边误判。
 */
export class SensorEngine {
  private buf: SampleBuffer;
  private grav = new GravityEstimator();
  private cd: CooldownTable;
  private presets: PresetWithWindow[];
  private gates: GatesConfig;

  private noiseFloor: number;
  private sensitivity = 1.0;

  private mode: ModeId;
  private placement: Placement;

  /** 一次手势只出一次事件：action 维度的锁定 + 连续不满足帧数的抖动容忍。 */
  private lock: { action: string; presetId: string; miss: number } | null = null;
  /** 倾斜这类"保持型"动作的持续计时。 */
  private holdSince = new Map<string, number>();

  private calibrating = false;
  private calibMags: number[] = [];
  private calibReliable = true;

  // ── 节律与环境态 ──────────────────────────────
  /** 最近一次节律成立（判定为跑步）的时刻。自动暂停按它计时。 */
  private lastActiveAt = 0;
  /** 节律是否持续成立（瞬时判定，来自 2s 节律窗口）。 */
  private running = false;
  /** 自动暂停只在"真的跑过一段"之后才武装，避免刚进跑步模式站着就自动暂停。 */
  private autoPauseArmed = false;
  /** 日常模式下检测到持续跑步、准备提示的起始时刻。 */
  private suggestSince: number | null = null;
  /** 每段持续跑步只提示一次。 */
  private suggestShown = false;

  // ── 硬件步检测（TYPE_STEP_DETECTOR）─────────────────
  /** 步检测通道是否可用（机型支持 + 用户授权）。不可用 = 一切行为同旧版。 */
  private stepDetectorOn = false;
  /** 最近 4s 内的步事件时刻。只需要回答"有没有在走/跑"，不需要精确步频。 */
  private stepTimes: number[] = [];

  constructor(opts: EngineOptions = {}) {
    this.gates = opts.gates ?? DEFAULT_GATES;
    this.presets = (opts.presets ?? DEFAULT_PRESETS).map((p) => ({ ...p }));
    this.presets.sort((a, b) => b.priority - a.priority);
    this.cd = new CooldownTable(opts.cooldownRules ?? DEFAULT_COOLDOWN_RULES);
    this.noiseFloor = opts.noiseFloor ?? FALLBACK_NOISE_FLOOR;
    this.mode = opts.mode ?? MODES.default_mode;
    this.placement = opts.placement ?? 'hand';
    this.buf = new SampleBuffer(BUFFER_SEC, BUFFER_CAPACITY);
  }

  /** 静止门随噪声底浮动：噪声大的机型，静止门也跟着抬高。 */
  get stillGate(): number {
    return Math.max(0.8, this.noiseFloor * 3);
  }

  get currentNoiseFloor(): number {
    return this.noiseFloor;
  }

  get isCalibrating(): boolean {
    return this.calibrating;
  }

  /** 上一次静置校准是否可信。跑动中校准会被判为不可信并回退默认值。 */
  get calibrationReliable(): boolean {
    return this.calibReliable;
  }

  get currentMode(): ModeId {
    return this.mode;
  }

  get currentPlacement(): Placement {
    return this.placement;
  }

  get modeDefinition() {
    return MODES.modes[this.mode];
  }

  /**
   * 切换使用模式。
   *
   * 模式是用户显式选择的结果，不做自动切换——自动切换会在用户正用手势时
   * 突然把它们关掉，用户只会认为是 bug（或者更糟：以为动作失灵了）。
   */
  setMode(mode: ModeId, placement?: Placement): void {
    if (placement) this.placement = placement;
    if (this.mode === mode) return;
    this.mode = mode;
    // 换模式等于换一套物理语义，锁与持续计时全部失效，必须清空。
    this.lock = null;
    this.holdSince.clear();
    this.suggestSince = null;
    this.suggestShown = false;
    this.autoPauseArmed = false;
    // 缓冲必须一起清。不清的话，切到跑步的瞬间窗口里还留着"上一秒在甩手机"
    // 的位移累积——新模式的预设会拿着旧模式的数据做第一次判定。
    this.buf.clear();
    // 进入跑步模式时把基准时间设成"现在"，否则站着不动 6 秒就会被判定为
    // "跑完停下"而自动暂停，而用户其实还没开始跑。
    const t = this.buf.last?.t ?? 0;
    this.lastActiveAt = t;
  }

  setPlacement(p: Placement): void {
    if (this.placement === p) return;
    this.placement = p;
    this.lock = null;
    this.holdSince.clear();
  }

  /**
   * 硬件步检测可用性。机型不支持 / 权限被拒 / 退回 web 通道时都是 false，
   * 此时所有步相关的否决逻辑整体旁路——降级路径必须与"从未有过这个功能"完全一致，
   * 否则排障时要同时考虑两套行为。
   */
  setStepDetectorAvailable(available: boolean): void {
    this.stepDetectorOn = available;
    if (!available) this.stepTimes = [];
  }

  /**
   * 喂一个硬件步事件（脚触地时刻，与样本同一时钟基准）。
   * 只在提示跑步时做"有没有在走"的存在性判断，不算硬件步频——
   * 检测器有最高 2s 的上报延迟，间隔测量在这个延迟下抖得没法用。
   */
  noteStep(t: number): void {
    this.stepTimes.push(t);
    const cutoff = t - 4;
    while (this.stepTimes.length > 0 && this.stepTimes[0] < cutoff) this.stepTimes.shift();
  }

  /**
   * 硬件步确认：步检测可用时，最近 stale_ms 内必须有步事件才允许提示跑步。
   *
   * 这是对"乘车颠簸凑出假跑步节律"的否决——Android 文档明确步检测器
   * 不应在车内触发，加速度节律说在跑而硬件没有步，信硬件。
   * stale_ms（默认 3s）大于检测器官方延迟上限（2s），真跑步不会被误伤。
   */
  private hwConfirmsRun(t: number): boolean {
    if (!this.stepDetectorOn) return true;
    if (this.stepTimes.length === 0) return false;
    const stale = (MODES.activity.hw_step?.stale_ms ?? 3000) / 1000;
    return t - this.stepTimes[this.stepTimes.length - 1] <= stale;
  }

  /** 当前模式下、当前放置位置可用的预设（含 enabled 状态，供 UI 渲染）。 */
  getModePresets(): Preset[] {
    return this.activePresets().map((p) => ({ ...p }));
  }

  /** 全部预设，跨模式。设置页或调试用。 */
  getPresets(): Preset[] {
    return this.presets.map((p) => ({ ...p }));
  }

  /** 喂一个原始采样，返回补齐了去重力结果的样本。 */
  ingest(raw: RawSample): Sample {
    const prev = this.buf.last;
    const dt = prev ? clampDt(raw.t - prev.t) : 1 / 60;
    const g = this.grav.update(raw.ax, raw.ay, raw.az, dt);
    const lx = raw.ax - g[0];
    const ly = raw.ay - g[1];
    const lz = raw.az - g[2];
    const sample: Sample = {
      ...raw,
      lx,
      ly,
      lz,
      grav: [g[0], g[1], g[2]],
    };
    this.buf.push(sample);
    if (this.calibrating) this.calibMags.push(hypot3(lx, ly, lz));
    return sample;
  }

  /**
   * 开始静置校准。
   *
   * 跑步模式下直接拒绝：跑动中测到的不是噪声底，是冲击幅度。写进去之后
   * 噪声底会被抬到 3.0，stillGate 变成 9，加速度永远落不进"静止"，
   * 速度衰减永不触发，位移无界漂移——而且退回日常模式后也恢复不了，
   * 因为校准值已经持久化了。这一条比"校准不准"严重得多，所以是硬拒绝。
   */
  startCalibration(): boolean {
    if (this.mode === 'run') return false;
    this.calibrating = true;
    this.calibMags = [];
    this.calibReliable = true;
    this.buf.clear();
    this.grav.reset();
    this.lock = null;
    this.holdSince.clear();
    return true;
  }

  /**
   * 结束校准，返回本机噪声底。
   * 取 90 分位而不是 RMS：手不可能完全不动，偶尔的磕碰尖峰会把 RMS 拉高，
   * 分位数对这类离群点不敏感。
   */
  finishCalibration(): number {
    this.calibrating = false;
    if (this.calibMags.length >= 10) {
      const s = [...this.calibMags].sort((a, b) => a - b);
      const p90 = s[Math.floor(s.length * 0.9)];
      if (p90 > CALIBRATION_MOTION_LIMIT) {
        // 校准期间在动（隔着一层 startCalibration 的检查也挡不住这种情况：
        // 用户点了校准之后才开始跑，或者边跑边点）。宁可放弃这次校准值，
        // 也不要写一个会让后续判定永久失准的数进去。
        this.calibReliable = false;
        this.noiseFloor = FALLBACK_NOISE_FLOOR;
      } else {
        this.calibReliable = true;
        this.noiseFloor = clamp(p90, 0.15, 3.0);
      }
    }
    this.reset();
    return this.noiseFloor;
  }

  /** 灵敏度倍率 0.6~1.6：只作用于幅度/跨度门的下限，不改契约数据。 */
  setSensitivity(scale: number): void {
    this.sensitivity = clamp(scale, 0.6, 1.6);
  }

  get currentSensitivity(): number {
    return this.sensitivity;
  }

  setPresetEnabled(id: string, enabled: boolean): void {
    const p = this.presets.find((x) => x.id === id);
    if (p) p.enabled = enabled;
  }

  /** 每帧调一次：算出特征、跑完所有门、最多返回一个事件。 */
  evaluate(): FrameDiagnostics {
    const rejections: FrameDiagnostics['rejections'] = [];
    const cooled: string[] = [];
    const last = this.buf.last;
    if (!last || this.calibrating) {
      return {
        features: null,
        rejections,
        event: null,
        cooled,
        modeTransition: null,
        running: this.running,
      };
    }

    const hz = this.buf.sampleHz();
    const t = last.t;

    // 1) 节律与环境态。每帧都算，即使当前模式用不到：
    //    日常模式要靠它提示"你好像在跑步"，跑步模式要靠它判断是否已停下。
    const rhythm = computeRhythm(this.buf.tail(RHYTHM_WINDOW_SEC), this.noiseFloor);
    const runNow = this.rhythmIsRun(rhythm);
    // 跑步态的维护（退出判定、自动暂停）只信节律，硬件步检测不参与——
    // 不对称是刻意的，见 hwConfirmsRun 的注释与 motion_modes.json 的 hw_step_note。
    this.running = runNow;
    if (runNow) {
      this.lastActiveAt = t;
      if (this.mode === 'run') this.autoPauseArmed = true;
    }
    // 提示跑步才做硬件交叉验证：乘车颠簸能凑出节律，但凑不出硬件步事件。
    let modeTransition = this.updateSuggest(runNow && this.hwConfirmsRun(t), t);

    // 2) 只给当前模式的预设算特征。不在模式里的预设连特征都不参与，
    //    这是"互斥"的落地处：不是算出来再丢弃，而是根本不算。
    const actives = this.activePresets();
    const cache = new Map<number, SensorFeatures>();
    for (const p of actives) {
      const w = windowMsOf(p);
      if (cache.has(w)) continue;
      const f = computeFeatures(this.buf.tail(w / 1000), {
        windowSec: w / 1000,
        stillGate: this.stillGate,
        noiseFloor: this.noiseFloor,
      });
      f.sampleHz = hz;
      cache.set(w, f);
    }
    // 兜底：当前模式下一个预设都没启用时，UI 仍要能显示数据。
    if (cache.size === 0) {
      const f = computeFeatures(this.buf.tail(0.4), {
        windowSec: 0.4,
        stillGate: this.stillGate,
        noiseFloor: this.noiseFloor,
      });
      f.sampleHz = hz;
      cache.set(400, f);
    }

    // 节律字段合并进每个窗口特征：这些量来自独立的 2s 窗口，
    // 与动作窗口的时间尺度本来就不同，合并是为了让契约条件能统一读取。
    for (const f of cache.values()) {
      f.cadenceHz = rhythm.cadenceHz;
      f.cadenceStability = rhythm.stability;
      f.impactRms = rhythm.impactRms;
      f.narrowPulseWidthMs = rhythm.narrowPulseWidthMs;
      f.tapPairs = rhythm.tapPairs;
      // 抬腕用瞬时量（距上一个冲击峰多久），不用基于 2s 窗口继续增长的量——
      // 后者有整个窗口的滞后，等它满足时手已经放下了。
      f.rhythmBreakMs = rhythm.sinceLastPeakMs;
    }

    this.updateLock(cache);

    let event: ActionEvent | null = null;
    for (const p of actives) {
      if (!p.enabled) continue;
      const f = cache.get(windowMsOf(p));
      if (!f) continue;

      if (this.lock && this.lock.action === p.action) {
        rejections.push({ presetId: p.id, gate: 'lock', detail: '同一动作尚未结束' });
        continue;
      }

      const r = this.checkPreset(p, f, t);
      if (!r.ok) {
        rejections.push({ presetId: p.id, gate: r.gate ?? 'unknown', detail: r.detail ?? '' });
        continue;
      }

      if (!this.cd.allow(p.action, t)) {
        cooled.push(p.action);
        rejections.push({
          presetId: p.id,
          gate: 'cooldown',
          detail: `还需 ${this.cd.remaining(p.action, t).toFixed(2)}s`,
        });
        // 冷却拒绝就整体中止，不让低优先级预设"捡漏"同一个动作
        break;
      }

      event = { action: p.action, presetId: p.id, presetName: p.name, t, features: f };
      this.lock = { action: p.action, presetId: p.id, miss: 0 };
      break;
    }

    // 3) 自动暂停。只在没有其它事件时做，避免抢占用户主动操作。
    if (!event) {
      const auto = this.maybeAutoPause(t);
      if (auto) {
        event = auto;
        modeTransition = 'auto_paused';
      }
    }

    const firstEnabled = actives.find((p) => p.enabled);
    let features: SensorFeatures | null = null;
    if (firstEnabled) features = cache.get(windowMsOf(firstEnabled)) ?? null;
    if (!features && cache.size > 0) features = Array.from(cache.values())[0];

    return { features, rejections, event, cooled, modeTransition, running: this.running };
  }

  reset(): void {
    this.buf.clear();
    this.grav.reset();
    this.cd.reset();
    this.lock = null;
    this.holdSince.clear();
    this.suggestSince = null;
    this.suggestShown = false;
    this.autoPauseArmed = false;
    // 步事件清空但可用性保留：校准后用户还在不在走，与通道是否可用无关。
    this.stepTimes = [];
  }

  // ── 内部 ──────────────────────────────────────────────

  /** 当前模式 + 放置位置允许的预设。这是"互斥"的唯一入口。 */
  private activePresets(): PresetWithWindow[] {
    const def = MODES.modes[this.mode];
    const allowed = def.placements?.[this.placement]?.presets;
    return this.presets.filter((p) => {
      if (!modesOf(p).includes(this.mode)) return false;
      if (allowed && !allowed.includes(p.id)) return false;
      return true;
    });
  }

  /** 是否满足跑步节律：步频在区间内、间隔稳定、冲击幅度足够。 */
  private rhythmIsRun(r: RhythmResult): boolean {
    const c = MODES.activity.enter;
    return (
      r.periodic &&
      r.cadenceHz >= c.cadence_hz_min &&
      r.cadenceHz <= c.cadence_hz_max &&
      r.stability <= c.stability_max &&
      r.impactRms >= c.impact_rms_min
    );
  }

  /**
   * 日常模式下检测到持续跑步时提示一次。
   * 只提示不切换：切换会立刻关掉用户正在用的全部手势，必须由用户决定。
   */
  private updateSuggest(runNow: boolean, t: number): FrameDiagnostics['modeTransition'] {
    if (this.mode === 'run') {
      this.suggestSince = null;
      return null;
    }
    if (!runNow) {
      // 节律断掉就重新武装提示。加 3s 的隐式间隔在 suggestShown 上体现：
      // 同一次连续跑步只提示一次，跑完再跑才会提示第二次。
      this.suggestSince = null;
      if (this.lastActiveAt > 0 && (t - this.lastActiveAt) * 1000 > 3000) {
        this.suggestShown = false;
      }
      return null;
    }
    if (this.suggestSince === null) this.suggestSince = t;
    if (this.suggestShown) return null;
    if ((t - this.suggestSince) * 1000 < MODES.activity.enter.hold_ms) return null;
    this.suggestShown = true;
    return 'suggest_run';
  }

  /**
   * 退出跑步态且静止够久 → 自动暂停。
   *
   * idle 时长取自模式契约的 on_exit.idle_ms（6s），比退出判定（5s）更晚，
   * 这样短暂停顿不会误暂停：状态可以先退回，音乐不必跟着停。
   */
  private maybeAutoPause(t: number): ActionEvent | null {
    const def = MODES.modes[this.mode];
    if (!def.on_exit) return null;
    if (!this.autoPauseArmed) return null;
    if ((t - this.lastActiveAt) * 1000 < def.on_exit.idle_ms) return null;
    if (!this.cd.allow(def.on_exit.action, t)) return null;

    this.autoPauseArmed = false;
    const f: SensorFeatures = {
      timestamp: t,
      source: 'sensor',
      sampleHz: this.buf.sampleHz(),
      windowFull: true,
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
      noiseFloor: this.noiseFloor,
      holdMs: {},
      cadenceHz: 0,
      cadenceStability: 0,
      impactRms: 0,
      narrowPulseWidthMs: NARROW_PULSE_UNMEASURABLE,
      tapPairs: 0,
      rhythmBreakMs: Math.min(99999, (t - this.lastActiveAt) * 1000),
    };
    return {
      action: def.on_exit.action,
      presetId: '__mode_exit__',
      presetName: '停下自动暂停',
      t,
      features: f,
    };
  }

  /**
   * 动作锁定的解除：
   * 触发后只要"动作形态"仍然成立就一直锁着，连续 direction_lock_frames 帧不成立才解锁。
   * 这与 PC 端 MISS_FRAMES_TO_UNLOCK = 3 是同一个思路。
   *
   * 模式切换后原锁定预设可能已经不在匹配池里，此时必须直接解锁，
   * 否则会永久卡住那个 action。
   */
  private updateLock(cache: Map<number, SensorFeatures>): void {
    if (!this.lock) return;
    const p = this.presets.find((x) => x.id === this.lock!.presetId);
    if (!p || !modesOf(p).includes(this.mode) || cache.get(windowMsOf(p)) === undefined) {
      this.lock = null;
      return;
    }
    const f = cache.get(windowMsOf(p));
    const stillInGesture = f ? this.morphologySatisfied(p, f) : false;
    if (stillInGesture) {
      this.lock.miss = 0;
      return;
    }
    this.lock.miss++;
    if (this.lock.miss >= this.gates.direction_lock_frames) this.lock = null;
  }

  /** 只看"形态"条件（不含 hold_ms），用于判断手势是否仍在进行。 */
  private morphologySatisfied(p: Preset, f: SensorFeatures): boolean {
    for (const [key, cond] of Object.entries(p.conditions)) {
      if (key === 'hold_ms') continue;
      const getter = FEATURE_GETTERS[key];
      if (!getter) return false;
      const v = getter(f);
      if (v === null) return false;
      if (!this.satisfies(key, v, cond)) return false;
    }
    return true;
  }

  private checkPreset(p: Preset, f: SensorFeatures, t: number): GateResult {
    if (this.gates.require_window_full && !f.windowFull) {
      return { ok: false, gate: 'window', detail: '窗口未填满' };
    }
    if (f.sampleHz > 0 && f.sampleHz < this.gates.min_sample_hz) {
      return {
        ok: false,
        gate: 'frequency',
        detail: `${f.sampleHz.toFixed(0)}Hz < ${this.gates.min_sample_hz}Hz`,
      };
    }

    for (const [key, cond] of Object.entries(p.conditions)) {
      if (key === 'hold_ms') continue;
      const getter = FEATURE_GETTERS[key];
      if (!getter) return { ok: false, gate: key, detail: '契约里没有这个特征' };
      const v = getter(f);
      if (v === null) return { ok: false, gate: key, detail: '特征不可用' };
      if (!this.satisfies(key, v, cond)) {
        this.holdSince.delete(p.id);
        return { ok: false, gate: key, detail: describeMiss(v, cond) };
      }
    }

    const hold = p.conditions['hold_ms'];
    if (hold?.min !== undefined) {
      let since = this.holdSince.get(p.id);
      if (since === undefined) {
        since = t;
        this.holdSince.set(p.id, since);
      }
      const held = (t - since) * 1000;
      f.holdMs[p.id] = held;
      if (held < hold.min) {
        return { ok: false, gate: 'hold_ms', detail: `保持 ${held.toFixed(0)}/${hold.min}ms` };
      }
    }

    return { ok: true };
  }

  private satisfies(key: string, v: number | string, cond: Condition): boolean {
    if (cond.in) return cond.in.includes(v);
    if (typeof v !== 'number') return true;

    let lo = cond.min;
    let hi = cond.max;

    // 两个"主动作量"的下限受噪声底与灵敏度共同约束。
    // 这是运行时可调参数，不是把契约阈值抄一份到这里——契约值仍是基准。
    if (key === 'lin_acc_peak' && lo !== undefined) {
      lo = Math.max(lo, this.noiseFloor * this.gates.noise_floor_multiplier);
    }
    if ((key === 'lin_acc_peak' || key === 'lateral_span') && lo !== undefined) {
      lo = lo / this.sensitivity;
    }

    if (lo !== undefined && v < lo) return false;
    if (hi !== undefined && v > hi) return false;
    return true;
  }
}

function describeMiss(v: number | string, cond: Condition): string {
  if (cond.in) return `${String(v)} 不在 ${cond.in.join('/')} 内`;
  if (typeof v !== 'number') return String(v);
  const parts: string[] = [];
  if (cond.min !== undefined) parts.push(`≥${cond.min}`);
  if (cond.max !== undefined) parts.push(`≤${cond.max}`);
  return `${v.toFixed(3)} 未达 ${parts.join(' 且 ')}`;
}
