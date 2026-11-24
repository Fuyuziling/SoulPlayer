/**
 * 传感器判定链的类型定义。
 *
 * 字段命名与 contracts/sensor_feature_schema.json 一一对应，改字段先改契约再改这里。
 * 坐标系统一采用 W3C DeviceMotion 约定：屏幕竖直握持、正面朝向用户时，
 * x 向右、y 向上、z 垂直屏幕向外。
 */

/**
 * 使用模式。
 *
 * 二者是**互斥的判定基线**，不是"可选动作集"：跑步时的持续节律会同时破坏
 * 日常手势的幅度门（冲击 15~25 m/s^2 与甩动重叠）、稳定性门（摆臂让 roll
 * 在 ±30~50° 波动，std 远大于倾斜判定允许的 3°）和校准基线（跑动中测到的
 * 噪声底是冲击幅度，会把 afterwards 的精度永久拉低）。
 * 所以切换模式时，不属于当前模式的预设整体移出匹配池。
 */
export type ModeId = 'still' | 'run';

/**
 * 手机在跑动中的放置位置。决定跑步模式下哪些手势物理上做得出：
 * 口袋里敲不到机身背面，抬腕也没有姿态意义。
 */
export type Placement = 'hand' | 'pocket';

/** 一个原始 IMU 采样点。 */
export interface RawSample {
  /** 单调时钟秒数（performance.now() / 1000）。 */
  t: number;
  /** 含重力的加速度，m/s^2。 */
  ax: number;
  ay: number;
  az: number;
  /** 角速度，rad/s（原始事件是 deg/s，入队前已转换）。 */
  gx: number;
  gy: number;
  gz: number;
  /** 设备是否提供了 rotationRate（部分机型/浏览器没有）。 */
  hasRotation: boolean;
}

/** 入队时补齐派生字段的样本。 */
export interface Sample extends RawSample {
  /** 去重力后的线性加速度，m/s^2。 */
  lx: number;
  ly: number;
  lz: number;
  /** 该样本入队时刻的重力估计（设备坐标）。姿态角从这里算，不是从原始读数算。 */
  grav: [number, number, number];
}

/**
 * 窗口特征。全部由 core/features.ts 在滑动窗口上重算，
 * 不做跨窗口累加——积分漂移一旦累积，判定就会越来越飘。
 */
export interface SensorFeatures {
  timestamp: number;
  source: 'sensor';
  sampleHz: number;
  windowFull: boolean;

  linAccPeak: number;
  linAccRms: number;
  velMagPeak: number;
  velHPeak: number;
  verticalRatio: number;

  dominantAxis: 'x' | 'y' | 'z' | null;
  lateralSpan: number;
  lateralDirection: -1 | 0 | 1;
  /**
   * 主轴速度穿过零点的次数。这是"甩动"与"摇动"最本质的区分：
   * 甩动是单向的，速度从 0 升到峰值再回 0，反向 0~1 次；
   * 摇一摇是往复的，速度反复正负翻转，0.36s 内就能翻转 3 次以上。
   * 只看位移跨度会被"边摇边走"骗过去，看速度方向翻转不会。
   */
  velReversals: number;
  /**
   * 主轴加速度的符号反转次数（带 3 m/s^2 死区）。
   *
   * 比 velReversals 更早、更本质：甩动是单次冲量，加速度全程同号，反转 0 次；
   * 摇动是往复，0.36s 内加速度必然正负交替 3 次以上。
   * 而速度反转要等积分累积出方向变化才看得出来，动作刚起步和刚结束时
   * 都会短暂回到 0~1 次，正好在两端留出误判窗口。
   */
  accSignChanges: number;
  dispX: number;
  dispY: number;

  gyroMagPeak: number;
  pitchDeg: number;
  rollDeg: number;
  /**
   * 窗口内横滚角的标准差（度）。
   *
   * 用来区分"有意倾斜"与"甩动后的重力估计恢复期"：两者都会让瞬时 roll
   * 越过阈值，但真倾斜时手机静止、roll 几乎不动（std < 2°），
   * 而甩动之后重力估计要从被带偏的状态慢慢收敛回真实值，
   * roll 会连续滑过几十度（std > 10°）。
   * 只卡瞬时值的话，快速甩动会被判成"倾斜调音量"。
   */
  rollStd: number;
  flipAngleDeg: number;
  /** 窗口首末俯仰角之差（不带陀螺仪补偿），用于把"翻面"和"甩动"分开。 */
  tiltDeltaDeg: number;
  faceState: 'up' | 'down' | 'edge' | 'unknown';
  /**
   * 窗口首末的"屏幕朝向"是否发生了穿越朝下的变化。
   * 这是翻面能被可靠识别、而"把手机放到桌上"不会被误判的关键：
   *   竖直握持(90°) → 朝下(180°)   = to_down    ← 翻面
   *   朝下(180°) → 竖直(90°)       = from_down  ← 翻回来
   *   竖直握持(90°) → 桌上朝上(0°)  = none       ← 放下手机，不算
   * 转角都是 90°，只有朝向变化的方向能区分这两件事。
   */
  faceTransition: 'none' | 'to_down' | 'from_down';

  peakCount: number;
  peakIntervalMs: number;
  stillRatio: number;
  isStill: boolean;
  noiseFloor: number;

  // ── 节律特征（来自独立的 2s 窗口，不随预设 window_ms 变化）──────
  /**
   * 步频（Hz）。0 表示节律不成立（峰值太少或间隔不稳）。
   * 跑步 2.4~3.6Hz（144~216 步/分），散步约 1.8Hz。
   */
  cadenceHz: number;
  /**
   * 峰值间隔的 MAD / 中位数。
   * 跑步的步频非常稳定（通常 <0.15），随手摇晃没有稳定节律（>0.3）。
   * 这是区分"真在跑"和"手在乱动"的关键——两者幅度可以一样大。
   */
  cadenceStability: number;
  /** 节律窗口内线性加速度的 RMS，m/s^2。跑步 5~15，静置 0.3~1.0，轻晃 1~3。 */
  impactRms: number;
  /**
   * 最近一次窄脉冲的宽度（ms）。测不到时为 NARROW_PULSE_UNMEASURABLE。
   *
   * 这是"敲击机身"与"脚步冲击"唯一可靠的区分量：两者峰值都落在
   * 15~25 m/s^2，但指尖敲击只持续 20~50ms，脚步经身体与衣物传导后
   * 宽度在 80ms 以上。峰值完全没有判别力，宽度有。
   */
  narrowPulseWidthMs: number;
  /** 成对窄脉冲的对数（两击间隔 120~320ms）。脚步是单侧周期性的，凑不出"对"。 */
  tapPairs: number;
  /**
   * 节律中断时长（ms）。由引擎跨帧维护后写回，特征层不重复计算——
   * "中断了多久"是状态，不是某一帧的窗口属性。
   */
  rhythmBreakMs: number;

  /** 条件持续满足的毫秒数，由引擎按预设逐帧累计（不在特征层算）。 */
  holdMs: Record<string, number>;
}

/** 一次判定输出的动作事件。 */
export interface ActionEvent {
  action: string;
  presetId: string;
  presetName: string;
  /** 触发时刻（秒）。 */
  t: number;
  features: SensorFeatures;
}

/** 一帧判定的诊断信息，用于 UI 回显"为什么没触发"。 */
export interface FrameDiagnostics {
  features: SensorFeatures | null;
  /** 每个预设被拒的原因（按优先级排列，只保留第一道未通过的门）。 */
  rejections: Array<{ presetId: string; gate: string; detail: string }>;
  /** 本帧是否有动作触发。 */
  event: ActionEvent | null;
  /** 冷却导致的额外拒绝。 */
  cooled: string[];
  /**
   * 本帧需要告知 UI 的提示。
   *
   * 刻意不做"自动切换模式"：模式由用户显式选择，自动切换会在用户正用手势时
   * 突然关掉它们，用户只会认为是 bug。所以检测到持续跑步时只提示（suggest_run），
   * 切换动作留给用户。
   */
  modeTransition: 'suggest_run' | 'auto_paused' | null;
  /** 当前节律是否成立（正在跑）。UI 用来显示"已检测到跑步节律"。 */
  running: boolean;
}

/** 单条预设的条件比较器。 */
export interface Condition {
  min?: number;
  max?: number;
  in?: Array<string | number>;
}

export interface Preset {
  id: string;
  name: string;
  gesture: string;
  action: string;
  enabled: boolean;
  priority: number;
  description?: string;
  calibrate?: string;
  conditions: Record<string, Condition>;
  /**
   * 归属模式。缺省视为 ['still']——保守默认，新增预设不会因为忘记声明
   * 就悄悄出现在跑步模式里，那是最难查的一类误触。
   */
  modes?: ModeId[];
}

export interface CooldownRule {
  self_cd?: number;
  peer_cd?: number;
  peers?: string[];
  also_block?: string[];
  /**
   * also_block 的压制时长（秒），缺省取 self_cd。
   *
   * 必须显式给，因为 PC 端"连带登记时间戳"的做法有个隐含缺陷：压制时长
   * 实际上由**被压制动作自己的 self_cd** 决定，而不是由压制者决定。
   * 摇一摇想压制切歌 1.8s，但 player_next 的 self_cd 只有 0.35s，
   * 于是压制 0.35s 就失效了，余波照样切歌。压多久应该由压制者说了算。
   */
  block_cd?: number;
}

export interface GatesConfig {
  require_window_full: boolean;
  min_sample_hz: number;
  noise_floor_multiplier: number;
  direction_lock_frames: number;
  steady_reset_ms: number;
  /** 缓冲保留时长（毫秒）。各预设在此缓冲内取自己的子窗口。 */
  buffer_ms?: number;
}

/** 某个模式下、某种放置位置可用的预设。 */
export interface ModePlacement {
  label: string;
  presets: string[];
  note?: string;
}

export interface ModeDefinition {
  label: string;
  desc: string;
  mutex_note?: string;
  /** static=启动时静置校准；rhythm=禁止静置，改用节律基线。 */
  calibration: 'static' | 'rhythm';
  calibration_note?: string;
  /** waveform=三轴波形；cadence=节律条与步频。 */
  scope: 'waveform' | 'cadence';
  scope_note?: string;
  /** 动作 id → 振动节奏（毫秒数组）。 */
  haptic?: Record<string, number[]>;
  haptic_note?: string;
  placements?: Partial<Record<Placement, ModePlacement>>;
  /** 退出本模式后，若持续静止到 idle_ms 则发出该动作。 */
  on_exit?: { action: string; idle_ms: number; note?: string };
  /** 跑步模式的步频乐单匹配（三个预设文件夹切换）。 */
  music?: RunMusicConfig;
}

/**
 * 跑步乐单文件夹。三个文件夹在 contracts/motion_modes.json 里预设，
 * 曲库页直接把它们当容器展示——用户选文件夹再导入，不用逐首打标签。
 */
export type MusicGroup = 'calm' | 'light' | 'intense';
/** 曲目归属：三个文件夹之一，或 'none'（未分组，不参与跑步乐单）。 */
export type TrackGroup = MusicGroup | 'none';

export interface MusicFolder {
  id: MusicGroup;
  label: string;
  hint?: string;
}

export interface RunMusicLevels {
  /** 升入激烈档的步频（步/分）。 */
  enter_intense_bpm: number;
  /** 从激烈档退回轻松档的步频（步/分），低于 enter 构成滞回。 */
  exit_intense_bpm: number;
  /** 从舒缓档升入轻松档的步频（步/分）。 */
  exit_calm_bpm: number;
  /** 退回舒缓档的步频（步/分），低于 exit_calm 构成滞回。 */
  enter_calm_bpm: number;
}

export interface RunMusicConfig {
  /** 三个预设文件夹，顺序即 UI 展示顺序。 */
  folders: MusicFolder[];
  /** 三档切换的四阈值（两段双向滞回）。 */
  levels: RunMusicLevels;
  /** 目标档需连续满足的时长（ms）才触发切换。 */
  dwell_ms: number;
  /** 正在播的歌剩余超过该值就立即淡出切换，否则等播完（ms）。 */
  switch_remaining_ms: number;
  note?: string;
}

/**
 * 步频乐单匹配参数。滞回双阈值 + 连续驻留，见 motion_modes.json 的
 * modes.run.music.note——阈值含义与取舍理由都在那里，代码不重复。
 */

export interface ActivityConfig {
  enter: {
    cadence_hz_min: number;
    cadence_hz_max: number;
    stability_max: number;
    impact_rms_min: number;
    hold_ms: number;
  };
  exit: { idle_ms: number };
  enter_note?: string;
  exit_note?: string;
  impact_rms_min_note?: string;
  /**
   * 硬件步检测交叉验证（Android TYPE_STEP_DETECTOR）。
   * 只否决 suggest_run（乘车抑制），不参与跑步态退出——不对称是刻意的，
   * 见 motion_modes.json 的 hw_step_note。
   */
  hw_step?: { stale_ms: number };
  hw_step_note?: string;
}

export interface ModesConfig {
  version: number;
  default_mode: ModeId;
  modes: Record<ModeId, ModeDefinition>;
  activity: ActivityConfig;
}
