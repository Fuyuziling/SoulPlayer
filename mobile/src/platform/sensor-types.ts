/**
 * 传感器通道的公共类型。
 *
 * 单独抽一个文件是为了避免循环引用：web 通道（sensors.ts）和原生通道
 * （native-sensors.ts）都要用这些类型，但两者互不引用，由 motion.ts 负责选择。
 */

export interface SensorStatus {
  supported: boolean;
  secureContext: boolean;
  permission: 'granted' | 'denied' | 'prompt' | 'unsupported';
  active: boolean;
  hz: number;
  hasRotation: boolean;
  message: string;
  /** 实际走的是哪条采集通道。真机排障时这一项比任何日志都直接。 */
  channel: 'web-devicemotion' | 'native-sensor';
}

/** 两条通道必须共同满足的契约，app.ts 只依赖这个接口。 */
export interface MotionSourceLike {
  onSample: ((s: import('../core/types').RawSample) => void) | null;
  onHz: ((hz: number) => void) | null;
  /**
   * 硬件步事件（Android TYPE_STEP_DETECTOR，脚触地时刻，与样本同一时钟基准）。
   * 只有原生通道有；web 通道没有这个属性，保持 undefined = 判定链中性降级。
   */
  onStep?: ((t: number) => void) | null;
  /** 步检测可用性变化（权限批准/拒绝、机型不支持）。 */
  onStepDetector?: ((available: boolean) => void) | null;
  getStatus(): SensorStatus;
  start(): Promise<SensorStatus>;
  stop(): void;
}
