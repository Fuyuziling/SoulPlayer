import { Capacitor } from '@capacitor/core';
import { MotionSource } from './sensors';
import { NativeMotionSource } from './native-sensors';
import type { MotionSourceLike, SensorStatus } from './sensor-types';
import type { RawSample } from '../core/types';

/**
 * 采集通道选择器。
 *
 * APK 内用原生 SensorManager（见 native-sensors.ts 里为什么不用 devicemotion），
 * 浏览器/PWA 用 devicemotion。判定链与 UI 完全不感知差别——这是当初把平台差异
 * 收进 platform/ 那一层的直接回报。
 */
export function createMotionSource(): MotionSourceLike {
  return Capacitor.isNativePlatform() ? new ResilientMotionSource() : new MotionSource();
}

/**
 * 原生优先、失败退回浏览器的双通道源。
 *
 * 为什么需要这层：APK 里理论上一定有原生插件，但现实中会遇到
 * 系统 WebView 被替换、插件桥接异常、个别 ROM 权限拦截等情况。
 * 那时如果只留"启动失败"一条路，用户看到的现象是应用打不开动作功能，
 * 而实际上 devicemotion 那条路在部分机型上仍然是通的。
 * 两条都试过才叫"这台设备真的不行"。
 */
class ResilientMotionSource implements MotionSourceLike {
  onSample: ((s: RawSample) => void) | null = null;
  onHz: ((hz: number) => void) | null = null;
  /** 只有原生通道有硬件步检测。退回 web 通道时它是 null——判定链按"不可用"中性降级。 */
  onStep: ((t: number) => void) | null = null;
  onStepDetector: ((available: boolean) => void) | null = null;

  private native = new NativeMotionSource();
  private web = new MotionSource();
  private active: MotionSourceLike = this.native;

  constructor() {
    for (const src of [this.native, this.web]) {
      src.onSample = (s) => this.onSample?.(s);
      src.onHz = (hz) => this.onHz?.(hz);
    }
    // 步检测事件只从原生通道来。注意：退回 web 通道时不撤掉已有回调，
    // 原生通道已停采不会再发事件，残留的"最后状态"由引擎自己解释。
    this.native.onStep = (t) => this.onStep?.(t);
    this.native.onStepDetector = (available) => this.onStepDetector?.(available);
  }

  getStatus(): SensorStatus {
    return this.active.getStatus();
  }

  async start(): Promise<SensorStatus> {
    const nativeStatus = await this.native.start();
    if (nativeStatus.active) {
      this.active = this.native;
      return nativeStatus;
    }

    const why = nativeStatus.message;
    const webStatus = await this.web.start();
    this.active = this.web;
    if (webStatus.active) {
      return { ...webStatus, message: `原生通道不可用（${why}），已退回浏览器通道` };
    }
    return { ...webStatus, message: webStatus.message || why };
  }

  stop(): void {
    this.native.stop();
    this.web.stop();
  }
}
