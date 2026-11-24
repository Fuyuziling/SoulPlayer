import type { RawSample } from '../core/types';

/**
 * 传感器接入层。
 *
 * 所有平台差异都收在这个文件里，core/ 完全不知道 DeviceMotion 的存在。
 * 这样后续换成 Capacitor 原生插件时，只需要替换这个文件。
 *
 * 三个必须处理好的现实问题：
 *
 * 1. **安全上下文**。DeviceMotion 只在 https 或 localhost 下可用。
 *    用局域网 http://192.168.x.x 打开会静默拿不到任何事件——不是报错，是没数据。
 *    这种情况必须明确告诉用户，否则会以为是传感器坏了。
 *
 * 2. **iOS 要用户手势授权**。iOS 13+ 的 requestPermission() 必须在用户点击的
 *    同一调用栈里触发，页面加载时自动调用会被拒绝。所以 start() 只能由按钮触发。
 *
 * 3. **旋转向量可能缺失**。部分机型/浏览器不给 rotationRate，此时 hasRotation
 *    为 false，翻面判定会退化（只能靠重力倾角），需要降级而不是崩。
 */

import type { MotionSourceLike, SensorStatus } from './sensor-types';

export type { SensorStatus };

const D2R = Math.PI / 180;

type MotionEventCtor = typeof DeviceMotionEvent & {
  requestPermission?: () => Promise<'granted' | 'denied'>;
};

export class MotionSource implements MotionSourceLike {
  private handler: ((e: DeviceMotionEvent) => void) | null = null;
  private hzWindow: number[] = [];
  private lastRotationSeen = false;
  private status: SensorStatus = {
    supported: typeof window !== 'undefined' && 'DeviceMotionEvent' in window,
    secureContext: typeof window !== 'undefined' ? window.isSecureContext : false,
    permission: 'prompt',
    active: false,
    hz: 0,
    hasRotation: false,
    message: '',
    channel: 'web-devicemotion',
  };

  onSample: ((s: RawSample) => void) | null = null;
  /** 采样频率变化时回调，UI 用来显示实时 Hz。 */
  onHz: ((hz: number) => void) | null = null;

  getStatus(): SensorStatus {
    // hasRotation 反映"实际收到过陀螺仪数据"，而不是"浏览器声称支持"。
    // 翻面判定强依赖它，UI 上要能看出这台设备是不是缺了这块。
    return { ...this.status, hz: this.currentHz(), hasRotation: this.lastRotationSeen && this.status.active };
  }

  /** 必须由用户手势（按钮点击）触发，iOS 的权限请求才有意义。 */
  async start(): Promise<SensorStatus> {
    if (!this.status.supported) {
      this.status = { ...this.status, message: '此浏览器不支持 DeviceMotion' };
      return this.getStatus();
    }
    if (!this.status.secureContext) {
      this.status = {
        ...this.status,
        permission: 'unsupported',
        message: '传感器需要 https 或 localhost 才能启用。当前是 http 局域网地址，浏览器会静默拒绝。',
      };
      return this.getStatus();
    }

    const ctor = window.DeviceMotionEvent as MotionEventCtor;
    if (typeof ctor.requestPermission === 'function') {
      try {
        // requestPermission 必须在用户手势中调用才有响应；无手势时（比如程序
        // 误触发、或某些桌面浏览器实现了这个 API 但没有权限弹窗）它会永久挂起。
        // 挂起会吊死整个启动流程，所以加 3.5s 兜底：超时按"继续走"处理——
        // 真手机上不会走到这一步，桌面浏览器反正也没有传感器数据。
        const granted = await Promise.race([
          ctor.requestPermission(),
          new Promise<'granted'>((resolve) => window.setTimeout(() => resolve('granted'), 3500)),
        ]);
        if (granted !== 'granted') {
          this.status = { ...this.status, permission: 'denied', message: '系统拒绝了动作传感器权限' };
          return this.getStatus();
        }
      } catch {
        this.status = { ...this.status, permission: 'denied', message: '权限请求失败，需在设置中允许"动作与方向访问"' };
        return this.getStatus();
      }
    }
    this.status = { ...this.status, permission: 'granted' };

    if (this.handler) return this.getStatus();

    this.handler = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity;
      if (!a || a.x === null || a.y === null || a.z === null) return;
      const r = e.rotationRate;
      const hasRot = !!r && r.alpha !== null && r.beta !== null && r.gamma !== null;
      if (hasRot) this.lastRotationSeen = true;
      const t = performance.now() / 1000;
      this.pushHz(t);
      this.onSample?.({
        t,
        ax: a.x,
        ay: a.y,
        az: a.z,
        // rotationRate 的原生单位是 deg/s，契约用 rad/s
        gx: hasRot ? (r!.beta ?? 0) * D2R : 0,
        gy: hasRot ? (r!.gamma ?? 0) * D2R : 0,
        gz: hasRot ? (r!.alpha ?? 0) * D2R : 0,
        hasRotation: hasRot,
      });
    };
    window.addEventListener('devicemotion', this.handler, { passive: true });
    this.status = { ...this.status, active: true, message: '传感器已启用' };
    return this.getStatus();
  }

  stop(): void {
    if (this.handler) {
      window.removeEventListener('devicemotion', this.handler);
      this.handler = null;
    }
    this.hzWindow = [];
    this.status = { ...this.status, active: false, message: '传感器已停止' };
  }

  /** 只统计最近 40 个时间戳，避免频率显示跳变。 */
  private pushHz(t: number): void {
    this.hzWindow.push(t);
    if (this.hzWindow.length > 40) this.hzWindow.shift();
    if (this.hzWindow.length >= 10) {
      const span = this.hzWindow[this.hzWindow.length - 1] - this.hzWindow[0];
      if (span > 0) this.onHz?.((this.hzWindow.length - 1) / span);
    }
  }

  private currentHz(): number {
    if (this.hzWindow.length < 2) return 0;
    const span = this.hzWindow[this.hzWindow.length - 1] - this.hzWindow[0];
    return span > 0 ? (this.hzWindow.length - 1) / span : 0;
  }
}
