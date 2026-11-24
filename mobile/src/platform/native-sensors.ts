import { Capacitor, registerPlugin, type PluginListenerHandle } from '@capacitor/core';
import type { RawSample } from '../core/types';
import type { MotionSourceLike, SensorStatus } from './sensor-types';

/**
 * 原生传感器通道（Android APK 内使用）。
 *
 * 为什么打包成 APK 后不走 devicemotion：
 *
 * 1. **可用性不确定**。WebView 对 devicemotion 的支持取决于系统 WebView 版本，
 *    在部分机型上事件根本不触发——不报错，就是没数据，用户只会看到"传感器坏了"。
 * 2. **频率被限**。Chromium 把 devicemotion 合并到渲染节拍，通常封顶 60Hz；
 *    甩动全程只有 0.2~0.3s，60Hz 意味着只有十几帧，跨度与速度特征都算不准。
 * 3. **语义漂移**。rotationRate 的轴向约定在不同浏览器的 alpha/beta/gamma 上
 *    并不稳定，原生 SensorManager 给的 x/y/z 才是确定的设备坐标。
 *
 * 所以原生侧直接读 SensorManager，按 ~100Hz 成批推送，JS 侧只做时间轴换算，
 * 判定链（core/）完全不改。
 */

/** 原生侧一帧的线上格式。字段短名是为了压低跨桥 JSON 的体积。 */
interface NativeSample {
  /** 秒，SystemClock.elapsedRealtimeNanos / 1e9，单调递增。 */
  t: number;
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  /** 该帧是否配到了陀螺仪数据。 */
  rot: boolean;
}

interface NativeProbe {
  available: boolean;
  hasGyro: boolean;
  hasStepDetector?: boolean;
  message?: string;
}

interface SoulSensorsPlugin {
  isAvailable(): Promise<NativeProbe>;
  start(): Promise<NativeProbe>;
  stop(): Promise<void>;
  /** Android 10+ 申请 ACTIVITY_RECOGNITION（步检测需要）。拒绝也是正常返回。 */
  requestStepPermission(): Promise<{ granted: boolean; reason?: string }>;
  addListener(
    event: 'samples',
    listener: (data: { samples: NativeSample[] }) => void,
  ): Promise<PluginListenerHandle>;
  addListener(
    event: 'step',
    listener: (data: { t: number }) => void,
  ): Promise<PluginListenerHandle>;
  addListener(
    event: 'stepdetector',
    listener: (data: { available: boolean }) => void,
  ): Promise<PluginListenerHandle>;
  removeAllListeners(): Promise<void>;
}

const SoulSensors = registerPlugin<SoulSensorsPlugin>('SoulSensors');

export class NativeMotionSource implements MotionSourceLike {
  private handle: PluginListenerHandle | null = null;
  private hzWindow: number[] = [];
  private lastRotationSeen = false;
  private status: SensorStatus = {
    supported: Capacitor.isNativePlatform(),
    // APK 内的 WebView 走 https://localhost，始终是安全上下文
    secureContext: true,
    permission: 'prompt',
    active: false,
    hz: 0,
    hasRotation: false,
    message: '',
    channel: 'native-sensor',
  };

  onSample: ((s: RawSample) => void) | null = null;
  onHz: ((hz: number) => void) | null = null;
  /** 硬件步事件（脚触地时刻，与样本同一系统时钟）。 */
  onStep: ((t: number) => void) | null = null;
  /** 步检测可用性（权限/机型）。 */
  onStepDetector: ((available: boolean) => void) | null = null;
  private stepHandle: PluginListenerHandle | null = null;
  private stepDetectorHandle: PluginListenerHandle | null = null;

  getStatus(): SensorStatus {
    return { ...this.status, hz: this.currentHz(), hasRotation: this.lastRotationSeen && this.status.active };
  }

  async start(): Promise<SensorStatus> {
    if (this.handle) return this.getStatus();

    try {
      const probe = await SoulSensors.isAvailable();
      if (!probe.available) {
        this.status = {
          ...this.status,
          permission: 'unsupported',
          message: probe.message || '这台设备没有可用的加速度计',
        };
        return this.getStatus();
      }

      this.handle = await SoulSensors.addListener('samples', ({ samples }) => {
        // 一批可能有 5~8 帧，逐帧送进判定链；时间轴来自系统时钟，不受 JS 调度抖动影响。
        for (const s of samples) {
          if (s.rot) this.lastRotationSeen = true;
          this.pushHz(s.t);
          this.onSample?.({
            t: s.t,
            ax: s.ax,
            ay: s.ay,
            az: s.az,
            gx: s.gx,
            gy: s.gy,
            gz: s.gz,
            hasRotation: s.rot,
          });
        }
      });

      // 硬件步检测（TYPE_STEP_DETECTOR）：权限弹窗的最终结果也走这条事件，
      // 所以必须先挂监听再 start()，否则授权回调会丢。
      this.stepHandle = await SoulSensors.addListener('step', ({ t }) => {
        this.onStep?.(t);
      });
      this.stepDetectorHandle = await SoulSensors.addListener('stepdetector', ({ available }) => {
        this.onStepDetector?.(available);
      });

      const res = await SoulSensors.start();
      this.status = {
        ...this.status,
        permission: 'granted',
        active: true,
        hasRotation: res.hasGyro,
        message: res.hasGyro ? '传感器已启用（原生通道）' : '已启用，但这台设备没有陀螺仪，翻面判定会退化',
      };
      // 启动门按钮本身就是用户手势，紧跟着弹权限框是自然时机。
      // 已授权/老系统/无传感器时立即返回，不弹任何东西。
      // 结果不在这里消费——可用性统一走 stepdetector 事件，失败也只是静默降级。
      void SoulSensors.requestStepPermission().catch(() => undefined);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.status = { ...this.status, permission: 'denied', message: `原生传感器启动失败：${msg}` };
    }
    return this.getStatus();
  }

  stop(): void {
    void this.handle?.remove().catch(() => undefined);
    this.handle = null;
    void this.stepHandle?.remove().catch(() => undefined);
    this.stepHandle = null;
    void this.stepDetectorHandle?.remove().catch(() => undefined);
    this.stepDetectorHandle = null;
    void SoulSensors.stop().catch(() => undefined);
    this.hzWindow = [];
    this.status = { ...this.status, active: false, message: '传感器已停止' };
  }

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
