/**
 * 重力估计与数值小工具。
 *
 * 为什么需要去重力：DeviceMotion 的 accelerationIncludingGravity 里躺着 9.81 的
 * 重力常量，直接算能量的话"把手机静止放在桌上"就已经有 9.81 的读数，任何幅度门
 * 都会被击穿。必须先减掉重力，只留"人施加的加速度"。
 *
 * 为什么用 EMA 而不是直接读 acceleration：Android Chrome 的 acceleration 字段在
 * 部分机型上不可靠（有的一直为 null，有的单位错）。用含重力数据自己做低通估计，
 * 是所有机型上都成立的做法。
 */

/** 时间常数（秒）。越小跟得越快，但静止噪声抑制越弱。
 *  0.25s 对应截止频率约 0.64Hz：翻面（0.5s 转 180°）能跟上大部分，
 *  而甩动（2~4Hz）被完整保留为线性加速度。
 *  这个值是从"翻面必须能被识别为 180° 而不是 90°"倒推出来的，不要随手调大。 */
export const GRAVITY_TAU_SEC = 0.25;

export const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

/** 采样间隔的保护性裁剪：丢帧时 dt 可能很大，放任会把积分算飞。 */
export const clampDt = (dt: number): number => clamp(dt, 0.001, 0.12);

export class GravityEstimator {
  private g: [number, number, number] = [0, 0, -9.81];
  private primed = false;

  /** 用第一帧直接初始化，避免从 (0,0,-9.81) 慢慢收敛的瞬态污染开头几帧。 */
  update(ax: number, ay: number, az: number, dt: number): [number, number, number] {
    if (!this.primed) {
      this.primed = true;
      const m = Math.hypot(ax, ay, az) || 1;
      const k = 9.81 / m;
      this.g = [ax * k, ay * k, az * k];
      return this.g;
    }
    const alpha = clamp(dt / GRAVITY_TAU_SEC, 0.02, 1);
    this.g = [
      this.g[0] + (ax - this.g[0]) * alpha,
      this.g[1] + (ay - this.g[1]) * alpha,
      this.g[2] + (az - this.g[2]) * alpha,
    ];
    return this.g;
  }

  get gravity(): [number, number, number] {
    return this.g;
  }

  reset(): void {
    this.primed = false;
    this.g = [0, 0, -9.81];
  }
}

/** 去掉重力分量，得到线性加速度。 */
export function removeGravity(
  a: readonly [number, number, number],
  g: readonly [number, number, number],
): [number, number, number] {
  return [a[0] - g[0], a[1] - g[1], a[2] - g[2]];
}

export function hypot3(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}

export function median(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * 设备俯仰角：屏幕法线（+z）与"重力反方向"的夹角。
 *   0°   = 屏幕朝上平放
 *   90°  = 竖直握持
 *   180° = 屏幕朝下
 * 用 acos(-gz/|g|) 而不是 atan2 组合，是因为 0~180 全覆盖且无分支跳变，
 * 正好是"翻面"需要的那根轴。
 */
export function pitchFromGravity(g: readonly [number, number, number]): number {
  const m = hypot3(g[0], g[1], g[2]) || 9.81;
  return (Math.acos(clamp(-g[2] / m, -1, 1)) * 180) / Math.PI;
}

/**
 * 设备横滚角。定义为 + 表示"向右倾斜"（手机右边缘向下），
 * 与"向右倾斜 = 音量增大"的产品语义保持同向，UI 上不用再做符号反转。
 */
export function rollFromGravity(g: readonly [number, number, number]): number {
  return (Math.atan2(g[0], Math.hypot(g[1], g[2])) * 180) / Math.PI;
}
