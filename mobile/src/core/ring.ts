import type { Sample } from './types';

/**
 * 滑动样本缓冲。
 *
 * 设计取舍：不按"固定帧数"截断，而按"时间跨度 + 硬上限"双重约束。
 * 原因是 DeviceMotion 的实际频率在机型间差别很大（30~120Hz 都见过），
 * 固定帧数在不同手机上对应的时间长度完全不同，那样标定出来的阈值不可移植。
 */
export class SampleBuffer {
  private buf: Sample[] = [];

  constructor(
    /** 保留多久的样本（秒）。取最慢动作的时间尺度再留余量。 */
    private readonly maxAgeSec = 1.2,
    /** 硬上限，防高频机型把内存撑满。 */
    private readonly capacity = 160,
  ) {}

  push(s: Sample): void {
    this.buf.push(s);
    const cutoff = s.t - this.maxAgeSec;
    // 小数组，从头剪比用索引环更简单，且不会有环绕边界的边界条件 bug。
    while (this.buf.length > 0 && this.buf[0].t < cutoff) this.buf.shift();
    if (this.buf.length > this.capacity) this.buf.shift();
  }

  /** 取最近 windowSec 秒的样本（含端点）。不足则返回现有全部。 */
  tail(windowSec: number): Sample[] {
    if (this.buf.length === 0) return [];
    const end = this.buf[this.buf.length - 1].t;
    const start = end - windowSec;
    // 从尾部往前找，找到第一个早于 start 的位置
    let i = this.buf.length - 1;
    while (i > 0 && this.buf[i - 1].t >= start) i--;
    return this.buf.slice(i);
  }

  get size(): number {
    return this.buf.length;
  }

  get last(): Sample | undefined {
    return this.buf[this.buf.length - 1];
  }

  clear(): void {
    this.buf = [];
  }

  /** 估计最近的实际采样频率（Hz）。低于 0 的情况不出现，无样本返回 0。 */
  sampleHz(): number {
    if (this.buf.length < 4) return 0;
    const n = Math.min(30, this.buf.length);
    const last = this.buf[this.buf.length - 1];
    const first = this.buf[this.buf.length - n];
    const dt = last.t - first.t;
    return dt > 0 ? (n - 1) / dt : 0;
  }
}
