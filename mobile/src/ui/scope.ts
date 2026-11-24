import type { Sample, SensorFeatures } from '../core/types';

/**
 * 实时三轴示波。
 *
 * 为什么值得做：手势识别的调试体验完全依赖"看得见"。
 * 阈值调不对的时候，用户在屏幕上只能感受到"没反应"，无法区分是
 * 传感器没数据、幅度不够、还是方向判反了。把三轴曲线画出来，
 * 这些问题一眼就能定位——这也正是 PC 端加了 SOULPLAYER_WAVE_DIAG
 * 日志才把挥手阈值调准的同一个道理。
 */
export class Scope {
  private ctx: CanvasRenderingContext2D;
  private buf: Sample[] = [];
  private readonly maxSamples = 140;

  constructor(private canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('canvas 2d 不可用');
    this.ctx = ctx;
    this.resize();
  }

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const w = this.canvas.clientWidth || 320;
    const h = this.canvas.clientHeight || 62;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  push(s: Sample): void {
    this.buf.push(s);
    if (this.buf.length > this.maxSamples) this.buf.shift();
  }

  clear(): void {
    this.buf = [];
  }

  draw(features: SensorFeatures | null): void {
    const { ctx } = this;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    // 中轴与幅度门参考线
    ctx.strokeStyle = '#3a332d';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (features) {
      const gate = features.noiseFloor * 3;
      const scale = h / 2 / 26; // 纵向满量程 ±26 m/s^2
      if (gate * scale < h / 2 - 2) {
        ctx.strokeStyle = 'rgba(217, 160, 63, 0.22)';
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(0, h / 2 - gate * scale);
        ctx.lineTo(w, h / 2 - gate * scale);
        ctx.moveTo(0, h / 2 + gate * scale);
        ctx.lineTo(w, h / 2 + gate * scale);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (this.buf.length < 2) return;

    const n = this.buf.length;
    const scale = h / 2 / 26;
    const step = w / (this.maxSamples - 1);
    const offset = (this.maxSamples - n) * step;

    const drawAxis = (pick: (s: Sample) => number, color: string, width: number): void => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const v = pick(this.buf[i]);
        const x = offset + i * step;
        const y = h / 2 - Math.max(-26, Math.min(26, v)) * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    drawAxis((s) => s.lz, 'rgba(122, 156, 198, 0.55)', 1.2);
    drawAxis((s) => s.ly, 'rgba(134, 176, 111, 0.7)', 1.2);
    drawAxis((s) => s.lx, '#c4956a', 1.8);
  }
}
