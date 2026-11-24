import { describe, expect, it } from 'vitest';
import { MusicMatcher } from '../src/core/music-match';
import type { RunMusicConfig } from '../src/core/types';

/**
 * 步频乐单匹配器（三档：舒缓 / 轻松 / 激烈）。
 * 参数与 contracts/motion_modes.json 的 modes.run.music.levels 一致
 * （172/165/158/148，驻留 20s），断言的是滞回与驻留语义而不是具体数值。
 */
const cfg: RunMusicConfig = {
  folders: [
    { id: 'calm', label: '舒缓' },
    { id: 'light', label: '轻松' },
    { id: 'intense', label: '激烈' },
  ],
  levels: {
    enter_intense_bpm: 172,
    exit_intense_bpm: 165,
    exit_calm_bpm: 158,
    enter_calm_bpm: 148,
  },
  dwell_ms: 20000,
  switch_remaining_ms: 60000,
};

type Fired = 'calm' | 'light' | 'intense';

/** 以 100ms 一帧的节奏喂固定步频，返回每次触发的文件夹序列。 */
function feed(m: MusicMatcher, bpm: number | null, seconds: number, startMs = 0): Fired[] {
  const out: Fired[] = [];
  for (let t = 0; t <= seconds * 1000; t += 100) {
    const r = m.update(bpm, startMs + t);
    if (r) out.push(r);
  }
  return out;
}

describe('MusicMatcher 三档步频乐单', () => {
  it('高步频连续 20s 触发一次激烈，之后不再重复触发', () => {
    const m = new MusicMatcher(cfg);
    expect(feed(m, 175, 25)).toEqual(['intense']);
    expect(m.target).toBe('intense');
    expect(feed(m, 175, 40, 25000)).toEqual([]);
  });

  it('低步频连续 20s 触发舒缓', () => {
    const m = new MusicMatcher(cfg);
    expect(feed(m, 140, 25)).toEqual(['calm']);
    expect(m.target).toBe('calm');
  });

  it('刚进跑步态、步频在中间区默认落轻松档（同样要驻留）', () => {
    const m = new MusicMatcher(cfg);
    expect(feed(m, 160, 19)).toEqual([]);
    expect(feed(m, 160, 2, 19000)).toEqual(['light']);
    expect(m.target).toBe('light');
  });

  it('轻松档的两段滞回带（165~172 / 148~158）都不动作', () => {
    const m = new MusicMatcher(cfg);
    feed(m, 160, 25); // 锁定轻松
    expect(feed(m, 168, 30, 25000)).toEqual([]); // 升不到激烈
    expect(feed(m, 152, 30, 55000)).toEqual([]); // 降不到舒缓
    expect(m.target).toBe('light');
  });

  it('舒缓 → 轻松需 ≥158；148~158 滞回带里保持舒缓', () => {
    const m = new MusicMatcher(cfg);
    feed(m, 140, 25);
    expect(feed(m, 152, 25, 25000)).toEqual([]);
    expect(m.target).toBe('calm');
    expect(feed(m, 160, 25, 50000)).toEqual(['light']);
  });

  it('激烈 → 轻松需 ≤165；165~172 滞回带里保持激烈', () => {
    const m = new MusicMatcher(cfg);
    feed(m, 175, 25);
    expect(feed(m, 168, 25, 25000)).toEqual([]);
    expect(m.target).toBe('intense');
    expect(feed(m, 160, 25, 50000)).toEqual(['light']);
  });

  it('从散步直接冲刺可以跨档：一步到位落激烈，不等两次驻留', () => {
    const m = new MusicMatcher(cfg);
    feed(m, 140, 25); // 舒缓
    expect(feed(m, 178, 21, 25000)).toEqual(['intense']);
  });

  it('从冲刺直接放松同理，一步落舒缓', () => {
    const m = new MusicMatcher(cfg);
    feed(m, 175, 25);
    expect(feed(m, 135, 21, 25000)).toEqual(['calm']);
  });

  it('驻留途中离开该侧阈值就重置计时', () => {
    const m = new MusicMatcher(cfg);
    feed(m, 160, 25); // 轻松
    feed(m, 175, 15, 25000); // 候选激烈已累计 15s
    feed(m, 168, 5, 40000); // 回落滞回带 → 候选清零
    expect(feed(m, 175, 15, 45000)).toEqual([]);
    expect(feed(m, 175, 6, 60000)).toEqual(['intense']);
  });

  it('节律不成立（停下 / 步频 0）清候选计时，但已锁定档位不变', () => {
    const m = new MusicMatcher(cfg);
    feed(m, 160, 25); // 轻松
    feed(m, 175, 15, 25000); // 候选激烈 15s
    feed(m, null, 5, 40000); // 停下
    expect(m.target).toBe('light');
    expect(feed(m, 175, 15, 45000)).toEqual([]);
    expect(feed(m, 175, 6, 60000)).toEqual(['intense']);

    const m2 = new MusicMatcher(cfg);
    expect(feed(m2, 0, 30)).toEqual([]); // 0 不误判成"≤148 的舒缓"
    expect(m2.target).toBeNull();
  });

  it('reset 清空判定状态，上次跑步不泄漏到下次', () => {
    const m = new MusicMatcher(cfg);
    feed(m, 175, 25);
    expect(m.target).toBe('intense');
    m.reset();
    expect(m.target).toBeNull();
    expect(feed(m, 160, 19)).toEqual([]);
    expect(feed(m, 160, 2, 19000)).toEqual(['light']);
  });

  it('边界值：172 升激烈、165 降轻松、158 升轻松、148 落舒缓', () => {
    const a = new MusicMatcher(cfg);
    feed(a, 160, 25);
    expect(feed(a, 172, 21, 25000)).toEqual(['intense']);

    const b = new MusicMatcher(cfg);
    feed(b, 175, 25);
    expect(feed(b, 165, 21, 25000)).toEqual(['light']);

    const c = new MusicMatcher(cfg);
    feed(c, 140, 25);
    expect(feed(c, 158, 21, 25000)).toEqual(['light']);

    const d = new MusicMatcher(cfg);
    feed(d, 160, 25);
    expect(feed(d, 148, 21, 25000)).toEqual(['calm']);
  });
});
