import { describe, it, expect } from 'vitest';
import { SensorEngine } from '../src/core/engine';
import { ImuSim, flickImpulse, shakeWave, smallJiggle } from './synthetic';
import type { ActionEvent } from '../src/core/types';

function run(sim: ImuSim, noiseFloor = 0.3): ActionEvent[] {
  const eng = new SensorEngine({ noiseFloor });
  const out: ActionEvent[] = [];
  for (const s of sim.samples()) {
    eng.ingest(s);
    const d = eng.evaluate();
    if (d.event) out.push(d.event);
  }
  return out;
}

const acts = (e: ActionEvent[]): string[] => e.map((x) => x.action);

describe('传感器判定链', () => {
  it('静置不触发任何动作', () => {
    expect(acts(run(new ImuSim().still(3)))).toEqual([]);
  });

  it('轻晃不触发（幅度与跨度都在门内）', () => {
    const sim = new ImuSim().still(1).move(0.18, { linear: smallJiggle(5) }).still(1);
    expect(acts(run(sim))).toEqual([]);
  });

  it('向左甩只触发一次下一首', () => {
    const sim = new ImuSim().still(1).move(0.22, { linear: flickImpulse(-1) }).still(1);
    expect(acts(run(sim))).toEqual(['player_next']);
  });

  it('向右甩只触发一次上一首', () => {
    const sim = new ImuSim().still(1).move(0.22, { linear: flickImpulse(1) }).still(1);
    expect(acts(run(sim))).toEqual(['player_prev']);
  });

  it('甩动的回程不会二次触发', () => {
    // 甩出去再甩回来，中间只停 80ms：方向锁存 + 配对冷却必须只放行一次
    const sim = new ImuSim()
      .still(1)
      .move(0.22, { linear: flickImpulse(-1) })
      .still(0.08)
      .move(0.22, { linear: flickImpulse(1) })
      .still(1);
    expect(acts(run(sim))).toHaveLength(1);
  });

  it('翻面朝下触发播放/暂停切换', () => {
    const sim = new ImuSim().still(1).poseTo(0.45, { pitch: 180, roll: 0 }).still(1);
    expect(acts(run(sim))).toEqual(['player_toggle']);
  });

  it('把手机放到桌上不算翻面', () => {
    // 俯仰角同样变化 90°，但朝向是 edge -> up 而不是穿过 down
    const sim = new ImuSim().still(1).poseTo(0.45, { pitch: 0, roll: 0 }).still(1);
    expect(acts(run(sim))).toEqual([]);
  });

  it('翻回来再次切换，两次间隔受冷却约束', () => {
    const sim = new ImuSim()
      .still(1)
      .poseTo(0.45, { pitch: 180, roll: 0 })
      .still(1.5)
      .poseTo(0.45, { pitch: 90, roll: 0 })
      .still(1);
    const got = acts(run(sim));
    expect(got).toEqual(['player_toggle', 'player_toggle']);
  });

  it('摇一摇只随机，不附带切歌', () => {
    const sim = new ImuSim().still(1).move(1.0, { linear: shakeWave(15, 4.2) }).still(1);
    expect(acts(run(sim))).toEqual(['player_random']);
  });

  it('向右倾斜保持触发音量增，且是连续可重复的', () => {
    const sim = new ImuSim().still(1).poseTo(0.3, { pitch: 90, roll: 35 }).still(1.2);
    const got = acts(run(sim));
    expect(got.length).toBeGreaterThanOrEqual(1);
    expect(got.every((a) => a === 'volume_up')).toBe(true);
  });

  it('间隔足够的两次同向甩动各触发一次', () => {
    // 这条守的是"动作锁定 + 冷却不会把引擎永久卡死"：
    // 锁靠"形态连续不满足"解除，冷却靠时间过期，两者都必须能恢复。
    const sim = new ImuSim()
      .still(0.7)
      .move(0.22, { linear: flickImpulse(-1) })
      .still(0.6)
      .move(0.22, { linear: flickImpulse(-1) })
      .still(1);
    expect(acts(run(sim))).toEqual(['player_next', 'player_next']);
  });

  it('窗口未填满时不出判定', () => {
    // 启动瞬间就动手，不应立刻出事件
    const sim = new ImuSim().move(0.15, { linear: flickImpulse(-1) });
    expect(acts(run(sim))).toEqual([]);
  });

  it('采样率不足时宁可不出判定', () => {
    const sim = new ImuSim(20).still(1.5).move(0.44, { linear: flickImpulse(-1) }).still(1);
    expect(acts(run(sim))).toEqual([]);
  });

  it('高噪声底机型上，轻晃仍不触发', () => {
    const sim = new ImuSim(60, 0.8).still(1).move(0.18, { linear: smallJiggle(5) }).still(1);
    expect(acts(run(sim, 1.2))).toEqual([]);
  });
});

describe('噪声底校准', () => {
  it('静置校准得到的噪声底接近真实噪声', () => {
    const eng = new SensorEngine();
    eng.startCalibration();
    for (const s of new ImuSim(60, 0.4).still(1.5).samples()) eng.ingest(s);
    const floor = eng.finishCalibration();
    expect(floor).toBeGreaterThan(0.15);
    expect(floor).toBeLessThan(1.5);
  });

  it('校准期间不出判定', () => {
    const eng = new SensorEngine();
    eng.startCalibration();
    const sim = new ImuSim().still(0.5).move(0.22, { linear: flickImpulse(-1) }).still(0.5);
    let fired = 0;
    for (const s of sim.samples()) {
      eng.ingest(s);
      if (eng.evaluate().event) fired++;
    }
    expect(fired).toBe(0);
  });
});
