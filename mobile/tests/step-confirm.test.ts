import { describe, it, expect } from 'vitest';
import { SensorEngine } from '../src/core/engine';
import { CooldownTable } from '../src/core/cooldown';
import { ImuSim } from './synthetic';
import type { ActionEvent, ModeId } from '../src/core/types';

/**
 * 硬件步检测（TYPE_STEP_DETECTOR）交叉验证 + 双击冷却固定窗口语义的回归。
 *
 * 设计要点见 contracts/motion_modes.json 的 hw_step_note：
 *   - 硬件只否决"提示跑步"（suggest_run），不参与跑步态退出与自动暂停；
 *   - 不对称是刻意的：某机型漏步时，用它判"跑完了"会造成跑步中误暂停，
 *     比漏一次提示严重得多；
 *   - 步检测不可用（机型不支持/权限被拒/web 通道）必须与旧版行为完全一致。
 */

interface CollectOpts {
  /** 步检测通道是否可用。 */
  available?: boolean;
  /** 步事件间隔（秒）。不给 = 不喂步事件（车辆场景 / 漏步机型）。 */
  stepEvery?: number;
  stepFrom?: number;
}

function collect(sim: ImuSim, mode: ModeId, opts: CollectOpts = {}) {
  const eng = new SensorEngine({ noiseFloor: 0.3, mode, placement: 'hand' });
  eng.setStepDetectorAvailable(opts.available ?? false);
  const events: ActionEvent[] = [];
  const transitions: string[] = [];
  let next = opts.stepFrom ?? 1.0;
  for (const s of sim.samples()) {
    eng.ingest(s);
    if (opts.stepEvery !== undefined) {
      while (next <= s.t) {
        eng.noteStep(next);
        next += opts.stepEvery;
      }
    }
    const d = eng.evaluate();
    if (d.event) events.push(d.event);
    if (d.modeTransition) transitions.push(d.modeTransition);
  }
  return { events, transitions, eng };
}

const acts = (e: ActionEvent[]): string[] => e.map((x) => x.action);

describe('硬件步检测交叉验证（suggest_run）', () => {
  it('真跑步 + 硬件步事件：提示照常出现', () => {
    // 176 步/分 → 步事件间隔 60/176 ≈ 0.34s，与 ImuSim.run 的脚步节律一致。
    const sim = new ImuSim(100).run(9);
    const r = collect(sim, 'still', { available: true, stepEvery: 60 / 176 });
    expect(r.transitions).toContain('suggest_run');
  });

  it('乘车场景：节律成立但硬件无步事件 → 不提示', () => {
    // 公交颠簸能凑出假跑步节律（这是 suggest_run 最大的误报源），
    // 但 Android 步检测器按官方文档不应在车内触发——信硬件，否决提示。
    const sim = new ImuSim(100).run(9);
    const r = collect(sim, 'still', { available: true });
    expect(r.transitions).not.toContain('suggest_run');
    expect(r.eng.currentMode).toBe('still');
  });

  it('步检测不可用：行为与旧版完全一致（提示照常）', () => {
    // 机型不支持 / 权限被拒 / web 通道。降级路径必须旁路一切否决逻辑，
    // 否则排障时要同时考虑两套行为。
    const sim = new ImuSim(100).run(9);
    const r = collect(sim, 'still', { available: false });
    expect(r.transitions).toContain('suggest_run');
  });

  it('步事件断档 2.9 秒（检测器延迟抖动上限附近）不影响提示', () => {
    // stale_ms=3000 > 官方延迟上限 2s。人为断 2.9s——贴着 stale 边界但不超过，
    // 真跑步不应被延迟抖动误伤。
    const sim = new ImuSim(100).run(12);
    const eng = new SensorEngine({ noiseFloor: 0.3, mode: 'still', placement: 'hand' });
    eng.setStepDetectorAvailable(true);
    const transitions: string[] = [];
    const skipFrom = 4.0;
    const skipTo = 6.9;
    let next = 1.0;
    for (const s of sim.samples()) {
      eng.ingest(s);
      while (next <= s.t) {
        if (!(next >= skipFrom && next < skipTo)) eng.noteStep(next);
        next += 60 / 176;
      }
      const d = eng.evaluate();
      if (d.modeTransition) transitions.push(d.modeTransition);
    }
    expect(transitions).toContain('suggest_run');
  });
});

describe('硬件步检测不参与跑步态退出（不对称）', () => {
  it('跑步模式下漏步机型（无步事件）跑步中不误暂停', () => {
    // 某些机型/佩戴位置下步检测器漏步。跑步态维护只信节律：
    // 跑步全程（前 8 秒）不应有任何事件。
    const sim = new ImuSim(100).run(8);
    const r = collect(sim, 'run', { available: true });
    expect(acts(r.events)).toEqual([]);
  });

  it('跑步模式下漏步机型停下后自动暂停照常工作', () => {
    // 退出与自动暂停不受硬件步缺失影响——rhythm-only。
    const sim = new ImuSim(100).run(8).still(11);
    const r = collect(sim, 'run', { available: true });
    const pauses = acts(r.events).filter((a) => a === 'player_pause_auto');
    expect(pauses).toHaveLength(1);
  });
});

describe('双击冷却的固定窗口语义', () => {
  it('CooldownTable：窗口内被拒的尝试不延长冷却', () => {
    // 微信摇一摇验证过的模式：状态锁 + 固定窗口，不重置计时。
    // 若实现成"每次尝试都重置"，第三次（t=1.05）会被第二次（t=0.9）顶掉。
    const cd = new CooldownTable({ player_toggle: { self_cd: 1.0 } });
    expect(cd.allow('player_toggle', 0)).toBe(true);
    expect(cd.allow('player_toggle', 0.5)).toBe(false);
    expect(cd.allow('player_toggle', 0.9)).toBe(false);
    expect(cd.allow('player_toggle', 1.05)).toBe(true);
  });

  it('引擎级：冷却窗口内的连击只响应一次', () => {
    // 用户在 1s 冷却内连敲两组双击，只应切换一次播放。
    const sim = new ImuSim(100).run(6).doubleTap().still(0.1).doubleTap().run(2);
    const r = collect(sim, 'run', {});
    expect(acts(r.events).filter((a) => a === 'player_toggle')).toHaveLength(1);
  });
});
