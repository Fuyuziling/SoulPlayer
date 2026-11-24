import { describe, it, expect } from 'vitest';
import { SensorEngine } from '../src/core/engine';
import { ImuSim, flickImpulse } from './synthetic';
import type { ActionEvent, ModeId, Placement } from '../src/core/types';

/**
 * 模式互斥与跑步模板的回归。
 *
 * 这里最重要的一条不是"双击能触发"，而是"跑步态下日常六条预设一次都不触发"。
 * 跑步的步频节律（约 2.9Hz）与摇一摇（4.2Hz 往复）、甩动（位移跨度）落在
 * 同一特征区间，只要两套预设同时参与匹配，就必然互相误触。互斥是在
 * 匹配池层面做的，所以测试也要在整段真实跑步数据上验证"零事件"。
 */

interface Collected {
  events: ActionEvent[];
  transitions: string[];
  eng: SensorEngine;
}

function collect(
  sim: ImuSim,
  mode: ModeId,
  placement: Placement = 'hand',
  noiseFloor = 0.3,
): Collected {
  const eng = new SensorEngine({ noiseFloor, mode, placement });
  const events: ActionEvent[] = [];
  const transitions: string[] = [];
  for (const s of sim.samples()) {
    eng.ingest(s);
    const d = eng.evaluate();
    if (d.event) events.push(d.event);
    if (d.modeTransition) transitions.push(d.modeTransition);
  }
  return { events, transitions, eng };
}

const acts = (e: ActionEvent[]): string[] => e.map((x) => x.action);

describe('模式互斥', () => {
  it('跑步模式下只有两条跑步预设参与匹配', () => {
    const eng = new SensorEngine({ mode: 'run', placement: 'hand' });
    expect(eng.getModePresets().map((p) => p.id)).toEqual([
      'sensor_tap_double',
      'sensor_wrist_raise',
    ]);
  });

  it('口袋档关闭抬腕，只留双击', () => {
    const eng = new SensorEngine({ mode: 'run', placement: 'pocket' });
    expect(eng.getModePresets().map((p) => p.id)).toEqual(['sensor_tap_double']);
  });

  it('日常模式不包含任何跑步预设', () => {
    const eng = new SensorEngine({ mode: 'still' });
    const ids = eng.getModePresets().map((p) => p.id);
    expect(ids).toHaveLength(6);
    expect(ids).not.toContain('sensor_tap_double');
    expect(ids).not.toContain('sensor_wrist_raise');
  });

  it('跑步态下跑 20 秒，不产生任何事件', () => {
    // 这条是整个设计的核心断言。合成数据里横向摆臂峰值 8，位移约 9.4cm，
    // 已经逼近甩动的 14cm 跨度门；脚步冲击峰值 20，与敲击的峰值完全重叠。
    // 如果日常预设还留在匹配池里，这里必然出现误触发。
    const sim = new ImuSim(100).still(1).run(20, { bpm: 176 }).still(1);
    const r = collect(sim, 'run', 'hand');
    expect(acts(r.events)).toEqual([]);
  });

  it('跑步态下停止判定是"移出池子"而不是"阈值放宽"', () => {
    // 即便把灵敏度拉到最高（幅度门和跨度门都除以 1.6），跑步也不该触发——
    // 因为日常预设根本不在匹配池里，灵敏度对它没有作用点。
    const sim = new ImuSim(100).still(1).run(12, { bpm: 176 }).still(1);
    const eng = new SensorEngine({ noiseFloor: 0.3, mode: 'run', placement: 'hand' });
    eng.setSensitivity(1.6);
    const out: string[] = [];
    for (const s of sim.samples()) {
      eng.ingest(s);
      const d = eng.evaluate();
      if (d.event) out.push(d.event.action);
    }
    expect(out).toEqual([]);
  });
});

describe('跑步模板动作', () => {
  it('跑步中双击触发一次播放/暂停', () => {
    const sim = new ImuSim(100).run(6).doubleTap().run(2);
    const r = collect(sim, 'run', 'hand');
    expect(acts(r.events)).toEqual(['player_toggle']);
  });

  it('跑步中抬腕触发播报曲目', () => {
    const sim = new ImuSim(100).run(6).raisePhone(1.0);
    const r = collect(sim, 'run', 'hand');
    expect(acts(r.events)).toContain('announce_track');
  });

  it('口袋档下抬腕不响应（口袋里没有姿态意义）', () => {
    const sim = new ImuSim(100).run(6).raisePhone(1.0);
    const r = collect(sim, 'run', 'pocket');
    expect(acts(r.events)).not.toContain('announce_track');
  });

  it('跑步本身的脚步冲击不会被当成双击', () => {
    // 脚步声是单侧周期性的，凑不出"一对窄脉冲"；宽度也超过 70ms 的窄脉冲上限。
    const sim = new ImuSim(100).run(15, { bpm: 190, amp: 28 }).still(1);
    const r = collect(sim, 'run', 'hand');
    expect(acts(r.events)).toEqual([]);
  });
});

describe('停下自动暂停', () => {
  it('跑动后停下足够久会自动暂停', () => {
    // 停下后节律窗口还有约 2s 的跑步数据，所以 6s 的 idle 实际要等约 8s。
    const sim = new ImuSim(100).run(8).still(11);
    const r = collect(sim, 'run', 'hand');
    expect(acts(r.events)).toContain('player_pause_auto');
  });

  it('刚进跑步模式站着不动，不会自动暂停', () => {
    // autoPauseArmed 只有"真的跑过一段"之后才武装，否则用户刚点开
    // 跑步模式、还没起跑，音乐就被自己停了。
    const sim = new ImuSim(100).still(12);
    const r = collect(sim, 'run', 'hand');
    expect(acts(r.events)).toEqual([]);
  });
});

describe('模式切换', () => {
  it('切换模式后缓冲清空，旧模式的动作不再参与判定', () => {
    const eng = new SensorEngine({ noiseFloor: 0.3, mode: 'still' });
    const sim = new ImuSim().still(1).move(0.22, { linear: flickImpulse(-1) }).still(0.3);
    for (const s of sim.samples()) eng.ingest(s);
    eng.setMode('run', 'hand');
    // 刚切过去，缓冲是空的，不该有任何判定结果
    const d = eng.evaluate();
    expect(d.event).toBeNull();
    expect(d.features).toBeNull();
  });

  it('日常模式检测到持续跑步只提示，不自动切换', () => {
    const sim = new ImuSim(100).run(9);
    const r = collect(sim, 'still', 'hand');
    expect(r.transitions).toContain('suggest_run');
    // 模式必须仍是用户选的那个：自动切换会在用户正用手势时把它们关掉
    expect(r.eng.currentMode).toBe('still');
  });
});

describe('跑步模式的校准保护', () => {
  it('跑步模式下 startCalibration 被硬拒绝', () => {
    const eng = new SensorEngine({ mode: 'run' });
    expect(eng.startCalibration()).toBe(false);
    expect(eng.isCalibrating).toBe(false);
  });

  it('校准期间检测到运动时回退默认噪声底', () => {
    // 跑动中校准测到的不是噪声底而是冲击幅度。写进去会让 stillGate 变成 9，
    // 加速度永远落不进"静止"，位移无界漂移，而且这个值是持久化的。
    const eng = new SensorEngine();
    eng.startCalibration();
    for (const s of new ImuSim(100).run(2).samples()) eng.ingest(s);
    const floor = eng.finishCalibration();
    expect(eng.calibrationReliable).toBe(false);
    expect(floor).toBeCloseTo(0.6, 5);
  });
});
