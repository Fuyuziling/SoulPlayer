import { describe, it } from 'vitest';
import { SensorEngine } from '../src/core/engine';
import { ImuSim, flickImpulse, shakeWave, smallJiggle } from './synthetic';
import type { SensorFeatures } from '../src/core/types';

/**
 * 阈值标定探针：把各场景的**全程极值**与事件序列打出来。
 * 注意要取全程极值而不是"峰值帧"——事件往往发生在特征刚过门的那一帧，
 * 而特征最大值出现在稍后，两者不是同一帧。
 */
function profile(name: string, sim: ImuSim): void {
  const eng = new SensorEngine({ noiseFloor: 0.3 });
  const events: string[] = [];
  const m = {
    span: 0,
    linPeak: 0,
    peaks: 0,
    rev: 0,
    flip: 0,
    tilt: 0,
    roll: 0,
    gyro: 0,
    vRatio: 0,
    still: 1,
    hz: 0,
  };
  let lastReject = '';
  for (const s of sim.samples()) {
    eng.ingest(s);
    const d = eng.evaluate();
    if (d.event) events.push(d.event.presetName);
    if (d.rejections.length > 0) {
      lastReject = d.rejections.map((r) => `${r.presetId}:${r.gate}`).join(' ');
    }
    const f: SensorFeatures | null = d.features;
    if (!f) continue;
    m.span = Math.max(m.span, f.lateralSpan);
    m.linPeak = Math.max(m.linPeak, f.linAccPeak);
    m.peaks = Math.max(m.peaks, f.peakCount);
    m.rev = Math.max(m.rev, f.velReversals);
    m.flip = Math.max(m.flip, f.flipAngleDeg);
    m.tilt = Math.max(m.tilt, f.tiltDeltaDeg);
    if (Math.abs(f.rollDeg) > Math.abs(m.roll)) m.roll = f.rollDeg;
    m.gyro = Math.max(m.gyro, f.gyroMagPeak);
    m.vRatio = Math.max(m.vRatio, f.verticalRatio);
    m.still = Math.min(m.still, f.stillRatio);
    m.hz = Math.max(m.hz, f.sampleHz);
  }
  console.log(
    name.padEnd(16),
    `span=${m.span.toFixed(3)} lin=${m.linPeak.toFixed(1)} peak=${m.peaks} rev=${m.rev} ` +
      `flip=${m.flip.toFixed(0)} tilt=${m.tilt.toFixed(0)} roll=${m.roll.toFixed(0)} ` +
      `gyro=${m.gyro.toFixed(2)} vR=${m.vRatio.toFixed(2)} still=${m.still.toFixed(2)} hz=${m.hz.toFixed(0)} ` +
      `| 事件=${events.length ? events.join(',') : '无'}`,
  );
  if (lastReject) console.log(' '.repeat(17) + '最后拒因: ' + lastReject);
}

describe('特征标定探针', () => {
  it('打印各场景全程极值', () => {
    profile('静置', new ImuSim().still(3));
    profile('轻晃5/0.18', new ImuSim().still(1).move(0.18, { linear: smallJiggle(5) }).still(1));
    profile('左甩-22', new ImuSim().still(1).move(0.22, { linear: flickImpulse(-1) }).still(1));
    profile('右甩+22', new ImuSim().still(1).move(0.22, { linear: flickImpulse(1) }).still(1));
    profile('摇一摇15/4.2', new ImuSim().still(1).move(1.0, { linear: shakeWave(15, 4.2) }).still(1));
    profile('翻面90->180', new ImuSim().still(1).poseTo(0.45, { pitch: 180, roll: 0 }).still(1));
    profile('放桌上90->0', new ImuSim().still(1).poseTo(0.45, { pitch: 0, roll: 0 }).still(1));
    profile('右倾保持35', new ImuSim().still(1).poseTo(0.3, { pitch: 90, roll: 35 }).still(1));
    profile('左甩x2快', new ImuSim()
      .still(1)
      .move(0.22, { linear: flickImpulse(-1) })
      .still(0.1)
      .move(0.22, { linear: flickImpulse(-1) })
      .still(1));
  });
});
