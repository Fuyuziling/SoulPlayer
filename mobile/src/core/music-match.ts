import type { MusicGroup, RunMusicConfig } from './types';
export type { MusicGroup } from './types';

/**
 * 步频乐单匹配状态机：三档 + 两段双向滞回 + 连续驻留。
 *
 * 三个文件夹（舒缓 / 轻松 / 激烈）在契约里预设。为什么不"过线就切"：
 * 步频在配速临界点附近天然抖动（±5 步/分很常见），单阈值会导致在边界
 * 来回横跳、每 20 秒切一次歌。所以每条边界有进/出两个阈值：
 *
 *   舒缓 ──(≥158 升 / ≤148 降)── 轻松 ──(≥172 升 / ≤165 降)── 激烈
 *
 * 165~172 与 148~158 是滞回带（保持当前档），158~165 是轻松档稳定区。
 *
 * 与引擎的关系：引擎只产出 cadenceHz（2s 节律窗口），匹配器消费它并发出
 * 目标文件夹；怎么切（立即/播完切/淡出）是播放策略，归 app 层。心率是
 * 预留的第二输入位——将来接手环时把 update() 的入参从单指标扩成结构体
 * 即可，本状态机的滞回与驻留逻辑不用动。
 */
export class MusicMatcher {
  private readonly cfg: RunMusicConfig;
  /** 已生效的目标文件夹。null = 尚未判定（进跑步模式初期）。 */
  private current: MusicGroup | null = null;
  /** 候选文件夹及其连续满足的起始时刻。null = 当前保持现状。 */
  private pending: MusicGroup | null = null;
  private pendingSince = 0;

  constructor(cfg: RunMusicConfig) {
    this.cfg = cfg;
  }

  /** 跑步模式开启/重进时复位到"未判定"，避免上次跑步的状态泄漏到这次。 */
  reset(): void {
    this.current = null;
    this.pending = null;
    this.pendingSince = 0;
  }

  get target(): MusicGroup | null {
    return this.current;
  }

  /** 候选档已连续满足的时长（ms），UI 展示"×秒后切换"用。 */
  pendingMs(tMs: number): number {
    return this.pending ? tMs - this.pendingSince : 0;
  }

  get pendingGroup(): MusicGroup | null {
    return this.pending;
  }

  /**
   * 每帧喂一次步频。
   *
   * @param cadenceBpm 步频（步/分）。null 或 0 = 节律不成立（停下/晃动），
   *                   清掉候选计时——等红灯那 30 秒不该把驻留进程冻着
   *                   等恢复后立刻结算；已锁定的文件夹保持不变。
   * @param tMs 当前时刻（performance.now()）。
   * @returns 需要切换到的文件夹；每次状态变化只发一次，其余帧返回 null。
   */
  update(cadenceBpm: number | null, tMs: number): MusicGroup | null {
    const desired = this.desiredLevel(cadenceBpm);

    if (desired === null || desired === this.current) {
      // 死区 / 节律中断 / 已在目标档：候选归零。
      if (this.pending !== null) {
        this.pending = null;
        this.pendingSince = tMs;
      }
      return null;
    }
    if (desired !== this.pending) {
      this.pending = desired;
      this.pendingSince = tMs;
      return null;
    }
    if (tMs - this.pendingSince < this.cfg.dwell_ms) return null;

    this.current = desired;
    this.pending = null;
    return desired;
  }

  /** 依据当前档位与步频算出"应该去哪档"；null = 保持现状（含滞回带）。 */
  private desiredLevel(cadenceBpm: number | null): MusicGroup | null {
    if (cadenceBpm === null || cadenceBpm <= 0) return null;
    const L = this.cfg.levels;

    // 极值直接落档，不逐级爬——从散步直接起跑冲刺不该等两次 20s 驻留。
    if (cadenceBpm >= L.enter_intense_bpm) return 'intense';
    if (cadenceBpm <= L.enter_calm_bpm) return 'calm';

    switch (this.current) {
      case 'intense':
        // 165~172 是滞回带：不降档。降到 165 以下才回轻松。
        return cadenceBpm <= L.exit_intense_bpm ? 'light' : null;
      case 'calm':
        // 148~158 是滞回带：不升档。升到 158 以上才进轻松。
        return cadenceBpm >= L.exit_calm_bpm ? 'light' : null;
      case null:
        // 刚进跑步态、步频落在中间区：默认落轻松档（最中性），
        // 同样要驻留 20s 才锁定，避免开跑瞬间的假步频直接定档。
        return 'light';
      default:
        // 当前轻松档：158~165 稳定区与两段滞回带都保持不动。
        return null;
    }
  }
}
