import type { CooldownRule } from './types';

/**
 * 声明式冷却表，与 PC 端 main.py 的 ACTION_COOLDOWN_RULES 同构。
 *
 * 三层防重复在手机端同样成立：
 *   采集层 —— 引擎的"动作锁定"（一次手势只出一次）
 *   判定层 —— 本表的 self_cd（同动作重复）
 *   执行层 —— 本表的 peers（双向动作互斥）与 also_block（形态相近的动作连带压制）
 *
 * 为什么冷却必须集中成一张表而不是散落在各处 if 里：PC 端曾经因为冷却散落，
 * 改了一个动作的间隔却漏改另一个，导致"改了没效果"。集中成数据之后，
 * 调整只需要改 contracts/actions.json。
 */
export class CooldownTable {
  private last = new Map<string, number>();
  /** 被别的动作连带压制到的时刻。与 last 分开存，因为语义不同：
   *  last 是"这个动作自己触发过"，suppressed 是"别的动作要求它别动"。 */
  private suppressed = new Map<string, number>();

  constructor(private readonly rules: Record<string, CooldownRule> = {}) {}

  /** 放行则返回 true，并登记时间戳（含 also_block 的连带压制）。 */
  allow(action: string, now: number): boolean {
    if ((this.suppressed.get(action) ?? 0) > now) return false;

    const rule = this.rules[action];
    if (!rule) return true; // 未登记的动作不设冷却，与 PC 端一致

    const selfCd = rule.self_cd ?? 0;
    if (now - (this.last.get(action) ?? -Infinity) < selfCd) return false;

    const peerCd = rule.peer_cd ?? 0;
    for (const peer of rule.peers ?? []) {
      if (now - (this.last.get(peer) ?? -Infinity) < peerCd) return false;
    }

    this.last.set(action, now);
    const blockFor = rule.block_cd ?? rule.self_cd ?? 0;
    for (const blocked of rule.also_block ?? []) {
      this.suppressed.set(blocked, now + blockFor);
    }
    return true;
  }

  /** 距离下次可放行还有多久（秒），用于 UI 提示。 */
  remaining(action: string, now: number): number {
    const sup = (this.suppressed.get(action) ?? 0) - now;
    const rule = this.rules[action];
    if (!rule) return Math.max(0, sup);
    let wait = Math.max(0, sup, (rule.self_cd ?? 0) - (now - (this.last.get(action) ?? -Infinity)));
    for (const peer of rule.peers ?? []) {
      wait = Math.max(wait, (rule.peer_cd ?? 0) - (now - (this.last.get(peer) ?? -Infinity)));
    }
    return Math.max(0, wait);
  }

  reset(): void {
    this.last.clear();
    this.suppressed.clear();
  }
}
