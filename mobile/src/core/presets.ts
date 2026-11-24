import presetsJson from '../../../contracts/sensor_presets.json';
import actionsJson from '../../../contracts/actions.json';
import modesJson from '../../../contracts/motion_modes.json';
import type { CooldownRule, GatesConfig, ModeId, ModesConfig, Preset } from './types';

/**
 * 契约加载。数据全部来自 ../contracts/*.json，代码里不重复写一份阈值——
 * 这是本项目最贵的一次教训（两处各写一份，改了一处以为生效）。
 */

export const DEFAULT_GATES: GatesConfig = presetsJson.gates as unknown as GatesConfig;

export const MODES: ModesConfig = modesJson as unknown as ModesConfig;

/**
 * 按优先级从高到低排序。
 * 跨模式比较优先级没有意义（同一时刻只有一个模式的预设参与匹配），
 * 排序只是为了同一模式内部的匹配顺序确定。
 */
export const DEFAULT_PRESETS: Preset[] = (
  presetsJson.presets as unknown as Preset[]
)
  .slice()
  .sort((a, b) => b.priority - a.priority);

/** 预设归属的模式。缺省 ['still']——保守默认见 types.ts 的注释。 */
export function modesOf(preset: Preset): ModeId[] {
  return preset.modes ?? ['still'];
}

export const DEFAULT_COOLDOWN_RULES: Record<string, CooldownRule> = Object.fromEntries(
  Object.entries(actionsJson.actions as Record<string, { cooldown?: CooldownRule }>)
    .filter(([, v]) => v.cooldown)
    .map(([k, v]) => [k, v.cooldown as CooldownRule]),
);

export const ACTION_META = actionsJson.actions as Record<
  string,
  { label: string; how: string; note?: string }
>;

/** 动作 → 中文短标签，UI 与日志共用。 */
export function actionLabel(action: string): string {
  return ACTION_META[action]?.label ?? action;
}

/** 每个预设默认使用的窗口，缺省 400ms。 */
export function windowMsOf(preset: Preset & { window_ms?: number }): number {
  return preset.window_ms ?? 400;
}
