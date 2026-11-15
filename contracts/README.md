# contracts/ — 跨端唯一真源

这三份 JSON 是 PC 端（Python）与手机端（TypeScript）**共用**的契约。改阈值、改动作、加特征，
只改这里，两端按同一份数据解释，禁止在任一端的代码里手抄一份常量——这正是本项目反复踩过的坑
（改了源码没进包、两处阈值各写一遍慢慢漂移）。

| 文件 | 作用 | 谁消费 |
|---|---|---|
| `sensor_feature_schema.json` | 传感器特征的字段名、量纲、坐标系、窗口长度 | 手机端 `core/features.ts` 产出；PC 端联动接收时按名读取 |
| `actions.json` | 动作 id、语义、冷却表 | 手机端 `core/cooldown.ts`；PC 端 `ACTION_COOLDOWN_RULES` 的对照参考 |
| `sensor_presets.json` | 传感器动作预设：条件、动作、优先级、阈值标定依据 | 手机端 `core/engine.ts` |

## 两条命名约定

1. **动作 id 两端同名同义**。`player_next` 在 PC 端表示"挥手切下一首"，在手机端表示"甩动切下一首"，
   语义一致、方向一致（向左 = 下一首），这样用户换设备不需要重新学动作。
2. **特征名映射而非复用**。传感器有 `lateral_span`，视觉有 `arm_horizontal_motion`，两者语义对应
   （都是"横向位移跨度"）但名不同，因为量纲不同（米 vs 屏宽比例）。映射关系写在
   `sensor_feature_schema.json` 的 `derived_from_python` 里，不做强行合并。

## 阈值怎么改

每条预设都带 `calibrate` 字段，写明这个数值为什么是这个数、量级依据是什么。
调参流程：手机端打开"传感器"面板看实时特征 → 做 3 次目标动作记下 `lateral_span` 与
`lin_acc_peak` → 取最小值再留 20% 余量 → 改这里 → 刷新页面即生效（构建时内联）。

不要凭感觉改。阈值定在"错误动作的实际分布"与"正确动作的实际分布"之间的空档里，
不是定在"看起来合理"的整数上。
