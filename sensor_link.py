"""手机联动接收端 —— 接口骨架。

背景：手机端（mobile/）已经有一套完整的 IMU 判定链，产出与 PC 端同名的
动作 id（player_next / player_toggle / volume_up …，见 contracts/actions.json）。
本模块是 PC 端接收这些动作事件的落点。

本轮只定义协议与状态机，**不实现网络传输**——传输层（局域网 WebSocket 还是
UDP 组播、要不要配对码加密）在手机端发布为线上服务之后再定，因为那个决定
影响配对方式，先实现大概率要返工。

设计约束（与现有架构对齐）：

1. 手机推过来的是**动作 id**，不是原始传感器数据。判定在手机端完成，
   PC 端只做冷却与执行。这样 PC 端的 ACTION_COOLDOWN_RULES 对手机动作
   同样生效，两端行为一致，也不用在 PC 上再跑一遍特征计算。
2. 事件必须带 preset 与时间戳，用于面板回显"刚才那次是谁触发的"，
   这是排查误判的第一现场，PC 端挥手判定的排查经验已经证明它的价值。
3. 协议版本号放第一位。以后手机端和 PC 端版本错开时，旧协议要能被拒绝
   而不是解析出垃圾动作。
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

PROTOCOL_VERSION = 1

#: 允许从手机端进入 PC 的动作白名单。就算将来传输层被打进垃圾数据，
#: 也只有这些动作会被执行——白名单在这里比在执行处更集中。
ALLOWED_ACTIONS = frozenset(
    {
        "player_next",
        "player_prev",
        "player_toggle",
        "player_random",
        "volume_up",
        "volume_down",
    }
)

#: 事件队列上限。主循环卡顿时手机端可能连发多个事件（音量类约 5.5 次/秒），
#: 上限防止积压后一次性回放造成"幽灵连按"。
MAX_QUEUE = 16


@dataclass(frozen=True)
class SensorEvent:
    """一条来自手机端的动作事件。"""

    action: str
    preset_id: str
    preset_name: str
    device: str
    #: 手机端的时间戳（秒）。与 PC 端时钟无关，只用于排序与延迟估算。
    device_time: float
    received_at: float
    features: Dict[str, Any] = field(default_factory=dict)


class LinkState:
    OFFLINE = "未连接"
    PAIRING = "等待配对"
    LINKED = "已连接"
    ERROR = "异常"


class SensorLinkHub:
    """手机端事件的汇聚点。

    用法（主循环集成点，传输层就位后启用）::

        self.sensor_hub = SensorLinkHub()
        # 网络线程收到 payload 后:
        self.sensor_hub.handle_payload(raw_json)
        # 主循环每帧:
        for ev in self.sensor_hub.poll():
            self._handle_action(ev.action)   # 走既有冷却表，与视觉动作同路
    """

    def __init__(self) -> None:
        self.state: str = LinkState.OFFLINE
        self.status_text: str = "手机联动为预留功能：判定链已在手机端就绪，传输层待接入"
        self._queue: Deque[SensorEvent] = deque(maxlen=MAX_QUEUE)
        self._last_event: Optional[SensorEvent] = None
        self._reject_count = 0

    # ── 传输层回调 ────────────────────────────

    def handle_payload(self, raw: str) -> Optional[SensorEvent]:
        """解析并校验一条 JSON payload，合法则入队。

        任何解析失败、版本不符、动作不在白名单的情况都只计数不抛异常——
        网络输入是不可信的，这里绝不能让主循环崩掉。
        """
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("payload 不是对象")
            if data.get("v") != PROTOCOL_VERSION:
                raise ValueError(f"协议版本不符: {data.get('v')}")
            action = data.get("action")
            if action not in ALLOWED_ACTIONS:
                raise ValueError(f"动作不在白名单: {action!r}")

            ev = SensorEvent(
                action=action,
                preset_id=str(data.get("preset", "")),
                preset_name=str(data.get("preset_name", "")),
                device=str(data.get("device", "phone")),
                device_time=float(data.get("t", 0.0)),
                received_at=time.time(),
                features=data.get("features") or {},
            )
            self._queue.append(ev)
            self._last_event = ev
            if self.state != LinkState.LINKED:
                self.state = LinkState.LINKED
                self.status_text = f"已连接 · 来自 {ev.device}"
            return ev
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._reject_count += 1
            self.state = LinkState.ERROR
            self.status_text = f"最近一次数据被拒绝: {exc}"
            return None

    # ── 主循环消费 ────────────────────────────

    def poll(self) -> List[SensorEvent]:
        """取出并清空事件队列。每帧调用，顺序保持到达顺序。"""
        out = list(self._queue)
        self._queue.clear()
        return out

    # ── 面板展示 ──────────────────────────────

    @property
    def last_event(self) -> Optional[SensorEvent]:
        return self._last_event

    @property
    def reject_count(self) -> int:
        return self._reject_count

    def summary(self) -> Dict[str, Any]:
        """给「手机联动」面板的状态汇总。"""
        last = self._last_event
        return {
            "state": self.state,
            "status": self.status_text,
            "last": (
                f"{last.preset_name or last.preset_id} → {last.action}"
                if last
                else "暂无"
            ),
            "last_age": (time.time() - last.received_at) if last else None,
            "rejected": self._reject_count,
            "pending": len(self._queue),
        }
