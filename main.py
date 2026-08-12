#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoulPlayer - 基于肢体动作捕捉的智能音乐播放器
主程序入口（main.py）

职责：
- 初始化并整合所有子模块（动作捕捉、预设匹配、音乐播放）
- 提供基于 pygame 的图形用户界面
- 运行主循环，协调各模块间的数据流与控制流
- 处理用户输入与系统事件

模块间接口契约参见 ARCHITECTURE.md
"""

import sys
import os
import time
import json
import random
import shutil
import threading
import traceback
import inspect
from pathlib import Path
from typing import Optional, Dict, Any

# ==================== 未捕获异常自动写入日志 ====================
# 这样即使窗口闪退，也能在 error.log 中查看错误信息
def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    try:
        log_path = Path(__file__).parent / "error.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 未捕获异常:\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            f.write(f"{'='*50}\n")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = _log_uncaught_exception

# ==================== 第三方依赖 ====================
try:
    import pygame
    import numpy as np
except ImportError as e:
    print(f"[致命错误] 缺少依赖: {e}")
    print("请运行: pip install -r requirements.txt")
    input("按回车键退出...")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None

# ==================== 项目模块导入（带降级桩） ====================
# 若项目子模块尚未实现，则使用内置的最小桩实现，确保主程序可独立运行。
# 当真正的模块文件就位后，桩实现会自动被覆盖。

try:
    from motion_capture import MotionCapture
except ImportError:
    class MotionCapture:
        """
        动作捕捉模块最小桩实现。
        当真正的 motion_capture.py 就位后，此类自动被替换。
        在桩模式下，通过正弦波生成模拟动作数据，便于调试 UI 与主循环。
        """

        def __init__(self, camera_id=0, model_complexity=1,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5):
            self._camera_id = camera_id
            self._model_complexity = model_complexity
            self._min_detection_confidence = min_detection_confidence
            self._min_tracking_confidence = min_tracking_confidence

            self._running = False
            self._lock = threading.Lock()
            self._latest_features = self._default_features()
            self._frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self._thread: Optional[threading.Thread] = None

        def _default_features(self) -> Dict[str, Any]:
            return {
                'timestamp': time.time(),
                'motion_energy': 0.0,
                'body_speed': 0.0,
                'arm_amplitude': 0.0,
                'body_rhythm': 0.0,
                'posture': 'unknown',
                'hand_position': 'unknown',
                'is_active': False,
                'landmarks': [],
            }

        def start(self):
            """启动模拟捕获线程"""
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

        def _capture_loop(self):
            """后台线程：周期性更新模拟帧与模拟特征"""
            while self._running:
                t = time.time()
                with self._lock:
                    self._frame = self._generate_simulated_frame(t)
                    self._latest_features = self._simulate_features(t)
                time.sleep(0.033)  # 约 30 FPS

        def _generate_simulated_frame(self, t: float) -> np.ndarray:
            """生成动态彩色模拟图像（无需摄像头）"""
            h, w = 480, 640
            x = np.linspace(0, 4 * np.pi, w)
            y = np.linspace(0, 4 * np.pi, h)
            X, Y = np.meshgrid(x, y)
            r = np.sin(X + t) * 0.5 + 0.5
            g = np.sin(Y + t * 0.7) * 0.5 + 0.5
            b = np.sin((X + Y) * 0.5 + t * 1.3) * 0.5 + 0.5
            frame = np.stack([r, g, b], axis=-1) * 255
            return frame.astype(np.uint8)

        def _simulate_features(self, t: float) -> Dict[str, Any]:
            """基于正弦波生成模拟动作特征，用于测试不同预设触发"""
            # 周期约 20 秒，依次经过 安静→轻度→活跃→跳动
            phase = (t % 20.0) / 20.0
            if phase < 0.25:
                energy = 0.05 + 0.05 * np.sin(t * 2)
            elif phase < 0.5:
                energy = 0.25 + 0.05 * np.sin(t * 2)
            elif phase < 0.75:
                energy = 0.55 + 0.05 * np.sin(t * 2)
            else:
                energy = 0.85 + 0.1 * np.sin(t * 3)

            energy = float(np.clip(energy, 0.0, 1.0))
            return {
                'timestamp': t,
                'motion_energy': energy,
                'body_speed': float(np.clip(energy * 0.9, 0.0, 1.0)),
                'arm_amplitude': float(np.clip(0.1 + 0.3 * np.sin(t * 0.8), 0.0, 1.0)),
                'body_rhythm': float(np.clip(0.2 + 0.2 * np.sin(t * 1.2), 0.0, 1.0)),
                'posture': 'standing',
                'hand_position': 'center',
                'is_active': energy > 0.15,
                'landmarks': [],
            }

        def stop(self):
            """停止捕获线程并释放资源"""
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.5)
            self._thread = None

        def get_latest_motion_features(self) -> Dict[str, Any]:
            """线程安全地获取最新动作特征"""
            with self._lock:
                return self._latest_features.copy()

        def get_camera_frame(self) -> np.ndarray:
            """线程安全地获取当前帧（用于 UI 预览）"""
            with self._lock:
                return self._frame.copy()

        @property
        def is_running(self) -> bool:
            return self._running


try:
    from gesture_presets import GesturePreset, PresetEngine
except ImportError:
    class GesturePreset:
        """
        单个动作预设定义（桩实现）。
        参见 ARCHITECTURE.md 接口契约。
        """

        def __init__(self, name: str, description: str,
                     conditions: dict, action: str, priority: int = 0):
            self.name = name
            self.description = description
            self.conditions = conditions
            self.action = action
            self.priority = priority

    class PresetEngine:
        """
        预设匹配引擎（桩实现）。
        完整实现 duration 阈值逻辑、优先级排序与持续计时器。
        """

        def __init__(self):
            self.presets: list[GesturePreset] = []
            self.active_preset: Optional[str] = None
            self.hold_timers: Dict[str, float] = {}
            self._last_check_time = time.time()

        def load_presets(self, filepath: str):
            """从 JSON 文件加载预设列表"""
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"预设文件不存在: {filepath}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for p in data.get('presets', []):
                self.add_preset(GesturePreset(
                    name=p['name'],
                    description=p.get('description', ''),
                    conditions=p['conditions'],
                    action=p['action'],
                    priority=p.get('priority', 0)
                ))

        def save_presets(self, filepath: str):
            """将当前预设列表保存为 JSON"""
            data = {'presets': []}
            for p in self.presets:
                data['presets'].append({
                    'name': p.name,
                    'description': p.description,
                    'conditions': p.conditions,
                    'action': p.action,
                    'priority': p.priority,
                })
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        def add_preset(self, preset: GesturePreset):
            self.presets.append(preset)
            self.hold_timers.setdefault(preset.name, 0.0)

        def remove_preset(self, name: str):
            self.presets = [p for p in self.presets if p.name != name]
            self.hold_timers.pop(name, None)

        def match(self, motion_features: Dict[str, Any]) -> Optional[str]:
            """
            根据动作特征匹配预设。
            - 仅当条件持续满足达到 duration 阈值时才触发
            - 多个预设同时满足时，返回 priority 最高的 action
            - 不满足条件的预设计时器清零
            """
            current_time = time.time()
            dt = current_time - self._last_check_time
            self._last_check_time = current_time

            best_preset: Optional[GesturePreset] = None
            best_priority = -1

            for preset in self.presets:
                is_match = self._check_conditions(preset, motion_features)
                if is_match:
                    self.hold_timers[preset.name] = self.hold_timers.get(preset.name, 0.0) + dt
                    duration = preset.conditions.get('duration', 0.0)
                    if self.hold_timers[preset.name] >= duration:
                        if preset.priority > best_priority:
                            best_priority = preset.priority
                            best_preset = preset
                else:
                    self.hold_timers[preset.name] = 0.0

            if best_preset:
                self.active_preset = best_preset.name
                return best_preset.action

            self.active_preset = None
            return None

        def _check_conditions(self, preset: GesturePreset, features: Dict[str, Any]) -> bool:
            """检查单个预设的所有条件是否满足"""
            conds = preset.conditions

            # 连续数值型条件
            for key in ('motion_energy', 'body_speed', 'arm_amplitude', 'body_rhythm'):
                if key in conds:
                    val = features.get(key, 0.0)
                    if val < conds[key]['min'] or val > conds[key]['max']:
                        return False

            # 离散型条件
            if 'posture' in conds:
                if features.get('posture') not in conds['posture']:
                    return False
            if 'hand_position' in conds:
                if features.get('hand_position') not in conds['hand_position']:
                    return False
            if 'is_active' in conds:
                if features.get('is_active') != conds['is_active']:
                    return False

            return True

        def get_preset_list(self) -> list:
            return self.presets


try:
    from music_player import MusicPlayer
except ImportError:
    class MusicPlayer:
        """
        音乐播放器最小桩实现（基于 pygame.mixer）。
        当真正的 music_player.py 就位后，此类自动被替换。
        """

        def __init__(self):
            # 避免重复初始化
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._playlist: list[str] = []
            self._current_index = -1
            self._volume = 0.5
            self._mood_playlists: Dict[str, list[str]] = {}

        def load_playlist(self, folder_path: str):
            """扫描文件夹加载本地音乐文件"""
            exts = {'.mp3', '.wav', '.ogg', '.flac'}
            path = Path(folder_path)
            if not path.exists():
                return
            self._playlist = sorted([
                str(p) for p in path.iterdir() if p.suffix.lower() in exts
            ])
            if self._playlist:
                self._current_index = 0

        def load_playlist_from_spotify(self, playlist_id: str, token: str):
            # 桩实现：不执行实际 Spotify 调用
            pass

        def play(self, track_index=None):
            if track_index is not None:
                self._current_index = track_index
            if self._playlist and 0 <= self._current_index < len(self._playlist):
                try:
                    pygame.mixer.music.load(self._playlist[self._current_index])
                    pygame.mixer.music.play()
                    pygame.mixer.music.set_volume(self._volume)
                except Exception as e:
                    print(f"[MusicPlayer] 播放失败: {e}")

        def play_track(self, filepath: str):
            """直接加载并播放指定文件"""
            try:
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                pygame.mixer.music.set_volume(self._volume)
                self._current_index = -1
            except Exception as e:
                print(f"[MusicPlayer] 播放失败: {e}")

        def pause(self):
            pygame.mixer.music.pause()

        def stop(self):
            pygame.mixer.music.stop()

        def next_track(self):
            if not self._playlist:
                return
            self._current_index = (self._current_index + 1) % len(self._playlist)
            self.play()

        def previous_track(self):
            if not self._playlist:
                return
            self._current_index = (self._current_index - 1) % len(self._playlist)
            self.play()

        def set_volume(self, volume: float):
            self._volume = float(np.clip(volume, 0.0, 1.0))
            pygame.mixer.music.set_volume(self._volume)

        def crossfade_to(self, track_index: int, duration_ms: int = 2000):
            # 简化桩实现：直接切换
            self.play(track_index)

        def set_playlist_by_mood(self, mood: str):
            # 桩实现：仅记录 mood，不实际切换播放列表
            # 真正的实现应根据 mood 映射到不同的文件夹或播放列表
            if mood in self._mood_playlists:
                self._playlist = self._mood_playlists[mood]
                self._current_index = 0

        @property
        def is_playing(self) -> bool:
            return pygame.mixer.music.get_busy()

        def get_playback_position(self) -> float:
            return 0.0

        def get_track_duration(self) -> float:
            return -1.0

        def seek_to(self, position_sec: float) -> None:
            pass

        @property
        def current_track(self) -> dict:
            if self._playlist and 0 <= self._current_index < len(self._playlist):
                p = Path(self._playlist[self._current_index])
                return {
                    'title': p.stem,
                    'filename': p.name,
                    'path': str(p),
                    'index': self._current_index,
                }
            return {'title': '无曲目', 'filename': '', 'path': '', 'index': -1}


class _MusicPlayerStub:
    """
    pygame.mixer 初始化失败时的降级播放器。

    接口与 MusicPlayer 对齐，但全部为安全空操作。此时 mixer 不可用，
    任何 pygame.mixer 调用都会抛 pygame.error，因此这里不做任何底层调用。
    """

    def __init__(self):
        self._playlists: dict = {}
        self._current_playlist_key = None
        self._tracks: list = []
        self._current_index = -1
        self._playing = False
        self._paused = False
        self._volume = 0.5

    def load_playlist(self, folder_path: str):
        print(f"[播放器不可用] 忽略加载: {folder_path}")

    def play(self, track_index=None):
        pass

    def pause(self):
        pass

    def stop(self):
        pass

    def next_track(self):
        pass

    def previous_track(self):
        pass

    def play_track(self, filepath: str):
        pass

    def set_volume(self, volume: float):
        self._volume = float(max(0.0, min(1.0, volume)))

    def crossfade_to(self, *args, **kwargs):
        pass

    def set_playlist_by_mood(self, mood: str):
        pass

    def sync_folder(self, folder_path: str):
        return [], []

    def get_playlist_info(self) -> dict:
        return {
            "total_tracks": 0,
            "current_index": -1,
            "playlist_key": None,
            "tracks": [],
        }

    def register_playlist(self, key: str, tracks: list):
        pass

    def unload(self):
        pass

    @property
    def is_playing(self) -> bool:
        return False

    def get_playback_position(self) -> float:
        return 0.0

    def get_track_duration(self) -> float:
        return -1.0

    def seek_to(self, position_sec: float) -> None:
        pass

    @property
    def current_track(self) -> dict:
        return {}


# 自定义预设
try:
    from custom_preset import CustomPresetManager
except ImportError:
    CustomPresetManager = None

# 语音台词触发（B 方案：音频模板比对，无需语言模型）
try:
    from voice_match import VoiceMatcher as VoiceTrigger
except ImportError:
    class VoiceTrigger:
        """
        语音触发模块最小桩实现。
        当 voice_match.py 就位后自动被替换。
        """
        def __init__(self, model_path=None, storage_path="voice_templates.json"):
            self.triggers: dict[str, str] = {}
            self.available = False
            self._queue = None
        def load(self): pass
        def save(self): pass
        def add_trigger(self, phrase, action, tracks=None, start_pct=0.0): pass
        def remove_trigger(self, phrase): pass
        def get_tracks(self, phrase): return []
        def get_start_pct(self, phrase): return 0.0
        def start_capture(self): pass
        def stop_capture(self): return ""
        def start_listening(self): pass
        def stop_listening(self): pass
        def get_triggered_action(self): return None
        def get_status(self):
            return {"state": "unavailable", "error": "voice_match 模块未安装",
                    "listening": False, "available": False, "trigger_count": 0}

# UI 主题与动画系统
try:
    from ui_theme import SoulTheme, UIRenderer, ICONS, draw_pose_skeleton, value_to_color, \
        draw_hand_skeleton
    from ui_animations import AnimationManager, Easing
except ImportError:
    SoulTheme = UIRenderer = ICONS = draw_pose_skeleton = value_to_color = None
    draw_hand_skeleton = None
    AnimationManager = Easing = None


# ==================== 主程序 ====================

class _VoiceTemplateRef:
    """语音模板适配对象：让模板可以复用预设删除确认弹窗。"""
    _is_voice_template = True

    def __init__(self, name: str, tracks: list):
        self.name = name
        self.tracks = tracks


class SoulPlayerApp:
    """
    SoulPlayer 主应用程序类。

    整合动作捕捉、预设匹配与音乐播放三大模块，
    提供 pygame 渲染的图形界面与主循环。
    """

    # 配色方案（低饱和度暖色调）
    COLOR_BG = (43, 43, 43)  # #2B2B2B 深暖灰背景
    COLOR_PANEL = (54, 54, 54)  # #363636 面板
    COLOR_TEXT = (232, 232, 232)  # #E8E8E8 主文字
    COLOR_TEXT_DIM = (170, 170, 170)  # #AAAAAA 次要文字
    COLOR_ACCENT = (196, 149, 106)  # #C4956A 暖琥珀色强调
    COLOR_BAR_BG = (85, 85, 85)  # #555555 进度条背景
    COLOR_BAR_LOW = (136, 136, 136)  # #888888 低值条
    COLOR_WHITE = (255, 255, 255)

    # 布局常量
    WIN_WIDTH = 1280
    WIN_HEIGHT = 720
    PREVIEW_W = 320
    PREVIEW_H = 240
    FPS = 60

    def __init__(self):
        """
        初始化 Pygame、各子模块及界面状态。
        """
        print("[DEBUG] 开始初始化...")

        # 关键修复：Windows 微软拼音等 IME 的候选框不显示问题。
        # SDL 在 Windows 上默认以 UILess 模式抑制系统原生 IME UI（候选框），
        # 导致改名时"拼音能键入、却看不到候选字"（约等于盲打）。
        # SDL hint 需在 SDL 初始化（pygame.init）之前生效，故用环境变量设置：
        # SDL_HINT_IME_SHOW_UI=1 → SDL 显示系统原生 IME 候选框
        #（SDL 2.0.18+ 支持，pygame 2.6.1 / SDL 2.28.4 已包含）。
        # 仅当进入改名等文本输入场景（start_text_input）时 IME 才接管；
        # 平时 stop_text_input 仍保证 R/V 等字母快捷键直达程序，互不影响。
        os.environ["SDL_HINT_IME_SHOW_UI"] = "1"

        pygame.init()
        print("[DEBUG] pygame 初始化完成")
        
        pygame.display.set_caption("SoulPlayer")
        self.screen = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        # 关键修复：关闭 SDL 文本输入模式。
        # SDL 默认开启文本输入，中文输入法（IME）激活时会拦截 R/V 等字母键
        # 作为拼音候选，导致 pygame 完全收不到按键事件。
        # 本项目不需要文本输入，关闭后字母键可直达程序。
        pygame.key.stop_text_input()
        print("[DEBUG] 已关闭文本输入模式（防止输入法拦截字母键）")
        print("[DEBUG] 显示窗口创建完成")

        # 字体初始化
        self._fonts = {}
        print("[DEBUG] 开始初始化字体...")
        self._init_fonts()
        print("[DEBUG] 字体初始化完成")

        # 初始化子模块
        print("[DEBUG] 开始初始化动作捕捉模块...")
        try:
            self.motion = MotionCapture(camera_id=0, model_complexity=1)
            print("[DEBUG] 动作捕捉模块创建完成")
            self.preset_engine = PresetEngine()
            print("[DEBUG] 预设引擎创建完成")
            preset_path = Path(__file__).parent / "presets" / "default_presets.json"
            print(f"[DEBUG] 加载预设文件: {preset_path}")
            self.preset_engine.load_presets(str(preset_path))
            print("[DEBUG] 预设加载完成")
        except Exception as e:
            print(f"[初始化错误] 模块加载失败: {e}")
            raise

        # 运行状态（提前初始化，避免异常处理中 _add_log 失败）
        self.current_features: Optional[Dict[str, Any]] = None
        self.last_action: Optional[str] = None
        self.action_timestamp = 0.0
        self._action_cooldowns: Dict[str, float] = {}  # 动作冷却期记录
        self.log_messages: list[tuple[str, float]] = []

        # 初始化播放器
        print("[DEBUG] 开始初始化音乐播放器...")
        try:
            self.player = MusicPlayer()
            print("[DEBUG] 音乐播放器初始化完成")

            # 自动加载音乐文件夹（放在 try 块内部）
            music_folder = self._music_dir()
            print(f"[DEBUG] 音乐目录定位: {music_folder}")
            if music_folder.exists():
                try:
                    self.player.load_playlist(str(music_folder))
                    info = self.player.get_playlist_info()
                    total = info.get('total_tracks', 0)
                    if total > 0:
                        print(f"[DEBUG] 自动加载 {total} 首曲目")
                        self._add_log(f"已加载 {total} 首曲目")
                    else:
                        print("[DEBUG] 音乐文件夹为空，请放入 .mp3 文件")
                        self._add_log("音乐文件夹为空，请放入 .mp3 文件")
                except Exception as e:
                    print(f"[DEBUG] 自动加载音乐失败: {e}")
                    self._add_log("音乐加载失败")
            else:
                print(f"[DEBUG] 未找到音乐文件夹: {music_folder}")
                self._add_log("未找到音乐文件夹，请创建 music/ 目录")
        except Exception as e:
            print(f"[警告] 音频播放器初始化失败: {e}")
            self.player = _MusicPlayerStub()
            print("[DEBUG] 音乐播放器降级完成")

        # 自定义预设管理器
        self.custom_manager = None
        if CustomPresetManager is not None:
            try:
                self.custom_manager = CustomPresetManager(
                    storage_dir=str(self._data_dir("custom_presets")))
                self.custom_manager.inject_into_engine(self.preset_engine)
                print(f"[DEBUG] 自定义预设加载完成: {len(self.custom_manager.presets)} 个")
            except Exception as e:
                print(f"[DEBUG] 自定义预设加载失败: {e}")

        # 语音台词触发（B 方案：音频模板比对引擎，构造时无需加载重型模型）
        self.voice_trigger = None
        if VoiceTrigger is not None:
            try:
                vt_dir = self._data_dir("voice_templates")
                self.voice_trigger = VoiceTrigger(
                    templates_dir=str(vt_dir),
                    storage_path=str(vt_dir.parent / "voice_templates.json"))
                if self.voice_trigger.available:
                    self.voice_trigger.start_listening()
                    self._add_log("🎤 语音监听已启动")
                    print(f"[DEBUG] 语音触发初始化完成: {len(self.voice_trigger.triggers)} 条模板")
                else:
                    status = self.voice_trigger.get_status()
                    err = status.get("error") or status.get("status") or "未知原因"
                    self._add_log(f"🎤 语音未就绪: {err}")
                    print(f"[DEBUG] 语音引擎未就绪: {err}")
            except Exception as e:
                print(f"[DEBUG] 语音触发初始化失败: {e}")
                self._add_log("语音引擎初始化失败")

        # 录制状态
        self.recording_mode = False
        self.recording_preset_name = ""

        # 重命名状态（界面内联编辑，不弹窗）
        self._renaming_preset = None
        self._rename_text = ""
        # 删除确认状态（界面内联弹窗，与重命名风格一致）
        self._deleting_preset = None
        # 录制后选歌面板状态（界面内联弹窗，替代 exe 中不可靠的 tkinter 对话框）
        # 结构: {"name": str, "conditions": dict, "songs": [str], "selected": set[int], "scroll": int,
        #        "mode": "motion"|"voice", "phrase": str}
        self._song_select = None
        self._song_select_rects = []
        # 语音面板模板行的删除按钮矩形（每帧重算）
        self._voice_actions = []
        self._voice_scroll = 0
        self._voice_list_rect = None
        # 预览框动作骨架显隐（默认显示，按钮可切换）
        self._skeleton_visible = True
        # 语音台词录制状态机: None | {"phase": "countdown"|"recording", "t0": float, "auto_started": bool}
        self._voice_rec = None
        # 播放栏进度条拖拽状态
        self._seek_dragging = False
        self._seek_hovered = False
        # 音量条拖拽状态与热区（每帧刷新）
        self._vol_dragging = False
        self._vol_bar_rect = None
        # 录制后段落选择面板（选歌后的非强制步骤）
        # 结构: {"tracks": [str], "sections": {track_idx: {"start_pct": float, "label": str}},
        #        "current_track_idx": int, "scroll": int, "mode": "motion"|"voice", "phrase": str,
        #        "pending_preset_name": str, "pending_conditions": dict}
        self._section_select = None
        self._section_select_rects = []
        # 段落选择进度条拖拽状态（点击时间轴后置位，松开鼠标复位）
        self._section_dragging = False
        # 人声/扬声器分离：说话期间自动压低音乐音量（记录原音量，None 表示未压低）
        self._voice_ducked = None
        self._duck_until = 0.0
        # music 文件夹动态监测：上次扫描时间戳
        self._last_music_scan = 0.0
        # 音乐列表弹窗（手动选歌播放）：True 为打开
        self._music_list_open = False
        self._music_list_scroll = 0
        self._music_list_rects = []
        self._music_list_btn_rect = None
        # 常驻歌单面板滚动状态
        self._play_scroll = 0
        self._playlist_rect = None
        # 底部信息区页签：动作特征 / 事件日志（折叠切换）
        self._bottom_tab = "features"
        self._bottom_tab_rects = []
        self._tab_anim = {"features": 0.0, "log": 0.0}  # 页签 hover 缩放进度
        # 音量调整后的数值回显弹标截止时间（0 表示不显示）
        self._vol_popup_until = 0.0

        # 运行状态（已提前初始化，此处可省略）

        # 初始化 UI 渲染器与动画系统
        if UIRenderer is not None:
            self.ui = UIRenderer(self.screen)
            print("[DEBUG] UI 渲染器初始化完成")
        else:
            self.ui = None
        if AnimationManager is not None:
            self.anim = AnimationManager()
            print("[DEBUG] 动画系统初始化完成")
        else:
            self.anim = None

        # 启动动作捕捉
        print("[DEBUG] 启动动作捕捉...")
        try:
            self.motion.start()
            print("[DEBUG] 动作捕捉线程启动完成")
            self._add_log("系统已启动")
        except RuntimeError as e:
            self._add_log(f"摄像头不可用: {e}")
            print(f"[DEBUG] 摄像头不可用: {e}")
        
        print("[DEBUG] 初始化完成，即将进入主循环")

    def _init_fonts(self):
        """
        字体初始化，完全不硬编码 C:\\ 路径。
        查找顺序：
        1. 项目目录 fonts/ 子目录（方便用户自备字体，绿色便携）
        2. 系统字体目录（通过 WINDIR / SystemRoot 环境变量动态获取）
        3. pygame 内置默认字体（跨平台回退）
        """
        self._fonts = {}

        # 1️⃣ 优先：项目目录下的 fonts/ 子目录（绿色便携，不依赖系统）
        project_dir = Path(__file__).parent.resolve()
        project_fonts_dir = project_dir / "fonts"
        project_candidates = []
        if project_fonts_dir.exists():
            project_candidates = sorted(
                [str(f) for f in project_fonts_dir.iterdir()
                 if f.suffix.lower() in {".ttf", ".ttc", ".otf"}]
            )

        # 2️⃣ 系统字体目录（动态获取，不硬编码 C:\\）
        sys_fonts = []
        windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
        if windir:
            sys_font_dir = Path(windir) / "Fonts"
            sys_fonts = [
                str(sys_font_dir / "msyh.ttc"),    # 微软雅黑
                str(sys_font_dir / "msyh.ttf"),
                str(sys_font_dir / "simhei.ttf"),  # 黑体
                str(sys_font_dir / "simsun.ttc"),  # 宋体
                str(sys_font_dir / "msjhl.ttc"),   # 微软雅黑 Light
                str(sys_font_dir / "malgun.ttf"),  # 韩文 but 有中文
            ]

        # 合并候选列表（项目优先）
        all_candidates = project_candidates + sys_fonts

        font_path = None
        for fp in all_candidates:
            if os.path.exists(fp):
                font_path = fp
                break

        sizes = [("large", 36), ("medium", 22), ("small", 16), ("tiny", 13)]

        if font_path:
            for key, size in sizes:
                self._fonts[key] = pygame.font.Font(font_path, size)
            print(f"[DEBUG] 使用字体: {font_path}")
        else:
            print("[DEBUG] 未找到中文字体，回退到 pygame 默认字体（可能显示为方框）")
            # 回退到 pygame 内置的默认字体
            default_name = pygame.font.get_default_font()  # 通常是 "freesansbold.ttf"
            # 尝试找到完整路径
            pygame_dir = os.path.dirname(pygame.__file__)
            full_path = os.path.join(pygame_dir, default_name)
            if os.path.exists(full_path):
                default = full_path
            else:
                default = default_name

            for key, size in sizes:
                try:
                    self._fonts[key] = pygame.font.Font(default, size)
                except Exception:
                    self._fonts[key] = pygame.font.Font(None, size)  # None 表示 pygame 默认字体
    def _font(self, key: str):
        return self._fonts.get(key, self._fonts['medium'])

    def _add_log(self, message: str):
        """向日志缓冲区追加消息，保留最近 15 条"""
        self.log_messages.append((message, time.time()))
        if len(self.log_messages) > 15:
            self.log_messages.pop(0)

    def run(self):
        """主事件循环：处理输入 → 更新逻辑 → 渲染画面"""
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self._heartbeat()
            self.clock.tick(self.FPS)
        self._cleanup()

    def _heartbeat(self):
        """诊断心跳：每秒在窗口标题更新帧计数与键盘焦点状态。

        - 帧计数持续增加 = 主循环存活
        - 焦点状态 = 键盘事件能否进入本窗口（False 时按键无效，需点击窗口）
        """
        self._frame_count = getattr(self, '_frame_count', 0) + 1
        if self._frame_count % 30 == 0:
            focused = pygame.key.get_focused()
            state = "焦点OK" if focused else "无焦点-请点击本窗口"
            pygame.display.set_caption(
                f"SoulPlayer | 帧:{self._frame_count} | {state}"
            )

    def _handle_events(self):
        """处理 Pygame 事件：窗口关闭、键盘快捷键"""
        for event in pygame.event.get():
            # 调试：打印所有键盘相关事件
            if event.type in (pygame.KEYDOWN, pygame.KEYUP, pygame.TEXTINPUT):
                print(f"[EVENT] type={event.type}, key={getattr(event, 'key', 'N/A')}, unicode={getattr(event, 'unicode', 'N/A')}")
            
            if event.type == pygame.QUIT:
                self.running = False
                continue

            # 重命名输入模式：键盘事件全部进入输入框，不触发任何快捷键
            if self._renaming_preset is not None:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._cancel_rename()
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self._confirm_rename()
                    elif event.key == pygame.K_BACKSPACE:
                        self._rename_text = self._rename_text[:-1]
                elif event.type == pygame.TEXTINPUT:
                    self._rename_text += event.text
                continue

            # 删除确认模式：Enter 确认 / Esc 取消，或点击弹窗按钮
            if self._deleting_preset is not None:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self._confirm_delete()
                    elif event.key == pygame.K_ESCAPE:
                        self._cancel_delete()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, act in getattr(self, '_delete_btn_rects', []):
                        if rect.collidepoint(event.pos):
                            if act == 'confirm':
                                self._confirm_delete()
                            else:
                                self._cancel_delete()
                            break
                continue

            # 音乐列表弹窗：点击行播放 / 点击关闭按钮 / Esc 关闭 / 滚轮翻页
            if self._music_list_open:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._music_list_open = False
                        if self.anim:
                            self.anim.close_modal()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, act in self._music_list_rects:
                        if rect.collidepoint(event.pos):
                            if act == "close":
                                self._music_list_open = False
                                if self.anim:
                                    self.anim.close_modal()
                            elif isinstance(act, int):
                                self._play_from_music_list(act)
                            break
                elif event.type == pygame.MOUSEWHEEL:
                    self._music_list_scroll = max(0, self._music_list_scroll - event.y)
                continue

            # 选歌面板：Enter 保存 / Esc 取消 / 点击行切换勾选 / 滚轮翻页
            if self._song_select is not None:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self._confirm_song_select()
                    elif event.key == pygame.K_ESCAPE:
                        self._cancel_song_select()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, (act, idx) in self._song_select_rects:
                        if rect.collidepoint(event.pos):
                            if act == "toggle":
                                self._toggle_song_select_row(idx)
                            elif act == "confirm":
                                self._confirm_song_select()
                            else:
                                self._cancel_song_select()
                            break
                elif event.type == pygame.MOUSEWHEEL:
                    self._song_select["scroll"] = max(
                        0, self._song_select["scroll"] - event.y)
                continue

            # 段落选择面板：Enter 保存 / Esc 跳过 / 点击按钮或拖动进度条选择段落
            if self._section_select is not None:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self._confirm_section_select()
                    elif event.key == pygame.K_ESCAPE:
                        self._skip_section_select()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, act in getattr(self, '_section_select_rects', []):
                        if rect.collidepoint(event.pos):
                            if act == "confirm":
                                self._confirm_section_select()
                            elif act == "skip":
                                self._skip_section_select()
                            elif act == "timeline":
                                # 点击时间轴：按鼠标 x 定位段落并进入拖拽
                                self._section_dragging = True
                                pct = max(0.0, min(1.0, (event.pos[0] - rect.x) / max(1, rect.width)))
                                self._section_select_set_start(pct)
                            elif act.startswith("preset:"):
                                pct = float(act.split(":")[1])
                                self._section_select_set_start(pct)
                            break
                elif event.type == pygame.MOUSEMOTION and self._section_dragging:
                    # 拖拽中：跟随鼠标 x 实时更新段落
                    for rect, act in getattr(self, '_section_select_rects', []):
                        if act == "timeline":
                            pct = max(0.0, min(1.0, (event.pos[0] - rect.x) / max(1, rect.width)))
                            self._section_select_set_start(pct)
                            break
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._section_dragging = False
                continue

            elif event.type == pygame.MOUSEWHEEL:
                vlr = getattr(self, '_voice_list_rect', None)
                if vlr and vlr.collidepoint(pygame.mouse.get_pos()):
                    templates = []
                    if self.voice_trigger and hasattr(self.voice_trigger, 'list_templates'):
                        templates = self.voice_trigger.list_templates()
                    max_scroll = max(0, len(templates) - 3)
                    self._voice_scroll = max(0, min(self._voice_scroll - event.y, max_scroll))
                plr = getattr(self, '_playlist_rect', None)
                if plr and plr.collidepoint(pygame.mouse.get_pos()):
                    self._play_scroll = max(0, self._play_scroll - event.y)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # 底部信息区页签切换
                for rect, tab in getattr(self, '_bottom_tab_rects', []):
                    if rect.collidepoint(event.pos):
                        self._bottom_tab = tab
                        break
                # 音乐列表按钮：打开歌单弹窗
                btn_rect = getattr(self, '_music_list_btn_rect', None)
                if btn_rect and btn_rect.collidepoint(event.pos):
                    self._music_list_open = True
                    self._music_list_scroll = 0
                    if self.anim:
                        self.anim.open_modal()
                    continue
                # 音量条点击/拖拽调节（热区在进度条上方，优先检测）
                vol_rect = getattr(self, '_vol_bar_rect', None)
                if vol_rect and vol_rect.collidepoint(event.pos):
                    self._vol_dragging = True
                    self._update_volume_from_mouse(event.pos[0])
                    continue
                # 优先检测进度条点击
                seek_rect = getattr(self, '_seek_bar_rect', None)
                if seek_rect and seek_rect.collidepoint(event.pos):
                    self._seek_dragging = True
                    self._update_seek_from_mouse(event.pos[0])
                    continue
                # 左键点击：优先预览框骨架显隐，其次语音模板编辑按钮，再次自定义预设
                sk_rect = getattr(self, '_skeleton_btn_rect', None)
                if sk_rect and sk_rect.collidepoint(event.pos):
                    self._skeleton_visible = not self._skeleton_visible
                    self._add_log(f"🦴 动作骨架已{'显示' if self._skeleton_visible else '隐藏'}")
                    continue
                voice_hit = False
                for rect, tpl_name, action in getattr(self, '_voice_actions', []):
                    if rect.collidepoint(event.pos):
                        tracks = (self.voice_trigger.get_tracks(tpl_name)
                                  if self.voice_trigger else [])
                        if action == "rename":
                            # 用 Tk 原生对话框改名（SDL 无法显示微软拼音候选框，
                            # 见 _ask_rename_dialog 说明）；Enter/Esc 后直接确认/取消
                            ref = _VoiceTemplateRef(tpl_name, tracks)
                            new_name = self._ask_rename_dialog(ref)
                            if new_name:
                                self._renaming_preset = ref
                                self._rename_text = new_name
                                self._confirm_rename()
                            else:
                                self._add_log(f"✏️ 已取消重命名 [{tpl_name}]")
                        elif action == "delete":
                            self._deleting_preset = _VoiceTemplateRef(tpl_name, tracks)
                            self._add_log(f"🗑 确认删除语音模板 [{tpl_name}]？（Enter 确认 / Esc 取消）")
                        elif action == "favorite":
                            if hasattr(self.voice_trigger, 'toggle_favorite'):
                                state = self.voice_trigger.toggle_favorite(tpl_name)
                                if state:
                                    self._add_log(f"⭐ 已收藏台词 [{tpl_name}]，置顶显示")
                                else:
                                    self._add_log(f"☆ 已取消收藏台词 [{tpl_name}]")
                        elif action == "move_left":
                            if hasattr(self.voice_trigger, 'move_template'):
                                if self.voice_trigger.move_template(tpl_name, -1):
                                    self._add_log(f"← [{tpl_name}] 已前移")
                        elif action == "move_right":
                            if hasattr(self.voice_trigger, 'move_template'):
                                if self.voice_trigger.move_template(tpl_name, +1):
                                    self._add_log(f"→ [{tpl_name}] 已后移")
                        voice_hit = True
                        break
                if not voice_hit:
                    self._handle_custom_preset_click(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self._vol_dragging:
                    self._vol_dragging = False
                    self._update_volume_from_mouse(event.pos[0])
                if self._seek_dragging:
                    self._seek_dragging = False
                    self._update_seek_from_mouse(event.pos[0])

            elif event.type == pygame.MOUSEMOTION:
                if self._vol_dragging:
                    self._update_volume_from_mouse(event.pos[0])
                elif self._seek_dragging:
                    self._update_seek_from_mouse(event.pos[0])

            elif event.type == pygame.KEYDOWN:
                # 退出
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                # 播放控制
                elif event.key == pygame.K_SPACE:
                    if self.player.is_playing:
                        self.player.pause()
                        self._add_log("手动暂停")
                    else:
                        self.player.play()
                        self._add_log("手动播放")
                elif event.key == pygame.K_RIGHT:
                    self.player.next_track()
                    self._add_log("手动下一首")
                elif event.key == pygame.K_LEFT:
                    self.player.previous_track()
                    self._add_log("手动上一首")
                # 音量
                elif event.key == pygame.K_UP:
                    self.player.set_volume(self.player._volume + 0.05)
                    self._vol_popup_until = time.time() + 1.2
                    self._add_log(f"音量上调至 {self.player._volume:.0%}")
                elif event.key == pygame.K_DOWN:
                    self.player.set_volume(self.player._volume - 0.05)
                    self._vol_popup_until = time.time() + 1.2
                    self._add_log(f"音量下调至 {self.player._volume:.0%}")
                # 自定义预设录制（KEYDOWN 事件路径 + get_pressed 轮询双保险，
                # 内部 0.3 秒防抖防止两条路径重复触发同一次按键）
                # 语音台词录制期间屏蔽 R/V，防止打断录制
                elif event.key == pygame.K_r and self._voice_rec is None:
                    self._handle_recording_key()
                # 语音台词录制（倒计时 → 录台词 → 选歌绑定）
                elif event.key == pygame.K_t:
                    self._handle_voice_record_key()
                # 语音触发监听切换
                elif event.key == pygame.K_v and self._voice_rec is None:
                    self._toggle_voice_listening()
                # 音乐列表开关
                elif event.key == pygame.K_m:
                    self._music_list_open = not self._music_list_open
                    if self._music_list_open:
                        self._music_list_scroll = 0
                        if self.anim:
                            self.anim.open_modal()
                    else:
                        if self.anim:
                            self.anim.close_modal()

    def _update_volume_from_mouse(self, mouse_x: int):
        """根据鼠标 X 坐标设置音量（0.0~1.0），并触发数值回显弹标。"""
        bar = getattr(self, '_vol_bar_rect', None)
        if bar is None or bar.width <= 0:
            return
        vol = (mouse_x - bar.x) / bar.width
        vol = min(1.0, max(0.0, vol))
        try:
            self.player.set_volume(vol)
            self._vol_popup_until = time.time() + 1.2
        except Exception:
            pass

    def _update_seek_from_mouse(self, mouse_x: int):
        """根据鼠标 X 坐标计算并设置播放进度（0.0~1.0）。"""
        seek_rect = getattr(self, '_seek_bar_rect', None)
        if seek_rect is None or seek_rect.width <= 0:
            return
        progress = (mouse_x - seek_rect.x) / seek_rect.width
        progress = min(1.0, max(0.0, progress))
        try:
            dur = self.player.get_track_duration()
            if dur > 0:
                self.player.seek_to(progress * dur)
        except Exception:
            pass

    def _update(self):
        """
        核心更新逻辑：
        1. 获取最新动作特征
        2. 传入预设引擎匹配
        3. 若触发动作，调用处理器
        4. 更新动画系统
        """
        # 备选：检测按键边沿（只在按键"刚按下"时触发，避免按住时反复触发）
        keys = pygame.key.get_pressed()
        # 重命名/删除/选歌弹窗或语音录制期间，字母键不触发 R/V 快捷键
        renaming = (self._renaming_preset is not None
                    or self._deleting_preset is not None
                    or self._song_select is not None
                    or self._section_select is not None
                    or self._voice_rec is not None
                    or self._music_list_open)

        # R 键边沿检测
        r_now = keys[pygame.K_r] and not renaming
        r_prev = getattr(self, '_r_key_prev', False)
        self._r_key_prev = r_now
        if r_now and not r_prev:
            self._handle_recording_key()

        # V 键边沿检测
        v_now = keys[pygame.K_v] and not renaming
        v_prev = getattr(self, '_v_key_prev', False)
        self._v_key_prev = v_now
        if v_now and not v_prev:
            self._toggle_voice_listening()

        # T 键边沿检测（语音台词录制；录制进行中也允许按 T 结束，故不受 renaming 守卫限制，
        # 但弹窗打开时仍屏蔽）
        t_blocked = (self._renaming_preset is not None
                     or self._deleting_preset is not None
                     or self._song_select is not None
                     or self._section_select is not None
                     or self._music_list_open)
        t_now = keys[pygame.K_t] and not t_blocked
        t_prev = getattr(self, '_t_key_prev', False)
        self._t_key_prev = t_now
        if t_now and not t_prev:
            self._handle_voice_record_key()

        # 每 2 秒扫描一次 music 文件夹，动态检测新增/移除的歌曲
        now_scan = time.time()
        if now_scan - self._last_music_scan > 2.0:
            self._last_music_scan = now_scan
            self._sync_music_folder()

        try:
            self.current_features = self.motion.get_latest_motion_features()
        except Exception as e:
            self._add_log(f"获取动作特征异常: {e}")
            return
        # 更新动画系统（特征值平滑过渡等）
        if self.anim and self.current_features:
            self.anim.set_feature_targets(self.current_features)
        if self.anim:
            self.anim.update()

        action = self.preset_engine.match(self.current_features)

        if action:
            self._handle_action(action)
            self.last_action = action
            self.action_timestamp = time.time()
            # 兼容真正的 PresetEngine（active_preset 是对象）和桩实现（active_preset 是字符串）
            active = self.preset_engine.active_preset
            if active is not None:
                preset_name = getattr(active, 'name', active)  # 对象取 .name，字符串直接取
            else:
                preset_name = "未知"
            self._add_log(f"触发 [{preset_name}] → {action}")
            # 触发动作闪光动画
            if self.anim:
                self.anim.trigger_action_flash(980, 200)

        # 语音台词触发：非阻塞检查识别结果
        if self.voice_trigger is not None:
            voice_action = self.voice_trigger.get_triggered_action()
            if voice_action:
                self._handle_action(voice_action)
                self.last_action = voice_action
                self.action_timestamp = time.time()
                self._add_log(f"🎤 语音触发 → {voice_action}")
                # 触发动作闪光动画
                if self.anim:
                    self.anim.trigger_action_flash(980, 200)

        # 录制模式：每帧传入特征
        if self.recording_mode and self.custom_manager:
            self._update_recording()

        # 语音台词录制状态机：每帧推进（倒计时→录制→超时）
        if self._voice_rec is not None:
            self._update_voice_recording()

        # 人声/扬声器分离：说话期间自动压低音乐音量
        self._update_voice_ducking()

    def _update_voice_ducking(self):
        """检测到说话（或录制台词）时自动压低音乐音量，说完约 1.5 秒后恢复。

        降低音量可减少扬声器音乐被麦克风拾取导致的模板误匹配/录音混杂；
        1.5 秒保持时间避免 VAD 抖动引起音量来回跳动。"""
        now = time.time()
        try:
            speech = bool(
                self.voice_trigger is not None
                and getattr(self.voice_trigger, 'is_listening', False)
                and getattr(self.voice_trigger, 'speech_active', False)
            )
            playing = bool(self.player.is_playing)
        except Exception:
            return
        if speech and playing:
            self._duck_until = now + 1.5
            if self._voice_ducked is None:
                self._voice_ducked = float(getattr(self.player, '_volume', 0.5))
                try:
                    self.player.set_volume(max(0.12, self._voice_ducked * 0.45))
                except Exception:
                    self._voice_ducked = None
        elif self._voice_ducked is not None and now >= self._duck_until:
            try:
                self.player.set_volume(self._voice_ducked)
            except Exception:
                pass
            self._voice_ducked = None

    def _music_dir(self) -> Path:
        """
        统一的音乐文件夹定位。

        打包成 exe 后，PyInstaller 会把 music 复制到 _internal 里成为"打包时的旧副本"，
        用户新增的歌曲在源码目录 SoulPlayer/music 或 exe 旁边，导致"文件夹有 6 首只检测到 4 首"。
        因此按优先级探测：
        1. 项目源码目录的 music（exe 上两级，即 dist/SoulPlayer/../../music）
        2. exe 旁边的 music（整包拷贝到别的电脑时的标准位置）
        3. 脚本目录/_internal 里的 music（打包副本，兜底）
        返回第一个"存在且含音频文件"的目录；都不含则返回第一个存在的目录；
        全不存在时创建并返回首选目录。
        """
        candidates = []
        if getattr(sys, 'frozen', False):
            exe_dir = Path(sys.executable).parent
            candidates.append(exe_dir.parent.parent / "music")  # 项目源码目录
            candidates.append(exe_dir / "music")                # exe 旁边
        candidates.append(Path(__file__).parent / "music")      # 源码 / _internal

        for c in candidates:
            try:
                if c.is_dir() and any(
                    p.is_file() and p.suffix.lower() in self.AUDIO_EXTS
                    for p in c.rglob("*")
                ):
                    return c
            except OSError:
                continue
        for c in candidates:
            if c.is_dir():
                return c
        fallback = candidates[0]
        try:
            fallback.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return fallback

    def _data_dir(self, name: str) -> Path:
        """
        统一的用户数据目录定位（自定义动作预设、语音台词模板等）。

        与 _music_dir() 同理：打包成 exe 后数据若只存在 _internal 里，
        重新打包时整个 dist 会被删除导致数据丢失；且源码运行与 exe 运行
        读到的不是同一份数据，导致"上次录的台词下次打开触发别的模板"。
        因此按优先级探测：项目源码目录 > exe 旁边 > _internal/脚本目录，
        选第一个"存在且非空"的目录；若它不是首选目录，则把内容
        （连同配套的 <name>.json 映射文件，位于目录的上一级）复制到
        首选目录后返回首选；迁移失败时沿用原目录。
        """
        candidates = []
        if getattr(sys, 'frozen', False):
            exe_dir = Path(sys.executable).parent
            candidates.append(exe_dir.parent.parent / name)  # 项目源码目录
            candidates.append(exe_dir / name)                # exe 旁边
        candidates.append(Path(__file__).parent / name)      # 源码 / _internal

        def _has_content(d: Path) -> bool:
            try:
                return d.is_dir() and any(d.iterdir())
            except OSError:
                return False

        chosen = None
        for c in candidates:
            if _has_content(c):
                chosen = c
                break
        if chosen is None:
            for c in candidates:
                if c.is_dir():
                    chosen = c
                    break
        if chosen is None:
            chosen = candidates[0]
            try:
                chosen.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            return chosen

        preferred = candidates[0]
        if chosen != preferred:
            try:
                preferred.mkdir(parents=True, exist_ok=True)
                for item in chosen.iterdir():
                    dest = preferred / item.name
                    if dest.exists():
                        continue
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                # 配套映射文件（如 voice_templates.json）与目录同级
                sidecar_src = chosen.parent / f"{name}.json"
                sidecar_dst = preferred.parent / f"{name}.json"
                if sidecar_src.is_file() and not sidecar_dst.exists():
                    shutil.copy2(sidecar_src, sidecar_dst)
                print(f"[数据迁移] {name}: {chosen} -> {preferred}")
                return preferred
            except OSError as e:
                print(f"[数据迁移失败，沿用原目录] {name}: {e}")
        return chosen

    def _sync_music_folder(self):
        """每 2 秒调用一次：检测 music 文件夹中新增/移除的歌曲并同步播放列表。"""
        if not hasattr(self.player, 'sync_folder'):
            return
        try:
            music_dir = self._music_dir()
            added, removed = self.player.sync_folder(str(music_dir))
            if added:
                names = ", ".join(os.path.basename(t) for t in added[:3])
                suffix = f" 等 {len(added)} 首" if len(added) > 3 else ""
                self._add_log(f"🎵 检测到新歌曲: {names}{suffix}")
            if removed:
                self._add_log(f"🗑 {len(removed)} 首歌曲已从文件夹移除")
        except Exception as e:
            print(f"[DEBUG] 音乐文件夹扫描异常: {e}")

    def _handle_action(self, action: str):
        """
        将预设动作映射到播放器指令。
        与 ARCHITECTURE.md 中 handle_action 定义保持一致。
        对瞬时动作（pause/play/next/prev）添加 1 秒冷却期，防止连续触发。
        """
        # 挥手方向冷却（同向/反向分离）：
        # - 同向冷却 0.35s：防止同方向机关枪式连发（match 每帧返回），
        #   但间隔 >0.35s 的同向连续挥手可正常触发 → 连贯切歌不受影响；
        # - 反向冷却 0.9s：挥完收手的"回正"是一次紧跟触发的反向横扫
        #   （挥右切下一首后把手收回 = 向左横扫），若冷却太短会被误判为
        #   "上一首"——"收手时最容易串成反向"的根因。0.9s 覆盖完整
        #   收手过程（挥出+收回约 0.4~0.8s）；真正想切反方向需间隔 >0.9s，
        #   与 direction 首段判定（motion_capture 侧）双保险。
        _WAVE_ACTIONS = {"player_next", "player_prev"}
        if action in _WAVE_ACTIONS:
            now = time.time()
            other = "player_prev" if action == "player_next" else "player_next"
            last_same = self._action_cooldowns.get(action, 0.0)
            last_other = self._action_cooldowns.get(other, 0.0)
            if now - last_same < 0.35 or now - last_other < 0.9:
                return
            self._action_cooldowns[action] = now
            # 挥手触发后同步屏蔽自定义：挥手的收手/回正余波与自定义动作
            # 的轨迹/特征模板相近，若不屏蔽，余波会被识别为自定义动作
            # 播放（"做通用动作却串自定义动作"的根因，与下方 custom_play
            # 屏蔽 player_wave 形成双向隔离）。
            self._action_cooldowns["custom_play"] = now

        # 瞬时动作冷却期：1 秒内不重复触发同一动作
        # custom_play 也在内：自定义条件在触发后仍会持续满足若干帧，
        # 不冷却会每帧重启歌曲（"机关枪"式重触发）
        _COOLDOWN_ACTIONS = {"player_pause", "player_play", "custom_play"}
        if action in _COOLDOWN_ACTIONS:
            now = time.time()
            last_trigger = self._action_cooldowns.get(action, 0.0)
            if now - last_trigger < 1.0:
                return  # 冷却期内，忽略本次触发
            self._action_cooldowns[action] = now

        # 自定义动作与挥手切歌互斥：自定义动作触发后 1 秒内屏蔽下一首/上一首。
        # 自定义动作的收手/回正余波与挥手特征相似，若不屏蔽会被识别为挥手误切歌。
        #（player_wave 旧共用 key 已拆分为 player_next/player_prev 双向）
        if action == "custom_play":
            _cw = time.time()
            self._action_cooldowns["player_next"] = _cw
            self._action_cooldowns["player_prev"] = _cw

        if action.startswith("mood_"):
            mood = action.replace("mood_", "")
            try:
                self.player.set_playlist_by_mood(mood)
                self._add_log(f"切换情绪列表: {mood}")
            except Exception as e:
                self._add_log(f"切换列表失败: {e}")
        elif action == "player_pause":
            self.player.pause()
        elif action == "player_play":
            self.player.play()
        elif action == "player_next":
            try:
                self.player.next_track()
            except Exception as e:
                self._add_log(f"下一首失败: {e}")
        elif action == "player_prev":
            try:
                self.player.previous_track()
            except Exception as e:
                self._add_log(f"上一首失败: {e}")
        elif action == "custom_play":
            if self.custom_manager:
                preset_name = self.preset_engine.active_preset
                if preset_name is not None:
                    preset_name = getattr(preset_name, 'name', preset_name)
                track = self.custom_manager.pick_track(preset_name)
                if track:
                    try:
                        self.player.play_track(track)
                        # 从录制时选择的起始段落开始播放
                        start_pct = 0.0
                        preset = self.custom_manager.get_preset(preset_name)
                        if preset is not None:
                            start_pct = getattr(preset, 'start_pct', 0.0) or 0.0
                        self._seek_to_start_pct(start_pct)
                        self._add_log(f"自定义播放: {os.path.basename(track)}")
                    except Exception as e:
                        self._add_log(f"播放失败: {e}")
                else:
                    self._add_log("无有效歌曲")
        elif action.startswith("voice_play:"):
            # 语音台词触发：从该台词关联的歌曲中随机播放一首
            phrase = action.split(":", 1)[1]
            if self.voice_trigger is not None:
                tracks = self.voice_trigger.get_tracks(phrase)
                if tracks:
                    track = random.choice(tracks)
                    try:
                        self.player.play_track(track)
                        # 从录制时选择的起始段落开始播放
                        start_pct = self.voice_trigger.get_start_pct(phrase)
                        self._seek_to_start_pct(start_pct)
                        self._add_log(f"🎤 台词 [{phrase}] 播放: {os.path.basename(track)}")
                    except Exception as e:
                        self._add_log(f"播放失败: {e}")
                else:
                    self._add_log(f"台词 [{phrase}] 没有关联歌曲")
        else:
            self._add_log(f"未知动作: {action}")

    def _seek_to_start_pct(self, start_pct: float):
        """按起始段落百分比（0.0~1.0）跳转到对应播放位置；0.0 或时长未知时忽略。"""
        try:
            if not start_pct:
                return
            duration = self.player.get_track_duration()
            if duration and duration > 0:
                self.player.seek_to(start_pct * duration)
        except Exception:
            pass

    # ---------- 语音监听控制 ----------

    def _play_from_music_list(self, idx: int):
        """从音乐列表弹窗播放指定索引曲目（保持弹窗打开，便于继续换歌）。"""
        try:
            self.player.play(idx)
            track = self.player.current_track
            title = (track.get('title') or track.get('filename')
                     or f"曲目 {idx + 1}") if isinstance(track, dict) else f"曲目 {idx + 1}"
            self._add_log(f"手动选择播放: {title}")
        except Exception as e:
            self._add_log(f"播放失败: {e}")

    def _toggle_voice_listening(self):
        """切换语音监听开关状态（V 键触发，0.3 秒防抖避免双路径重复触发）。"""
        now = time.time()
        if now - getattr(self, '_last_v_handle_time', 0.0) < 0.3:
            return
        self._last_v_handle_time = now

        if self.voice_trigger is None:
            self._add_log("语音模块未加载")
            return
        status = self.voice_trigger.get_status()
        if status.get("listening"):
            self.voice_trigger.stop_listening()
            self._add_log("🎤 语音监听已关闭")
        elif self.voice_trigger.available:
            self.voice_trigger.start_listening()
            self._add_log("🎤 语音监听已启动")
        else:
            err = status.get("status", "未知原因")
            self._add_log(f"🎤 语音未就绪: {err}")

    # ---------- 自定义预设录制方法 ----------

    def _handle_recording_key(self):
        """处理录制按键：idle→倒计时→录制→结束 的状态转换。

        可由 KEYDOWN 事件或 get_pressed 边沿检测两条路径触发，
        用 0.3 秒防抖避免同一次按键被两条路径重复处理。
        """
        # 防抖：0.3 秒内的重复触发视为同一次按键
        now = time.time()
        if now - getattr(self, '_last_r_handle_time', 0.0) < 0.3:
            return
        self._last_r_handle_time = now

        if not self.custom_manager:
            self._add_log("自定义预设模块未加载")
            return

        # 诊断日志：确认按键已被接收（同时输出到控制台）
        state = self.custom_manager.recorder.state
        self._add_log(f"[R键] 当前录制状态: {state}")
        print(f"[R键] 已接收，录制状态: {state}")

        if self.custom_manager.is_countdown():
            # 倒计时中按 R：取消
            self.custom_manager.cancel_recording()
            self.recording_mode = False
            self._add_log("❌ 录制已取消")
        elif self.custom_manager.is_recording():
            # 录制中按 R：结束录制并保存
            self._finish_recording_preset()
        else:
            # idle 或 finished 状态：开始倒计时
            self.custom_manager.start_recording()
            self.recording_mode = True
            self.recording_preset_name = f"custom_{time.strftime('%H%M%S')}"
            self._countdown_done_notified = False  # 重置通知标志，确保下次录制会显示通知
            self._add_log("⏳ 3秒倒计时：请保持静止（采集中性基线）... 屏幕提示符结束后开始做动作")


    def _update_recording(self):
        """每帧调用：推进录制状态机（倒计时→录制→结束）。"""
        if not self.recording_mode or not self.custom_manager:
            return

        self.custom_manager.feed_recording(self.current_features)
        
        # 每30帧打印一次状态（避免刷屏）
        frame_count = getattr(self, '_debug_frame_count', 0)
        self._debug_frame_count = frame_count + 1
        if frame_count % 30 == 0:
            st = self.custom_manager.recorder.state if self.custom_manager.recorder else "no_recorder"
            rem = self.custom_manager.get_countdown_remaining() if hasattr(self.custom_manager, 'get_countdown_remaining') else 0
            print(f"[DEBUG-UPDATE] state={st}, remaining={rem:.1f}s, recording_mode={self.recording_mode}")

        # 倒计时结束后自动通知用户
        if self.custom_manager.is_recording() and not getattr(self, '_countdown_done_notified', False):
            self._add_log("⏺ 开始录制！请保持动作，按 R 结束")
            self._countdown_done_notified = True
            print("[DEBUG] 倒计时结束，开始录制")

        # 超时自动结束
        if self.custom_manager.is_recording_finished():
            print("[DEBUG] 录制自动结束（超时）")
            self._finish_recording_preset()
            self._countdown_done_notified = False

    def _finish_recording_preset(self):
        """录制完成：打开界面内联选歌面板，保存预设（不使用 tkinter 弹窗）。"""
        if not self.custom_manager:
            return
        if not self.recording_mode:
            # 防卡死：若录制器仍在录制但标志位丢失（异常导致状态不一致），
            # 强制恢复标志位继续走完成流程，避免 R 键永久失灵
            if self.custom_manager.is_recording():
                self.recording_mode = True
            else:
                return

        self.recording_mode = False
        conditions = self.custom_manager.finish_recording()
        summary = self.custom_manager.get_recording_summary()
        self._add_log(f"录制完成: {summary}")

        # 方案A：录制已同时采集"中性基线+动作模板"，触发端用距离匹配
        # （距动作近、距静止基线远才触发），静止误触/判定点不足由匹配
        # 逻辑天然免疫，不再需要 is_idle_matching / count_discriminators 硬校验。
        # 此处仅兜底：完全没有动作样本时直接提示重新录制。
        if not conditions.get("action"):
            self._add_log("⚠️ 未采到有效动作样本，已取消保存（请重新录制）")
            return

        # 打开界面内联选歌面板（替代 tkinter 文件对话框——
        # 打包成 exe 后 tkinter 对话框经常弹不出来/藏在窗口后面）
        self._open_song_select(conditions)

    # ---------- 语音台词录制（T 键：倒计时 → 录台词 → 选歌绑定） ----------
    VOICE_REC_COUNTDOWN = 3.0   # 倒计时时长（秒）
    VOICE_REC_TIMEOUT = 10.0    # 录制超时自动结束（秒）

    def _handle_voice_record_key(self):
        """处理 T 键：idle→倒计时→录制中→结束；倒计时期间再按为取消。
        0.3 秒防抖避免 KEYDOWN 与 get_pressed 双路径重复触发。"""
        now = time.time()
        if now - getattr(self, '_last_t_handle_time', 0.0) < 0.3:
            return
        self._last_t_handle_time = now

        if self.voice_trigger is None:
            self._add_log("语音模块未加载")
            return
        if not self.voice_trigger.available:
            err = (self.voice_trigger.get_status().get("error") or "未知原因")
            self._add_log(f"🎤 语音未就绪: {err}")
            return

        if self._voice_rec is None:
            # 录音前临时暂停正在播放的音乐，避免扬声器声音被麦克风录入
            try:
                playing_now = bool(self.player.is_playing)
            except Exception:
                playing_now = False
            self._voice_rec = {"phase": "countdown", "t0": now,
                               "auto_started": False, "resume_play": playing_now}
            if playing_now:
                try:
                    self.player.pause()
                    self._add_log("⏸ 已临时暂停音乐，保证录音清晰")
                except Exception as e:
                    print(f"[临时暂停失败] {e}")
                    self._voice_rec["resume_play"] = False
            self._add_log("⏳ 3秒后开始语音录制，请准备台词...")
            print("[T键] 语音录制倒计时开始")
        elif self._voice_rec["phase"] == "countdown":
            vr = self._voice_rec
            self._voice_rec = None
            self._restore_music_after_voice_rec(vr)
            self._add_log("❌ 语音录制已取消")
        else:
            self._finish_voice_recording()

    def _update_voice_recording(self):
        """每帧推进语音录制状态机。"""
        vr = self._voice_rec
        if vr is None:
            return
        now = time.time()
        if vr["phase"] == "countdown":
            if now - vr["t0"] >= self.VOICE_REC_COUNTDOWN:
                # 倒计时结束：确保语音监听在运行，进入捕获模式
                status = self.voice_trigger.get_status()
                if not status.get("listening"):
                    self.voice_trigger.start_listening()
                    vr["auto_started"] = True
                self.voice_trigger.start_capture()
                vr["phase"] = "recording"
                vr["t0"] = now
                self._add_log("⏺ 开始语音录制！请说出台词，按 T 结束")
                print("[T键] 倒计时结束，开始语音录制")
        elif vr["phase"] == "recording":
            if now - vr["t0"] >= self.VOICE_REC_TIMEOUT:
                self._add_log("⏱ 语音录制超时，自动结束")
                self._finish_voice_recording()

    def _restore_music_after_voice_rec(self, vr):
        """语音录制结束/取消后，恢复录制前被临时暂停的音乐。"""
        if vr and vr.get("resume_play"):
            try:
                self.player.play()  # 暂停状态下 play() 即恢复播放
                self._add_log("▶ 音乐已恢复播放")
            except Exception as e:
                print(f"[恢复播放失败] {e}")

    def _finish_voice_recording(self):
        """结束语音录制：保存音频模板，打开选歌面板绑定歌曲。"""
        vr = self._voice_rec
        self._voice_rec = None
        if vr is None:
            return
        self._restore_music_after_voice_rec(vr)
        name = self.voice_trigger.stop_capture()   # 返回模板名，无效录音返回 ""
        if not name:
            self._add_log("⚠️ 录音太短或没有声音，录制取消（请靠近麦克风清晰说出）")
            return
        # 复用选歌面板（voice 模式，phrase 即模板名）
        self._open_song_select(None, mode="voice", phrase=name)

    # ---------- 录制后选歌面板（内联覆盖层） ----------
    AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac"}

    def _open_song_select(self, conditions, mode="motion", phrase=""):
        """扫描 music 文件夹并进入选歌模式。

        :param mode: "motion"（动作预设）或 "voice"（语音台词）。
        :param phrase: voice 模式下的台词文本。
        """
        music_dir = self._music_dir()
        music_dir.mkdir(parents=True, exist_ok=True)
        songs = sorted(
            str(p) for p in music_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.AUDIO_EXTS
        )
        if not songs:
            self._add_log("⚠️ music 文件夹中没有歌曲，预设未保存")
            return
        self._song_select = {
            "mode": mode,
            "phrase": phrase,
            "name": phrase if mode == "voice" else self.recording_preset_name,
            "conditions": conditions,
            "songs": songs,
            "selected": set(),  # 默认无勾选：需手动选择，避免误绑全部歌曲
            "scroll": 0,
        }
        self._add_log(f"🎵 请选择关联歌曲（单选 1 首，点击选择，Enter 确认 / Esc 取消）")

    def _cancel_song_select(self):
        self._song_select = None
        self._song_select_rects = []
        self._add_log("已取消，预设未保存")

    def _confirm_song_select(self):
        sel = self._song_select
        self._song_select = None
        self._song_select_rects = []
        if sel is None:
            return
        tracks = [sel["songs"][i] for i in sorted(sel["selected"])]
        if not tracks:
            self._add_log("未选择歌曲，预设未保存")
            return
        # 打开段落选择（非强制步骤）
        self._open_section_select(sel, tracks)

    def _open_section_select(self, song_sel, tracks):
        """打开录制后的段落选择覆盖层（非强制）。"""
        self._section_select = {
            "mode": song_sel.get("mode", "motion"),
            "phrase": song_sel.get("phrase", ""),
            "name": song_sel.get("name", ""),
            "conditions": song_sel.get("conditions"),
            "tracks": tracks,
            "start_pct": 0.0,   # 默认从头开始（0.0~1.0）
            "label": "从头播放",
        }
        self._add_log("🎵 请选择播放起始段落（Enter 确认 / Esc 跳过）")

    def _section_select_set_start(self, pct: float, label: str = ""):
        """设置段落选择的起始百分比。"""
        if self._section_select is None:
            return
        # 自动映射常用标签
        if not label:
            preset_labels = {0.0: "从头播放", 0.15: "前奏结束", 0.30: "主歌开始",
                             0.50: "副歌开始", 0.70: "第二段"}
            label = preset_labels.get(round(pct, 2), f"{int(pct * 100)}%")
        self._section_select["start_pct"] = pct
        self._section_select["label"] = label
        self._add_log(f"段落选择: {label}")

    def _confirm_section_select(self):
        """确认段落选择并保存预设/台词（start_pct 写入存储）。"""
        sec = self._section_select
        self._section_select = None
        self._section_select_rects = []
        if sec is None:
            return
        tracks = sec["tracks"]
        start_pct = sec.get("start_pct", 0.0)
        label = sec.get("label", "从头播放")
        if sec.get("mode") == "voice":
            self.voice_trigger.add_trigger(sec["phrase"], "voice_play", tracks, start_pct=start_pct)
            self._add_log(f"✅ 台词 [{sec['phrase']}] 已保存，关联 {len(tracks)} 首歌曲，起始段落: {label}（按 V 开启监听后生效）")
            return
        preset = self.custom_manager.add_preset(
            name=sec["name"],
            conditions=sec["conditions"],
            tracks=tracks,
            start_pct=start_pct,
        )
        try:
            self.preset_engine.add_preset(preset)
            self._add_log(f"✅ 预设 [{preset.name}] 已保存，关联 {len(tracks)} 首歌曲，起始段落: {label}")
        except ValueError:
            self._add_log(f"⚠️ 预设 [{preset.name}] 已保存，但引擎中已有同名预设")

    def _skip_section_select(self):
        """跳过段落选择（默认从头播放）并保存。"""
        sec = self._section_select
        self._section_select = None
        self._section_select_rects = []
        if sec is None:
            return
        tracks = sec["tracks"]
        if sec.get("mode") == "voice":
            self.voice_trigger.add_trigger(sec["phrase"], "voice_play", tracks, start_pct=0.0)
            self._add_log(f"✅ 台词 [{sec['phrase']}] 已保存，关联 {len(tracks)} 首歌曲（按 V 开启监听后生效）")
            return
        preset = self.custom_manager.add_preset(
            name=sec["name"],
            conditions=sec["conditions"],
            tracks=tracks,
            start_pct=0.0,
        )
        try:
            self.preset_engine.add_preset(preset)
            self._add_log(f"✅ 预设 [{preset.name}] 已保存，关联 {len(tracks)} 首歌曲")
        except ValueError:
            self._add_log(f"⚠️ 预设 [{preset.name}] 已保存，但引擎中已有同名预设")

    def _toggle_song_select_row(self, index):
        sel = self._song_select
        if sel is None or not (0 <= index < len(sel["songs"])):
            return
        # 单选：一次自定义动作只关联 1 首歌曲
        if index in sel["selected"]:
            sel["selected"].discard(index)   # 再次点击取消
        else:
            sel["selected"] = {index}        # 选中当前项，替换之前的选择

    def _draw_song_select_overlay(self):
        """绘制选歌覆盖层（与重命名/删除弹窗同风格）。"""
        if self._song_select is None:
            self._song_select_rects = []
            return

        sel = self._song_select
        accent = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
        bg = SoulTheme.BG_PANEL if SoulTheme else self.COLOR_PANEL
        bg_input = SoulTheme.BG_DEEP if SoulTheme else (25, 25, 25)
        text_c = SoulTheme.TEXT_PRIMARY if SoulTheme else self.COLOR_TEXT
        muted = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM

        # 半透明遮罩
        overlay = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        # 面板
        box = pygame.Rect(340, 190, 600, 400)
        pygame.draw.rect(self.screen, bg, box, border_radius=8)
        pygame.draw.rect(self.screen, accent, box, 2, border_radius=8)

        title_text = "选择关联歌曲（语音台词）" if sel.get("mode") == "voice" else "选择关联歌曲"
        title = self._font('medium').render(title_text, True, accent)
        self.screen.blit(title, (box.x + 20, box.y + 14))
        sub = self._font('tiny').render(
            f"预设 [{sel['name']}]  |  已选 {len(sel['selected'])} 首（单选）  |  滚轮翻页",
            True, muted)
        self.screen.blit(sub, (box.x + 20, box.y + 46))

        # 歌曲列表（滚动）
        list_rect = pygame.Rect(box.x + 20, box.y + 70, 560, 270)
        row_h = 26
        visible = list_rect.height // row_h
        max_scroll = max(0, len(sel["songs"]) - visible)
        sel["scroll"] = max(0, min(sel["scroll"], max_scroll))
        mouse_pos = pygame.mouse.get_pos()

        old_clip = self.screen.get_clip()
        self.screen.set_clip(list_rect)
        self._song_select_rects = []
        for row in range(visible):
            idx = sel["scroll"] + row
            if idx >= len(sel["songs"]):
                break
            y = list_rect.y + row * row_h
            row_rect = pygame.Rect(list_rect.x, y, list_rect.width, row_h - 2)
            hovered = row_rect.collidepoint(mouse_pos)
            if hovered:
                pygame.draw.rect(self.screen, bg_input, row_rect, border_radius=4)
            # 复选框
            cb = pygame.Rect(row_rect.x + 4, y + 4, 16, 16)
            checked = idx in sel["selected"]
            pygame.draw.rect(self.screen, accent if checked else muted, cb, 2, border_radius=3)
            if checked:
                pygame.draw.line(self.screen, accent, (cb.x + 3, cb.y + 8), (cb.x + 7, cb.y + 12), 2)
                pygame.draw.line(self.screen, accent, (cb.x + 7, cb.y + 12), (cb.x + 13, cb.y + 3), 2)
            name = Path(sel["songs"][idx]).name
            if len(name) > 52:
                name = name[:49] + "..."
            surf = self._font('tiny').render(name, True, text_c if checked else muted)
            self.screen.blit(surf, (cb.right + 8, y + 5))
            self._song_select_rects.append((row_rect, ("toggle", idx)))
        self.screen.set_clip(old_clip)

        # 滚动条
        if max_scroll > 0:
            bar_h = max(30, int(list_rect.height * visible / len(sel["songs"])))
            bar_y = list_rect.y + int((list_rect.height - bar_h) * sel["scroll"] / max_scroll)
            pygame.draw.rect(self.screen, muted,
                             (list_rect.right - 6, bar_y, 4, bar_h), border_radius=2)

        # 按钮：保存 / 取消
        btn_confirm = pygame.Rect(box.x + 20, box.bottom - 46, 120, 32)
        btn_cancel = pygame.Rect(box.x + 156, box.bottom - 46, 120, 32)
        self._song_select_rects.append((btn_confirm, ("confirm", None)))
        self._song_select_rects.append((btn_cancel, ("cancel", None)))
        for rect, label, color in ((btn_confirm, "保存预设", accent), (btn_cancel, "取消", muted)):
            hovered = rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, bg_input, rect, border_radius=6)
            pygame.draw.rect(self.screen, color, rect, 2 if hovered else 1, border_radius=6)
            surf = self._font('small').render(label, True, color)
            self.screen.blit(surf, surf.get_rect(center=rect.center))

        hint = self._font('tiny').render("点击歌曲切换勾选  |  Enter 保存  |  Esc 取消", True, muted)
        self.screen.blit(hint, (box.x + 300, box.bottom - 38))

    def _draw_music_list_overlay(self):
        """绘制音乐列表弹窗（手动选歌播放，与选歌弹窗同风格）。"""
        if not self._music_list_open:
            self._music_list_rects = []
            return

        accent = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
        bg = SoulTheme.BG_PANEL if SoulTheme else self.COLOR_PANEL
        bg_input = SoulTheme.BG_DEEP if SoulTheme else (25, 25, 25)
        text_c = SoulTheme.TEXT_PRIMARY if SoulTheme else self.COLOR_TEXT
        muted = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM

        # 弹窗入场动画：透明度淡入 + 从下方 24px 上滑（ease_out_back 轻微回弹）
        anim_t = 1.0
        if self.anim:
            anim_t = float(getattr(self.anim.modal_in, 'value', 1.0))

        # 半透明遮罩
        overlay = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, int(120 * anim_t)))
        self.screen.blit(overlay, (0, 0))

        # 面板（双层嵌套风格：玻璃层 + 强调色描边）
        box = pygame.Rect(280, 90 + int((1 - anim_t) * 24), 720, 540)
        if self.ui and SoulTheme:
            self.ui.glass_panel(box, color=SoulTheme.BG_PANEL, alpha=235)
            self.ui.corner_brackets(box, SoulTheme.BORDER_LIGHT, length=16)
            self.ui.glow_rect(box, SoulTheme.ACCENT_GLOW, radius=16, glow=2)
            pygame.draw.rect(self.screen, accent, box, 1, border_radius=16)
        else:
            pygame.draw.rect(self.screen, bg, box, border_radius=8)
            pygame.draw.rect(self.screen, accent, box, 2, border_radius=8)

        info = {}
        try:
            info = self.player.get_playlist_info()
        except Exception:
            pass
        tracks = info.get("tracks", []) or []
        current_idx = info.get("current_index", -1)

        title = self._font('medium').render(f"{ICONS.get('music', '♪')} 音乐列表", True, accent)
        self.screen.blit(title, (box.x + 20, box.y + 14))
        sub = self._font('tiny').render(
            f"共 {len(tracks)} 首  |  单击曲目立即播放  |  滚轮翻页  |  Esc 关闭",
            True, muted)
        self.screen.blit(sub, (box.x + 20, box.y + 46))

        list_rect = pygame.Rect(box.x + 20, box.y + 70, 680, 400)
        row_h = 28
        visible = list_rect.height // row_h
        max_scroll = max(0, len(tracks) - visible)
        self._music_list_scroll = max(0, min(self._music_list_scroll, max_scroll))
        mouse_pos = pygame.mouse.get_pos()

        if not tracks:
            hint = self._font('small').render("暂无曲目，请将音乐文件放入 music/ 文件夹", True, muted)
            self.screen.blit(hint, hint.get_rect(center=list_rect.center))
            self._music_list_rects = []
        else:
            old_clip = self.screen.get_clip()
            self.screen.set_clip(list_rect)
            self._music_list_rects = []
            for row in range(visible):
                idx = self._music_list_scroll + row
                if idx >= len(tracks):
                    break
                y = list_rect.y + row * row_h
                row_rect = pygame.Rect(list_rect.x, y, list_rect.width, row_h - 2)
                hovered = row_rect.collidepoint(mouse_pos)
                is_current = (idx == current_idx)
                if hovered:
                    pygame.draw.rect(self.screen, bg_input, row_rect, border_radius=4)
                elif is_current:
                    hl = pygame.Surface(row_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(hl, (*accent, 40), hl.get_rect(), border_radius=4)
                    self.screen.blit(hl, row_rect.topleft)
                # 序号
                num = self._font('tiny').render(f"{idx + 1:>3}", True, muted)
                self.screen.blit(num, (row_rect.x + 4, y + 6))
                # 播放中标记
                icon_x = row_rect.x + 34
                if is_current:
                    mark = self._font('small').render(ICONS.get("play", "▶"), True, accent)
                    self.screen.blit(mark, (icon_x, y + 5))
                # 曲目名
                name = tracks[idx]
                if len(name) > 46:
                    name = name[:43] + "..."
                color = accent if is_current else (text_c if hovered else muted)
                surf = self._font('small').render(name, True, color)
                self.screen.blit(surf, (icon_x + 18, y + 5))
                self._music_list_rects.append((row_rect, idx))
            self.screen.set_clip(old_clip)

            # 滚动条
            if max_scroll > 0:
                bar_h = max(30, int(list_rect.height * visible / len(tracks)))
                bar_y = list_rect.y + int((list_rect.height - bar_h) * self._music_list_scroll / max_scroll)
                pygame.draw.rect(self.screen, muted, (list_rect.right - 6, bar_y, 4, bar_h),
                                 border_radius=2)

        # 底部：关闭按钮
        btn_close = pygame.Rect(box.x + 20, box.bottom - 46, 120, 32)
        self._music_list_rects.append((btn_close, "close"))
        hovered = btn_close.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, bg_input, btn_close, border_radius=6)
        pygame.draw.rect(self.screen, accent, btn_close, 2 if hovered else 1, border_radius=6)
        btn_surf = self._font('small').render("关闭", True, accent)
        self.screen.blit(btn_surf, btn_surf.get_rect(center=btn_close.center))

        hint2 = self._font('tiny').render("当前播放曲目带 ▶ 标记，单击曲目即切换播放", True, muted)
        self.screen.blit(hint2, (box.x + 160, box.bottom - 38))

    def _draw_section_select_overlay(self):
        """绘制段落选择覆盖层（选歌后的非强制步骤）。"""
        if self._section_select is None:
            self._section_select_rects = []
            return

        sec = self._section_select
        accent = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
        bg = SoulTheme.BG_PANEL if SoulTheme else self.COLOR_PANEL
        bg_input = SoulTheme.BG_DEEP if SoulTheme else (25, 25, 25)
        text_c = SoulTheme.TEXT_PRIMARY if SoulTheme else self.COLOR_TEXT
        muted = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM

        # 半透明遮罩
        overlay = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        # 面板
        box = pygame.Rect(340, 200, 600, 320)
        pygame.draw.rect(self.screen, bg, box, border_radius=8)
        pygame.draw.rect(self.screen, accent, box, 2, border_radius=8)

        title = self._font('medium').render("选择播放起始段落（可选）", True, accent)
        self.screen.blit(title, (box.x + 20, box.y + 14))
        sub = self._font('tiny').render(
            f"已选 {len(sec['tracks'])} 首歌曲  |  当前设置: {sec.get('label', '从头播放')}",
            True, muted)
        self.screen.blit(sub, (box.x + 20, box.y + 46))

        mouse_pos = pygame.mouse.get_pos()
        self._section_select_rects = []

        # 预设段落按钮
        presets = [
            ("从头播放", 0.0, 100),
            ("前奏结束", 0.15, 100),
            ("主歌开始", 0.30, 100),
            ("副歌开始", 0.50, 100),
            ("第二段", 0.70, 100),
        ]
        btn_y = box.y + 80
        for i, (label, pct, bw) in enumerate(presets):
            bx = box.x + 20 + (i % 3) * (bw + 10)
            by = btn_y + (i // 3) * 40
            rect = pygame.Rect(bx, by, bw, 32)
            selected = abs(sec.get("start_pct", 0.0) - pct) < 0.01
            hovered = rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, accent if selected else bg_input, rect, border_radius=6)
            pygame.draw.rect(self.screen, accent, rect, 2 if hovered or selected else 1, border_radius=6)
            surf = self._font('small').render(label, True, text_c if selected else (accent if hovered else muted))
            self.screen.blit(surf, surf.get_rect(center=rect.center))
            self._section_select_rects.append((rect, f"preset:{pct}"))

        # 时间轴（可拖拽进度条）：点击或拖动滑块自由选择起始段落
        timeline_y = box.y + 175
        tl_rect = pygame.Rect(box.x + 20, timeline_y - 8, 560, 22)  # 点击/拖拽热区
        pygame.draw.rect(self.screen, muted, (box.x + 20, timeline_y, 560, 6), border_radius=3)
        start_pct = sec.get("start_pct", 0.0)
        fill_w = int(560 * start_pct)
        if fill_w > 0:
            pygame.draw.rect(self.screen, accent, (box.x + 20, timeline_y, fill_w, 6), border_radius=3)
        thumb_x = box.x + 20 + fill_w
        # 滑块：悬停/拖拽时放大提示可交互
        hover_tl = tl_rect.collidepoint(mouse_pos)
        pygame.draw.circle(self.screen, accent, (thumb_x, timeline_y + 3),
                           8 if (hover_tl or self._section_dragging) else 6)
        pct_label = self._font('tiny').render(f"起始位置: {int(start_pct * 100)}%", True, muted)
        self.screen.blit(pct_label, (box.x + 20, timeline_y + 18))
        self._section_select_rects.append((tl_rect, "timeline"))

        # 按钮：确认 / 跳过
        btn_confirm = pygame.Rect(box.x + 20, box.bottom - 50, 120, 34)
        btn_skip = pygame.Rect(box.x + 156, box.bottom - 50, 120, 34)
        self._section_select_rects.append((btn_confirm, "confirm"))
        self._section_select_rects.append((btn_skip, "skip"))
        for rect, label, color in ((btn_confirm, "确认保存", accent), (btn_skip, "跳过", muted)):
            hovered = rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, bg_input, rect, border_radius=6)
            pygame.draw.rect(self.screen, color, rect, 2 if hovered else 1, border_radius=6)
            surf = self._font('small').render(label, True, color)
            self.screen.blit(surf, surf.get_rect(center=rect.center))

        hint = self._font('tiny').render("Enter 确认  |  Esc 跳过（默认从头播放）", True, muted)
        self.screen.blit(hint, (box.x + 300, box.bottom - 42))

    def _render(self):
        """渲染一帧画面到屏幕"""
        # 使用新主题背景色（近黑深底）+ 环境辉光（Ethereal Glass 氛围）
        bg = SoulTheme.BG_DEEP if SoulTheme else self.COLOR_BG
        self.screen.fill(bg)
        if self.ui is not None and hasattr(self.ui, 'ambient'):
            self.screen.blit(self.ui.ambient, (0, 0))

        # 绘制顶部标题栏
        self._draw_header()

        # ── 空壳三列布局 ──
        # 左列：摄像头预览 + 系统遥测 + 常驻歌单
        self._draw_camera_preview(16, 64)
        self._draw_telemetry_panel(16, 356)
        self._draw_playlist_panel(16, 420)
        # 中列：播放控制 + 语音台词 + 底部页签
        self._draw_player_panel(364, 64)
        vt_count = 0
        if self.voice_trigger is not None and hasattr(self.voice_trigger, 'list_templates'):
            vt_count = min(len(self.voice_trigger.list_templates()), 3)
        voice_h = 70 + (vt_count * 22 + 6 if vt_count else 0)
        self._draw_voice_trigger_panel(364, 234, voice_h)
        tab_y = 234 + voice_h + 15
        # 底部信息区：动作特征 / 事件日志 页签（折叠显示）
        self._draw_bottom_tabs(364, tab_y, max(60, min(165, 690 - tab_y)))
        # 右列：动作预设 + 自定义预设
        self._draw_preset_list(936, 64)
        self._draw_custom_preset_panel(936, 223)

        # 绘制底部帮助栏
        self._draw_footer()

        # 绘制选歌覆盖层与段落选择覆盖层（模态，覆盖在普通 UI 之上）
        self._draw_song_select_overlay()
        self._draw_section_select_overlay()
        # 音乐列表弹窗（模态，覆盖在最上层）
        self._draw_music_list_overlay()

        # 绘制动作闪光特效
        if self.anim and self.anim.action_flash and self.anim.action_flash.is_active():
            flash = self.anim.action_flash
            radius = int(flash.get_radius())
            alpha = flash.get_alpha()
            flash_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(flash_surf, (*SoulTheme.ACCENT, alpha), (radius, radius), radius)
            pygame.draw.circle(flash_surf, (*SoulTheme.ACCENT, min(255, alpha + 50)), (radius, radius), radius // 2)
            self.screen.blit(flash_surf, (flash.x - radius, flash.y - radius))

        # 录制状态覆盖层（倒计时 / REC 指示）
        if self.recording_mode and self.custom_manager:
            is_cd = self.custom_manager.is_countdown()
            is_rec = self.custom_manager.is_recording()
            print(f"[DEBUG-RENDER] recording_mode=True, countdown={is_cd}, recording={is_rec}")
            if self.custom_manager.is_countdown():
                remaining = self.custom_manager.get_countdown_remaining()
                count = int(remaining) + 1
                # 半透明黑色遮罩
                overlay = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                self.screen.blit(overlay, (0, 0))
                # 大数字倒计时（视觉提示符）
                countdown_text = self._font('large').render(str(count), True, (255, 80, 80))
                rect = countdown_text.get_rect(center=(self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2))
                self.screen.blit(countdown_text, rect)
                # 提示文字：倒计时期保持静止采集中性基线
                hint = self._font('small').render("保持静止（采集中性基线）...", True, (200, 200, 200))
                h_rect = hint.get_rect(center=(self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2 + 50))
                self.screen.blit(hint, h_rect)
            elif self.custom_manager.is_recording():
                # REC 指示
                rec_text = self._font('large').render("● REC", True, (255, 80, 80))
                rect = rec_text.get_rect(center=(self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2))
                self.screen.blit(rec_text, rect)
                # 提示开始做动作（视觉提示符：REC 亮起即开始）+ 按 R 结束
                hint = self._font('small').render("开始做动作！按 R 结束录制", True, (200, 200, 200))
                h_rect = hint.get_rect(center=(self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2 + 50))
                self.screen.blit(hint, h_rect)

        # 语音台词录制状态覆盖层（倒计时 / REC 指示）
        if self._voice_rec is not None:
            vr = self._voice_rec
            if vr["phase"] == "countdown":
                remaining = self.VOICE_REC_COUNTDOWN - (time.time() - vr["t0"])
                count = max(1, int(remaining) + 1)
                overlay = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                self.screen.blit(overlay, (0, 0))
                countdown_text = self._font('large').render(str(count), True, (80, 200, 255))
                rect = countdown_text.get_rect(center=(self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2))
                self.screen.blit(countdown_text, rect)
                hint = self._font('small').render("准备语音录制...", True, (200, 200, 200))
                h_rect = hint.get_rect(center=(self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2 + 50))
                self.screen.blit(hint, h_rect)
            elif vr["phase"] == "recording":
                overlay = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 90))
                self.screen.blit(overlay, (0, 0))
                rec_text = self._font('large').render("🎤 ● REC", True, (80, 200, 255))
                rect = rec_text.get_rect(center=(self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2))
                self.screen.blit(rec_text, rect)
                hint = self._font('small').render("请说出台词，按 T 结束录制", True, (200, 200, 200))
                h_rect = hint.get_rect(center=(self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2 + 50))
                self.screen.blit(hint, h_rect)

        # 重命名覆盖层（内联输入，不弹窗）
        self._draw_rename_overlay()
        # 删除确认覆盖层（与重命名风格一致）
        self._draw_delete_overlay()
        # 录制后选歌覆盖层
        self._draw_song_select_overlay()

        pygame.display.flip()

    # ---------- 各区域绘制方法 ----------

    def _draw_header(self):
        """绘制标题栏（未来科技 HUD 版）"""
        if self.ui and SoulTheme:
            # HUD 字标（大写 + 强调色）
            title = self._font('large').render("SOULPLAYER", True, SoulTheme.ACCENT)
            self.screen.blit(title, (40, 16))
            # 副标题（等宽风：小号灰色）
            subtitle = self._font('tiny').render(
                "BODY-CONTROLLED MUSIC · 基于肢体动作捕捉的智能音乐播放器", True,
                SoulTheme.TEXT_MUTED)
            self.screen.blit(subtitle, (42, 64))
            # 右侧 HUD 状态条：CAM / VOICE / VOL（点指示 + 胶囊底色）
            vol = getattr(self.player, '_volume', 0.5)
            mic_ok = (self.voice_trigger is not None
                      and getattr(self.voice_trigger, 'available', False))
            cam_ok = bool(getattr(self.motion, 'is_running', False))
            chips = [
                ("CAM", SoulTheme.SUCCESS if cam_ok else SoulTheme.TEXT_MUTED,
                 "●" if cam_ok else "○"),
                ("VOICE", SoulTheme.SUCCESS if mic_ok else SoulTheme.TEXT_MUTED,
                 "●" if mic_ok else "○"),
                ("VOL", SoulTheme.TEXT_SECONDARY, f"{vol:.0%}"),
            ]
            cx = 1240
            for name, col, val in reversed(chips):
                chip_txt = f"{name} {val}"
                chip_w = max(72, len(chip_txt) * 12 + 26)
                chip_rect = pygame.Rect(cx - chip_w, 32, chip_w, 24)
                self.ui.alpha_rect(chip_rect, (*SoulTheme.BG_PANEL, 180), radius=6)
                pygame.draw.circle(self.screen, col, (chip_rect.x + 10, chip_rect.y + 12), 4)
                self.ui.label(chip_rect.x + 19, chip_rect.y + 6, chip_txt,
                              self._font('tiny'), color=col)
                cx -= chip_w + 8
            # 渐变分隔线
            for i in range(40, 1240, 2):
                t = (i - 40) / 1200
                line_color = value_to_color(t, SoulTheme.ACCENT, SoulTheme.INFO, SoulTheme.SUCCESS) if value_to_color else SoulTheme.BORDER
                pygame.draw.line(self.screen, line_color, (i, 102), (i + 1, 102), 2)
        else:
            # 降级：旧主题
            title = self._font('large').render("SoulPlayer", True, self.COLOR_ACCENT)
            self.screen.blit(title, (40, 25))
            subtitle = self._font('medium').render(
                "基于肢体动作捕捉的智能音乐播放器", True, self.COLOR_TEXT_DIM
            )
            self.screen.blit(subtitle, (40, 68))
            pygame.draw.line(self.screen, self.COLOR_PANEL, (40, 105), (1240, 105), 2)
            # 降级主题状态文字
            vol = getattr(self.player, '_volume', 0.5)
            mic_ok = (self.voice_trigger is not None
                      and getattr(self.voice_trigger, 'available', False))
            cam_ok = bool(getattr(self.motion, 'is_running', False))
            status_txt = self._font('tiny').render(
                f"语音{'开' if mic_ok else '关'} | 摄像头{'开' if cam_ok else '关'} | 音量 {vol:.0%}",
                True, self.COLOR_TEXT_DIM)
            self.screen.blit(status_txt, (1240 - status_txt.get_width(), 42))

    def _draw_camera_preview(self, x: int, y: int):
        """绘制摄像头预览窗口（新主题 + 骨架叠加）"""
        if self.ui and SoulTheme:
            # 圆角玻璃面板 + HUD 角标 + 标识
            panel_rect = pygame.Rect(x, y, self.PREVIEW_W + 20, self.PREVIEW_H + 45)
            self.ui.glass_panel(panel_rect, radius=SoulTheme.RADIUS_PANEL)
            self.ui.corner_brackets(panel_rect, SoulTheme.BORDER_LIGHT, length=12)
            self.ui.label(x + 12, y + 10, "摄像头预览", self._font('medium'),
                          color=SoulTheme.TEXT_SECONDARY)
            self.ui.label(panel_rect.right - 12, y + 12, "CAM-01",
                          self._font('tiny'), color=SoulTheme.ACCENT, anchor="topright")
            # 骨架显隐切换按钮（面板标题栏右侧，避开 CAM-01 角标）
            sk_on = self._skeleton_visible
            sk_rect = pygame.Rect(panel_rect.right - 146, y + 10, 74, 20)
            ok_c = SoulTheme.SUCCESS if SoulTheme else (0, 200, 100)
            muted_c = SoulTheme.TEXT_MUTED if SoulTheme else (150, 150, 150)
            sk_border = ok_c if sk_on else muted_c
            if sk_on:
                hl = pygame.Surface(sk_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(hl, (*ok_c, 26), hl.get_rect(), border_radius=3)
                self.screen.blit(hl, sk_rect.topleft)
            pygame.draw.rect(self.screen, sk_border, sk_rect, 1, border_radius=3)
            sk_lbl = "🦴 骨架:显" if sk_on else "🦴 骨架:隐"
            sk_surf = self._font('tiny').render(sk_lbl, True, ok_c if sk_on else muted_c)
            self.screen.blit(sk_surf, sk_surf.get_rect(center=sk_rect.center))
            self._skeleton_btn_rect = sk_rect
        else:
            pygame.draw.rect(self.screen, self.COLOR_PANEL, (x, y, self.PREVIEW_W + 20, self.PREVIEW_H + 40))
            label = self._font('medium').render("摄像头预览", True, self.COLOR_TEXT)
            self.screen.blit(label, (x + 10, y + 8))

        # 获取帧并渲染
        try:
            frame = self.motion.get_camera_frame()
            if frame is not None and frame.size > 0:
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if cv2 is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        rgb = frame
                    surf = pygame.image.frombuffer(
                        rgb.tobytes(), (frame.shape[1], frame.shape[0]), 'RGB'
                    )
                    surf = pygame.transform.scale(surf, (self.PREVIEW_W, self.PREVIEW_H))
                    preview_rect = (x + 10, y + 38)
                    self.screen.blit(surf, preview_rect)

                    # 内部 HUD 网格（叠加在画面上）
                    if SoulTheme:
                        grid_ov = pygame.Surface((self.PREVIEW_W, self.PREVIEW_H), pygame.SRCALPHA)
                        for gx in range(0, self.PREVIEW_W, 28):
                            pygame.draw.line(grid_ov, (*SoulTheme.GRID_DIM, 40),
                                             (gx, 0), (gx, self.PREVIEW_H), 1)
                        for gy in range(0, self.PREVIEW_H, 28):
                            pygame.draw.line(grid_ov, (*SoulTheme.GRID_DIM, 40),
                                             (0, gy), (self.PREVIEW_W, gy), 1)
                        self.screen.blit(grid_ov, preview_rect)
                    # 骨架叠加（MediaPipe 归一化坐标映射到预览尺寸，异常时静默跳过）
                    if (self._skeleton_visible and self.current_features
                            and draw_pose_skeleton and SoulTheme):
                        landmarks = self.current_features.get('landmarks', [])
                        try:
                            if landmarks and len(landmarks) == 33:
                                draw_pose_skeleton(self.screen, landmarks,
                                                   self.PREVIEW_W, self.PREVIEW_H,
                                                   line_color=SoulTheme.ACCENT,
                                                   line_width=2, dot_radius=3,
                                                   offset=preview_rect)
                        except Exception:
                            pass
                        # 手部 21 点细骨架（HandLandmarker，虚拟博主式）
                        if draw_hand_skeleton:
                            for _hk in ('hand_landmarks_L', 'hand_landmarks_R'):
                                _hp = self.current_features.get(_hk, []) or []
                                if len(_hp) >= 21:
                                    try:
                                        draw_hand_skeleton(
                                            self.screen, _hp,
                                            self.PREVIEW_W, self.PREVIEW_H,
                                            line_color=(255, 200, 80),
                                            line_width=2, dot_radius=2,
                                            offset=preview_rect)
                                    except Exception:
                                        pass
                else:
                    placeholder = self._font('small').render("帧格式不支持", True,
                                                              SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM)
                    self.screen.blit(placeholder, (x + 100, y + 120))
            else:
                placeholder = self._font('small').render("等待摄像头...", True,
                                                              SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM)
                self.screen.blit(placeholder, (x + 100, y + 120))
        except Exception as e:
            err = self._font('small').render(f"预览错误: {e}", True,
                                              SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM)
            self.screen.blit(err, (x + 10, y + 120))

    # ── HUD 系统遥测面板（空壳同款） ──
    def _draw_telemetry_panel(self, x: int, y: int):
        """HUD 系统遥测：FPS / LAT / THR（等宽数字）。"""
        if not (self.ui and SoulTheme):
            return
        rect = pygame.Rect(x, y, 336, 56)
        self.ui.alpha_rect(rect, (*SoulTheme.BG_PANEL, 180), radius=SoulTheme.RADIUS_PANEL)
        self.ui.label(x + 12, y + 8, "SYS.TELEMETRY", self._font('tiny'),
                      color=SoulTheme.TEXT_MUTED)
        f = self._font('tiny')
        fps = int(self.clock.get_fps())
        vals = [(f"FPS {fps:03d}", SoulTheme.INFO),
                (f"LAT {42 + int(8 * (time.time() % 1)):03d}ms", SoulTheme.WARNING),
                ("THR 0.62", SoulTheme.SUCCESS)]
        vx = x + 12
        for s, c in vals:
            self.ui.label(vx, y + 28, s, f, color=c)
            vx += f.size(s)[0] + 18

    # ── HUD 常驻歌单面板（空壳同款，真实曲目） ──
    def _draw_playlist_panel(self, x: int, y: int):
        """常驻歌单：真实曲目列表 + 滚轮滚动 + 当前播放高亮。"""
        if not (self.ui and SoulTheme):
            return
        rect = pygame.Rect(x, y, 336, 178)
        self.ui.alpha_rect(rect, (*SoulTheme.BG_PANEL, 200), radius=SoulTheme.RADIUS_PANEL)
        self.ui.label(x + 16, y + 10, "♪ PLAYLIST.歌单", self._font('small'),
                      color=SoulTheme.TEXT_PRIMARY)
        info = {}
        try:
            info = self.player.get_playlist_info()
        except Exception:
            pass
        tracks = info.get("tracks", []) or []
        current_idx = info.get("current_index", -1)
        self.ui.label(rect.right - 16, y + 14, f"{len(tracks)} TRACKS",
                      self._font('tiny'), color=SoulTheme.TEXT_MUTED, anchor="topright")
        mouse = pygame.mouse.get_pos()
        VISIBLE = 5
        row_h = 26
        list_y = y + 44
        list_h = VISIBLE * row_h
        self._playlist_rect = pygame.Rect(x + 14, list_y, rect.width - 28, list_h)
        max_scroll = max(0, len(tracks) - VISIBLE)
        self._play_scroll = max(0, min(getattr(self, '_play_scroll', 0), max_scroll))
        ry = list_y
        for i in range(self._play_scroll, self._play_scroll + VISIBLE):
            if i >= len(tracks):
                break
            name = tracks[i]
            if isinstance(name, dict):
                name = name.get('title', '?')
            row = pygame.Rect(x + 14, ry, rect.width - 28, row_h - 4)
            hovered = row.collidepoint(mouse)
            current = (i == current_idx)
            if current:
                hl = pygame.Surface(row.size, pygame.SRCALPHA)
                pygame.draw.rect(hl, (*SoulTheme.ACCENT, 26), hl.get_rect(), border_radius=4)
                self.screen.blit(hl, row.topleft)
                pygame.draw.rect(self.screen, SoulTheme.BORDER_LIGHT, row, 1, border_radius=4)
            elif hovered:
                hl = pygame.Surface(row.size, pygame.SRCALPHA)
                pygame.draw.rect(hl, (*SoulTheme.ACCENT, 16), hl.get_rect(), border_radius=4)
                self.screen.blit(hl, row.topleft)
            mark = "▶" if current else f"{i + 1:02d}"
            col = (SoulTheme.ACCENT if current
                   else (SoulTheme.TEXT_PRIMARY if hovered else SoulTheme.TEXT_SECONDARY))
            self.ui.label(row.x + 8, row.y + 5, mark, self._font('tiny'), color=col)
            nm = str(name)
            f = self._font('tiny')
            if f.size(nm)[0] > row.w - 96:
                while f.size(nm + "…")[0] > row.w - 96 and len(nm) > 1:
                    nm = nm[:-1]
                nm += "…"
            self.ui.label(row.x + 46, row.y + 4, nm, f,
                          color=SoulTheme.ACCENT if current else SoulTheme.TEXT_PRIMARY)
            ry += row_h
        if max_scroll > 0:
            bar_x = rect.right - 10
            bar_h = max(10, int(list_h * VISIBLE / len(tracks)))
            bar_y = list_y + int((list_h - bar_h) * self._play_scroll / max_scroll)
            pygame.draw.rect(self.screen, SoulTheme.BORDER, (bar_x, list_y, 3, list_h),
                             border_radius=2)
            pygame.draw.rect(self.screen, SoulTheme.ACCENT, (bar_x, bar_y, 3, bar_h),
                             border_radius=2)

    def _draw_feature_panel(self, x: int, y: int, title: bool = True):
        """绘制动作特征数值面板（新主题 + 动画过渡）。

        title=False 时仅绘制内容（供底部"动作特征"页签嵌入，标题由页签提供）。
        """
        if title:
            if self.ui and SoulTheme:
                self.ui.label(x, y, "动作特征", self._font('medium'), color=SoulTheme.TEXT_SECONDARY)
            else:
                label = self._font('medium').render("动作特征", True, self.COLOR_TEXT)
                self.screen.blit(label, (x, y))

        if not self.current_features:
            hint_color = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM
            hint = self._font('small').render("等待数据...", True, hint_color)
            self.screen.blit(hint, (x, y + 30))
            return

        # 使用动画平滑后的值（如果动画系统可用）
        if self.anim:
            items = [
                ("运动能量", self.anim.get_feature_animated('motion_energy')),
                ("身体速度", self.anim.get_feature_animated('body_speed')),
                ("手臂幅度", self.anim.get_feature_animated('arm_amplitude')),
                ("身体节奏", self.anim.get_feature_animated('body_rhythm')),
            ]
        else:
            features = self.current_features
            items = [
                ("运动能量", features.get('motion_energy', 0.0)),
                ("身体速度", features.get('body_speed', 0.0)),
                ("手臂幅度", features.get('arm_amplitude', 0.0)),
                ("身体节奏", features.get('body_rhythm', 0.0)),
            ]

        row_gap = 38 if title else 30
        for i, (name, value) in enumerate(items):
            row_y = y + (30 if title else 4) + i * row_gap
            if self.ui and SoulTheme:
                # 标签
                self.ui.label(x, row_y, name, self._font('small'), color=SoulTheme.TEXT_MUTED)
                # 数值
                self.ui.label(x + 80, row_y, f"{value:.2f}", self._font('small'),
                              color=SoulTheme.TEXT_PRIMARY)
                # 渐变进度条（缩短以适应右侧布局）
                self.ui.gradient_bar(x + 130, row_y + 2, 150, 16, value)
            else:
                txt = self._font('small').render(f"{name}", True, self.COLOR_TEXT_DIM)
                self.screen.blit(txt, (x, row_y))
                val_txt = self._font('small').render(f"{value:.2f}", True, self.COLOR_TEXT)
                self.screen.blit(val_txt, (x + 90, row_y))
                bar_x = x + 130
                pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (bar_x, row_y, 150, 16))
                bar_width = int(150 * float(np.clip(value, 0.0, 1.0)))
                color = self.COLOR_ACCENT if value > 0.3 else self.COLOR_BAR_LOW
                if bar_width > 0:
                    pygame.draw.rect(self.screen, color, (bar_x, row_y, bar_width, 16))

        # 离散状态
        features = self.current_features
        posture = features.get('posture', 'unknown')
        hand = features.get('hand_position', 'unknown')
        active = "是" if features.get('is_active') else "否"
        info_text = f"姿态: {posture}  |  手部: {hand}  |  活动: {active}"
        info_y = y + (185 if title else 128)
        if SoulTheme:
            self.ui.label(x, info_y, info_text, self._font('tiny'), color=SoulTheme.TEXT_MUTED)
        else:
            info = self._font('tiny').render(info_text, True, self.COLOR_TEXT_DIM)
            self.screen.blit(info, (x, info_y))

    def _draw_player_panel(self, x: int, y: int):
        """绘制播放器控制面板（新主题 + 进度条）"""
        panel_w = 560
        panel_h = 160
        if self.ui and SoulTheme:
            panel_rect = pygame.Rect(x, y, panel_w, panel_h)
            self.ui.glass_panel(panel_rect, radius=SoulTheme.RADIUS_PANEL)
            self.ui.label(x + 15, y + 10, "播放控制", self._font('medium'),
                          color=SoulTheme.TEXT_SECONDARY)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PANEL, (x, y, panel_w, panel_h))
            title = self._font('medium').render("播放控制", True, self.COLOR_TEXT)
            self.screen.blit(title, (x + 15, y + 10))

        # 右上角：音乐列表按钮（点击打开歌单弹窗）
        btn_rect = pygame.Rect(x + panel_w - 92, y + 8, 80, 26)
        self._music_list_btn_rect = btn_rect
        btn_hover = btn_rect.collidepoint(pygame.mouse.get_pos())
        if self.ui and SoulTheme:
            btn_bg = SoulTheme.BG_HOVER if btn_hover else SoulTheme.BG_CARD
            self.ui.rounded_rect(btn_rect, btn_bg, radius=SoulTheme.RADIUS_BTN,
                                 border=1,
                                 border_color=SoulTheme.ACCENT if btn_hover else SoulTheme.BORDER_LIGHT)
            self.ui.label(btn_rect.centerx, btn_rect.centery,
                          f"{ICONS.get('music', '♪')} 歌单", self._font('small'),
                          color=SoulTheme.ACCENT if btn_hover else SoulTheme.TEXT_SECONDARY,
                          align="center")
        else:
            pygame.draw.rect(self.screen, self.COLOR_ACCENT, btn_rect, 1, border_radius=6)
            btn_txt = self._font('small').render("♪ 歌单", True, self.COLOR_ACCENT)
            self.screen.blit(btn_txt, btn_txt.get_rect(center=btn_rect.center))

        track = self.player.current_track
        track_title = track.get('title', '无曲目') if isinstance(track, dict) else '无曲目'
        if self.ui and SoulTheme:
            self.ui.label(x + 15, y + 38, track_title, self._font('medium'),
                          color=SoulTheme.TEXT_PRIMARY)
        else:
            track_name = self._font('medium').render(f"{track_title}", True, self.COLOR_TEXT)
            self.screen.blit(track_name, (x + 15, y + 38))

        # 左侧：播放状态（播放中带呼吸脉冲点）
        status = "播放中" if self.player.is_playing else "已暂停"
        if self.ui and SoulTheme:
            status_color = SoulTheme.ACCENT if self.player.is_playing else SoulTheme.TEXT_MUTED
            icon = ICONS.get("play" if self.player.is_playing else "pause", "")
            if self.player.is_playing:
                # 三角波呼吸：半径 4~7px 匀速脉动
                breath = abs(2 * ((time.time() * 2) % 1) - 1)
                pygame.draw.circle(self.screen, SoulTheme.ACCENT,
                                   (x + 11, y + 72), 4 + int(3 * breath))
            self.ui.label(x + 15, y + 66, f"{icon} {status}", self._font('small'),
                          color=status_color)
        else:
            status_color = self.COLOR_ACCENT if self.player.is_playing else self.COLOR_TEXT_DIM
            status_txt = self._font('small').render(status, True, status_color)
            self.screen.blit(status_txt, (x + 15, y + 66))

        # 右侧：音量条
        vol = self.player._volume if hasattr(self.player, '_volume') else 0.5
        if self.ui and SoulTheme:
            self.ui.label(x + 330, y + 66, f"{ICONS.get('volume_high', '')} 音量", self._font('small'),
                          color=SoulTheme.TEXT_MUTED)
            self.ui.gradient_bar(x + 390, y + 69, 120, 12, vol,
                                 bg_color=SoulTheme.BAR_BG)
            self.ui.label(x + 520, y + 67, f"{vol:.0%}", self._font('tiny'),
                          color=SoulTheme.TEXT_MUTED)
        else:
            vol_label = self._font('small').render(f"音量", True, self.COLOR_TEXT_DIM)
            self.screen.blit(vol_label, (x + 330, y + 66))
            pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (x + 390, y + 69, 120, 12))
            vol_w = int(120 * vol)
            if vol_w > 0:
                pygame.draw.rect(self.screen, self.COLOR_ACCENT, (x + 390, y + 69, vol_w, 12))
            vol_pct = self._font('tiny').render(f"{vol:.0%}", True, self.COLOR_TEXT_DIM)
            self.screen.blit(vol_pct, (x + 520, y + 67))
        # 音量条热区（点击/拖拽调节，每帧刷新）
        self._vol_bar_rect = pygame.Rect(x + 388, y + 65, 126, 20)

        # 进度条（可拖拽）
        self._draw_seek_bar(x + 15, y + 95, panel_w - 30)

        # 空状态引导：无曲目时提示放音乐（替代最近动作位置）
        if not isinstance(track, dict) or not track.get('title'):
            hint = self._font('small').render(
                "暂无曲目：将音乐放入 music/ 文件夹，按 M 打开歌单", True,
                SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT)
            self.screen.blit(hint, (x + 15, y + 130))
            return

        # 最近触发动作（带闪烁效果）
        if self.last_action:
            age = time.time() - self.action_timestamp
            alpha = max(50, 255 - int(age * 30))
            if self.ui and SoulTheme:
                # 使用主题色但带透明度
                action_text = f"{ICONS.get('sparkles', '')} 最近动作: {self.last_action}"
                action_surf = self._font('small').render(action_text, True, SoulTheme.ACCENT)
                # 创建带 alpha 的 surface
                action_surf.set_alpha(alpha)
                self.screen.blit(action_surf, (x + 15, y + 130))
            else:
                action_color = (
                    self.COLOR_ACCENT[0] * alpha // 255,
                    self.COLOR_ACCENT[1] * alpha // 255,
                    self.COLOR_ACCENT[2] * alpha // 255,
                )
                action_txt = self._font('small').render(f"最近动作: {self.last_action}", True, action_color)
                self.screen.blit(action_txt, (x + 15, y + 130))

        # 音量数值回显弹标（按键调整后短暂显示，带淡出）
        if time.time() < self._vol_popup_until:
            vol = getattr(self.player, '_volume', 0.5)
            remain = self._vol_popup_until - time.time()
            pop_alpha = 255 if remain > 0.6 else int(255 * remain / 0.6)
            pop_txt = f"{ICONS.get('volume_high', '🔊')} 音量 {vol:.0%}"
            pop_surf = self._font('small').render(
                pop_txt, True, SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT)
            pop_surf.set_alpha(max(40, pop_alpha))
            pill = pygame.Rect(0, 0, pop_surf.get_width() + 24, 28)
            pill.center = (x + panel_w // 2, y - 8)
            if self.ui and SoulTheme:
                self.ui.rounded_rect(pill, SoulTheme.BG_CARD, radius=14,
                                     border=1, border_color=SoulTheme.ACCENT_DIM)
            else:
                pygame.draw.rect(self.screen, self.COLOR_PANEL, pill, border_radius=14)
            self.screen.blit(pop_surf, pop_surf.get_rect(center=pill.center))

    def _draw_seek_bar(self, x: int, y: int, width: int):
        """绘制可拖拽的音频播放进度条（UI 占位，seek 后端待完善）。"""
        bar_h = 10
        thumb_r = 6
        accent = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
        bg = SoulTheme.BAR_BG if SoulTheme else self.COLOR_BAR_BG
        muted = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM
        text_c = SoulTheme.TEXT_PRIMARY if SoulTheme else self.COLOR_TEXT

        # 获取播放进度（0.0~1.0），若无法获取则显示 0
        progress = 0.0
        try:
            pos = self.player.get_playback_position()
            dur = self.player.get_track_duration()
            if dur > 0:
                progress = min(1.0, max(0.0, pos / dur))
        except Exception:
            pass

        mouse_pos = pygame.mouse.get_pos()
        bar_rect = pygame.Rect(x, y, width, bar_h + 4)
        # 悬停检测区域稍大
        self._seek_hovered = bar_rect.collidepoint(mouse_pos)

        # HUD 波形进度：伪波形柱（播放过=强调色）+ 播放头 + 悬停滑块
        track_y = y + (bar_h // 2) + 2
        fill_w = int(width * progress)
        wav_h = bar_h + 4
        bars = 120
        for i in range(bars):
            wf = 0.5 + 0.5 * abs(np.sin(i * 0.35 + time.time() * 3))
            hgt = max(2, int(wav_h * wf))
            bx = x + i * (width / bars)
            done = (i / bars) <= progress
            pygame.draw.rect(self.screen, accent if done else bg,
                             (bx, track_y + (wav_h - hgt) // 2, 3, hgt))
        # 播放头（竖线 + 三角）
        hx = x + fill_w
        pygame.draw.line(self.screen, text_c, (hx, track_y), (hx, track_y + wav_h), 2)
        hds = pygame.Surface((9, 14), pygame.SRCALPHA)
        pygame.draw.polygon(hds, (*accent, 230), [(0, 0), (9, 7), (0, 14)])
        self.screen.blit(hds, (hx - 5, track_y - 2))
        # 滑块（悬停或拖拽时显示）
        if self._seek_hovered or self._seek_dragging:
            thumb_x = x + fill_w
            pygame.draw.circle(self.screen, accent, (thumb_x, track_y + bar_h // 2), thumb_r)

        # 时间文本
        try:
            pos = self.player.get_playback_position()
            dur = self.player.get_track_duration()
            if dur > 0:
                pos_str = f"{int(pos // 60)}:{int(pos % 60):02d}"
                dur_str = f"{int(dur // 60)}:{int(dur % 60):02d}"
                time_text = f"{pos_str} / {dur_str}"
            else:
                time_text = "--:-- / --:--"
        except Exception:
            time_text = "--:-- / --:--"
        # 拖拽中时间文本用强调色（数值回显反馈）
        if self._seek_dragging:
            time_color = accent
        elif self._seek_hovered:
            time_color = text_c
        else:
            time_color = muted
        time_surf = self._font('tiny').render(time_text, True, time_color)
        self.screen.blit(time_surf, (x + width - time_surf.get_width(), y - 2))

        # 注册进度条热区供事件处理使用（每帧刷新）
        self._seek_bar_rect = bar_rect

    def _draw_preset_list(self, x: int, y: int):
        """绘制预设列表（新主题 + 脉冲高亮 + 高度自适应）"""
        presets = self.preset_engine.presets
        # 面板高度随预设数量自适应：标题38 + 每行24 + 底部自定义提示34
        rows = max(1, min(len(presets), 6))
        panel_h = 38 + rows * 24 + 34

        if self.ui and SoulTheme:
            panel_rect = pygame.Rect(x, y, 328, panel_h)
            self.ui.glass_panel(panel_rect, radius=SoulTheme.RADIUS_PANEL)
            self.ui.label(x + 15, y + 12, "动作预设", self._font('medium'),
                          color=SoulTheme.TEXT_SECONDARY)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PANEL, (x, y, 328, panel_h))
            title = self._font('medium').render("动作预设", True, self.COLOR_TEXT)
            self.screen.blit(title, (x + 15, y + 12))

        # 空状态引导：无预设时提示录制
        if not presets:
            hint_color = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM
            hint = self._font('small').render("暂无预设：按 R 录制第一个自定义动作", True, hint_color)
            self.screen.blit(hint, (x + 15, y + 52))

        col1_x = x + 15
        col2_x = x + 270

        # 脉冲效果（激活预设的缩放）
        pulse_scale = 1.0
        if self.anim and self.anim.preset_pulse:
            pulse_scale = self.anim.get_preset_pulse_scale()

        for i, preset in enumerate(presets):
            col_x = col1_x if i < 6 else col2_x
            row_y = y + 38 + (i % 6) * 24

            active = self.preset_engine.active_preset
            if active is None:
                is_active = False
            elif isinstance(active, str):
                is_active = preset.name == active
            else:
                is_active = preset.name == getattr(active, 'name', None)

            if is_active and SoulTheme:
                # 激活项：脉冲效果 + 高亮背景
                prefix = f"{ICONS.get('fire', '●')} "
                color = SoulTheme.ACCENT
                # 绘制小高亮条
                highlight = pygame.Surface((6, 14), pygame.SRCALPHA)
                pygame.draw.rect(highlight, (*SoulTheme.ACCENT, 200), (0, 0, 6, 14), border_radius=3)
                self.screen.blit(highlight, (col_x - 8, row_y + 1))
            else:
                prefix = "  "
                color = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM

            name = f"{prefix}{preset.name}"
            if self.ui and SoulTheme:
                self.ui.label(col_x, row_y, name, self._font('tiny'), color=color)
            else:
                txt = self._font('tiny').render(name, True, color)
                self.screen.blit(txt, (col_x, row_y))

        # 自定义预设按钮（带图标），位置随面板高度自适应
        if self.custom_manager and self.custom_manager.presets:
            btn_y = y + panel_h - 22
            if SoulTheme and ICONS:
                btn_text = f"{ICONS.get('plus', '+')} 自定义预设 ({len(self.custom_manager.presets)})"
                self.ui.label(x + 15, btn_y, btn_text, self._font('tiny'), color=SoulTheme.ACCENT)
            else:
                btn_text = "➕ 自定义预设"
                btn_txt = self._font('tiny').render(btn_text, True, self.COLOR_ACCENT)
                self.screen.blit(btn_txt, (x + 15, btn_y))

    def _draw_voice_trigger_panel(self, x: int, y: int, h: int = 70):
        """绘制语音触发状态面板（含模板列表与悬停删除按钮）"""
        if not self.voice_trigger:
            return

        w = 560
        status = self.voice_trigger.get_status()

        if self.ui and SoulTheme:
            panel_rect = pygame.Rect(x, y, w, h)
            self.ui.glass_panel(panel_rect, radius=SoulTheme.RADIUS_PANEL)
            self.ui.label(x + 15, y + 8, "🎤 语音触发", self._font('medium'),
                          color=SoulTheme.TEXT_SECONDARY)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PANEL, (x, y, w, h))
            title = self._font('medium').render("🎤 语音触发", True, self.COLOR_TEXT)
            self.screen.blit(title, (x + 15, y + 8))

        # 状态指示圆点
        if not status['available']:
            dot_color = SoulTheme.DANGER if SoulTheme else (255, 80, 80)
            status_text = "引擎不可用"
        elif status['listening']:
            dot_color = SoulTheme.SUCCESS if SoulTheme else (0, 200, 100)
            status_text = "监听中"
        else:
            dot_color = SoulTheme.TEXT_MUTED if SoulTheme else (150, 150, 150)
            status_text = "已停止"

        pygame.draw.circle(self.screen, dot_color, (x + 20, y + 38), 5)

        if self.ui and SoulTheme:
            self.ui.label(x + 32, y + 32, status_text, self._font('small'),
                          color=SoulTheme.TEXT_PRIMARY)
            trigger_info = f"{status['trigger_count']} 条台词"
            self.ui.label(x + 120, y + 32, trigger_info, self._font('small'),
                          color=SoulTheme.TEXT_MUTED)
        else:
            status_surf = self._font('small').render(status_text, True, self.COLOR_TEXT)
            self.screen.blit(status_surf, (x + 32, y + 32))
            trigger_info = f"{status['trigger_count']} 条台词"
            trigger_surf = self._font('small').render(trigger_info, True, self.COLOR_TEXT_DIM)
            self.screen.blit(trigger_surf, (x + 120, y + 32))

        # 最近识别结果（仅监听中时显示）
        last = status.get('last_result', '')
        if last and status['listening']:
            if self.ui and SoulTheme:
                self.ui.label(x + 220, y + 32, f"识别: {last}", self._font('small'),
                              color=SoulTheme.ACCENT)
            else:
                last_surf = self._font('small').render(f"识别: {last}", True, self.COLOR_ACCENT)
                self.screen.blit(last_surf, (x + 220, y + 32))

        # 快捷键提示
        if self.ui and SoulTheme:
            self.ui.label(x + 15, y + 52, "按 V 切换监听 | 按 T 录制台词", self._font('tiny'),
                          color=SoulTheme.TEXT_MUTED)
        else:
            hint = self._font('tiny').render("按 V 切换监听 | 按 T 录制台词", True, self.COLOR_TEXT_DIM)
            self.screen.blit(hint, (x + 15, y + 52))

        # 语音模板列表（悬停显示编辑工具栏：改/删/收藏/左移/右移，支持滚动）
        self._voice_actions = []
        templates = []
        if hasattr(self.voice_trigger, 'list_templates'):
            templates = self.voice_trigger.list_templates()
            # 按收藏+排序整理（与自定义预设一致）
            templates.sort(key=lambda t: (-int(t.get("favorite", False)), t.get("order", 9999)))
        if templates:
            accent = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
            text_c = SoulTheme.TEXT_PRIMARY if SoulTheme else self.COLOR_TEXT
            muted = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM
            danger = SoulTheme.DANGER if SoulTheme else (220, 90, 90)
            mouse_pos = pygame.mouse.get_pos()
            VISIBLE_ROWS = 3
            row_h = 26
            list_y = y + 70
            list_h = VISIBLE_ROWS * row_h
            self._voice_list_rect = pygame.Rect(x + 15, list_y, w - 30, list_h)
            max_scroll = max(0, len(templates) - VISIBLE_ROWS)
            self._voice_scroll = max(0, min(self._voice_scroll, max_scroll))
            row_y = list_y
            for tpl in templates[self._voice_scroll:self._voice_scroll + VISIBLE_ROWS]:
                name = tpl["name"]
                tc = tpl["track_count"]
                is_fav = tpl.get("favorite", False)
                row_rect = pygame.Rect(x + 15, row_y, w - 30, 22)
                hovered = row_rect.collidepoint(mouse_pos)
                if hovered:
                    # HUD 悬停：青辉高亮 + 高亮描边（SRCALPHA 临时表面保证真半透明）
                    hl = pygame.Surface(row_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(hl, (*accent, 34), hl.get_rect(), border_radius=4)
                    self.screen.blit(hl, row_rect.topleft)
                    pygame.draw.rect(self.screen,
                                     SoulTheme.BORDER_LIGHT if SoulTheme else (150, 150, 150),
                                     row_rect, 1, border_radius=4)
                # 收藏左侧色条
                if is_fav:
                    pygame.draw.rect(self.screen, accent,
                                     (row_rect.x, row_rect.y + 4, 3, row_rect.h - 8),
                                     border_radius=2)
                # 收藏标记 + 名称
                fav_mark = "★ " if is_fav else ""
                name_color = accent if is_fav else (text_c if hovered else muted)
                label = self._font('tiny').render(f"{fav_mark}🎵 {name}  ({tc} 首)", True, name_color)
                self.screen.blit(label, (row_rect.x + 4, row_rect.y + 3))
                if hovered:
                    # 工具栏：改 / 删 / 收藏 / 左移 / 右移（与自定义预设卡片保持一致）
                    btn_y = row_rect.y
                    bx = row_rect.right - 140
                    btn_specs = [
                        ("改", "rename", 28, accent),
                        ("删", "delete", 28, danger),
                        ("★" if is_fav else "☆", "favorite", 28, accent),
                        ("←", "move_left", 24, accent),
                        ("→", "move_right", 24, accent),
                    ]
                    for lbl, act, bw, col in btn_specs:
                        brect = pygame.Rect(bx, btn_y, bw, 20)
                        bhover = brect.collidepoint(mouse_pos)
                        if bhover:
                            hl = pygame.Surface(brect.size, pygame.SRCALPHA)
                            pygame.draw.rect(hl, (*accent, 50), hl.get_rect(), border_radius=3)
                            self.screen.blit(hl, brect.topleft)
                        pygame.draw.rect(self.screen, col, brect, 1, border_radius=3)
                        bsurf = self._font('tiny').render(lbl, True, col)
                        self.screen.blit(bsurf, bsurf.get_rect(center=brect.center))
                        self._voice_actions.append((brect, name, act))
                        bx += bw + 3
                row_y += row_h
            if max_scroll > 0:
                bar_w = 3
                bar_x = x + w - 12
                track_h = list_h
                bar_h = max(10, int(track_h * VISIBLE_ROWS / len(templates)))
                bar_y = list_y + int((track_h - bar_h) * self._voice_scroll / max_scroll)
                pygame.draw.rect(self.screen, muted, (bar_x, list_y, bar_w, track_h), border_radius=2)
                pygame.draw.rect(self.screen, accent, (bar_x, bar_y, bar_w, bar_h), border_radius=2)

    def _draw_bottom_tabs(self, x: int, y: int, h: int = 100):
        """底部信息区页签面板：动作特征 / 事件日志 折叠切换。"""
        self._bottom_tab_rects = []
        panel_rect = pygame.Rect(x, y, 560, h)
        if self.ui and SoulTheme:
            self.ui.glass_panel(panel_rect, radius=SoulTheme.RADIUS_PANEL)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PANEL, (x, y, 560, h))

        # 页签栏（点击切换 + hover 缩放动效）
        tabs = [("features", "动作特征"), ("log", "事件日志")]
        tx = x + 15
        mouse_pos = pygame.mouse.get_pos()
        for key, label in tabs:
            active = (self._bottom_tab == key)
            base = pygame.Rect(tx, y + 8, 88, 24)
            hovered = base.collidepoint(mouse_pos)
            if self.ui and SoulTheme:
                # hover 缩放动效：进度 0~1 平滑趋近（每帧指数插值），以中心放大
                p = self._tab_anim[key]
                target = 1.0 if (hovered or active) else 0.0
                self._tab_anim[key] = p + (target - p) * 0.25
                grow = int(5 * p)
                tab_rect = base.inflate(grow * 2, grow * 2)
                if p > 0.02:
                    self.ui.glow_rect(tab_rect, SoulTheme.ACCENT_GLOW,
                                      radius=SoulTheme.RADIUS_BTN, glow=2)
                bg = SoulTheme.BG_HOVER if (active or hovered) else SoulTheme.BG_CARD
                self.ui.rounded_rect(tab_rect, bg, radius=SoulTheme.RADIUS_BTN,
                                     border=1,
                                     border_color=SoulTheme.ACCENT if active else SoulTheme.BORDER_LIGHT)
                self.ui.label(tab_rect.centerx, tab_rect.centery, label,
                              self._font('small'),
                              color=SoulTheme.ACCENT if active else SoulTheme.TEXT_MUTED,
                              align="center")
            else:
                tab_rect = base
                pygame.draw.rect(self.screen, self.COLOR_PANEL, tab_rect, border_radius=8)
                tab_txt = self._font('small').render(
                    label, True, self.COLOR_ACCENT if active else self.COLOR_TEXT_DIM)
                self.screen.blit(tab_txt, (tx + 12, y + 14))
            self._bottom_tab_rects.append((tab_rect, key))
            tx += 96

        # 内容区
        content_y = y + 40
        if self._bottom_tab == "features":
            self._draw_feature_panel(x + 15, content_y, title=False)
        else:
            max_lines = max(1, min(4, (h - 48) // 17))
            for i, (msg, t) in enumerate(self.log_messages[-max_lines:]):
                age = time.time() - t
                if age < 1.0:
                    color = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
                elif age < 3.0:
                    color = SoulTheme.TEXT_SECONDARY if SoulTheme else self.COLOR_TEXT_DIM
                else:
                    color = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM
                if self.ui and SoulTheme:
                    self.ui.label(x + 15, content_y + i * 17, msg, self._font('tiny'), color=color)
                else:
                    line = self._font('tiny').render(msg, True, color)
                    self.screen.blit(line, (x + 15, content_y + i * 17))

    def _draw_log_panel(self, x: int, y: int, h: int = 100):
        """绘制实时日志面板（新主题，高度可随语音面板扩展而压缩）"""
        max_lines = max(1, min(4, (h - 40) // 17))
        if self.ui and SoulTheme:
            panel_rect = pygame.Rect(x, y, 520, h)
            self.ui.glass_panel(panel_rect, radius=SoulTheme.RADIUS_PANEL)
            self.ui.label(x + 15, y + 10, "事件日志", self._font('medium'),
                          color=SoulTheme.TEXT_SECONDARY)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PANEL, (x, y, 520, h))
            title = self._font('medium').render("事件日志", True, self.COLOR_TEXT)
            self.screen.blit(title, (x + 15, y + 12))

        for i, (msg, t) in enumerate(self.log_messages[-max_lines:]):
            age = time.time() - t
            fade = max(80, 255 - int(age * 50))
            if SoulTheme:
                # 新消息用强调色，旧消息用灰色
                if age < 1.0:
                    color = SoulTheme.ACCENT
                elif age < 3.0:
                    color = SoulTheme.TEXT_SECONDARY
                else:
                    color = SoulTheme.TEXT_MUTED
                txt = self._font('tiny').render(msg, True, color)
                self.screen.blit(txt, (x + 15, y + 36 + i * 17))
            else:
                color = (fade, fade, fade)
                txt = self._font('tiny').render(msg, True, color)
                self.screen.blit(txt, (x + 15, y + 38 + i * 17))

    def _voice_status_hint(self, themed: bool):
        """计算底部语音状态提示（新主题/降级主题共用，避免两处重复逻辑）"""
        vt_status = self.voice_trigger.get_status()
        mode = vt_status.get("mode") or "MFCC"
        if vt_status.get("listening"):
            return f"🎤 语音监听中（{mode}）", SoulTheme.SUCCESS if themed else (100, 200, 100)
        if not self.voice_trigger.available:
            if themed:
                err = vt_status.get("last_error", "")[:20]
                return f"🎤 语音未就绪: {err}...", SoulTheme.WARNING
            return "🎤 语音未就绪", (200, 180, 80)
        if themed:
            return f"🎤 语音就绪（{mode}，按 V 开启）", SoulTheme.TEXT_MUTED
        return f"🎤 按 V 开启语音监听（{mode}）", (170, 170, 170)

    def _draw_footer(self):
        """绘制底部帮助栏与帧率信息（新主题 + 录制动画）"""
        if SoulTheme:
            # 帮助信息
            help_text = (
                f"{ICONS.get('arrow_left', '')}{ICONS.get('arrow_right', '')} 下一首  |  "
                f"{ICONS.get('arrow_up', '')}{ICONS.get('arrow_down', '')} 音量  |  "
                f"空格 播放/暂停  |  R 录制  |  T 台词  |  V 语音  |  M 歌单  |  ESC 退出"
            )
            self.ui.label(40, 692, help_text, self._font('tiny'), color=SoulTheme.TEXT_MUTED)

            # 录制状态提示（脉冲动画）
            if self.recording_mode and self.custom_manager:
                progress = self.custom_manager.get_recording_progress()
                if self.anim and self.anim.recording:
                    pulse = self.anim.get_recording_phase()
                    # 脉冲圆点
                    dot_x = 980
                    dot_y = 700
                    self.ui.pulse_dot(dot_x, dot_y, 6, SoulTheme.DANGER, pulse_phase=pulse)
                    # 文字
                    rec_text = f"录制中... {progress:.0%}"
                    self.ui.label(dot_x + 15, dot_y - 6, rec_text, self._font('tiny'),
                                  color=SoulTheme.DANGER)
                    # 进度条
                    self.ui.gradient_bar(dot_x + 15, dot_y + 8, 100, 4, progress,
                                        color_high=SoulTheme.DANGER)
                else:
                    rec_text = f"⏺ 录制中... {progress:.0%}"
                    rec_surf = self._font('tiny').render(rec_text, True, SoulTheme.DANGER)
                    self.screen.blit(rec_surf, (1000, 695))
            else:
                # 显示语音状态（优先）或录制快捷键提示
                if self.voice_trigger is not None:
                    hint, hint_color = self._voice_status_hint(themed=True)
                    self.ui.label(1000, 695, hint, self._font('tiny'), color=hint_color)
                else:
                    hint = f"按 R 录制自定义动作预设"
                    self.ui.label(1000, 695, hint, self._font('tiny'), color=SoulTheme.TEXT_MUTED)

            # FPS
            fps = int(self.clock.get_fps())
            fps_color = SoulTheme.SUCCESS if fps >= 25 else SoulTheme.WARNING if fps >= 15 else SoulTheme.DANGER
            self.ui.label(1190, 695, f"FPS: {fps}", self._font('tiny'), color=fps_color)
        else:
            # 降级：旧主题
            help_text = (
                "快捷键:  [ESC/Q] 退出  |  [空格] 播放/暂停  |  [<-/->] 下一首  |  [上/下] 音量  |  [R] 录制预设  |  [T] 录制台词  |  [V] 语音监听  |  [M] 歌单"
            )
            help_surf = self._font('tiny').render(help_text, True, (100, 100, 100))
            self.screen.blit(help_surf, (40, 695))
            if self.recording_mode and self.custom_manager:
                progress = self.custom_manager.get_recording_progress()
                rec_text = f"⏺ 录制中... {progress:.0%}"
                rec_surf = self._font('tiny').render(rec_text, True, (255, 80, 80))
                self.screen.blit(rec_surf, (1000, 695))
            elif self.voice_trigger is not None:
                hint, hint_color = self._voice_status_hint(themed=False)
                hint_surf = self._font('tiny').render(hint, True, hint_color)
                self.screen.blit(hint_surf, (1000, 695))
            fps = int(self.clock.get_fps())
            fps_txt = self._font('tiny').render(f"FPS: {fps}", True, self.COLOR_TEXT_DIM)
            self.screen.blit(fps_txt, (1190, 695))

    def _draw_custom_preset_panel(self, x: int, y: int):
        """自定义预设面板（右列纵向卡片，空壳样式 + 悬停内联工具栏）"""
        if not self.custom_manager or not self.custom_manager.presets:
            self._custom_cards = []
            self._custom_actions = []
            return

        w = 328
        h = 250
        if self.ui and SoulTheme:
            panel_rect = pygame.Rect(x, y, w, h)
            self.ui.glass_panel(panel_rect, radius=SoulTheme.RADIUS_PANEL)
            self.ui.label(x + 16, y + 10, "✧ CUSTOM.自定义", self._font('small'),
                          color=SoulTheme.TEXT_PRIMARY)
        else:
            pygame.draw.rect(self.screen, self.COLOR_PANEL, (x, y, w, h))
            label = self._font('small').render("自定义预设", True, self.COLOR_TEXT)
            self.screen.blit(label, (x + 10, y + 8))

        presets = self.custom_manager.get_display_presets()[:4]  # 右列最多 4 张
        mouse_pos = pygame.mouse.get_pos()
        self._custom_cards = []    # [(Rect, preset)] 供点击命中检测
        self._custom_actions = []  # [(Rect, action, preset)] 编辑按钮
        hovered_preset = None

        def track_title(p):
            t = getattr(p, 'tracks', None)
            if t:
                try:
                    return Path(str(t[0])).stem
                except Exception:
                    return "未绑定曲目"
            return "未绑定曲目"

        card_w = w - 28
        card_h = 42
        for i, preset in enumerate(presets):
            card_rect = pygame.Rect(x + 14, y + 46 + i * 50, card_w, card_h)
            self._custom_cards.append((card_rect, preset))
            hovered = card_rect.collidepoint(mouse_pos)
            if hovered:
                hovered_preset = preset

            is_fav = getattr(preset, 'favorite', False)
            fav_mark = "★ " if is_fav else ""
            nm = preset.name
            f_tiny = self._font('tiny')
            # 悬停时右侧留出内联工具栏空间
            name_w = card_w - (150 if hovered else 16)
            if f_tiny.size(nm)[0] > name_w:
                while f_tiny.size(nm + "…")[0] > name_w and len(nm) > 1:
                    nm = nm[:-1]
                nm += "…"

            if self.ui and SoulTheme:
                bg = SoulTheme.BG_PANEL if hovered else SoulTheme.BG_CARD
                if hovered:
                    self.ui.glow_rect(card_rect, SoulTheme.ACCENT_GLOW,
                                      radius=SoulTheme.RADIUS_BTN, glow=2)
                self.ui.rounded_rect(card_rect, bg, radius=SoulTheme.RADIUS_BTN)
                pygame.draw.rect(self.screen,
                                 SoulTheme.BORDER_LIGHT if hovered else SoulTheme.BORDER,
                                 card_rect, 1, border_radius=SoulTheme.RADIUS_BTN)
                name_color = SoulTheme.ACCENT if is_fav else SoulTheme.TEXT_PRIMARY
                self.ui.label(card_rect.x + 10, card_rect.y + 5, f"{fav_mark}{nm}", f_tiny,
                              color=name_color)
                self.ui.label(card_rect.x + 10, card_rect.y + 22,
                              f"播放: {track_title(preset)}", f_tiny,
                              color=SoulTheme.TEXT_MUTED)
            else:
                pygame.draw.rect(self.screen, self.COLOR_PANEL, card_rect)
                txt = self._font('tiny').render(f"{fav_mark}{nm}", True, self.COLOR_TEXT_DIM)
                self.screen.blit(txt, (card_rect.x + 5, card_rect.y + 8))

            # 悬停 → 内联工具栏（改/删/★/←/→）
            if hovered and self.ui and SoulTheme:
                bx = card_rect.right - 132
                btns = [("改", "rename", SoulTheme.TEXT_SECONDARY),
                        ("删", "delete", SoulTheme.DANGER),
                        ("★" if is_fav else "☆", "favorite", SoulTheme.ACCENT),
                        ("←", "move_left", SoulTheme.TEXT_SECONDARY),
                        ("→", "move_right", SoulTheme.TEXT_SECONDARY)]
                for s, act, col in btns:
                    bb = pygame.Rect(bx, card_rect.y + 9, 24, 24)
                    bh = bb.collidepoint(mouse_pos)
                    if bh:
                        hl = pygame.Surface(bb.size, pygame.SRCALPHA)
                        pygame.draw.rect(hl, (*SoulTheme.ACCENT, 50), hl.get_rect(), border_radius=3)
                        self.screen.blit(hl, bb.topleft)
                    pygame.draw.rect(self.screen, SoulTheme.BORDER_LIGHT, bb, 1,
                                     border_radius=3)
                    self.ui.label(bb.centerx, bb.centery, s, self._font('tiny'),
                                  color=col, align="center")
                    self._custom_actions.append((bb, act, preset))
                    bx += 27

    def _draw_preset_edit_toolbar(self, card_rect: pygame.Rect, preset, mouse_pos):
        """在悬停的预设卡片下方绘制编辑工具栏，并注册按钮热区。"""
        accent = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
        bg = SoulTheme.BG_CARD if SoulTheme else self.COLOR_PANEL
        bg_hover = SoulTheme.BG_PANEL if SoulTheme else (72, 72, 72)
        text_color = SoulTheme.TEXT_PRIMARY if SoulTheme else self.COLOR_TEXT

        is_fav = getattr(preset, 'favorite', False)
        buttons = [
            ("改", "rename", 30),
            ("删", "delete", 30),
            ("★" if is_fav else "☆", "favorite", 30),
            ("←", "move_left", 26),
            ("→", "move_right", 26),
        ]

        bx = card_rect.x
        by = card_rect.bottom + 4
        for label, action, bw in buttons:
            rect = pygame.Rect(bx, by, bw, 24)
            hovered = rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, bg_hover if hovered else bg, rect, border_radius=4)
            pygame.draw.rect(self.screen, accent if hovered else (*accent[:3],), rect, 1, border_radius=4)
            surf = self._font('tiny').render(label, True, accent if hovered else text_color)
            self.screen.blit(surf, surf.get_rect(center=rect.center))
            self._custom_actions.append((rect, action, preset))
            bx += bw + 4

    def _handle_custom_preset_click(self, pos):
        """处理自定义预设编辑工具栏的鼠标点击。"""
        if (self._renaming_preset is not None or self._deleting_preset is not None
                or self._song_select is not None or self._section_select is not None):
            return  # 弹窗期间忽略面板点击
        for rect, action, preset in getattr(self, '_custom_actions', []):
            if not rect.collidepoint(pos):
                continue
            if action == "rename":
                self._rename_custom_preset(preset)
            elif action == "delete":
                self._delete_custom_preset(preset)
            elif action == "favorite":
                state = self.custom_manager.toggle_favorite(preset.name)
                if state:
                    self._add_log(f"⭐ 已收藏 [{preset.name}]，置顶显示")
                else:
                    self._add_log(f"☆ 已取消收藏 [{preset.name}]")
            elif action == "move_left":
                if self.custom_manager.move_preset(preset.name, -1):
                    self._add_log(f"← [{preset.name}] 已前移")
            elif action == "move_right":
                if self.custom_manager.move_preset(preset.name, +1):
                    self._add_log(f"→ [{preset.name}] 已后移")
            return

    def _rename_custom_preset(self, preset):
        """用 Tk 原生对话框重命名自定义预设。

        SDL 在 Windows 上无法显示微软拼音候选框（UILess 模式抑制，
        Win11 22000+ 已知问题，hint 亦无效），内联输入等于盲打；
        Tk 对话框是系统原生窗口，候选框必然正常显示（见
        _ask_rename_dialog 说明）。Enter/Esc 后直接确认/取消。
        """
        new_name = self._ask_rename_dialog(preset)
        if new_name:
            self._renaming_preset = preset
            self._rename_text = new_name
            self._confirm_rename()
        else:
            self._add_log(f"✏️ 已取消重命名 [{preset.name}]")

    def _cancel_rename(self):
        """取消重命名，恢复快捷键。"""
        self._renaming_preset = None
        self._rename_text = ""
        pygame.key.stop_text_input()
        self._add_log("已取消重命名")

    def _ask_rename_dialog(self, preset) -> Optional[str]:
        """用主题化 Tk 原生输入对话框获取新名称（返回 None = 用户取消）。

        背景：SDL 在 Windows 上无法显示微软拼音的 IME 候选框
        （UILess 模式抑制原生候选 UI，Win11 22000+ 已知问题，hint
        SDL_HINT_IME_SHOW_UI 亦无效），pygame 内联输入等于盲打。
        Tk 对话框是系统原生窗口，微软拼音候选框必然正常显示；
        为与主界面 UI 统一，对话框套用 SoulTheme 同款配色与微软雅黑
        字体（深色面板 + 琥珀/青绿强调 + 输入框高亮边框）。
        """
        try:
            import tkinter as tk
        except Exception:
            print("[DEBUG] Tk 输入对话框不可用，退回内联输入")
            return None

        is_voice = bool(getattr(preset, '_is_voice_template', False))
        old_name = getattr(preset, 'name', '') or ''

        # 与 pygame 主界面同款主题色（SoulTheme，缺省时回退类常量）
        accent  = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
        bg      = SoulTheme.BG_DEEP if SoulTheme else self.COLOR_BG
        panel   = SoulTheme.BG_PANEL if SoulTheme else self.COLOR_PANEL
        text_c  = SoulTheme.TEXT_PRIMARY if SoulTheme else self.COLOR_TEXT
        muted   = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM
        input_bg = SoulTheme.BG_DEEP if SoulTheme else (25, 25, 25)

        def _hex(c):
            return '#%02x%02x%02x' % (int(c[0]), int(c[1]), int(c[2]))

        root = tk.Tk()
        root.title("重命名台词" if is_voice else "重命名预设")
        root.configure(bg=_hex(bg))
        root.resizable(False, False)
        root.attributes("-topmost", True)

        # 居中于 pygame 窗口
        W, H = 480, 196
        px = self.WIN_WIDTH // 2 - W // 2
        py = self.WIN_HEIGHT // 2 - H // 2
        root.geometry(f"{W}x{H}+{px}+{py}")

        f = "Microsoft YaHei UI"   # 与主界面 msyh.ttc 同族

        # 面板容器
        frame = tk.Frame(root, bg=_hex(panel), padx=24, pady=18)
        frame.pack(fill="both", expand=True)

        # 标题（强调色，与主界面面板标题一致）
        tk.Label(frame, text="重命名台词" if is_voice else "重命名预设",
                 bg=_hex(panel), fg=_hex(accent),
                 font=(f, 13, "bold")).pack(anchor="w")

        # 输入框（深底 + 强调色高亮边框，同主界面输入风格）
        var = tk.StringVar(value=old_name)
        entry = tk.Entry(frame, textvariable=var, bg=_hex(input_bg),
                         fg=_hex(text_c), insertbackground=_hex(accent),
                         relief="flat", font=(f, 13),
                         highlightthickness=1,
                         highlightbackground=_hex(accent),
                         highlightcolor=_hex(accent))
        entry.pack(fill="x", pady=(14, 10), ipady=6)
        entry.focus_force()
        root.update()
        entry.select_range(0, tk.END)   # 全选旧名，直接输入即覆盖

        # 提示行（次要文字色）
        tk.Label(frame, text="Enter 确认  |  Esc 取消  |  支持中文输入",
                 bg=_hex(panel), fg=_hex(muted),
                 font=(f, 9)).pack(anchor="w")

        result = {}

        def _ok(_e=None):
            result["val"] = var.get().strip()
            root.destroy()

        def _cancel(_e=None):
            result["val"] = None
            root.destroy()

        entry.bind("<Return>", _ok)
        entry.bind("<Escape>", _cancel)
        root.protocol("WM_DELETE_WINDOW", _cancel)

        root.wait_window(root)          # 模态阻塞，直到用户确认/取消
        new_name = result.get("val")
        return new_name or None

    def _confirm_rename(self):
        """确认重命名（同步引擎与存储），恢复快捷键。"""
        preset = self._renaming_preset
        new_name = self._rename_text.strip()
        self._renaming_preset = None
        self._rename_text = ""
        pygame.key.stop_text_input()

        if preset is None or not new_name or new_name == preset.name:
            return
        old_name = preset.name

        # 语音模板重命名
        if getattr(preset, '_is_voice_template', False):
            if self.voice_trigger is None:
                return
            if not self.voice_trigger.rename_trigger(old_name, new_name):
                self._add_log(f"⚠️ 重命名失败：[{new_name}] 已存在")
                return
            self._add_log(f"✏️ 台词已重命名: {old_name} → {new_name}")
            return

        if not self.custom_manager.rename_preset(old_name, new_name):
            self._add_log(f"⚠️ 重命名失败：[{new_name}] 已存在")
            return

        # 同步预设引擎（引擎内持有旧对象的引用）
        try:
            self.preset_engine.remove_preset(old_name)
        except ValueError:
            pass
        try:
            self.preset_engine.add_preset(preset)
        except ValueError:
            pass
        self._add_log(f"✏️ 预设已重命名: {old_name} → {new_name}")

    def _draw_rename_overlay(self):
        """绘制内联重命名输入框（居中覆盖层）。"""
        if self._renaming_preset is None:
            return

        accent = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
        bg = SoulTheme.BG_PANEL if SoulTheme else self.COLOR_PANEL
        bg_input = SoulTheme.BG_DEEP if SoulTheme else (25, 25, 25)
        text_c = SoulTheme.TEXT_PRIMARY if SoulTheme else self.COLOR_TEXT
        muted = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM

        # 半透明遮罩
        overlay = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        # 输入面板
        box = pygame.Rect(400, 280, 480, 130)
        pygame.draw.rect(self.screen, bg, box, border_radius=8)
        pygame.draw.rect(self.screen, accent, box, 2, border_radius=8)

        title_text = "重命名台词" if getattr(self._renaming_preset, '_is_voice_template', False) else "重命名预设"
        title = self._font('medium').render(title_text, True, accent)
        self.screen.blit(title, (box.x + 20, box.y + 14))

        # 输入框（带闪烁光标）
        input_rect = pygame.Rect(box.x + 20, box.y + 50, 440, 34)
        pygame.draw.rect(self.screen, bg_input, input_rect, border_radius=4)
        cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
        txt = self._font('small').render(self._rename_text + cursor, True, text_c)
        self.screen.blit(txt, (input_rect.x + 8, input_rect.y + 7))

        hint = self._font('tiny').render("Enter 确认  |  Esc 取消  |  支持中文输入", True, muted)
        self.screen.blit(hint, (box.x + 20, box.y + 96))

    def _delete_custom_preset(self, preset):
        """进入删除确认模式（界面内联弹窗，与重命名风格一致）。"""
        self._deleting_preset = preset
        self._add_log(f"🗑 确认删除 [{preset.name}]？（Enter 确认 / Esc 取消）")

    def _cancel_delete(self):
        """取消删除。"""
        self._deleting_preset = None
        self._add_log("已取消删除")

    def _confirm_delete(self):
        """确认删除（同步引擎与存储）。"""
        preset = self._deleting_preset
        self._deleting_preset = None
        if preset is None:
            return

        name = preset.name
        # 语音模板：走语音引擎的删除通道（含 wav 文件）
        if getattr(preset, '_is_voice_template', False):
            if self.voice_trigger is not None:
                self.voice_trigger.remove_trigger(name)
            self._add_log(f"🗑 已删除语音模板 [{name}]")
            return

        self.custom_manager.remove_preset(name)
        try:
            self.preset_engine.remove_preset(name)
        except ValueError:
            pass
        self._add_log(f"🗑 已删除预设 [{name}]")

    def _draw_delete_overlay(self):
        """绘制删除确认弹窗（居中覆盖层，与重命名风格一致）。"""
        if self._deleting_preset is None:
            self._delete_btn_rects = []
            return

        preset = self._deleting_preset
        accent = SoulTheme.ACCENT if SoulTheme else self.COLOR_ACCENT
        danger = SoulTheme.DANGER if SoulTheme else (220, 90, 90)
        bg = SoulTheme.BG_PANEL if SoulTheme else self.COLOR_PANEL
        bg_input = SoulTheme.BG_DEEP if SoulTheme else (25, 25, 25)
        text_c = SoulTheme.TEXT_PRIMARY if SoulTheme else self.COLOR_TEXT
        muted = SoulTheme.TEXT_MUTED if SoulTheme else self.COLOR_TEXT_DIM

        # 半透明遮罩
        overlay = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        # 面板
        box = pygame.Rect(400, 280, 480, 150)
        pygame.draw.rect(self.screen, bg, box, border_radius=8)
        pygame.draw.rect(self.screen, accent, box, 2, border_radius=8)

        title = self._font('medium').render("删除预设", True, danger)
        self.screen.blit(title, (box.x + 20, box.y + 14))

        msg = self._font('small').render(f"确定删除预设 [{preset.name}] 吗？", True, text_c)
        self.screen.blit(msg, (box.x + 20, box.y + 52))
        if getattr(preset, '_is_voice_template', False):
            sub_text = "模板音频文件将一并删除（歌曲文件不受影响）"
        else:
            sub_text = f"关联的 {len(preset.tracks)} 首歌曲文件不会被删除"
        sub = self._font('tiny').render(sub_text, True, muted)
        self.screen.blit(sub, (box.x + 20, box.y + 76))

        # 按钮：确认删除 / 取消
        mouse_pos = pygame.mouse.get_pos()
        btn_confirm = pygame.Rect(box.x + 20, box.y + 104, 120, 32)
        btn_cancel = pygame.Rect(box.x + 156, box.y + 104, 120, 32)
        self._delete_btn_rects = [(btn_confirm, 'confirm'), (btn_cancel, 'cancel')]

        for rect, label, color in ((btn_confirm, "确认删除", danger), (btn_cancel, "取消", muted)):
            hovered = rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, bg_input, rect, border_radius=6)
            pygame.draw.rect(self.screen, color, rect, 2 if hovered else 1, border_radius=6)
            surf = self._font('small').render(label, True, color)
            self.screen.blit(surf, surf.get_rect(center=rect.center))

        hint = self._font('tiny').render("Enter 确认  |  Esc 取消", True, muted)
        self.screen.blit(hint, (box.x + 300, box.y + 112))


    def _cleanup(self):
        """安全关闭所有资源"""
        self._add_log("正在关闭系统...")
        try:
            self.motion.stop()
        except Exception as e:
            print(f"关闭动作捕捉异常: {e}")
        try:
            if self.voice_trigger is not None:
                self.voice_trigger.stop_listening()
        except Exception as e:
            print(f"关闭语音触发异常: {e}")
        try:
            self.player.stop()
        except Exception as e:
            print(f"关闭播放器异常: {e}")
        pygame.quit()
        print("SoulPlayer 已安全退出。")


def main():
    """程序入口"""
    try:
        app = SoulPlayerApp()
        app.run()
    except KeyboardInterrupt:
        print("\n用户中断，程序退出。")
    except Exception as e:
        print(f"主程序异常: {e}")
        raise


if __name__ == "__main__":
    main()
