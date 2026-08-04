# -*- coding: utf-8 -*-
"""
SoulPlayer — 音乐播放引擎 (music_player.py)

负责：
    - 音乐文件扫描与播放列表管理
    - 播放 / 暂停 / 停止 / 切歌控制
    - 淡入淡出 (crossfade) 切换曲目
    - 按情绪 (mood) 切换播放列表
    - 音量控制

后端：pygame.mixer
"""

import os
import threading
import time
import random
from pathlib import Path
from typing import List, Optional

import pygame.mixer


# ---------------------------------------------------------------------------
#  常量
# ---------------------------------------------------------------------------
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac"}
DEFAULT_CROSSFADE_MS = 2000          # 默认淡入淡出时长（毫秒）
CROSSFADE_STEP_MS = 50               # 每次音量调整的间隔（毫秒）

# 情绪 → 子目录名 映射（供 set_playlist_by_mood 使用）
MOOD_SUBDIR_MAP = {
    "calm": "calm",
    "energetic": "energetic",
    "happy": "happy",
    "focus": "focus",
    "party": "party",
    "relaxed": "relaxed",
}


class MusicPlayerError(Exception):
    """音乐播放器内部异常基类"""
    pass


class PlaylistNotLoadedError(MusicPlayerError):
    """尚未加载任何播放列表"""
    pass


class TrackIndexError(MusicPlayerError):
    """曲目索引越界"""
    pass


class NoBackendsAvailableError(MusicPlayerError):
    """无可用音频后端"""
    pass


# ---------------------------------------------------------------------------
#  MusicPlayer
# ---------------------------------------------------------------------------
class MusicPlayer:
    """
    音乐播放引擎。

    基于 pygame.mixer.music 实现流式播放，支持多播放列表、
    淡入淡出切歌、情绪模式切换等功能。
    """

    def __init__(self):
        """
        初始化播放器，自动尝试启动 pygame.mixer 后端。
        """
        # 播放列表数据
        self._playlists: dict[str, List[str]] = {}   # mood_name -> [文件路径列表]
        self._current_playlist_key: Optional[str] = None
        self._tracks: List[str] = []                  # 当前激活的播放列表
        self._current_index: int = -1                 # 当前曲目索引

        # 播放状态
        self._playing: bool = False
        self._paused: bool = False
        self._volume: float = 0.8                      # 0.0 ~ 1.0
        self._last_volume_before_fade: float = 0.8

        # 淡入淡出线程控制
        self._fade_lock = threading.Lock()
        self._fade_thread: Optional[threading.Thread] = None
        self._fade_stop_event = threading.Event()

        # 跨曲目信息缓存（用于 current_track 属性）
        self._track_metadata_cache: dict[str, dict] = {}

        # 初始化 pygame.mixer
        self._init_backend()

    # ------------------------------------------------------------------
    #  内部：后端初始化
    # ------------------------------------------------------------------
    def _init_backend(self) -> None:
        """
        尝试初始化 pygame.mixer。若失败则抛出 NoBackendsAvailableError。
        如果 mixer 已被初始化，先关闭再重新初始化以确保参数一致。
        """
        try:
            # 若 mixer 已初始化，先关闭再重新初始化（确保使用指定参数）
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.music.set_volume(self._volume)
        except pygame.error as exc:
            raise NoBackendsAvailableError(
                f"pygame.mixer 初始化失败: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    #  播放列表管理
    # ------------------------------------------------------------------
    def load_playlist(self, folder_path: str) -> None:
        """
        从文件夹递归扫描受支持的音乐文件，并设为默认播放列表。

        :param folder_path: 音乐文件夹路径（绝对或相对）
        :raises PlaylistNotLoadedError: 如果文件夹不存在或为空
        """
        folder = Path(folder_path).expanduser().resolve()
        if not folder.exists():
            raise PlaylistNotLoadedError(f"播放列表路径不存在: {folder}")

        # 扫描所有音频文件
        tracks = []
        for ext in SUPPORTED_AUDIO_EXTS:
            tracks.extend(sorted(folder.rglob(f"*{ext}")))

        if not tracks:
            raise PlaylistNotLoadedError(
                f"在 {folder} 中未找到受支持的音频文件: {SUPPORTED_AUDIO_EXTS}"
            )

        # 去重并统一为字符串路径
        self._tracks = [str(t) for t in tracks]
        self._current_playlist_key = "default"
        self._playlists["default"] = self._tracks.copy()

        # 同时尝试加载 mood 子目录
        self._auto_load_mood_subdirs(folder)

        self._current_index = 0 if self._tracks else -1

    def _auto_load_mood_subdirs(self, base_folder: Path) -> None:
        """
        自动扫描 base_folder 下与 MOOD_SUBDIR_MAP 对应的子目录，
        将其作为 mood 播放列表加载。
        """
        for mood, subdir in MOOD_SUBDIR_MAP.items():
            mood_dir = base_folder / subdir
            if mood_dir.exists() and mood_dir.is_dir():
                mood_tracks = []
                for ext in SUPPORTED_AUDIO_EXTS:
                    mood_tracks.extend(sorted(mood_dir.rglob(f"*{ext}")))
                if mood_tracks:
                    self._playlists[mood] = [str(t) for t in mood_tracks]

    def sync_folder(self, folder_path: str) -> tuple:
        """
        重新扫描音乐文件夹，增量同步默认播放列表：
        新增的歌曲追加到列表末尾，已从文件夹删除的歌曲从列表移除。
        不打断当前播放；若正在播放的歌曲被删除，仅将其移出列表。

        :param folder_path: 音乐文件夹路径
        :return: (新增歌曲列表, 移除歌曲列表)
        """
        folder = Path(folder_path)
        if not folder.exists():
            return [], []

        scanned = sorted(
            str(p) for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
        )
        scanned_set = set(scanned)

        default = self._playlists.get("default")
        if default is None:
            # 尚无默认列表（如启动时文件夹为空）：直接整表加载
            self._playlists["default"] = scanned
            if self._current_playlist_key in (None, "default"):
                playing = None
                self._tracks = scanned.copy()
                self._current_playlist_key = "default"
                self._current_index = 0 if self._tracks else -1
            return scanned, []

        old_set = set(default)
        added = [t for t in scanned if t not in old_set]
        removed = [t for t in default if t not in scanned_set]
        if not added and not removed:
            return [], []

        # 保留原有顺序，移除失效项，新增项追加到末尾
        new_default = [t for t in default if t in scanned_set] + added
        self._playlists["default"] = new_default

        # 当前激活列表是 default 时才同步 _tracks 并修正当前索引
        if self._current_playlist_key in (None, "default"):
            playing = (
                self._tracks[self._current_index]
                if 0 <= self._current_index < len(self._tracks) else None
            )
            self._tracks = new_default.copy()
            self._current_playlist_key = "default"
            if playing and playing in self._tracks:
                self._current_index = self._tracks.index(playing)
            elif not self._tracks:
                self._current_index = -1
            elif self._current_index >= len(self._tracks) or playing not in self._tracks:
                # 当前歌曲已被移除：停留在原位置附近，不主动切换播放
                self._current_index = min(max(self._current_index, 0), len(self._tracks) - 1)

        return added, removed

    def load_playlist_from_spotify(self, playlist_id: str, token: str) -> None:
        """
        从 Spotify 加载播放列表（可选功能，当前版本为占位实现）。

        如需真实实现，可引入 spotipy 库并调用 Spotify Web API。
        """
        # TODO: 接入 spotipy 进行真实请求
        raise NotImplementedError(
            "Spotify 播放列表加载暂未实现。请使用本地文件夹加载。"
        )

    # ------------------------------------------------------------------
    #  播放控制
    # ------------------------------------------------------------------
    def play(self, track_index: Optional[int] = None) -> None:
        """
        播放 / 继续播放。

        :param track_index: 若指定，则播放该索引的曲目；否则继续当前或从头播放。
        :raises PlaylistNotLoadedError: 尚未加载播放列表
        :raises TrackIndexError: 索引越界
        """
        if not self._tracks:
            raise PlaylistNotLoadedError("播放列表为空，请先调用 load_playlist()")

        if track_index is not None:
            if not (0 <= track_index < len(self._tracks)):
                raise TrackIndexError(
                    f"曲目索引 {track_index} 超出范围 [0, {len(self._tracks) - 1}]"
                )
            self._current_index = track_index
            self._paused = False
            self._load_and_play(self._tracks[self._current_index])
        else:
            # 未指定索引：如果处于暂停则恢复，否则从头播放当前索引
            if self._paused:
                pygame.mixer.music.unpause()
                self._paused = False
                self._playing = True
            elif self._current_index >= 0:
                self._load_and_play(self._tracks[self._current_index])
            else:
                self._current_index = 0
                self._load_and_play(self._tracks[0])

    def pause(self) -> None:
        """暂停当前播放。若已暂停则无副作用。"""
        if self._playing and not self._paused:
            pygame.mixer.music.pause()
            self._paused = True
            self._playing = False

    def stop(self) -> None:
        """停止播放并重置状态。"""
        self._stop_fade()
        pygame.mixer.music.stop()
        self._playing = False
        self._paused = False

    def next_track(self) -> None:
        """
        切换到下一首。若已到达末尾，则循环到第一首。
        若播放列表为空，静默返回，不抛异常。
        """
        if not self._tracks:
            return
        if self._current_index < 0:
            self._current_index = 0
        else:
            self._current_index = (self._current_index + 1) % len(self._tracks)
        self._load_and_play(self._tracks[self._current_index])

    def previous_track(self) -> None:
        """
        切换到上一首。若已到达开头，则循环到最后一首。
        若播放列表为空，静默返回，不抛异常。
        """
        if not self._tracks:
            return
        if self._current_index < 0:
            self._current_index = len(self._tracks) - 1
        else:
            self._current_index = (self._current_index - 1) % len(self._tracks)
        self._load_and_play(self._tracks[self._current_index])

    def play_track(self, filepath: str) -> None:
        """
        直接加载并播放指定文件路径。

        :param filepath: 音频文件绝对路径
        """
        if not os.path.exists(filepath):
            raise MusicPlayerError(f"文件不存在: {filepath}")

        # 尝试将文件加入当前播放列表（如果不在列表中）
        if filepath not in self._tracks:
            self._tracks.append(filepath)

        # 更新当前索引
        self._current_index = self._tracks.index(filepath)
        self._load_and_play(filepath)

    def _load_and_play(self, filepath: str) -> None:
        """
        内部方法：加载文件并播放。

        :param filepath: 音频文件绝对路径
        """
        # 确保停止任何正在进行的淡入淡出
        self._stop_fade()

        try:
            pygame.mixer.music.load(filepath)
        except pygame.error as exc:
            # 若加载失败，尝试跳到下一首（避免卡死）
            # 但此处选择抛出异常，让上层处理
            raise MusicPlayerError(f"无法加载音频文件: {filepath} — {exc}") from exc

        pygame.mixer.music.set_volume(self._volume)
        pygame.mixer.music.play()
        self._playing = True
        self._paused = False

    # ------------------------------------------------------------------
    #  音量控制
    # ------------------------------------------------------------------
    def set_volume(self, volume: float) -> None:
        """
        设置音量，范围 0.0 ~ 1.0。

        :param volume: 音量值，会被钳制到 [0.0, 1.0]
        """
        volume = float(max(0.0, min(1.0, volume)))
        self._volume = volume
        self._last_volume_before_fade = volume
        pygame.mixer.music.set_volume(volume)

    # ------------------------------------------------------------------
    #  淡入淡出 (Crossfade)
    # ------------------------------------------------------------------
    def crossfade_to(self, track_index: int, duration_ms: int = DEFAULT_CROSSFADE_MS) -> None:
        """
        淡入淡出切换到指定曲目。

        实现策略：
            1. 先线性降低当前音量（淡出阶段）
            2. 切换曲目并播放新曲
            3. 再线性提升音量到目标值（淡入阶段）

        :param track_index: 目标曲目索引
        :param duration_ms: 过渡总时长（毫秒），默认 2000
        :raises PlaylistNotLoadedError: 播放列表为空
        :raises TrackIndexError: 索引越界
        """
        if not self._tracks:
            raise PlaylistNotLoadedError("播放列表为空")
        if not (0 <= track_index < len(self._tracks)):
            raise TrackIndexError(
                f"曲目索引 {track_index} 超出范围 [0, {len(self._tracks) - 1}]"
            )

        # 若正在淡入淡出，先中断
        self._stop_fade()

        self._fade_thread = threading.Thread(
            target=self._crossfade_worker,
            args=(track_index, duration_ms),
            daemon=True,
        )
        self._fade_thread.start()

    def _crossfade_worker(self, track_index: int, duration_ms: int) -> None:
        """
        在独立线程中执行淡出→切换→淡入。
        """
        fade_out_ms = duration_ms // 2
        fade_in_ms = duration_ms - fade_out_ms
        steps_out = max(1, fade_out_ms // CROSSFADE_STEP_MS)
        steps_in = max(1, fade_in_ms // CROSSFADE_STEP_MS)
        target_volume = self._last_volume_before_fade

        # ---- 淡出 ----
        current_vol = pygame.mixer.music.get_volume()
        for i in range(1, steps_out + 1):
            if self._fade_stop_event.is_set():
                return
            new_vol = current_vol * (1 - i / steps_out)
            pygame.mixer.music.set_volume(max(0.0, new_vol))
            time.sleep(CROSSFADE_STEP_MS / 1000.0)

        # ---- 切换曲目 ----
        if self._fade_stop_event.is_set():
            return
        self._current_index = track_index
        try:
            pygame.mixer.music.load(self._tracks[self._current_index])
            pygame.mixer.music.set_volume(0.0)
            pygame.mixer.music.play()
            self._playing = True
            self._paused = False
        except pygame.error:
            # 播放失败，恢复音量并退出
            pygame.mixer.music.set_volume(target_volume)
            return

        # ---- 淡入 ----
        for i in range(1, steps_in + 1):
            if self._fade_stop_event.is_set():
                return
            new_vol = target_volume * (i / steps_in)
            pygame.mixer.music.set_volume(min(target_volume, new_vol))
            time.sleep(CROSSFADE_STEP_MS / 1000.0)

        # 确保最终音量精确
        pygame.mixer.music.set_volume(target_volume)

    def _stop_fade(self) -> None:
        """
        中断当前正在进行的淡入淡出线程，并恢复音量。
        """
        with self._fade_lock:
            if self._fade_thread is not None and self._fade_thread.is_alive():
                self._fade_stop_event.set()
                self._fade_thread.join(timeout=1.0)
                self._fade_stop_event.clear()
            self._fade_thread = None
            # 恢复音量
            pygame.mixer.music.set_volume(self._last_volume_before_fade)

    # ------------------------------------------------------------------
    #  情绪播放列表切换
    # ------------------------------------------------------------------
    def set_playlist_by_mood(self, mood: str) -> None:
        """
        根据情绪/模式切换播放列表。

        :param mood: 如 'calm', 'energetic', 'happy', 'focus', 'party', 'relaxed'
        :raises PlaylistNotLoadedError: 若该 mood 没有预加载的曲目
        """
        mood_key = mood.lower().strip()
        if mood_key not in self._playlists or not self._playlists[mood_key]:
            raise PlaylistNotLoadedError(
                f"未找到 mood='{mood}' 的播放列表。"
                f"已加载的 mood 列表: {list(self._playlists.keys())}"
            )

        # 如果已经在该 mood 的播放列表中，则随机切歌（不重置）
        if self._current_playlist_key == mood_key and self._tracks:
            self._tracks = self._playlists[mood_key].copy()
            # 随机选择一首，避免连续播放同一首
            new_index = random.randint(0, len(self._tracks) - 1)
            if new_index == self._current_index and len(self._tracks) > 1:
                new_index = (new_index + 1) % len(self._tracks)
            self.crossfade_to(new_index, duration_ms=DEFAULT_CROSSFADE_MS)
            return

        # 切换到新 mood 播放列表
        self._current_playlist_key = mood_key
        self._tracks = self._playlists[mood_key].copy()
        self._current_index = 0
        self._load_and_play(self._tracks[0])

    # ------------------------------------------------------------------
    #  属性
    # ------------------------------------------------------------------
    @property
    def is_playing(self) -> bool:
        """当前是否正在播放（暂停不算正在播放）。"""
        return self._playing and not self._paused and pygame.mixer.music.get_busy()

    @property
    def current_track(self) -> dict:
        """
        返回当前曲目的信息字典。

        若未加载曲目，则返回空字典。
        """
        if not self._tracks or self._current_index < 0:
            return {}

        filepath = self._tracks[self._current_index]

        # 缓存解析，避免重复 I/O
        if filepath not in self._track_metadata_cache:
            path_obj = Path(filepath)
            self._track_metadata_cache[filepath] = {
                "index": self._current_index,
                "filepath": filepath,
                "filename": path_obj.name,
                "title": path_obj.stem,          # 默认使用文件名（无扩展名）作为标题
                "artist": "Unknown",              # 当前版本未解析 ID3 标签
                "duration": -1,                   # 当前版本未解析时长
                "playlist": self._current_playlist_key or "unknown",
            }

        # 返回副本，并确保 index 始终为最新
        info = self._track_metadata_cache[filepath].copy()
        info["index"] = self._current_index
        return info

    # ------------------------------------------------------------------
    #  其他工具方法
    # ------------------------------------------------------------------
    def get_playlist_info(self) -> dict:
        """
        获取当前播放列表的摘要信息（用于 UI 展示或调试）。

        :return: 包含 total_tracks, current_index, playlist_key, tracks 的字典
        """
        return {
            "total_tracks": len(self._tracks),
            "current_index": self._current_index,
            "playlist_key": self._current_playlist_key,
            "tracks": [Path(t).name for t in self._tracks],
        }

    def unload(self) -> None:
        """
        安全卸载播放器，释放 pygame.mixer 资源。
        建议在程序退出前调用。
        """
        self.stop()
        # pygame.mixer.music.unload() 在 pygame >= 2.1.0 才可用，旧版本回退到 stop
        if hasattr(pygame.mixer.music, 'unload'):
            pygame.mixer.music.unload()
        pygame.mixer.quit()

    def __del__(self):
        """
        析构时尝试安全释放资源。
        """
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  直接播放列表注册（供主程序或测试动态设置 mood 播放列表）
    # ------------------------------------------------------------------
    def register_playlist(self, key: str, tracks: List[str]) -> None:
        """
        手动注册一个播放列表，不通过文件扫描。

        :param key: 播放列表标识，如 'calm', 'energetic'
        :param tracks: 音频文件绝对路径列表
        """
        # 过滤掉不存在的文件
        valid_tracks = [t for t in tracks if os.path.exists(t)]
        self._playlists[key] = valid_tracks

        # 如果是第一个播放列表，自动设为当前
        if self._current_playlist_key is None and valid_tracks:
            self._current_playlist_key = key
            self._tracks = valid_tracks.copy()
            self._current_index = 0
