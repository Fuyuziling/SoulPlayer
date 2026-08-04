# -*- coding: utf-8 -*-
"""
SoulPlayer — 语音模板比对触发引擎 (voice_match.py)

B 方案：录制用户的声音作为音频模板，之后实时监听麦克风，
用 MFCC 特征 + DTW 动态时间规整计算相似度，命中模板即触发。
不依赖 Vosk 等语言模型，彻底避开原生 C 库在中文路径 / exe 打包下的崩溃面。

职责：
    - T 键录制台词：采集音频 → 静音裁剪 → 保存 WAV 模板
    - V 键实时监听：语音活动检测 (VAD) → MFCC+DTW 比对 → 触发动作
    - 模板与歌曲绑定的持久化（voice_templates.json + voice_templates/*.wav）

接口与 voice_trigger.VoiceTrigger 完全对齐，main.py 可无缝替换。

作者：SoulPlayer 开发团队
"""

from __future__ import annotations

import json
import queue
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 依赖检测（sounddevice 仅用于麦克风采集，崩溃面远小于 ASR 引擎）
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    sd = None  # type: ignore

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # 采集采样率
BLOCK_SIZE = 1024            # 每块约 64ms
RING_BLOCKS = 600            # 环形缓冲约 38 秒

MIN_SEGMENT_SEC = 0.25       # 最短语音段（更短视为噪音）
MAX_SEGMENT_SEC = 4.0        # 最长语音段（超出强制截断比对）
HANGOVER_SEC = 0.45          # 静音超过该时长判定语音结束
PRE_ROLL_BLOCKS = 4          # 语音起点前保留的块数

MATCH_THRESHOLD = 0.45       # DTW 余弦距离阈值（0~2 区间，越小越严格），命中日志会打印距离供调参
MATCH_MARGIN = 0.8           # 多个模板时，最佳距离须 ≤ 次佳距离 × 该比例（防相似发音误判）
COOLDOWN_SEC = 2.0           # 同一模板触发冷却，防止一句话重复触发


# ---------------------------------------------------------------------------
# 音频特征：MFCC（纯 numpy 实现）+ DTW
# ---------------------------------------------------------------------------
def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmax: float) -> np.ndarray:
    """构造 Mel 三角滤波器组，shape = (n_mels, n_fft//2+1)。"""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for j in range(left, center):
            fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            fb[i, j] = (right - j) / (right - center)
    return fb


# 滤波器组缓存（参数固定，只需算一次）
_FB_CACHE: Optional[np.ndarray] = None
_DCT_CACHE: Optional[np.ndarray] = None


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    """DCT-II 变换矩阵，shape = (n_mfcc, n_mels)。"""
    k = np.arange(n_mfcc)[:, None]
    n = np.arange(n_mels)[None, :]
    return np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))


def mfcc(samples: np.ndarray, sr: int = SAMPLE_RATE,
         n_mfcc: int = 13, n_fft: int = 512,
         win: int = 400, hop: int = 160, n_mels: int = 26) -> np.ndarray:
    """计算 MFCC 特征序列，shape = (帧数, n_mfcc)。含 CMVN 倒谱均值归一化。"""
    global _FB_CACHE, _DCT_CACHE
    if len(samples) < win:
        samples = np.pad(samples, (0, win - len(samples)))

    # 预加重 + 分帧 + 加窗
    y = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    n_frames = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * np.hamming(win)

    # 功率谱 → Mel 滤波 → log → DCT
    pow_spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2 / n_fft
    if _FB_CACHE is None:
        _FB_CACHE = _mel_filterbank(sr, n_fft, n_mels, sr / 2.0)
    mel = pow_spec @ _FB_CACHE.T
    mel = np.maximum(mel, 1e-10)
    if _DCT_CACHE is None:
        _DCT_CACHE = _dct_matrix(n_mfcc, n_mels)
    feat = np.log(mel) @ _DCT_CACHE.T

    # CMVN：减去本段均值，消除响度与声道差异
    feat -= feat.mean(axis=0, keepdims=True)
    return feat


def dtw_distance(a: np.ndarray, b: np.ndarray, band_ratio: float = 0.25) -> float:
    """
    带 Sakoe-Chiba 窗口约束的 DTW 距离，按路径长度归一化。
    a、b 为 (帧数, 特征维) 序列。
    帧间代价使用余弦距离（1 - cos），逐帧单位化后距离与响度/频谱规模无关，
    落在 0~2 区间：同一句台词通常 < 0.35，不同内容通常 > 0.6。
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    band = int(max(5, abs(n - m) + 2, band_ratio * max(n, m)))

    # 预计算余弦距离矩阵的帧范数（零向量防御）
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    L = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        ai = a[i - 1]
        an = a_norm[i - 1]
        for j in range(j_lo, j_hi + 1):
            denom = an * b_norm[j - 1]
            if denom < 1e-10:
                cost = 1.0
            else:
                cost = 1.0 - float(np.dot(ai, b[j - 1])) / denom
            # 三个前驱中取最小
            p = (D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            k = int(np.argmin(p))
            D[i, j] = cost + p[k]
            L[i, j] = 1 + (L[i - 1, j], L[i, j - 1], L[i - 1, j - 1])[k]

    if D[n, m] == INF or L[n, m] == 0:
        return INF
    return float(D[n, m] / L[n, m])


def trim_silence(samples: np.ndarray, frame: int = 400, hop: int = 160,
                 ratio: float = 0.08) -> np.ndarray:
    """按能量裁剪首尾静音。ratio 为相对峰值能量的阈值比例。"""
    if len(samples) < frame:
        return samples
    n_frames = 1 + (len(samples) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    energy = (samples[idx] ** 2).mean(axis=1)
    peak = energy.max()
    if peak <= 1e-8:
        return samples[:0]
    thr = peak * ratio
    active = np.where(energy > thr)[0]
    if len(active) == 0:
        return samples[:0]
    start = max(0, active[0] * hop - frame)
    end = min(len(samples), active[-1] * hop + frame * 2)
    return samples[start:end]


def rms_normalize(samples: np.ndarray, target: float = 0.1) -> np.ndarray:
    """RMS 响度归一化，消除离麦远近的差异。"""
    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    if rms < 1e-6:
        return samples
    gain = target / rms
    return np.clip(samples * gain, -1.0, 1.0)


# ---------------------------------------------------------------------------
# VoiceMatcher
# ---------------------------------------------------------------------------
class VoiceMatcher:
    """
    语音模板比对触发引擎。接口与 VoiceTrigger 对齐：

        vm = VoiceMatcher()
        vm.start_listening()
        action = vm.get_triggered_action()   # "voice_play:<模板名>"
        vm.stop_listening()

    录制台词（T 键流程）：
        vm.start_capture()
        ... 用户说话 ...
        name = vm.stop_capture()             # 返回自动命名的模板名，失败返回 ""
        vm.add_trigger(name, "voice_play", tracks=[...])
    """

    def __init__(
        self,
        templates_dir: Optional[str] = None,
        storage_path: str = "voice_templates.json",
    ):
        script_dir = Path(__file__).parent.resolve()

        td = Path(templates_dir) if templates_dir else script_dir / "voice_templates"
        if not td.is_absolute():
            td = script_dir / td
        self._templates_dir = td
        self._templates_dir.mkdir(parents=True, exist_ok=True)

        sp = Path(storage_path)
        if not sp.is_absolute():
            sp = script_dir / sp
        self._storage_path = sp

        # 兼容旧接口的数据结构
        self.triggers: dict[str, str] = {}          # name -> action
        self.trigger_tracks: dict[str, list[str]] = {}  # name -> 歌曲路径列表
        self._template_wav: dict[str, str] = {}     # name -> wav 文件路径
        self._template_feat: dict[str, np.ndarray] = {}  # name -> MFCC 特征缓存
        self.available: bool = SD_AVAILABLE

        # 音频流与线程
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ring: deque = deque(maxlen=RING_BLOCKS)
        self._ring_count = 0        # 累计入环块数（单调递增，用于定位）
        self._processed = 0         # VAD 已处理到的块序号

        # 录制模式
        self._capture: Optional[list] = None

        # VAD 状态
        self._vad_in_speech = False
        self._vad_speech_start = 0
        self._vad_last_voice = 0
        self._noise_floor = 1e-4

        # 触发冷却
        self._last_trigger: dict[str, float] = {}

        self._status: dict = {
            "state": "idle",
            "last_text": "",
            "last_action": None,
            "error": None,
            "model_loaded": SD_AVAILABLE,
            "model_path": str(self._templates_dir),
        }
        if not SD_AVAILABLE:
            self._status["error"] = "缺少 sounddevice，请执行: pip install sounddevice"
            print(f"[VoiceMatch] {self._status['error']}")

        self._load_storage()

    # ------------------------------------------------------------------
    # 存储
    # ------------------------------------------------------------------
    def _load_storage(self) -> None:
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}

        for item in data.get("templates", []):
            name = item.get("name", "").strip()
            wav = item.get("wav", "").strip()
            if not name or not wav:
                continue
            wav_path = Path(wav)
            if not wav_path.is_absolute():
                wav_path = Path(__file__).parent.resolve() / wav_path
            if not wav_path.exists():
                print(f"[VoiceMatch] 模板音频缺失，跳过: {wav_path}")
                continue
            self.triggers[name] = item.get("action", "voice_play")
            tracks = item.get("tracks") or []
            if tracks:
                self.trigger_tracks[name] = [str(t) for t in tracks]
            self._template_wav[name] = str(wav_path)
            try:
                self._template_feat[name] = self._extract_feature(str(wav_path))
            except Exception as e:
                print(f"[VoiceMatch] 模板特征提取失败 [{name}]: {e}")

        print(f"[VoiceMatch] 已加载 {len(self.triggers)} 条语音模板")

    def _save_storage(self) -> None:
        data = {"templates": []}
        for name, action in self.triggers.items():
            item = {"name": name, "action": action, "wav": self._template_wav.get(name, "")}
            tracks = self.trigger_tracks.get(name)
            if tracks:
                item["tracks"] = tracks
            data["templates"].append(item)
        try:
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[VoiceMatch] 保存失败: {e}")

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------
    def _extract_feature(self, wav_path: str) -> np.ndarray:
        with wave.open(wav_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        samples = rms_normalize(trim_silence(samples))
        return mfcc(samples)

    # ------------------------------------------------------------------
    # 映射管理（接口与 VoiceTrigger 对齐）
    # ------------------------------------------------------------------
    def add_trigger(self, phrase: str, action: str, tracks: Optional[list] = None) -> None:
        """为已录制的模板绑定动作/歌曲。phrase 即模板名。"""
        phrase = phrase.strip()
        if not phrase or phrase not in self._template_wav:
            print(f"[VoiceMatch] 模板不存在，无法绑定: {phrase}")
            return
        self.triggers[phrase] = action.strip() or "voice_play"
        if tracks:
            self.trigger_tracks[phrase] = [str(t) for t in tracks]
        self._save_storage()
        print(f"[VoiceMatch] 已绑定: \"{phrase}\" ({len(self.trigger_tracks.get(phrase, []))} 首歌曲)")

    def remove_trigger(self, phrase: str) -> None:
        """删除模板（含 wav 文件与歌曲绑定）。"""
        phrase = phrase.strip()
        if phrase not in self.triggers and phrase not in self._template_wav:
            return
        self.triggers.pop(phrase, None)
        self.trigger_tracks.pop(phrase, None)
        self._template_feat.pop(phrase, None)
        wav = self._template_wav.pop(phrase, None)
        if wav:
            try:
                Path(wav).unlink(missing_ok=True)
            except OSError:
                pass
        self._save_storage()
        print(f"[VoiceMatch] 已删除模板: \"{phrase}\"")

    def get_tracks(self, phrase: str) -> list:
        return list(self.trigger_tracks.get(phrase, []))

    def list_templates(self) -> list:
        """返回模板信息列表 [{name, track_count}]，供 UI 展示。"""
        return [
            {"name": n, "track_count": len(self.trigger_tracks.get(n, []))}
            for n in self._template_wav
        ]

    # ------------------------------------------------------------------
    # 录制模式（T 键流程）
    # ------------------------------------------------------------------
    def start_capture(self) -> None:
        """开始采集台词音频。若监听未运行会自动启动音频流。"""
        if not self._running:
            self.start_listening()
        with self._lock:
            self._capture = []
        print("[VoiceMatch] 开始录制台词音频")

    def stop_capture(self) -> str:
        """
        结束录制：裁剪静音、保存 WAV、建立模板。
        返回自动命名的模板名（如 voice_153012）；录制无效返回 ""。
        """
        with self._lock:
            blocks = self._capture
            self._capture = None
        if not blocks:
            print("[VoiceMatch] 未采集到音频")
            return ""

        samples = np.concatenate(blocks)
        samples = trim_silence(samples)
        if len(samples) < int(SAMPLE_RATE * MIN_SEGMENT_SEC):
            print(f"[VoiceMatch] 录音太短（{len(samples)/SAMPLE_RATE:.2f}s），模板无效")
            return ""

        name = f"voice_{time.strftime('%H%M%S')}"
        # 同一秒内重录则追加序号
        base = name
        i = 2
        while name in self._template_wav:
            name = f"{base}_{i}"
            i += 1

        wav_path = self._templates_dir / f"{name}.wav"
        samples = rms_normalize(samples)
        pcm = (samples * 32767.0).astype(np.int16)
        try:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())
        except OSError as e:
            print(f"[VoiceMatch] 模板保存失败: {e}")
            return ""

        self._template_wav[name] = str(wav_path)
        self._template_feat[name] = mfcc(samples)
        dur = len(samples) / SAMPLE_RATE
        print(f"[VoiceMatch] 模板已保存: {name}（{dur:.1f}s）")
        return name

    @property
    def is_capturing(self) -> bool:
        with self._lock:
            return self._capture is not None

    # ------------------------------------------------------------------
    # 监听控制
    # ------------------------------------------------------------------
    def start_listening(self) -> None:
        if not self.available:
            print("[VoiceMatch] 不可用：缺少 sounddevice")
            return
        if self._running:
            return
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self._status["error"] = f"麦克风打开失败: {e}"
            self._status["state"] = "error"
            print(f"[VoiceMatch] {self._status['error']}")
            return

        self._running = True
        self._status["state"] = "listening"
        self._status["error"] = None
        self._processed = self._ring_count
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[VoiceMatch] 开始监听麦克风（模板比对模式）...")

    def stop_listening(self) -> None:
        if not self._running:
            return
        self._running = False
        self._status["state"] = "idle"
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[VoiceMatch] 停止监听")

    @property
    def is_listening(self) -> bool:
        return self._running

    @property
    def speech_active(self) -> bool:
        """当前是否检测到说话声（或正处于台词录制捕获中）。

        供主程序做"人声/扬声器分离"：说话时自动压低音乐音量，
        避免扬声器播放的音乐被麦克风拾取后误匹配模板。"""
        return bool(self._running
                    and (self._vad_in_speech or self._capture is not None))

    # ------------------------------------------------------------------
    # 音频回调与工作线程
    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice 回调：音频块入环形缓冲，录制模式下同时入录制列表。"""
        try:
            block = np.array(indata[:, 0], dtype=np.float32, copy=True)
        except Exception:
            try:
                block = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                return
        with self._lock:
            self._ring.append(block)
            self._ring_count += 1
            if self._capture is not None:
                self._capture.append(block)

    def _energy(self, block: np.ndarray) -> float:
        return float(np.mean(block ** 2))

    def _worker(self) -> None:
        """工作线程：VAD 分段 → 段结束做模板比对。"""
        while self._running:
            try:
                self._process_new_blocks()
            except Exception as e:
                print(f"[VoiceMatch] 工作线程异常: {e}")
            time.sleep(0.03)

    def _process_new_blocks(self) -> None:
        with self._lock:
            if self._capture is not None:
                # 录制模式下不做触发比对，只推进已处理指针
                self._processed = self._ring_count
                self._vad_in_speech = False
                return
            start = self._processed
            end = self._ring_count
            blocks = list(self._ring)
            ring_oldest = self._ring_count - len(blocks)

        for idx in range(start, end):
            rel = idx - ring_oldest
            if rel < 0 or rel >= len(blocks):
                continue
            e = self._energy(blocks[rel])
            now_blocks = idx

            # 噪声底自适应（仅非语音段更新）
            if not self._vad_in_speech:
                self._noise_floor = 0.98 * self._noise_floor + 0.02 * e

            thr = max(self._noise_floor * 4.0, 4e-4)
            voiced = e > thr

            if not self._vad_in_speech:
                if voiced:
                    self._vad_in_speech = True
                    self._vad_speech_start = now_blocks
                    self._vad_last_voice = now_blocks
            else:
                if voiced:
                    self._vad_last_voice = now_blocks
                silent_for = (now_blocks - self._vad_last_voice) * BLOCK_SIZE / SAMPLE_RATE
                speech_len = (now_blocks - self._vad_speech_start) * BLOCK_SIZE / SAMPLE_RATE
                if silent_for >= HANGOVER_SEC or speech_len >= MAX_SEGMENT_SEC:
                    self._finalize_segment(now_blocks)
                    self._vad_in_speech = False

        with self._lock:
            self._processed = max(self._processed, end)

    def _finalize_segment(self, end_block: int) -> None:
        """语音段结束：提取音频并比对所有模板。"""
        with self._lock:
            blocks = list(self._ring)
            ring_oldest = self._ring_count - len(blocks)
            start_block = max(self._vad_speech_start - PRE_ROLL_BLOCKS, ring_oldest)
            rel_lo = start_block - ring_oldest
            rel_hi = end_block - ring_oldest + 1
            if rel_hi <= rel_lo:
                return
            seg = np.concatenate(blocks[rel_lo:rel_hi])

        if len(seg) < int(SAMPLE_RATE * MIN_SEGMENT_SEC):
            return
        seg = rms_normalize(trim_silence(seg))
        if len(seg) < int(SAMPLE_RATE * MIN_SEGMENT_SEC):
            return

        feat = mfcc(seg)
        best_name, best_dist = None, float("inf")
        distances = []
        for name, tfeat in self._template_feat.items():
            # 时长一致性过滤：帧数差异过大的模板直接跳过（防"串台"）
            ratio = len(feat) / max(1, len(tfeat))
            if not (0.5 <= ratio <= 2.0):
                print(f"[VoiceMatch] 跳过模板 [{name}]（时长比 {ratio:.2f} 超出 0.5~2.0）")
                continue
            d = dtw_distance(feat, tfeat)
            distances.append((name, d))
            if d < best_dist:
                best_name, best_dist = name, d

        if distances:
            dbg = ", ".join(f"{n}={d:.2f}" for n, d in sorted(distances, key=lambda x: x[1]))
            print(f"[VoiceMatch] 比对: {dbg}（阈值 {MATCH_THRESHOLD}）")

        if best_name is None or best_dist > MATCH_THRESHOLD:
            return

        # 多模板时要求最佳匹配与次佳保持足够差距，避免相似发音误判
        if len(distances) >= 2:
            sorted_d = sorted(d for _, d in distances)
            if best_dist > sorted_d[1] * MATCH_MARGIN:
                print(f"[VoiceMatch] 区分度不足，忽略（最佳 {best_dist:.2f} vs 次佳 {sorted_d[1]:.2f}）")
                return

        # 冷却期检查
        now = time.time()
        if now - self._last_trigger.get(best_name, 0.0) < COOLDOWN_SEC:
            return
        self._last_trigger[best_name] = now

        action = self.triggers.get(best_name, "voice_play")
        if action == "voice_play":
            action = f"voice_play:{best_name}"
        with self._lock:
            self._status["last_text"] = best_name
            self._status["last_action"] = action
        self._queue.put(action)
        print(f"[VoiceMatch] 命中模板 [{best_name}] dist={best_dist:.2f} → {action}")

    # ------------------------------------------------------------------
    # 结果与状态（接口与 VoiceTrigger 对齐）
    # ------------------------------------------------------------------
    def get_triggered_action(self) -> Optional[str]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def drain_queue(self) -> list:
        actions = []
        while True:
            try:
                actions.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return actions

    def get_status(self) -> dict:
        with self._lock:
            s = self._status.copy()
        s["available"] = self.available
        s["listening"] = self._running
        s["trigger_count"] = len(self._template_wav)
        s["status"] = s.get("error") or ("监听中" if self._running else "就绪" if self.available else "未就绪")
        s["last_result"] = s.get("last_text", "")
        s["last_error"] = s.get("error", "")
        return s
