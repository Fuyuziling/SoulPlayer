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
MAX_CAPTURE_SEC = 60.0       # 录制缓冲区上限（秒），防止忘记结束录制导致内存无限增长

MIN_SEGMENT_SEC = 0.25       # 最短语音段（更短视为噪音）
MAX_SEGMENT_SEC = 4.0        # 最长语音段（超出强制截断比对）
HANGOVER_SEC = 0.45          # 静音超过该时长判定语音结束
PRE_ROLL_BLOCKS = 4          # 语音起点前保留的块数

# 实测两个不同模板（约 3s 长录音）的余弦距离仅约 0.34，远低于理想区分度（>0.6），
# 而真实复述距离常落 0.3~0.5。原阈值 0.50 + margin 0.95 偏松，无关内容（0.45~0.6）
# 也可能命中 → 串播。收紧为：阈值 0.45、margin 0.90 + 绝对差距 0.06（见 _finalize_segment）。
# 阈值历史：0.50（初版，日语串播）→ 0.45（收紧防串）→ 0.50（恢复）→ 0.60（本轮，纯 MFCC 环境）。
# 0.45/0.50 在 lifter 之后余量不足：模拟同词复述距离 0.19~0.34 是"同段拉伸"的乐观估计，
# 真实独立复述（声调/韵律差异，中文尤甚）常落 0.40~0.60，0.45/0.50 会拦掉不少真实命中
#（实测中文/英文触发率偏低；本次 8 条真实模板"变速±12%+噪声"复述模拟：自身距离
#  0.43~0.60，0.50 阈值下约半数被拦 → 用户反馈"录制过的台词触发不了"）。
# 恢复并放宽到 0.60：覆盖真实复述距离区间；串播防护由 lifter 提升的
# 异词区分度（异词中位 0.67~0.95，远高于阈值）＋双重区分度门槛（MATCH_MARGIN/ABS_MARGIN）
# 承担，不再单靠阈值收紧。
MATCH_THRESHOLD = 0.60       # DTW 余弦距离阈值（0~2 区间，越小越严格），命中日志会打印距离供调参
MATCH_MARGIN = 0.90          # 多模板时，最佳距离须 ≤ 次佳距离 × 该比例（仅当次佳也是合格候选时生效）
ABS_MARGIN = 0.06            # 多模板时，最佳距离 + 该值 ≤ 次佳距离（绝对差距，防"最佳与次佳都像"）
COOLDOWN_SEC = 2.0           # 同一模板触发冷却，防止一句话重复触发

# ---------------------------------------------------------------------------
# 内容嵌入（标准模式，可选）：wav2vec2 自监督嵌入 + 余弦距离，与 MFCC+DTW 融合评分。
# 设计为"可插拔降级"：未安装 torch/transformers 或加载失败时，EMBED_AVAILABLE=False，
# 自动回退到纯 MFCC+DTW（现状）；依赖就绪后自动启用，无需改代码。
# 融合评分：d_final = w_emb * d_embed + (1 - w_emb) * d_mfcc（同为 0~2 尺度），
# 嵌入为主、MFCC 兜底；阈值/区分度/时长过滤沿用现有逻辑。
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "facebook/wav2vec2-base"   # 内容级自监督嵌入模型
EMBED_W = 0.6                                  # 融合权重：嵌入 60% + MFCC 40%
EMBED_AVAILABLE = False
_EMBED_MODEL = None
_EMBED_PROCESSOR = None
_EMBED_FAILED = False       # 哨兵：加载失败后不再重复尝试（避免逐模板重试卡顿）


def _load_embed_model():
    """懒加载 wav2vec2 模型（首次调用时）。失败返回 None，保持降级。"""
    global EMBED_AVAILABLE, _EMBED_MODEL, _EMBED_PROCESSOR, _EMBED_FAILED
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL, _EMBED_PROCESSOR
    if _EMBED_FAILED:
        return None, None
    try:
        import torch  # noqa: F401
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        _EMBED_PROCESSOR = Wav2Vec2Processor.from_pretrained(EMBED_MODEL_NAME)
        _EMBED_MODEL = Wav2Vec2Model.from_pretrained(EMBED_MODEL_NAME).eval()
        EMBED_AVAILABLE = True
        print("[VoiceMatch] 内容嵌入模型已加载（标准模式）")
    except Exception as e:
        EMBED_AVAILABLE = False
        _EMBED_FAILED = True
        print(f"[VoiceMatch] 内容嵌入不可用，回退 MFCC+DTW: {e}")
    return _EMBED_MODEL, _EMBED_PROCESSOR


def audio_embedding(samples: np.ndarray) -> Optional[np.ndarray]:
    """整段音频的均值池化内容嵌入向量（None = 不可用/失败，调用方降级）。"""
    try:
        model, proc = _load_embed_model()
        if model is None or proc is None:
            return None
        import torch
        inputs = proc(samples, sampling_rate=SAMPLE_RATE,
                      return_tensors="pt").input_values
        with torch.no_grad():
            hid = model(inputs).last_hidden_state  # (1, T, D)
        return hid.mean(dim=1).squeeze(0).numpy().astype(np.float64)
    except Exception:
        return None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """嵌入向量余弦距离（1 - cos，0~2 区间，与 DTW 距离同尺度可比）。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# 声纹嵌入（严格模式，可选双因子）：ECAPA-TDNN 说话人嵌入。
# 严格模式 = 内容嵌入（说了什么）**且** 声纹嵌入（是不是录这条声音的人）都命中
# 才触发——双因子 AND，比标准模式更防"他人模仿/环境声串播"。
# 同样可插拔降级：未装 speechbrain/torch 或加载失败 → 声纹因子自动关闭，
# 回到标准模式（内容嵌入 + MFCC）；依赖就绪后自动启用。
# ---------------------------------------------------------------------------
SPEAKER_MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"  # 说话人嵌入模型
STRICT_MODE = True           # 严格模式开关：True 且声纹可用时启用双因子
SPEAKER_AVAILABLE = False
# 声纹距离阈值（余弦距离 0~2）：同一人复述 ECAPA 声纹距离通常 <0.2，
# 不同人 >0.5——0.35 居中安全，比内容阈值 0.45 更严（说话人身份更难仿）
SPEAKER_THRESHOLD = 0.35
_SPEAKER_MODEL = None
# 缓存目录锚定脚本目录（相对 savedir 会随 CWD 漂移，导致重复下载/加载失败）
_SPEAKER_CACHE_DIR = str(Path(__file__).parent.resolve() / "speaker_model_cache")
_SPEAKER_FAILED = False      # 哨兵：加载失败后不再重复尝试（避免逐模板重试）


def _load_speaker_model():
    """懒加载 ECAPA-TDNN 声纹模型（首次调用时）。失败保持降级。"""
    global SPEAKER_AVAILABLE, _SPEAKER_MODEL, _SPEAKER_FAILED
    if _SPEAKER_MODEL is not None:
        return _SPEAKER_MODEL
    if _SPEAKER_FAILED:
        return None
    try:
        import speechbrain.inference.speaker  # noqa: F401
        _SPEAKER_MODEL = speechbrain.inference.speaker.EncoderClassifier.from_hparams(
            source=SPEAKER_MODEL_NAME, savedir=_SPEAKER_CACHE_DIR, run_opts={"device": "cpu"})
        SPEAKER_AVAILABLE = True
        print("[VoiceMatch] 声纹嵌入模型已加载（严格模式双因子）")
    except Exception as e:
        SPEAKER_AVAILABLE = False
        _SPEAKER_FAILED = True
        print(f"[VoiceMatch] 声纹嵌入不可用，保持标准模式: {e}")
    return _SPEAKER_MODEL


def speaker_embedding(samples: np.ndarray) -> Optional[np.ndarray]:
    """整段音频的说话人嵌入向量（None = 不可用/失败，调用方降级）。"""
    try:
        model = _load_speaker_model()
        if model is None:
            return None
        import torch
        wav = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(wav)  # (1, 1, D)
        return emb.squeeze().numpy().astype(np.float64)
    except Exception:
        return None


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
         win: int = 400, hop: int = 160, n_mels: int = 26,
         lifter_coef: float = 22.0) -> np.ndarray:
    """计算 MFCC 特征序列，shape = (帧数, n_mfcc)。含 CMVN 倒谱均值归一化。

    lifter_coef：倒谱提升系数（0 = 不提升）。浑厚/低频主导的嗓音（低音量大能量
    集中在低频）会让不同台词的 MFCC 低阶系数趋同、词间区分度被压小（实测异词
    危险对 16/28）；倒谱提升对低阶系数降权、突出高阶细节后，异词危险对降至 1/28
    （实测 voice_templates 8 模板，阈值 0.45）。对音量低 + 浑厚 + 日语的组合
    串播是针对性修复。
    """
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

    # 倒谱提升：压制低阶（浑厚低频主导）、突出高阶（词间区分细节）
    if lifter_coef and lifter_coef > 0.0:
        n = feat.shape[1]
        w = 1.0 + (lifter_coef / 2.0) * np.sin(np.pi * np.arange(n) / lifter_coef)
        feat = feat * w

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
        # name -> 内容嵌入向量（标准模式，可选；None/缺失 = 仅 MFCC 判定）
        self._template_embed: dict[str, Optional[np.ndarray]] = {}
        # name -> 声纹嵌入向量（严格模式双因子，可选；None/缺失 = 不启用声纹因子）
        self._template_speaker: dict[str, Optional[np.ndarray]] = {}
        # UI 状态（持久化到 voice_templates.json）
        self._template_favorite: dict[str, bool] = {}   # name -> 收藏
        self._template_order: dict[str, int] = {}       # name -> 排序序号
        self._template_start_pct: dict[str, float] = {} # name -> 播放起始段落百分比 0.0~1.0
        self.available: bool = SD_AVAILABLE

        # 音频流与线程
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._templates_lock = threading.Lock()   # 保护 triggers/_template_wav/_template_feat 跨线程访问
        self._ring: deque = deque(maxlen=RING_BLOCKS)
        self._ring_count = 0        # 累计入环块数（单调递增，用于定位）
        self._processed = 0         # VAD 已处理到的块序号

        # 录制模式（用有上限的 deque，防止用户忘记结束录制时内存无限增长）
        self._capture: Optional[deque] = None

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
            # 恢复 UI 状态（收藏 / 排序 / 起始段落）
            self._template_favorite[name] = bool(item.get("favorite", False))
            self._template_order[name] = int(item.get("order", 9999))
            start_pct = item.get("start_pct", 0.0)
            self._template_start_pct[name] = float(start_pct) if start_pct else 0.0
            try:
                samples = self._load_samples(str(wav_path))
                self._template_feat[name] = mfcc(samples)
                self._template_embed[name] = audio_embedding(samples)
                self._template_speaker[name] = speaker_embedding(samples)
            except Exception as e:
                print(f"[VoiceMatch] 模板特征提取失败 [{name}]: {e}")

        # 目录扫描补登：json 缺失但 wav 仍存在的旧模板（如 json 曾被覆盖/重建）
        # 自动补登并保留为可触发模板——否则"之前录制过的台词"因无 json 条目
        # 不被加载，表现为无法触发且无语音日志。补登仅登记模板与特征，
        # 不覆盖已存在的同名条目；歌曲绑定沿用默认 voice_play（可后续在 UI 重绑）。
        if self._templates_dir and self._templates_dir.is_dir():
            for wav_file in sorted(self._templates_dir.glob("*.wav")):
                nm = wav_file.stem
                if nm in self.triggers:
                    continue
                try:
                    samples = self._load_samples(str(wav_file))
                    self.triggers[nm] = "voice_play"
                    self._template_wav[nm] = str(wav_file)
                    self._template_feat[nm] = mfcc(samples)
                    self._template_embed[nm] = audio_embedding(samples)
                    self._template_speaker[nm] = speaker_embedding(samples)
                    print(f"[VoiceMatch] 补登未登记的旧模板: {nm}")
                except Exception as e:
                    print(f"[VoiceMatch] 补登模板失败 [{nm}]: {e}")

        print(f"[VoiceMatch] 已加载 {len(self.triggers)} 条语音模板")

    def _save_storage(self) -> None:
        data = {"templates": []}
        script_dir = Path(__file__).parent.resolve()
        for name, action in self.triggers.items():
            wav = self._template_wav.get(name, "")
            # 保存为相对脚本目录的路径（跨机器/换目录不失效；加载端负责解析）。
            # 自定义 templates_dir 在脚本目录外时回退绝对路径，避免保存失败。
            wav_rel = wav
            if wav:
                try:
                    wav_rel = str(Path(wav).resolve().relative_to(script_dir))
                except (ValueError, OSError):
                    pass
            item = {"name": name, "action": action, "wav": wav_rel}
            tracks = self.trigger_tracks.get(name)
            if tracks:
                item["tracks"] = tracks
            # 持久化 UI 状态（收藏 / 排序 / 起始段落）
            if self._template_favorite.get(name):
                item["favorite"] = True
            if self._template_order.get(name, 9999) != 9999:
                item["order"] = self._template_order[name]
            start_pct = self._template_start_pct.get(name, 0.0)
            if start_pct:
                item["start_pct"] = round(float(start_pct), 4)
            data["templates"].append(item)
        try:
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[VoiceMatch] 保存失败: {e}")

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------
    def _load_samples(self, wav_path: str) -> np.ndarray:
        """读取 wav → int16 → float32 → 静音裁剪 + RMS 归一化（特征/嵌入共用）。"""
        with wave.open(wav_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return rms_normalize(trim_silence(samples))

    def _extract_feature(self, wav_path: str) -> np.ndarray:
        return mfcc(self._load_samples(wav_path))

    # ------------------------------------------------------------------
    # 映射管理（接口与 VoiceTrigger 对齐）
    # ------------------------------------------------------------------
    def add_trigger(self, phrase: str, action: str, tracks: Optional[list] = None,
                    start_pct: float = 0.0) -> None:
        """为已录制的模板绑定动作/歌曲。phrase 即模板名。"""
        phrase = phrase.strip()
        if not phrase or phrase not in self._template_wav:
            print(f"[VoiceMatch] 模板不存在，无法绑定: {phrase}")
            return
        with self._templates_lock:
            self.triggers[phrase] = action.strip() or "voice_play"
            if tracks:
                self.trigger_tracks[phrase] = [str(t) for t in tracks]
            self._template_start_pct[phrase] = max(0.0, min(1.0, float(start_pct or 0.0)))
            self._save_storage()
        print(f"[VoiceMatch] 已绑定: \"{phrase}\" ({len(self.trigger_tracks.get(phrase, []))} 首歌曲)")

    def remove_trigger(self, phrase: str) -> None:
        """删除模板（含 wav 文件与歌曲绑定）。"""
        phrase = phrase.strip()
        if phrase not in self.triggers and phrase not in self._template_wav:
            return
        with self._templates_lock:
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

    def rename_trigger(self, old_name: str, new_name: str) -> bool:
        """重命名模板（含 wav 文件、特征缓存与歌曲绑定）。新名称已存在时返回 False。"""
        old_name = old_name.strip()
        new_name = new_name.strip()
        if not old_name or not new_name or old_name == new_name:
            return True
        if new_name in self._template_wav:
            return False
        with self._templates_lock:
            # 迁移所有关联数据
            self._template_wav[new_name] = self._template_wav.pop(old_name)
            self._template_feat[new_name] = self._template_feat.pop(old_name, None)
            if old_name in self.triggers:
                self.triggers[new_name] = self.triggers.pop(old_name)
            if old_name in self.trigger_tracks:
                self.trigger_tracks[new_name] = self.trigger_tracks.pop(old_name)
            # 重命名 wav 文件
            wav_path = Path(self._template_wav[new_name])
            if wav_path.exists() and wav_path.suffix.lower() == ".wav":
                new_wav = wav_path.with_name(f"{new_name}.wav")
                try:
                    wav_path.rename(new_wav)
                    self._template_wav[new_name] = str(new_wav)
                except OSError as e:
                    print(f"[VoiceMatch] wav 文件重命名失败: {e}")
            self._save_storage()
        print(f'[VoiceMatch] 已重命名模板: "{old_name}" → "{new_name}"')
        return True

    def list_templates(self) -> list:
        """返回模板信息列表 [{name, track_count}]，供 UI 展示。"""
        return [
            {"name": n, "track_count": len(self.trigger_tracks.get(n, [])),
             "favorite": self._template_favorite.get(n, False),
             "order": self._template_order.get(n, 9999)}
            for n in self._template_wav
        ]

    def toggle_favorite(self, name: str) -> Optional[bool]:
        """切换模板收藏状态并持久化。返回切换后的状态；模板不存在返回 None。"""
        with self._templates_lock:
            if name not in self._template_wav:
                return None
            new_state = not self._template_favorite.get(name, False)
            self._template_favorite[name] = new_state
            self._save_storage()
            return new_state

    def get_start_pct(self, name: str) -> float:
        """获取模板的播放起始段落百分比（0.0~1.0），未设置返回 0.0。"""
        return float(self._template_start_pct.get(name, 0.0) or 0.0)

    def set_start_pct(self, name: str, start_pct: float) -> bool:
        """设置模板的播放起始段落百分比并持久化。模板不存在返回 False。"""
        with self._templates_lock:
            if name not in self._template_wav:
                return False
            self._template_start_pct[name] = max(0.0, min(1.0, float(start_pct or 0.0)))
            self._save_storage()
            return True

    def move_template(self, name: str, direction: int) -> bool:
        """在显示顺序中移动模板（direction: -1 前移, +1 后移）并持久化。
        返回 True 成功；False 已到边界或模板不存在。"""
        with self._templates_lock:
            display = self.list_templates()
            idx = next((i for i, t in enumerate(display) if t["name"] == name), None)
            if idx is None:
                return False
            j = idx + direction
            if j < 0 or j >= len(display):
                return False
            # 交换 order
            other = display[j]
            self._template_order[name] = other.get("order", 9999)
            self._template_order[other["name"]] = display[idx].get("order", 9999)
            self._save_storage()
            return True

    # ------------------------------------------------------------------
    # 录制模式（T 键流程）
    # ------------------------------------------------------------------
    def start_capture(self) -> None:
        """开始采集台词音频。若监听未运行会自动启动音频流。"""
        if not self._running:
            self.start_listening()
        with self._lock:
            self._capture = deque(maxlen=int(MAX_CAPTURE_SEC * SAMPLE_RATE / BLOCK_SIZE))
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

        with self._templates_lock:
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

        with self._templates_lock:
            self._template_wav[name] = str(wav_path)
            self._template_feat[name] = mfcc(samples)
            self._template_embed[name] = audio_embedding(samples)
            self._template_speaker[name] = speaker_embedding(samples)
        dur = len(samples) / SAMPLE_RATE
        mode = "严格(嵌入+声纹)" if self._template_speaker.get(name) is not None \
            else ("嵌入+MFCC" if self._template_embed.get(name) is not None else "MFCC")
        print(f"[VoiceMatch] 模板已保存: {name}（{dur:.1f}s，{mode}）")
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
        # 复位 VAD 状态与噪声底：下次监听重新自适应，避免沿用旧的阈值基线
        with self._lock:
            self._vad_in_speech = False
            self._vad_speech_start = 0
            self._vad_last_voice = 0
            self._noise_floor = 1e-4
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
        # 内容嵌入（标准模式可选）：嵌入可用时与 MFCC 融合评分，
        # 不可用（未装 torch / 模型缺失 / 模板无嵌入）则回退纯 MFCC 判定。
        cur_embed = audio_embedding(seg) if EMBED_AVAILABLE else None
        # 声纹嵌入（严格模式双因子，可选）：声纹可用时对最佳候选追加声纹距离确认，
        # 不可用（未装 speechbrain / 模型缺失）则保持标准模式。
        cur_speaker = speaker_embedding(seg) if SPEAKER_AVAILABLE else None
        # 在锁内快照模板字典，避免与主线程 add_trigger/remove_trigger/stop_capture 的写入竞态
        with self._templates_lock:
            templates = list(self._template_feat.items())
            triggers = self.triggers.copy()
        best_name, best_dist = None, float("inf")
        best_speaker_dist = None
        distances = []
        for name, tfeat in templates:
            # 时长一致性过滤：帧数差异过大的模板直接跳过（防"串台"）。
            # 下限 0.5 曾导致"复述稍快/稍短"（时长比 0.45~0.46）被直接跳过、
            # 即使声音内容匹配也不触发——放宽到 0.4：容忍更快的复述语速，
            # 同时仍排除明显过短的杂音（0.24~0.28 类短段依旧被拦）。
            ratio = len(feat) / max(1, len(tfeat))
            if not (0.4 <= ratio <= 2.0):
                print(f"[VoiceMatch] 跳过模板 [{name}]（时长比 {ratio:.2f} 超出 0.4~2.0）")
                continue
            d_mfcc = dtw_distance(feat, tfeat)
            d = d_mfcc
            t_embed = self._template_embed.get(name)
            if cur_embed is not None and t_embed is not None:
                d_embed = _cosine_distance(cur_embed, t_embed)
                # 融合：嵌入为主（EMBED_W），MFCC 兜底（1-EMBED_W）
                d = EMBED_W * d_embed + (1.0 - EMBED_W) * d_mfcc
            distances.append((name, d))
            if d < best_dist:
                best_name, best_dist = name, d
                # 记录最佳候选的声纹距离（严格模式双因子用）
                t_spk = self._template_speaker.get(name)
                if cur_speaker is not None and t_spk is not None:
                    best_speaker_dist = _cosine_distance(cur_speaker, t_spk)
                else:
                    best_speaker_dist = None

        if distances:
            dbg = ", ".join(f"{n}={d:.2f}" for n, d in sorted(distances, key=lambda x: x[1]))
            print(f"[VoiceMatch] 比对: {dbg}（阈值 {MATCH_THRESHOLD}）")

        if best_name is None or best_dist > MATCH_THRESHOLD:
            return

        # 严格模式双因子：声纹可用（cur_speaker 与模板声纹均存在）时，
        # 要求说话人身份距离也达标（SPEAKER_THRESHOLD）才触发——
        # 防"他人模仿台词 / 环境声恰好相似"串播。声纹不可用则自动降级，
        # 保持标准模式判定（不阻断正常触发）。
        if (STRICT_MODE and best_speaker_dist is not None
                and best_speaker_dist > SPEAKER_THRESHOLD):
            print(f"[VoiceMatch] 声纹因子未通过（{best_speaker_dist:.2f} > {SPEAKER_THRESHOLD}），忽略")
            return

        # 多模板区分度门槛（防"最佳与次佳都像"时随便触发/串播）。
        # 仅当次佳在"临界区"（< MATCH_THRESHOLD + 0.10）时才启用——次佳远超
        # 阈值说明是唯一候选，不应因存在一个"最像"的其他模板而拒绝。
        # 双重差距：相对（最佳 ≤ 次佳×MATCH_MARGIN）且绝对（最佳+ABS_MARGIN ≤ 次佳），
        # 任一不满足即视为区分度不足。
        # 关键修正：最佳距离足够低（< CONFIDENT_DIST，同词复述强命中）时**直接放行**，
        # 不启用区分度门槛——日志实证"区分度不足 0.30 vs 0.30"把真实复述误拦。
        # CONFIDENT_DIST 由 0.35 → 0.55：真实独立复述（变速/音调差异）距离实测
        # 0.43~0.60（8 条模板变速±12%+噪声模拟），0.35 以下几乎无真实命中，
        # 反而让所有真实复述都落入区分度门槛、被"最佳≈次佳"误拦（用户反馈
        # "录制过的台词触发不了"）。0.55 覆盖真实复述主区间；仍低于阈值 0.60，
        # 0.55~0.60 的边缘命中继续走区分度门槛防串播。
        CONFIDENT_DIST = 0.55
        if best_dist >= CONFIDENT_DIST and len(distances) >= 2:
            sorted_d = sorted(d for _, d in distances)
            if (sorted_d[1] < MATCH_THRESHOLD + 0.10
                    and (best_dist > sorted_d[1] * MATCH_MARGIN
                         or best_dist + ABS_MARGIN > sorted_d[1])):
                print(f"[VoiceMatch] 区分度不足，忽略（最佳 {best_dist:.2f} vs 次佳 {sorted_d[1]:.2f}）")
                return

        # 冷却期检查
        now = time.time()
        if now - self._last_trigger.get(best_name, 0.0) < COOLDOWN_SEC:
            return
        self._last_trigger[best_name] = now

        action = triggers.get(best_name, "voice_play")
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
        # 实际判定模式：声纹严格双因子 > 内容嵌入融合 > 纯 MFCC（依赖实际可用性）
        with self._templates_lock:
            has_speaker = any(v is not None for v in self._template_speaker.values())
            has_embed = any(v is not None for v in self._template_embed.values())
        if SPEAKER_AVAILABLE and has_speaker:
            s["mode"] = "严格(嵌入+声纹)"
        elif EMBED_AVAILABLE and has_embed:
            s["mode"] = "嵌入+MFCC"
        else:
            s["mode"] = "MFCC"
        return s
