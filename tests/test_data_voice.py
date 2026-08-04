# -*- coding: utf-8 -*-
"""针对 2026-08-04 四项改动的单元测试：
A. _data_dir 统一定位 + 旧数据自动迁移
B. 语音匹配时长一致性过滤（防串台）
C. 说话时自动压低音乐音量（人声/扬声器分离）
D. 录音时自动暂停音乐并恢复
"""
import io
import queue
import sys
import tempfile
import time
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ==================== A. _data_dir ====================
def test_data_dir():
    import main as m

    tmp = Path(tempfile.mkdtemp())
    proj = tmp / "proj"
    internal = proj / "dist" / "SoulPlayer" / "_internal"
    internal.mkdir(parents=True)
    exe = proj / "dist" / "SoulPlayer" / "SoulPlayer.exe"
    exe.write_text("fake")

    old_frozen = getattr(sys, "frozen", None)
    old_exe = sys.executable
    old_file = m.__file__
    sys.frozen = True
    sys.executable = str(exe)
    m.__file__ = str(internal / "main.py")
    fn = m.SoulPlayerApp._data_dir
    try:
        # A1: 数据只在 _internal → 迁移到项目源码目录（含同级 sidecar json）
        d = internal / "custom_presets"
        d.mkdir()
        (d / "preset_a.json").write_text("{}", encoding="utf-8")
        (internal / "custom_presets.json").write_text("[]", encoding="utf-8")
        r = fn(None, "custom_presets")
        check("A1 迁移到首选目录", r == proj / "custom_presets", str(r))
        check("A1 内容已复制", (proj / "custom_presets" / "preset_a.json").exists())
        check("A1 sidecar 已复制", (proj / "custom_presets.json").exists())

        # A2: 首选目录已有数据 → 直接用首选，不覆盖
        (proj / "custom_presets" / "preset_b.json").write_text("b", encoding="utf-8")
        (d / "preset_c.json").write_text("c", encoding="utf-8")
        r = fn(None, "custom_presets")
        check("A2 返回首选目录", r == proj / "custom_presets")
        check("A2 原有内容未动", (proj / "custom_presets" / "preset_b.json").read_text() == "b")

        # A3: 全新目录 → 创建首选空目录
        r = fn(None, "voice_templates")
        check("A3 创建首选目录", r == proj / "voice_templates" and r.is_dir(), str(r))
    finally:
        if old_frozen is None:
            del sys.frozen
        else:
            sys.frozen = old_frozen
        sys.executable = old_exe
        m.__file__ = old_file


# ==================== B. 时长一致性过滤 ====================
def _speech(sig_dur, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sig_dur * sr)) / sr
    sig = (0.3 * np.sin(2 * np.pi * 220 * t)
           + 0.2 * np.sin(2 * np.pi * 440 * t * (1 + 0.1 * np.sin(2 * np.pi * 3 * t)))
           + 0.05 * rng.standard_normal(len(t)))
    return sig.astype(np.float32)


def _feed_segment(vm, seg, sr=16000, block=1024):
    """把 静音+语音段 推入环形缓冲并调用 _finalize_segment。"""
    silence = np.zeros(block, dtype=np.float32)
    n_pre = 6
    for _ in range(n_pre):
        vm._ring.append(silence)
        vm._ring_count += 1
    vm._vad_speech_start = vm._ring_count
    for i in range(0, len(seg), block):
        b = seg[i:i + block]
        if len(b) < block:
            b = np.pad(b, (0, block - len(b)))
        vm._ring.append(b.astype(np.float32))
        vm._ring_count += 1
    vm._finalize_segment(vm._ring_count - 1)


def test_duration_filter():
    from voice_match import VoiceMatcher, mfcc
    tmp = Path(tempfile.mkdtemp())
    vm = VoiceMatcher(templates_dir=str(tmp / "vt"), storage_path=str(tmp / "vt.json"))
    vm._last_trigger.clear()

    short_sig = _speech(0.8, seed=1)
    long_sig = _speech(3.5, seed=2)
    vm._template_feat["台词短"] = mfcc(short_sig)
    vm._template_feat["台词长"] = mfcc(long_sig)
    vm.triggers = {"台词短": "voice_play:台词短", "台词长": "voice_play:台词长"}

    # B1: 说 0.8s → 应命中"台词短"（"台词长"被时长过滤跳过）
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_segment(vm, short_sig)
    log = buf.getvalue()
    got = vm._queue.get_nowait() if not vm._queue.empty() else None
    check("B1 短句命中短模板", got == "voice_play:台词短", f"got={got}")
    check("B1 长模板被过滤日志", "跳过模板 [台词长]" in log, log.strip().splitlines()[-1:] )

    # B2: 说 3.5s → 应命中"台词长"
    vm._last_trigger.clear()
    vm._ring.clear(); vm._ring_count = 0
    _feed_segment(vm, long_sig)
    got = vm._queue.get_nowait() if not vm._queue.empty() else None
    check("B2 长句命中长模板", got == "voice_play:台词长", f"got={got}")


# ==================== C. 说话压低音量 ====================
def test_ducking():
    import main as m

    class P:
        _volume = 0.5
        is_playing = True
        def __init__(self): self.vol_history = []
        def set_volume(self, v): self._volume = v; self.vol_history.append(v)
    class VT:
        is_listening = True
        speech_active = True

    app = types.SimpleNamespace(player=P(), voice_trigger=VT(),
                                _voice_ducked=None, _duck_until=0.0)
    fn = m.SoulPlayerApp._update_voice_ducking

    fn(app)
    check("C1 说话时压低音量", abs(app.player._volume - max(0.12, 0.5 * 0.45)) < 1e-6,
          f"vol={app.player._volume}")
    check("C1 原音量已记录", app._voice_ducked == 0.5)

    # 说话停止，保持期内不恢复
    app.voice_trigger.speech_active = False
    fn(app)
    check("C2 保持期内不恢复", app.player._volume < 0.3)
    # 超过保持期恢复
    app._duck_until = time.time() - 0.1
    fn(app)
    check("C3 说完后恢复原音量", abs(app.player._volume - 0.5) < 1e-6 and app._voice_ducked is None)

    # 音乐未播放时不压低
    app.voice_trigger.speech_active = True
    app.player.is_playing = False
    fn(app)
    check("C4 未播放不压低", app._voice_ducked is None)


# ==================== D. 录音暂停/恢复 ====================
def test_rec_pause():
    import main as m

    calls = []
    class P:
        is_playing = True
        def pause(self): calls.append("pause"); self.is_playing = False
        def play(self): calls.append("play"); self.is_playing = True
    class VT:
        available = True
        def get_status(self): return {"listening": True}
        def start_capture(self): pass
        def stop_capture(self): return ""   # 无效录音

    app = types.SimpleNamespace(
        player=P(), voice_trigger=VT(), _voice_rec=None,
        _last_t_handle_time=0.0,
        _add_log=lambda s: None,
    )
    app._restore_music_after_voice_rec = (
        lambda vr: m.SoulPlayerApp._restore_music_after_voice_rec(app, vr))
    app._finish_voice_recording = (
        lambda: m.SoulPlayerApp._finish_voice_recording(app))
    m.SoulPlayerApp._handle_voice_record_key(app)
    check("D1 倒计时开始即暂停", calls == ["pause"] and app._voice_rec["resume_play"] is True)

    # 倒计时期间再按 T = 取消 → 恢复播放
    app._last_t_handle_time = 0.0
    m.SoulPlayerApp._handle_voice_record_key(app)
    check("D2 取消录制恢复播放", calls == ["pause", "play"] and app._voice_rec is None)

    # 录制结束（无效录音）也恢复播放
    calls.clear()
    app._last_t_handle_time = 0.0
    m.SoulPlayerApp._handle_voice_record_key(app)          # 开始
    app._voice_rec["phase"] = "recording"
    app._last_t_handle_time = 0.0
    m.SoulPlayerApp._handle_voice_record_key(app)          # 结束（stop_capture 返回 ""）
    check("D3 结束录制恢复播放", calls == ["pause", "play"])


if __name__ == "__main__":
    test_data_dir()
    test_duration_filter()
    test_ducking()
    test_rec_pause()
    print(f"\n===== {len(PASS)} 通过, {len(FAIL)} 失败 =====")
    sys.exit(1 if FAIL else 0)
