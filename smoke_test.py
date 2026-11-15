#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoulPlayer 冒烟验证脚本（改动代码后必跑）

两级防线：
  L1 静态检查（秒级）  ：语法编译 + 模块导入，能挡住 80% 的低级错误
  L2 运行时冒烟（~25s）：真实启动 → 窗口出现 → 帧计数持续增长 → 无 Traceback

用法：
    py smoke_test.py            # 跑全部（默认）
    py smoke_test.py --static   # 只跑 L1，秒级反馈
    py smoke_test.py --exe      # 验证 dist/SoulPlayer/SoulPlayer.exe 而非源码

退出码：0 = 全部通过；1 = 存在失败项
"""
import ctypes
import ctypes.wintypes as wt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT, "_smoke_last.log")
SHOT_PATH = os.path.join(ROOT, "smoke_shot.png")
ERROR_LOG = os.path.join(ROOT, "error.log")

# 应用自身的日志位置（SOULPLAYER_DATA_DIR 沙箱内），窗口模式 builds 的
# print 不再进 stdout，Traceback 检测靠它
_APP_DATA_DIR = None


def _app_log_path():
    if not _APP_DATA_DIR:
        return None
    return os.path.join(_APP_DATA_DIR, "logs", "soulplayer.log")

# 判定阈值（可按机器性能调整）
BOOT_TIMEOUT_SEC = 30      # 窗口必须在 30s 内出现
FRAME_GROWTH_MIN = 30      # 两次采样间帧计数至少增长这么多
INIT_WARN_SEC = 12         # 初始化超过该值则警告（黑屏回归的信号）
WAIT_AFTER_BOOT = 8        # 窗口出现后再观察多久

MAIN_MODULES = [
    "app_paths", "app_logging", "main", "motion_capture", "gesture_presets",
    "music_player", "custom_preset", "voice_match", "ui_theme",
    "ui_animations", "wave_learning", "wave_learning_rules",
]

results = []


def record(level, name, ok, detail=""):
    results.append((level, name, ok, detail))
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ───────────────────────── L1 静态检查 ─────────────────────────

def l1_static():
    print("\n── L1 静态检查 ──")
    targets = [m + ".py" for m in MAIN_MODULES] + ["build_exe.py", "run.py"]
    missing = [f for f in targets if not os.path.exists(os.path.join(ROOT, f))]
    record("L1", "源码文件齐全", not missing, f"缺失: {missing}" if missing else f"{len(targets)} 个")

    compile_err = []
    for f in [t for t in targets if os.path.exists(os.path.join(ROOT, t))]:
        r = subprocess.run([sys.executable, "-m", "py_compile", f],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            compile_err.append(f"{f}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?'}")

    # 在 dummy 驱动下导入，避免弹出窗口
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    import_err = []
    r = subprocess.run(
        [sys.executable, "-c",
         "import " + ", ".join(MAIN_MODULES) + "; print('OK')"],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=180)
    if r.returncode != 0:
        import_err = (r.stderr.strip().splitlines() or ["?"])[-3:]

    return (record("L1", "语法编译", not compile_err, "; ".join(compile_err))
            and record("L1", "模块导入", not import_err, " ".join(import_err)))


# ───────────────────────── L2 运行时冒烟 ─────────────────────────

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class BMIH(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


class BMI(ctypes.Structure):
    _fields_ = [("bmiHeader", BMIH), ("bmiColors", wt.DWORD * 3)]


ENUM_CB = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def find_window(acc):
    def cb(h, _):
        if user32.IsWindowVisible(h):
            n = user32.GetWindowTextLengthW(h)
            if n > 0:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(h, buf, n + 1)
                if buf.value.startswith("SoulPlayer"):
                    acc.append((h, buf.value))
        return True
    user32.EnumWindows(ENUM_CB(cb), 0)
    return acc


def grab(hwnd, path):
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    if w <= 0 or h <= 0:
        return None
    hdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    user32.PrintWindow(hwnd, mem, 2)
    bmi = BMI()
    bmi.bmiHeader.biSize = ctypes.sizeof(BMIH)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    try:
        from PIL import Image
        img = Image.frombuffer("RGBA", (w, h), buf.raw, "raw", "BGRA", 0, 1).convert("RGB")
        img.save(path)
        px = list(img.getdata())[::97]
        black = sum(1 for p in px if p == (0, 0, 0)) / len(px) * 100
        return {"size": (w, h), "colors": len(set(px)), "black_pct": black}
    except ImportError:
        return {"size": (w, h), "colors": -1, "black_pct": -1}
    finally:
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, hdc)


def l2_runtime(target_exe=False):
    print("\n── L2 运行时冒烟 ──")
    cmd = ([os.path.join(ROOT, "dist", "SoulPlayer", "SoulPlayer.exe")]
           if target_exe else [sys.executable, "main.py"])

    # 沙箱数据目录：隔离测试写入，且能确定性地读到应用日志。
    # 允许多实例：自动化测试可能与手工开的程序并存，互斥体会把测试进程挡掉。
    global _APP_DATA_DIR
    _APP_DATA_DIR = tempfile.mkdtemp(prefix="sp_smoke_")
    env = dict(os.environ, SOULPLAYER_ALLOW_MULTI="1",
               SOULPLAYER_DATA_DIR=_APP_DATA_DIR)

    with open(LOG_PATH, "w", encoding="utf-8", errors="replace") as logf:
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf,
                                stderr=subprocess.STDOUT, env=env)

        boot_at, title, hwnd = None, None, None
        frames = []
        while time.time() - t0 < BOOT_TIMEOUT_SEC:
            time.sleep(0.5)
            found = find_window([])
            if found:
                hwnd, title = found[0]
                m = re.search(r"帧:(\d+)", title)
                if m:
                    frames.append(int(m.group(1)))
                    if boot_at is None:
                        boot_at = time.time() - t0
                        break
            if proc.poll() is not None:
                break

        if boot_at is None:
            proc.kill()
            tail = _tail(LOG_PATH, 25)
            return (record("L2", "窗口启动", False,
                           f"{BOOT_TIMEOUT_SEC}s 内未出现主循环窗口")
                    and record("L2", "崩溃取证", False, "\n" + tail))

        # 窗口已出现，继续观察帧增长
        time.sleep(WAIT_AFTER_BOOT)
        found = find_window([])
        if found:
            hwnd, title = found[0]
            m = re.search(r"帧:(\d+)", title)
            if m:
                frames.append(int(m.group(1)))
        # 必须在 kill 之前截图：进程一旦被杀，窗口销毁中截图会得到纯黑
        shot = grab(hwnd, SHOT_PATH) if hwnd else None
        proc.kill()

    boot_ok = boot_at < INIT_WARN_SEC
    growth = frames[-1] - frames[0] if len(frames) >= 2 else 0
    grow_ok = growth >= FRAME_GROWTH_MIN

    shot_ok = bool(shot and shot["colors"] > 50 and shot["black_pct"] < 50)
    shot_detail = ""
    if shot:
        shot_detail = (f"{shot['size'][0]}x{shot['size'][1]}, "
                       f"{shot['colors']} 色, 纯黑 {shot['black_pct']:.1f}%")
    elif hwnd is None:
        shot_detail = "窗口已关闭"

    trace = _has_traceback_any([LOG_PATH, _app_log_path(), ERROR_LOG])
    if _APP_DATA_DIR:
        shutil.rmtree(_APP_DATA_DIR, ignore_errors=True)
    return all([
        record("L2", f"窗口启动({boot_at:.1f}s)", boot_ok,
               "" if boot_ok else f"超过 {INIT_WARN_SEC}s，疑似初始化阻塞（黑屏回归）"),
        record("L2", f"主循环运行(+{growth} 帧)", grow_ok,
               f"{WAIT_AFTER_BOOT}s 内应增长 ≥{FRAME_GROWTH_MIN} 帧"),
        record("L2", "画面渲染", shot_ok, shot_detail),
        record("L2", "无未捕获异常", not trace, trace or "日志干净"),
    ])


def _has_traceback_any(paths):
    for p in paths:
        if not p:
            continue
        hit = _has_traceback(p)
        if hit:
            return hit
    return ""


def _has_traceback(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return ""
    if "Traceback (most recent call last)" not in text:
        return ""
    idx = text.find("Traceback (most recent call last)")
    return "崩溃现场：" + text[idx:idx + 400].strip().replace("\n", "\n      ")


def _tail(path, n):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "      最近日志：\n      " + "      ".join(lines[-n:]).strip()
    except OSError:
        return ""


# ───────────────────────── 主流程 ─────────────────────────

def main():
    args = sys.argv[1:]
    print("=" * 60)
    print("  SoulPlayer 冒烟验证")
    print("=" * 60)

    ok = True
    if "--exe" not in args:
        ok &= l1_static()
    if "--static" not in args:
        ok &= l2_runtime(target_exe=("--exe" in args))

    print("\n" + "=" * 60)
    if ok:
        print("  全部通过 ✓  可以提交")
    else:
        failed = [f"{lv}/{n}" for lv, n, o, _ in results if not o]
        print(f"  存在失败项 ✗  {', '.join(failed)}")
        print(f"  回滚命令：git checkout -- .   （或 git diff 逐项检查后再定）")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
