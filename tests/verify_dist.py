# -*- coding: utf-8 -*-
"""
校验 dist/SoulPlayer/SoulPlayer.exe 的内容是否与预期一致。

用法（项目根目录）：
    py tests/verify_dist.py              # 代码一致性 + 资源 + 退出测试
    py tests/verify_dist.py --no-exit    # 跳过启动/退出测试（快速模式）

检查三层：
    1. PYZ 归档：项目模块是否都在包内、是否有重依赖混入
    2. 字节码特征串：关键修复是否真的编译进了 exe（不是只看源码）
    3. 资源文件：预设 JSON 的门槛值是否随包更新
    4. 运行时：能否启动窗口、WM_CLOSE 后能否自行退出

为什么不能只跑 smoke_test：
    smoke_test 只能证明"程序能起来、画面不是黑的"，证明不了"这次改的逻辑
    进了包"。特征串扫描正是补这一层——它从 exe 里解出字节码常量来比对。
"""

import ctypes
import ctypes.wintypes as wt
import json
import marshal
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
EXE = ROOT / "dist" / "SoulPlayer" / "SoulPlayer.exe"
INTERNAL = ROOT / "dist" / "SoulPlayer" / "_internal"

NO_EXIT_TEST = "--no-exit" in sys.argv or "--no-exit-test" in sys.argv

# 项目自身模块必须全部在包内（缺一个就是 hidden-import 漏配）
REQUIRED_MODULES = [
    "main", "motion_capture", "gesture_presets", "custom_preset",
    "music_player", "voice_match", "wave_learning", "wave_learning_rules",
    "ui_theme", "ui_animations", "app_paths", "app_logging",
]

# (模块, 期望出现的标识符, 说明) —— 每一项对应一次真实修复
CODE_CHECKS = [
    ("music_player", "SOULPLAYER_AUTO_ADVANCE", "自动切歌开关"),
    ("music_player", "MUSIC_END_EVENT", "播完事件"),
    ("voice_match", "SOULPLAYER_VOICE_SPEAKER_ONLINE", "声纹在线开关"),
    ("voice_match", "_SPEAKER_ONLINE_BUDGET_SEC", "声纹加载时间预算"),
    ("motion_capture", "_close_detector", "detector 超时关闭"),
    ("custom_preset", "已跳过", "坏预设跳过"),
    ("gesture_presets", "MISS_FRAMES_TO_UNLOCK", "解锁抖动容忍（连续 N 帧）"),
    ("main", "ACTION_COOLDOWN_RULES", "声明式动作冷却表"),
    ("main", "SOULPLAYER_WAVE_DIAG", "挥手灵敏度诊断开关"),
    ("main", "_acquire_single_instance", "单实例互斥"),
    ("app_paths", "LOCALAPPDATA", "数据根移出安装目录"),
    ("app_logging", "RotatingFileHandler", "日志轮转落盘"),
]

# (资源文件相对路径, 取值函数, 期望值, 说明)
def _wave_min(preset):
    """取'挥手下一首'预设的 arm_horizontal_motion 下限。"""
    for p in preset.get("presets", []):
        cond = p.get("conditions") or {}
        if p.get("action") == "player_next":
            return (cond.get("arm_horizontal_motion") or {}).get("min")
    return None


RESOURCE_CHECKS = [
    ("presets/default_presets.json", _wave_min, 0.19, "挥手跨度门槛"),
]


def fail(msg):
    print(f"  [FAIL] {msg}")
    return False


def ok(msg):
    print(f"  [PASS] {msg}")
    return True


def load_pyz(exe_path):
    """把 exe 内的 PYZ 归档解到临时文件，返回 (pyz_reader, tmp_path, carchive)。"""
    from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader

    ca = CArchiveReader(str(exe_path))
    data = ca.extract("PYZ.pyz")
    if isinstance(data, tuple):
        data = data[-1]
    tmp = os.path.join(tempfile.gettempdir(), "_sp_verify_pyz.pyz")
    with open(tmp, "wb") as f:
        f.write(data)
    return ZlibArchiveReader(tmp), tmp, ca


def strings_of(reader, name, all_names=None, carchive=None):
    """收集模块字节码中的字符串常量与全局名（co_names 含变量/函数名）。"""
    co = None
    if name == "main":
        # 入口脚本不进 PYZ：旧布局在 PYZ 叫 __main__，新布局（入口=main.py）
        # 在 CArchive 里叫 main（类型 's'）。两处都试，兼容两种打包形态。
        if carchive is not None:
            try:
                raw = carchive.extract("main")
                if isinstance(raw, tuple):
                    raw = raw[-1]
                if isinstance(raw, bytes):
                    import marshal
                    co = marshal.loads(raw)
                else:
                    co = raw
            except Exception:
                co = None
        if co is None and "__main__" in (all_names or set()):
            try:
                co = reader.extract("__main__")
            except Exception:
                return ""
    else:
        try:
            co = reader.extract(name)
        except Exception:
            return ""
    if co is None:
        return ""
    if isinstance(co, bytes):
        try:
            co = marshal.loads(co)
        except Exception:
            return ""
    out = []

    def walk(c):
        out.extend(k for k in getattr(c, "co_consts", ()) if isinstance(k, str))
        out.extend(getattr(c, "co_names", ()))
        for k in getattr(c, "co_consts", ()):
            if hasattr(k, "co_consts"):
                walk(k)

    walk(co)
    return "\n".join(out)


def check_archive(reader, names, carchive=None):
    print("── 1. 归档完整性 ──")
    okk = True
    # main 是入口脚本，不进 PYZ：旧布局在 PYZ 叫 __main__，
    # 新布局（入口=main.py）在 CArchive 叫 main（类型 's'）
    def _present(mod):
        if mod in names:
            return True
        if mod == "main":
            if "__main__" in names:
                return True
            if carchive is not None and "main" in carchive.toc:
                return True
        return False

    missing = [m for m in REQUIRED_MODULES if not _present(m)]
    if missing:
        okk &= fail(f"缺少模块：{', '.join(missing)}")
    else:
        okk &= ok(f"项目 {len(REQUIRED_MODULES)} 个模块全部在包内"
                  f"（PYZ 共 {len(names)} 个模块）")

    heavy = [n for n in names
             if n.split(".")[0] in ("torch", "transformers", "huggingface_hub")]
    if heavy:
        print(f"  [WARN] 检出重型依赖 {len(heavy)} 个（完整模式打包，体积会很大）")
    else:
        okk &= ok("未混入 torch/transformers（精简模式）")
    return okk


def check_code(reader, all_names, carchive=None):
    print("\n── 2. 修复是否真的编译进 exe ──")
    okk = True
    for mod, needle, desc in CODE_CHECKS:
        s = strings_of(reader, mod, all_names, carchive)
        if not s:
            okk &= fail(f"{desc}：模块 {mod} 无法读取")
            continue
        okk &= ok(f"{desc}  ({mod}.{needle})") if needle in s \
            else fail(f"{desc}  ({mod}.{needle} 未找到)")
    return okk


def check_resources():
    print("\n── 3. 随包资源与源码一致 ──")
    okk = True
    for rel, getter, expected, desc in RESOURCE_CHECKS:
        cands = [INTERNAL / rel, ROOT / "dist" / "SoulPlayer" / rel]
        path = next((p for p in cands if p.exists()), None)
        if path is None:
            okk &= fail(f"{desc}：找不到随包文件 {rel}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            okk &= fail(f"{desc}：{rel} 解析失败 {e}")
            continue
        got = getter(data)
        if got is not None and abs(float(got) - float(expected)) < 1e-9:
            okk &= ok(f"{desc} = {got}（期望 {expected}）")
        else:
            okk &= fail(f"{desc} = {got}（期望 {expected}，exe 里的配置是旧的）")
    return okk


def check_runtime():
    print("\n── 4. 启动与退出 ──")
    user32 = ctypes.windll.user32
    ENUM_CB = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    WM_CLOSE = 0x0010

    def find_hwnd():
        acc = []

        def cb(h, _):
            if user32.IsWindowVisible(h):
                n = user32.GetWindowTextLengthW(h)
                if n:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(h, buf, n + 1)
                    if buf.value.startswith("SoulPlayer"):
                        acc.append(h)
            return True

        user32.EnumWindows(ENUM_CB(cb), 0)
        return acc[0] if acc else None

    log = os.path.join(tempfile.gettempdir(), "_sp_verify_exit.log")
    # 沙箱数据目录 + 允许多实例：窗口模式的应用日志在数据目录里，Traceback 检测靠它
    sandbox = tempfile.mkdtemp(prefix="sp_verify_")
    app_log = os.path.join(sandbox, "logs", "soulplayer.log")
    env = dict(os.environ, SOULPLAYER_ALLOW_MULTI="1", SOULPLAYER_DATA_DIR=sandbox)
    okk = True
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        t0 = time.time()
        proc = subprocess.Popen([str(EXE)], cwd=str(EXE.parent),
                                stdout=f, stderr=subprocess.STDOUT, env=env)
        hwnd, boot = None, None
        while time.time() - t0 < 40:
            time.sleep(0.4)
            hwnd = find_hwnd()
            if hwnd:
                boot = time.time() - t0
                break
        if hwnd is None:
            proc.kill()
            return fail("40s 内窗口未出现")
        okk &= ok(f"窗口启动 {boot:.2f}s")

        time.sleep(5)
        user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        t1 = time.time()
        try:
            rc = proc.wait(timeout=20)
            okk &= ok(f"WM_CLOSE 后自行退出，用时 {time.time() - t1:.2f}s（退出码 {rc}）")
        except subprocess.TimeoutExpired:
            proc.kill()
            okk &= fail("20s 未退出（退出流程有阻塞）")

    # 顺便看看日志有没有 Traceback（stdout 与应用日志都查）
    try:
        for path in (log, app_log):
            txt = Path(path).read_text(encoding="utf-8", errors="replace")
            if "Traceback (most recent call last)" in txt:
                okk &= fail(f"运行日志含 Traceback，详见 {path}")
            else:
                okk &= ok(f"运行日志无 Traceback（{os.path.basename(path)}）")
    except Exception:
        pass
    finally:
        import shutil
        shutil.rmtree(sandbox, ignore_errors=True)
    return okk


def main():
    if not EXE.exists():
        print(f"[ERROR] exe 不存在：{EXE}")
        print("        先运行：py build_exe.py")
        return 2

    print("=" * 62)
    print("  SoulPlayer exe 内容校验")
    print("=" * 62)
    size_mb = EXE.stat().st_size / (1024 * 1024)
    print(f"  文件: {EXE}")
    print(f"  大小: {size_mb:.1f} MB\n")

    reader, tmp, carchive = load_pyz(EXE)
    try:
        names = set(reader.toc.keys())
        all_ok = check_archive(reader, names, carchive)
        all_ok &= check_code(reader, names, carchive)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

        # toc 用完后才可安全删除临时文件
    all_ok &= check_resources()

    if NO_EXIT_TEST:
        print("\n── 4. 启动与退出 ── 已跳过（--no-exit）")
    else:
        all_ok &= check_runtime()

    print()
    print("结论:", "全部通过" if all_ok else "存在失败项，见上方 [FAIL]")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
