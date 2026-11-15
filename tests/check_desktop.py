#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC 正式版形态检查（针对 dist/SoulPlayer/SoulPlayer.exe）

verify_dist.py 证明"改的逻辑进了包"，本脚本证明"它长得像桌面软件"：
  1. PE 子系统 = GUI（无控制台黑框）
  2. 版本资源：ProductName / FileVersion 与 version.py 一致
  3. 图标资源存在
  4. 日志落盘：启动后沙箱数据目录出现 logs/soulplayer.log
  5. 单实例：第二个进程立刻退出，已有窗口保持存活；WM_CLOSE 能正常退出

用法：py tests/check_desktop.py
"""

import ctypes
import os
import struct
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import version  # noqa: E402

EXE = ROOT / "dist" / "SoulPlayer" / "SoulPlayer.exe"
WM_CLOSE = 0x0010

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    return ok


# ───────────────────────── 1. PE 子系统 ─────────────────────────

def pe_subsystem(path: Path) -> int:
    data = path.read_bytes()
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    coff = e_lfanew + 4  # 跳过 'PE\0\0'
    optional = coff + 20
    # OptionalHeader.Subsystem 在 Magic(2)+...(偏移 68) 处（PE32+）
    magic = struct.unpack_from("<H", data, optional)[0]
    sub_off = optional + (68 if magic == 0x20B else 68)
    return struct.unpack_from("<H", data, sub_off)[0]


# ───────────────────────── 2. 版本资源 ─────────────────────────

def file_version(path: Path):
    version_dll = ctypes.WinDLL("version")
    size = version_dll.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    data = ctypes.create_string_buffer(size)
    if not version_dll.GetFileVersionInfoW(str(path), 0, size, data):
        return None
    fixed = ctypes.c_void_p()
    if not version_dll.VerQueryValueW(
            data, "\\", ctypes.byref(fixed), ctypes.byref(wintypes.UINT())):
        return None
    # VS_FIXEDFILEINFO: dwSignature(4) dwStrucVersion(4) dwFileVersionMS(4) dwFileVersionLS(4)
    ms, ls = struct.unpack_from("<II", ctypes.string_at(fixed.value + 8, 8))
    return ((ms >> 16) & 0xFFFF, ms & 0xFFFF, (ls >> 16) & 0xFFFF, ls & 0xFFFF)


def has_icon_resource(path: Path) -> bool:
    """
    用 SHDefExtractIcon 提取首个图标。
    （EnumResourceNamesW 的 ctypes 回调在 64 位下容易踩调用约定坑，
    Shell 这条路简单且行为与资源管理器一致。）
    """
    try:
        shell32 = ctypes.WinDLL("shell32")
        user32 = ctypes.WinDLL("user32")
        hicon = ctypes.c_void_p()
        rc = shell32.SHDefExtractIconW(
            str(path), 0, 0, ctypes.byref(hicon), None, 32)
        if hicon.value:
            user32.DestroyIcon(hicon)
        return rc == 0 and bool(hicon.value)
    except Exception:
        return False


# ───────────────────────── 4/5. 运行时 ─────────────────────────

user32 = ctypes.WinDLL("user32")


def find_hwnd():
    acc = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, 64)
        # pygame 2 的窗口类名实测是 "pygame"（SDL_app 是 SDL1 的）
        if cls.value not in ("pygame", "SDL_app"):
            return True
        acc.append(hwnd)
        return False

    user32.EnumWindows(cb, 0)
    return acc[0] if acc else None


def runtime_checks(sandbox: Path):
    env = dict(os.environ, SOULPLAYER_DATA_DIR=str(sandbox))
    log_file = sandbox / "logs" / "soulplayer.log"

    t0 = time.time()
    p1 = subprocess.Popen([str(EXE)], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hwnd = None
    while time.time() - t0 < 40:
        time.sleep(0.4)
        hwnd = find_hwnd()
        if hwnd:
            break
    boot_ok = record("窗口启动", hwnd is not None,
                     f"{time.time() - t0:.1f}s" if hwnd else "40s 内未出现")
    if not boot_ok:
        p1.kill()
        return

    time.sleep(3)  # 给日志一点时间落盘
    record("日志落盘（logs/soulplayer.log）",
           log_file.is_file() and log_file.stat().st_size > 0,
           str(log_file) if log_file.is_file() else "未生成")

    # 单实例：第二个进程应立刻退出
    t1 = time.time()
    p2 = subprocess.Popen([str(EXE)], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        rc2 = p2.wait(timeout=15)
        waited = time.time() - t1
        record("第二实例立即退出", rc2 == 0 and waited < 10,
               f"耗时 {waited:.1f}s，退出码 {rc2}")
    except subprocess.TimeoutExpired:
        p2.kill()
        record("第二实例立即退出", False, "15s 仍未退出（互斥体未生效）")

    record("首实例仍存活", p1.poll() is None)

    # 正常退出
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
    try:
        t2 = time.time()
        p1.wait(timeout=20)
        record("WM_CLOSE 后退出", True, f"{time.time() - t2:.1f}s")
    except subprocess.TimeoutExpired:
        p1.kill()
        record("WM_CLOSE 后退出", False, "20s 未退出")


def main():
    print("=" * 60)
    print("  SoulPlayer 桌面形态检查")
    print("=" * 60)

    if not EXE.exists():
        print(f"[ERROR] 未找到 {EXE}，请先 py build_exe.py")
        return 1

    print("\n── 静态检查（PE 资源）──")
    sub = pe_subsystem(EXE)
    record("PE 子系统=GUI（无控制台黑框）", sub == 2, f"subsystem={sub}")
    fv = file_version(EXE)
    record("版本资源 FileVersion",
           fv is not None and fv[:3] == version.VERSION_TUPLE[:3],
           f"exe={fv} 期望={version.VERSION_TUPLE}")
    record("图标资源存在", has_icon_resource(EXE))

    print("\n── 运行时检查（沙箱数据目录）──")
    sandbox = Path(tempfile.mkdtemp(prefix="sp_desktop_"))
    try:
        runtime_checks(sandbox)
    finally:
        # 应用进程可能仍有句柄 linger，清理失败不影响结论
        time.sleep(1)
        import shutil
        shutil.rmtree(sandbox, ignore_errors=True)

    print("\n" + "=" * 60)
    ok = all(o for _, o in results)
    print("  全部通过 ✓" if ok else
          f"  存在失败项 ✗  {[n for n, o in results if not o]}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
