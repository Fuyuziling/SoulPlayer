#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoulPlayer .exe 打包脚本

用法：在项目根目录运行
    py build_exe.py

说明：
- 使用 PyInstaller --onedir 模式（单文件夹）
- 自动包含所有资源文件（模型、预设、语音模型、音乐）
- 处理 MediaPipe / pygame / sounddevice 等依赖的隐藏导入
- 输出到 SoulPlayer/dist/SoulPlayer/
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ───────────────────────── 配置 ─────────────────────────

PROJECT_DIR = Path(__file__).parent.resolve()  # SoulPlayer 目录
ENTRY_SCRIPT = PROJECT_DIR / "main.py"
DIST_DIR = PROJECT_DIR / "dist"
BUILD_DIR = PROJECT_DIR / "build"
OUTPUT_NAME = "SoulPlayer"
ICON_FILE = PROJECT_DIR / "assets" / "soulplayer.ico"

# 正式版不带控制台窗口（用户不会看到一个黑框）。
# 打包前必须先完成 print -> 日志文件的重定向（app_logging），
# 否则 windowed 模式下 stdout 为 None，print 直接抛异常。
# 调试构建可用：set SOULPLAYER_CONSOLE=1 && py build_exe.py
CONSOLE_BUILD = os.environ.get("SOULPLAYER_CONSOLE", "") == "1"

# 需要打包进 exe 的资源文件/目录
# 格式：(源路径, 目标目录)
RESOURCES = [
    (PROJECT_DIR / "pose_landmarker_full.task", "."),
    (PROJECT_DIR / "hand_landmarker.task", "."),
    (PROJECT_DIR / "face_landmarker.task", "."),
    (PROJECT_DIR / "presets" / "default_presets.json", "presets"),
    (PROJECT_DIR / "voice_templates.json", "."),
    (PROJECT_DIR / "voice_templates", "voice_templates"),
    (PROJECT_DIR / "custom_presets", "custom_presets"),
    (PROJECT_DIR / "music", "music"),
    (PROJECT_DIR / "assets", "assets"),
]

# 隐藏导入（PyInstaller 可能无法自动检测到的模块）
HIDDEN_IMPORTS = [
    # 项目子模块（main.py 中用 try/except 动态导入）
    "app_paths",
    "app_logging",
    "motion_capture",
    "gesture_presets",
    "music_player",
    "custom_preset",
    "voice_match",
    "ui_theme",
    "ui_animations",
    # 学习模块
    "wave_learning",
    "wave_learning_rules",
    # OpenCV / NumPy
    "cv2",
    "numpy",
    "numpy.core._dtype_ctypes",
    # pygame
    "pygame",
    "pygame.mixer",
    "pygame.mixer.music",
    # MediaPipe（Tasks API 子模块）
    "mediapipe",
    "mediapipe.tasks",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
    "mediapipe.python._framework_bindings",
    "google.protobuf",
    # 语音（模板比对，依赖 sounddevice）
    "sounddevice",
    # tkinter（文件对话框）
    "tkinter",
    "tkinter.filedialog",
    "tkinter.commondialog",
    "tkinter.dialog",
    # 其他可能被动态导入的模块
    "queue",
    "json",
    "threading",
    "collections",
    "traceback",
    "inspect",
]

# 需要从 exe 中排除的重型依赖（可显著减小体积、加快启动）
# 说明：
#   torch + transformers 合计约 600MB，仅用于 wav2vec2 内容嵌入（语音匹配的高精度模式）。
#   voice_match.py 已做优雅降级：本地无 HF 缓存时根本不会 import 它们，直接用 MFCC+DTW。
#   因此在没有本地模型缓存的机器上，把它们打进 exe 只是白白增加 600MB 体积。
#   如需恢复（目标机器有 HF 缓存或可联网），设置环境变量后再打包：
#       set SOULPLAYER_BUNDLE_TORCH=1
#       py build_exe.py
_TORCH_EXCLUDES = [
    "torch",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "accelerate",
]
BUNDLE_TORCH = os.environ.get("SOULPLAYER_BUNDLE_TORCH", "") == "1"
EXCLUDE_MODULES = [] if BUNDLE_TORCH else _TORCH_EXCLUDES

# ───────────────────────── 辅助函数 ─────────────────────────

def clean_previous_builds():
    """清理旧的打包输出，避免冲突。"""
    for d in [DIST_DIR, BUILD_DIR]:
        if d.exists():
            print(f"[INFO] 清理旧目录: {d}")
            shutil.rmtree(d, ignore_errors=True)

def verify_resources():
    """检查所有资源文件是否存在。"""
    missing = []
    for src, _dest in RESOURCES:
        if not src.exists():
            missing.append(str(src))
    if missing:
        print("[WARN] 以下资源文件/目录不存在，将跳过：")
        for m in missing:
            print(f"       - {m}")
    return missing

def _write_version_file() -> Path:
    """由 version.py 生成 PyInstaller 版本资源文件（PE 属性页显示用）。"""
    import version

    txt = f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={version.VERSION_TUPLE},
    prodvers={version.VERSION_TUPLE},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable('080404b0', [
        StringStruct('CompanyName', '{version.COMPANY_NAME}'),
        StringStruct('FileDescription', '{version.FILE_DESCRIPTION}'),
        StringStruct('FileVersion', '{version.APP_VERSION}.0'),
        StringStruct('InternalName', '{version.PRODUCT_NAME}'),
        StringStruct('LegalCopyright', '{version.COPYRIGHT}'),
        StringStruct('OriginalFilename', '{OUTPUT_NAME}.exe'),
        StringStruct('ProductName', '{version.PRODUCT_NAME}'),
        StringStruct('ProductVersion', '{version.APP_VERSION}.0')
      ])
    ]),
    VarFileInfo([VarStruct('Translation', [2052, 1200])])
  ]
)
"""
    path = BUILD_DIR / "version_info.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")
    return path


def build_command():
    """构建 PyInstaller 命令行参数。"""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",           # 不确认覆盖
        "--onedir",              # 单文件夹模式（启动快、资源管理方便）
        "--console" if CONSOLE_BUILD else "--windowed",
        "--name", OUTPUT_NAME,
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        # 降低警告级别，减少无关输出
        "--log-level", "WARN",
    ]

    if ICON_FILE.exists():
        cmd.extend(["--icon", str(ICON_FILE)])
    if not CONSOLE_BUILD:
        try:
            cmd.extend(["--version-file", str(_write_version_file())])
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 版本资源生成失败（跳过）: {e}")

    # 添加资源文件
    for src, dest in RESOURCES:
        if src.exists():
            # Windows 上 --add-data 用分号分隔
            cmd.extend(["--add-data", f"{src};{dest}"])

    # 添加隐藏导入
    for mod in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", mod])

    # 排除重型依赖
    for mod in EXCLUDE_MODULES:
        cmd.extend(["--exclude-module", mod])

    # 尝试自动收集 mediapipe 的所有子模块和数据
    cmd.extend(["--collect-all", "mediapipe"])
    cmd.extend(["--collect-all", "pygame"])
    cmd.extend(["--collect-all", "sounddevice"])

    # 入口脚本
    cmd.append(str(ENTRY_SCRIPT))

    return cmd

def _git(*args):
    """执行 git 命令并返回去空白的输出；失败返回空串。"""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


# 不影响 exe 内容的路径：文档、测试、备份、构建工具自身。
# 指纹只关心"真正进包的东西"，否则写个日志、记个笔记就会把 exe 标成不可复现。
_FINGERPRINT_IGNORE_PREFIX = (
    ".workbuddy/", "tests/", "build/", "dist/", "dist_old_",
    "_dev_backup_", "Soul player提交材料/", "SoulPlayer/", "tools/",
    # 手机端是独立产品（Capacitor/TS），任何文件都不会进桌面 exe。
    # 不加这条的话，每次改手机端都会把 exe 误标成"过期"，久了就没人信这个检查了。
    "mobile/",
    # contracts/ 是手机端的跨端契约（JSON 阈值），Python 端不 import 它，
    # 改它不影响桌面 exe 的行为，同样不该触发同步误报。
    "contracts/",
)
_FINGERPRINT_IGNORE_NAMES = {
    "build_exe.py",        # 打包工具本身不进包
    "check_build_sync.py",
    "smoke_test.py",
    ".gitignore",
    "smoke_shot.png",
    "_smoke_last.log",
    "_wave_diag.log",
    "DIST_SYNC_REPORT.md",
    # PyInstaller 每次构建都会在项目根重新生成的 spec，属构建副产物
    "SoulPlayer.spec",
    # 仅被 copy_post_build_files 复制进 dist 的外围文件，
    # 不参与程序逻辑，改动不值得触发重打包
    "README.md",
    "requirements.txt",
    "run.bat",
}


def filter_build_relevant(paths):
    """从路径列表里滤掉不影响 exe 内容的（文档/测试/工具/备份/日志）。"""
    result = []
    for rel in paths:
        rel = rel.strip().replace("\\", "/")
        # git 对非 ASCII 路径默认输出 C 风格转义并加引号（中文文件名会变成
        # "/345/276/205..."）。不去引号的话，".md\"" 匹配不上 endswith(".md")，
        # 中文报告/待办文档就会绕过文档过滤，触发"exe 过期"误报。
        if len(rel) >= 2 and rel.startswith('"') and rel.endswith('"'):
            rel = rel[1:-1]
        if not rel:
            continue
        if rel.startswith(_FINGERPRINT_IGNORE_PREFIX):
            continue
        name = rel.rsplit("/", 1)[-1]
        if name in _FINGERPRINT_IGNORE_NAMES:
            continue
        # 所有 Markdown 文档都不进 exe（唯一例外 README.md 复制进 dist，
        # 但它已在 _FINGERPRINT_IGNORE_NAMES 里）。不忽略的话，每在根目录
        # 写一份调研报告/待办清单就会误报一次"exe 过期"——已实际踩坑
        # （2026-11-10：新增调研报告与待办文档触发误报）。
        if name.endswith(".md"):
            continue
        # 构建日志、冒烟截图等临时产物
        if name.startswith("_build_") and name.endswith(".log"):
            continue
        result.append(rel)
    return result


def relevant_dirty_files():
    """
    返回**会影响 exe 内容**的未提交文件列表。
    过滤掉文档/测试/备份/构建脚本，避免指纹误报"脏"。

    注意：不用 `git status --porcelain` 解析——它的固定列宽前缀会被
    _git() 的 strip() 破坏（首行前导空格被吃掉，导致路径少首字符）。
    改用两个不带格式前缀的命令拼接，路径原样返回。
    """
    paths = []
    # 已跟踪文件相对 HEAD 的改动（含已暂存与未暂存）
    tracked = _git("diff", "--name-only", "HEAD")
    if tracked:
        paths.extend(tracked.splitlines())
    # 未跟踪的新文件
    untracked = _git("ls-files", "--others", "--exclude-standard")
    if untracked:
        paths.extend(untracked.splitlines())

    return filter_build_relevant(paths)


def write_build_info():
    """
    在输出目录写入 BUILD_INFO.txt，作为"这个 exe 是从哪份源码打出来的"指纹。
    用途：check_build_sync.py 读取其中的 commit，与当前 HEAD 比对，
    不一致即说明 exe 落后于源码、需要重新打包。
    """
    import datetime

    output_dir = DIST_DIR / OUTPUT_NAME
    if not output_dir.exists():
        print(f"[WARN] 输出目录不存在，跳过 BUILD_INFO: {output_dir}")
        return

    commit = _git("rev-parse", "HEAD") or "unknown"
    short = _git("rev-parse", "--short", "HEAD") or commit[:7]
    commit_time = _git("log", "-1", "--format=%ci") or "unknown"
    subject = _git("log", "-1", "--format=%s") or "unknown"
    # 只看影响 exe 的改动；文档/日志/测试不算脏
    dirty_files = relevant_dirty_files()
    dirty = bool(dirty_files)

    try:
        import version as ver
    except Exception:  # noqa: BLE001
        ver = None

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        import PyInstaller
        pi_ver = PyInstaller.__version__
    except Exception:
        pi_ver = "unknown"

    lines = [
        "SoulPlayer 构建指纹",
        "=" * 50,
        f"打包时间   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"源码 commit: {commit}",
        f"commit 简称: {short}",
        f"commit 时间: {commit_time}",
        f"commit 说明: {subject}",
        f"工作区脏   : {'是（有未提交改动）' if dirty else '否'}",
        f"Python     : {py_ver}",
        f"PyInstaller: {pi_ver}",
        f"打包模式   : {'完整（含 torch）' if BUNDLE_TORCH else '精简（排除 torch）'}",
        f"窗口模式   : {'控制台（调试）' if CONSOLE_BUILD else '窗口（正式版）'}",
        f"版本       : {ver.APP_VERSION}",
    ]
    if dirty:
        lines.append("")
        lines.append("未提交文件：")
        lines.extend(f"  {ln}" for ln in dirty_files[:30])

    info_path = output_dir / "BUILD_INFO.txt"
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] 已写入构建指纹: {info_path}")
    print(f"       源码 commit {short}，工作区{'脏' if dirty else '干净'}")

    if dirty:
        print("[WARN] 工作区有未提交改动，exe 内容无法用 commit 精确复现。")


def copy_post_build_files():
    """
    打包后，将额外的文件复制到输出目录。
    如 run.bat、README 等。
    """
    output_dir = DIST_DIR / OUTPUT_NAME
    if not output_dir.exists():
        print(f"[WARN] 输出目录不存在: {output_dir}")
        return

    # 复制 run.bat（用户习惯双击 .bat）
    bat_src = PROJECT_DIR / "run.bat"
    if bat_src.exists():
        bat_dst = output_dir / "run.bat"
        shutil.copy2(bat_src, bat_dst)
        print(f"[INFO] 已复制: run.bat")

    # 复制 README
    readme_src = PROJECT_DIR / "README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, output_dir / "README.md")
        print(f"[INFO] 已复制: README.md")

    # 复制 requirements.txt（供参考）
    req_src = PROJECT_DIR / "requirements.txt"
    if req_src.exists():
        shutil.copy2(req_src, output_dir / "requirements.txt")
        print(f"[INFO] 已复制: requirements.txt")

def print_summary():
    """打印打包结果摘要。"""
    output_dir = DIST_DIR / OUTPUT_NAME
    exe_path = output_dir / f"{OUTPUT_NAME}.exe"

    print()
    print("=" * 60)
    print("  SoulPlayer 打包完成")
    print("=" * 60)
    print()

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"  可执行文件: {exe_path}")
        print(f"  文件大小:   {size_mb:.1f} MB")
    else:
        print(f"  [WARN] 未找到 exe 文件: {exe_path}")

    print(f"  输出目录:   {output_dir}")
    print()
    print("  使用方式:")
    print("  1. 将整个 SoulPlayer 文件夹复制到目标电脑")
    print("  2. 双击 SoulPlayer.exe 运行")
    print("  3. 如有问题，查看日志：%LOCALAPPDATA%\\SoulPlayer\\logs")
    print("     （便携模式：exe 旁的 logs 目录；数据目录里也放一个 portable.flag 即启用）")
    print()
    print("=" * 60)

# ───────────────────────── 主流程 ─────────────────────────

def main():
    print("=" * 60)
    print("  SoulPlayer .exe 打包脚本")
    print("=" * 60)
    print()

    # 1. 验证入口脚本
    if not ENTRY_SCRIPT.exists():
        print(f"[ERROR] 入口脚本不存在: {ENTRY_SCRIPT}")
        sys.exit(1)

    # 2. 验证资源
    verify_resources()

    # 2.5 显示打包模式
    if BUNDLE_TORCH:
        print("[INFO] 打包模式: 完整（包含 torch/transformers，体积较大）")
    else:
        print("[INFO] 打包模式: 精简（已排除 torch/transformers 等约 600MB 重型依赖）")
        print("       语音匹配将自动降级为 MFCC+DTW，功能不受影响。")
        print("       如需恢复，设置 SOULPLAYER_BUNDLE_TORCH=1 后重新打包。")
    print()

    # 3. 清理旧构建
    clean_previous_builds()

    # 4. 构建命令
    cmd = build_command()
    print("[INFO] PyInstaller 命令:")
    print("  " + " ".join(cmd))
    print()

    # 5. 执行打包
    print("[INFO] 开始打包，这可能需要几分钟...")
    print()
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("[ERROR] PyInstaller 未找到。请确保已安装：")
        print("        py -m pip install pyinstaller")
        sys.exit(1)

    if result.returncode != 0:
        print(f"[ERROR] PyInstaller 退出码: {result.returncode}")
        print("[INFO] 尝试以 ERROR 级别日志重新运行以查看详情...")
        sys.exit(1)

    # 6. 复制额外文件
    copy_post_build_files()

    # 6.5 写入构建指纹（供 check_build_sync.py 比对）
    write_build_info()

    # 7. 打印摘要
    print_summary()

if __name__ == "__main__":
    main()
