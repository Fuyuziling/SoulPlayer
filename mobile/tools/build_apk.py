#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 SoulPlayer Android 安装包（APK）。

一条命令走完：web 构建 → 同步进 Android 工程 → Gradle 打包 → 收集产物并写指纹。

用法：
    py mobile/tools/build_apk.py              # release（签名，可直接安装）
    py mobile/tools/build_apk.py --debug      # debug（调试用）

产物：
    mobile/release/soulplayer-<version>.apk
    mobile/release/apk-info.txt               # commit / 时间 / sha256 / 版本

为什么要有这个脚本而不是手敲三条命令：
    改动通常发生在 src/ 里，容易只 `npm run build` 忘了 `cap copy`——
    那样打出来的 APK 里跑的仍是上一版 web 产物，现象是"改了没效果"，
    与 PC 端踩过的"源码改了 exe 没重打包"是同一类坑。脚本把它们绑在一起。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

MOBILE = Path(__file__).resolve().parent.parent
ROOT = MOBILE.parent
ANDROID = MOBILE / "android"
RELEASE_DIR = MOBILE / "release"

DEFAULT_SDK = Path(r"D:\android-toolchain\sdk")
DEFAULT_JDK = Path(r"D:\android-toolchain\jdk-17.0.21+9")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sdk_dir() -> Path:
    """SDK 位置：环境变量优先，其次本项目约定的独立工具链目录。"""
    for key in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        v = os.environ.get(key)
        if v and Path(v).exists():
            return Path(v)
    return DEFAULT_SDK


def build_env() -> dict:
    """构造子进程环境。

    JAVA_HOME 必须是 Windows 路径形式（gradlew.bat 里是 %JAVA_HOME%\\bin\\java.exe），
    而 PATH 追加的条目不能带盘符冒号——PATH 以冒号分隔，'D:/x' 会被从 'D:' 处截断，
    命令直接找不到（踩过）。
    """
    env = os.environ.copy()
    jdk = Path(env.get("JAVA_HOME", "")) if env.get("JAVA_HOME") else DEFAULT_JDK
    if not (jdk / "bin" / "java.exe").exists():
        jdk = DEFAULT_JDK
    env["JAVA_HOME"] = str(jdk)
    sdk = sdk_dir()
    env["ANDROID_HOME"] = str(sdk)
    env["ANDROID_SDK_ROOT"] = str(sdk)
    extra = [str(jdk / "bin"), str(sdk / "platform-tools"), str(sdk / "build-tools" / "34.0.0")]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    log("$ " + " ".join(cmd))
    # npm/npx 在 Windows 上是 .cmd，必须 shell=True 才能被解析
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, shell=os.name == "nt")
    if proc.returncode != 0:
        raise SystemExit(f"命令失败（exit {proc.returncode}）：{' '.join(cmd)}")


def gradle_wrapper() -> str:
    return "gradlew.bat" if os.name == "nt" else "./gradlew"


def apk_version() -> str:
    """从 build.gradle 读 versionName，避免版本号在两处各写一遍。"""
    text = (ANDROID / "app" / "build.gradle").read_text(encoding="utf-8")
    m = re.search(r'versionName\s+"([^"]+)"', text)
    return m.group(1) if m else "0.0.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT), capture_output=True, text=True, check=False)
        # 只关心进包的文件，忽略构建产物与工具链目录
        ignore = ("mobile/node_modules", "mobile/dist", "mobile/release",
                  "mobile/android/build", "mobile/android/.gradle",
                  "mobile/android/app/build", "dist_old_")
        for line in out.stdout.splitlines():
            path = line[3:].strip()
            if not path or path.startswith(ignore):
                continue
            return True
        return False
    except Exception:
        return False


def verify_apk(apk: Path) -> None:
    """用 SDK 自带工具核对签名与清单，别只看"文件生成了"。"""
    sdk = sdk_dir()
    apksigner = sdk / "build-tools" / "34.0.0" / "apksigner.bat"
    aapt2 = sdk / "build-tools" / "34.0.0" / "aapt2.exe"
    env = build_env()

    if aapt2.exists():
        out = subprocess.run(
            [str(aapt2), "dump", "badging", str(apk)],
            env=env, capture_output=True, text=True, shell=False, check=False)
        head = out.stdout.splitlines()[:6]
        for line in head:
            log("  " + line)

    if apksigner.exists():
        out = subprocess.run(
            [str(apksigner), "verify", "--print-certs", str(apk)],
            env=env, capture_output=True, text=True, check=False)
        if out.returncode == 0:
            line = next((l for l in out.stdout.splitlines() if "SHA-256" in l), "")
            log(f"  签名校验通过 {line.strip()[:70]}")
        else:
            log("  ⚠ 签名校验失败：")
            log("  " + (out.stderr or out.stdout)[:300])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="打 debug 包（默认 release）")
    args = ap.parse_args()

    env = build_env()
    if not Path(env["JAVA_HOME"], "bin", "java.exe").exists():
        raise SystemExit(f"找不到 JDK：{env['JAVA_HOME']}")
    if not sdk_dir().exists():
        raise SystemExit(f"找不到 Android SDK：{sdk_dir()}")

    log("1/4 构建 web 产物")
    run(["npm", "run", "build"], MOBILE, env)

    log("2/4 同步进 Android 工程")
    run(["npx", "cap", "copy", "android"], MOBILE, env)

    variant = "debug" if args.debug else "release"
    log(f"3/4 Gradle 打包（{variant}）")
    task = "assembleDebug" if args.debug else "assembleRelease"
    run([gradle_wrapper(), task, "--no-daemon"], ANDROID, env)

    log("4/4 收集产物")
    src = ANDROID / "app" / "build" / "outputs" / "apk" / variant / f"app-{variant}.apk"
    if not src.exists():
        raise SystemExit(f"未找到 APK：{src}")
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    version = apk_version()
    suffix = "-debug" if args.debug else ""
    dst = RELEASE_DIR / f"soulplayer-{version}{suffix}.apk"
    shutil.copy2(src, dst)

    digest = sha256(dst)
    info = RELEASE_DIR / "apk-info.txt"
    info.write_text(
        f"APK      {dst.name}\n"
        f"版本     {version} ({'debug' if args.debug else 'release'})\n"
        f"大小     {dst.stat().st_size / 1024 / 1024:.1f} MB\n"
        f"sha256   {digest}\n"
        f"commit   {git_commit()}{'（工作区有未提交改动）' if git_dirty() else ''}\n"
        f"构建时间 {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8")

    log(f"产物：{dst}  ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
    verify_apk(dst)
    log(f"SHA256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
