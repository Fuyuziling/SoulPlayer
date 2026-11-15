#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoulPlayer 日志层

正式版不能带控制台窗口（--windowed），而 sys.stdout / sys.stderr 在
windowed 模式下是 None——项目里 190+ 处 print 会直接抛 AttributeError，
或在某些打包配置下静默丢弃。所以必须先把输出收口到文件，再切窗口模式。

本模块做四件事：
1. 把 print 的输出同时写入轮转日志文件（5MB × 3）与控制台（若有）
2. 接管 sys.excepthook / threading.excepthook，线程崩溃也能落盘
3. 启用 faulthandler，捕获 MediaPipe 这类原生库的硬崩溃
4. 启动时打印一行路径摘要，排查"数据到底存哪"不用猜

环境变量 SOULPLAYER_NO_LOG=1 可关闭文件落盘（测试用）。
"""

from __future__ import annotations

import faulthandler
import io
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import app_paths

ENV_NO_LOG = "SOULPLAYER_NO_LOG"
LOG_FILE_NAME = "soulplayer.log"
CRASH_FILE_NAME = "crash.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_initialized = False
_log_file: Optional[Path] = None
_crash_stream = None  # 必须持有引用，否则 faulthandler 的文件句柄会被回收


class _Tee(io.TextIOBase):
    """
    把写入转发给 logging 与原始流。

    注意：不能用 TextIOWrapper 包 stdout——windowed 模式下原始流是 None，
    包一层会立刻炸。这里只实现 write/flush 这两个真正被用到的接口。
    """

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, logger: logging.Logger, level: int, real=None):
        super().__init__()
        self._logger = logger
        self._level = level
        self._real = real
        self._buf = ""

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def fileno(self):  # 某些库会拿 fileno 探测终端，这里明确拒绝
        raise io.UnsupportedOperation("fileno")

    def write(self, s) -> int:
        if not isinstance(s, str):
            s = str(s)
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                try:
                    self._logger.log(self._level, line.rstrip())
                except Exception:
                    pass
        if self._real is not None:
            try:
                self._real.write(s)
                self._real.flush()
            except Exception:
                self._real = None  # 控制台失效后不再重试
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            try:
                self._logger.log(self._level, self._buf.rstrip())
            except Exception:
                pass
            self._buf = ""
        if self._real is not None:
            try:
                self._real.flush()
            except Exception:
                pass


def _log_path() -> Optional[Path]:
    return _log_file


def _handle_exception(exc_type, exc_value, exc_traceback) -> None:
    """未捕获异常：写日志。控制台输出由 logging 的 StreamHandler 负责。"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger("soulplayer").critical(
        "未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))
    # 兜底：若日志系统本身不可用，仍尝试原始输出
    if not logging.getLogger().handlers:
        try:
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        except Exception:
            pass


def _handle_thread_exception(args) -> None:
    """子线程崩溃默认只打一行 stderr；摄像头/语音线程崩了必须留证据。"""
    logging.getLogger("soulplayer").error(
        "子线程异常退出: %s",
        getattr(args.thread, "name", "?"),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def setup(console: Optional[bool] = None, level: int = logging.INFO) -> Path:
    """
    初始化日志。必须在任何 print 之前调用（main.py 顶部）。

    console=None 时自动判断：原始 stderr 存在就同时输出到控制台。
    返回日志文件路径；文件落盘不可用时返回空 Path。
    """
    global _initialized, _log_file, _crash_stream
    if _initialized:
        return _log_file or Path()

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_FORMAT, _DATEFMT)

    # 原始流要在替换前抓住
    real_out = sys.stdout
    real_err = sys.stderr
    if console is None:
        console = real_err is not None

    # SOULPLAYER_NO_LOG=1：纯透传，不装任何 handler（自动化测试用，
    # 避免测试输出被 Tee + StreamHandler 打成两份）
    no_log = os.environ.get(ENV_NO_LOG) == "1"

    if not no_log:
        try:
            log_dir = app_paths.logs_dir()
            path = log_dir / LOG_FILE_NAME
            file_handler = RotatingFileHandler(
                path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
                encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
            _log_file = path

            crash_path = log_dir / CRASH_FILE_NAME
            _crash_stream = open(crash_path, "a", encoding="utf-8")
            faulthandler.enable(file=_crash_stream, all_threads=True)
        except Exception:
            _log_file = None

        if console and real_err is not None:
            try:
                stream_handler = logging.StreamHandler(real_err)
                stream_handler.setFormatter(formatter)
                root.addHandler(stream_handler)
            except Exception:
                pass

    logger = logging.getLogger("soulplayer")
    sys.stdout = _Tee(logger, logging.INFO, real_out)
    sys.stderr = _Tee(logger, logging.ERROR, real_err)
    sys.excepthook = _handle_exception
    threading.excepthook = _handle_thread_exception
    _initialized = True

    logger.info("=" * 56)
    logger.info("SoulPlayer 启动")
    logger.info(app_paths.describe())
    if _log_file is None:
        logger.warning("日志文件不可用（目录不可写），仅控制台输出")
    else:
        logger.info(f"日志文件: {_log_file}")
    return _log_file or Path()
