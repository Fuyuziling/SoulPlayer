#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_paths 路径层单元测试（stdlib unittest，不引入新依赖，秒级）

锁住四种运行态的定位规则与迁移行为——这是"用户数据会不会丢"的根基：
  1. 开发态：数据根 = 源码目录（与旧行为一致）
  2. 环境变量覆盖最高优先
  3. 打包态默认：数据根 = %LOCALAPPDATA%/SoulPlayer（安装目录只读）
  4. 打包态便携：exe 旁有 portable.flag → 数据根 = exe 目录
  5. 旧数据一次性迁移（含同级 .json 配套文件）

用法：py tests/test_paths.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app_paths


class PathsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="sp_paths_")
        self.tmp = Path(self._tmp.name)
        # 每个用例独立的"安装目录"
        self.exe_dir = self.tmp / "inst" / "SoulPlayer"
        self.exe_dir.mkdir(parents=True)
        (self.exe_dir / "SoulPlayer.exe").write_bytes(b"fake")
        # 手动伪装打包态（sys 是内建模块，mock.patch 对动态属性不可靠）
        self._had_frozen = hasattr(sys, "frozen")
        self._old_frozen = getattr(sys, "frozen", None)
        self._old_executable = sys.executable
        sys.frozen = True
        sys.executable = str(self.exe_dir / "SoulPlayer.exe")
        self._old_env = os.environ.pop(app_paths.ENV_DATA_DIR, None)
        self._old_appdata = os.environ.get("LOCALAPPDATA")
        os.environ["LOCALAPPDATA"] = str(self.tmp / "appdata")
        app_paths.reset_cache()

    def tearDown(self):
        if self._had_frozen:
            sys.frozen = self._old_frozen
        else:
            try:
                del sys.frozen
            except AttributeError:
                pass
        sys.executable = self._old_executable
        if self._old_env is not None:
            os.environ[app_paths.ENV_DATA_DIR] = self._old_env
        if self._old_appdata is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = self._old_appdata
        app_paths.reset_cache()
        self._tmp.cleanup()

    def test_dev_mode_uses_project_dir(self):
        # 临时切回开发态
        if self._had_frozen:
            sys.frozen = self._old_frozen
        else:
            del sys.frozen
        app_paths.reset_cache()
        try:
            self.assertEqual(app_paths.data_root(),
                             Path(app_paths.__file__).resolve().parent)
        finally:
            sys.frozen = True
            app_paths.reset_cache()

    def test_env_override_highest_priority(self):
        custom = self.tmp / "mydata"
        os.environ[app_paths.ENV_DATA_DIR] = str(custom)
        app_paths.reset_cache()
        try:
            self.assertEqual(app_paths.data_root(), custom)
        finally:
            os.environ.pop(app_paths.ENV_DATA_DIR, None)
            app_paths.reset_cache()

    def test_frozen_defaults_to_localappdata(self):
        self.assertEqual(app_paths.data_root(),
                         self.tmp / "appdata" / app_paths.APP_NAME)
        # 关键断言：绝不往安装目录写数据
        self.assertFalse((self.exe_dir / "custom_presets").exists())

    def test_portable_flag_uses_exe_dir(self):
        (self.exe_dir / app_paths.PORTABLE_FLAG).write_text("", encoding="utf-8")
        app_paths.reset_cache()
        self.assertEqual(app_paths.data_root(), self.exe_dir)

    def test_migration_copies_content_and_sidecar(self):
        app_paths.reset_cache()
        legacy = self.exe_dir / "custom_presets"
        legacy.mkdir()
        (legacy / "p1.json").write_text("{}", encoding="utf-8")
        (self.exe_dir / "custom_presets.json").write_text("[]", encoding="utf-8")

        target = app_paths.data_dir(
            "custom_presets",
            migrate_from=app_paths.legacy_candidates("custom_presets"))
        self.assertEqual(target,
                         self.tmp / "appdata" / app_paths.APP_NAME / "custom_presets")
        self.assertEqual((target / "p1.json").read_text(encoding="utf-8"), "{}")
        # 同级配套映射文件也要跟过来
        self.assertEqual(
            (target.parent / "custom_presets.json").read_text(encoding="utf-8"), "[]")
        # 迁移是拷贝不是移动：源保留（迁移失败可回退）
        self.assertTrue((legacy / "p1.json").exists())

    def test_no_migration_when_target_has_content(self):
        app_paths.reset_cache()
        target = app_paths.data_root() / "voice_templates"
        target.mkdir(parents=True)
        (target / "keep.txt").write_text("new", encoding="utf-8")

        legacy = self.exe_dir / "voice_templates"
        legacy.mkdir()
        (legacy / "old.txt").write_text("old", encoding="utf-8")

        result = app_paths.data_dir(
            "voice_templates",
            migrate_from=app_paths.legacy_candidates("voice_templates"))
        self.assertEqual(result, target)
        self.assertFalse((target / "old.txt").exists())  # 有内容就不迁移，不覆盖


if __name__ == "__main__":
    unittest.main(verbosity=2)
