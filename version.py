#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoulPlayer 版本号唯一来源。

build_exe.py 读取它生成 PE 版本资源（文件属性页显示的产品名/版本/版权）。
改版本号只改这里，不要去动 version_info 模板。
"""

APP_VERSION = "2.2.0"
# filevers/prodvers 必须是四元组
VERSION_TUPLE = (2, 2, 0, 0)

COMPANY_NAME = "SoulPlayer"
PRODUCT_NAME = "SoulPlayer"
FILE_DESCRIPTION = "基于肢体动作捕捉的智能音乐播放器"
COPYRIGHT = "(c) 2026 SoulPlayer"
