# SoulPlayer

> 基于 AI 肢体动作捕捉与语音模板触发的智能音乐播放器。让身体成为遥控器，让音乐随律动自然流淌。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Platform-Windows-lightgrey" alt="Platform">
</p>

---

## ✨ 核心功能

- **🎥 动作感知播放** — MediaPipe 实时检测人体 33 个关键点，双手举起暂停，挥手切歌（支持方向区分：显像框中右→左下一首、左→右上一首）
- **🎤 语音模板触发** — 录下自己的声音当"遥控器"，再说一次即可播放绑定歌曲；MFCC + DTW 纯本地比对，无需语言模型、无需联网
- **🎨 自定义预设** — 动作（R 键）与台词（T 键）均可一键录制并绑定任意歌曲，同一预设多首歌随机播放
- **🎵 音乐库动态监测** — 运行中往 `music/` 文件夹放歌，2 秒内自动进播放列表
- **🔒 完全离线** — 所有 AI 推理均在本地完成，保护隐私

---

## 🚀 快速开始

### 1. 环境准备

- Windows 10/11，Python 3.9+
- 一个摄像头（动作捕捉）与麦克风（语音功能）

### 2. 安装依赖

```bash
git clone <仓库地址>
cd SoulPlayer
pip install -r requirements.txt
```

仓库已自带 MediaPipe 姿态模型 `pose_landmarker_full.task`，无需额外下载。

### 3. 放入音乐

把 `.mp3 / .wav / .ogg / .flac` 文件放进 `music/` 文件夹（支持子文件夹）。

### 4. 启动

```bash
python run.py
# 或双击 run.bat
```

---

## 🎮 使用指南

### 动作交互

| 动作 | 效果 |
|------|------|
| ✋ 张开手掌正对摄像头，悬停不动约 0.5 秒 | 暂停播放 |
| 👋 挥手（显像框中从右向左划） | 下一首 |
| 👋 挥手（显像框中从左向右划） | 上一首 |

> 显像框即界面左上角的摄像头预览画面，方向以你"看到"的为准。
> 暂停的判定是严格的"手掌悬停"：掌心朝镜头 + 五指展开 + 保持不动，
> 普通抬手、挥舞、上下移动都不会误触。自定义动作匹配期间，内置动作自动屏蔽。

### 语音交互

1. 按 **T** → 屏幕中央 3 秒倒计时 → 清晰说出一句台词（2~6 字短句最佳）→ 再按 **T** 结束
2. 在弹出的选歌面板勾选歌曲（点击切换，滚轮翻页），**Enter** 保存
3. 按 **V** 开启监听，之后说出这句台词就会随机播放绑定歌曲之一
4. 语音面板中悬停模板可删除

### 自定义动作预设

1. 按 **R** → 3 秒倒计时 → 做出你的动作 → 再按 **R** 结束
2. 选歌面板绑定歌曲，Enter 保存
3. 悬停预设卡片可重命名 / 删除 / 收藏 / 调整顺序

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| 空格 | 播放 / 暂停 |
| ← / → | 上一首 / 下一首 |
| ↑ / ↓ | 音量调节 |
| R | 录制动作预设 |
| T | 录制语音台词 |
| V | 开关语音监听 |
| ESC | 退出 |

---

## ❓ 常见问题

**Q: 语音触发不灵 / 容易误触？**
控制台会打印每次比对的距离（`比对: voice_xxxxxx=0.xx（阈值 0.60）`）。该触发没触发就把距离调大 `voice_match.py` 中的 `MATCH_THRESHOLD`，误触就调小。录音建议在安静环境、距麦克风 30cm 左右。

**Q: 挥手不触发？**
动作幅度大一些（横向跨度需超过 15% 屏宽）、速度自然即可，确保手腕在画面中可见。

**Q: 如何运行测试？**
```bash
python tests/test_wave.py        # 挥手检测仿真
python tests/test_palm.py         # 手掌悬停检测仿真
python tests/test_custom.py       # 自定义预设与内置动作屏蔽
python tests/test_data_voice.py   # 数据目录 / 语音匹配 / 音量分离
```

**Q: 打包成 exe？**
```bash
python build_exe.py
```
产物在 `dist/SoulPlayer/`。音乐与自定义数据（预设、语音模板）优先读写项目源码目录，其次是 exe 旁边；exe 内部的旧数据会在首次启动时自动迁移出来。

> 💡 本仓库 `release/SoulPlayer/` 已附带预编译好的 Windows 可执行版（onedir 模式，含 `SoulPlayer.exe` 与 `_internal/`），下载后运行 `SoulPlayer.exe` 或 `run.bat` 即可，无需安装 Python。

---

## 🏗️ 项目结构

```
SoulPlayer/
├── run.py                  # 启动器（依赖检查 + 拉起主程序）
├── main.py                 # 主程序：UI、主循环、交互逻辑
├── motion_capture.py       # 动作捕捉（MediaPipe + 特征提取 + 挥手/手掌悬停判定）
├── gesture_presets.py      # 动作预设引擎（条件匹配 + 自定义屏蔽内置动作）
├── music_player.py         # 音乐播放引擎（播放列表 + 动态文件夹同步）
├── custom_preset.py        # 自定义动作预设（录制 + 存储 + 随机选曲）
├── voice_match.py          # 语音模板比对（MFCC + DTW，录制 + 监听）
├── ui_theme.py / ui_animations.py  # 界面主题与动画
├── pose_landmarker_full.task       # MediaPipe 姿态模型
├── presets/default_presets.json    # 内置动作预设
├── tests/                  # 仿真单元测试（无需摄像头即可运行）
├── music/                  # 放你的歌曲（.gitignore 忽略内容）
└── build_exe.py            # PyInstaller 打包脚本
```

## 📄 开源协议

MIT License，详见 [LICENSE](LICENSE)。
