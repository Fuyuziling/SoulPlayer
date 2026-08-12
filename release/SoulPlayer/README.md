# SoulPlayer

> 基于 AI 肢体动作捕捉与语音台词触发的智能音乐播放器。让身体成为遥控器，让音乐随律动自然流淌。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platform">
</p>

---

## ✨ 核心功能

- **🎥 动作感知播放** — 通过 MediaPipe 实时检测人体 33 个关键点，自动根据你的动作幅度、姿态、手位切换音乐风格
- **🎤 语音台词触发** — 说出"灵魂觉醒""静心""下一首"等中文台词即可控制播放，离线识别，无需联网
- **🎨 自定义动作预设** — 录制专属动作并绑定任意歌单，实现千人千面的交互体验
- **🖥️ 实时可视化 UI** — Pygame 渲染的图形界面，实时显示摄像头画面、运动特征与触发状态
- **🔒 完全离线** — 所有 AI 推理均在本地完成，无需联网，保护隐私

---

## 📦 安装

### 环境要求

- Python 3.9+
- 摄像头（内置或外接）
- 麦克风（语音功能可选）

### 快速安装

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/SoulPlayer.git
cd SoulPlayer

# 2. 安装依赖
pip install -r requirements.txt
```

### Windows 一键启动

双击 `run.bat` 即可启动程序。

---

## 🚀 使用指南

### 启动程序

```bash
python run.py
```

### 动作交互

| 你的动作 | 播放器的回应 |
|---------|-------------|
| 静坐，几乎不动 | 播放舒缓音乐 🍃 |
| 轻微活动、摇头晃脑 | 播放轻松音乐 🌤️ |
| 起身走动、中等幅度动作 | 播放流行/动感音乐 🔥 |
| 大幅跳动、舞动 | 播放派对电子乐 🎉 |
| 🙌 双手举过头顶 | 暂停播放 ⏸️ |
| 👋 挥手 | 下一首 ⏭️ |

### 语音交互（录制你的声音作为台词模板）

按 `R` 录制台词声音模板，按 `V` 开启实时监听；当你说出与模板匹配的声音时触发对应动作。

| 你说 | 播放器执行 |
|------|-----------|
| "灵魂觉醒" | 切换动感模式 |
| "静心" | 切换安静模式 |
| "播放" / "暂停" | 播放 / 暂停 |
| "下一首" / "上一首" | 切歌 |
| "快乐" / "专注" / "放松" / "派对" | 切换对应情绪歌单 |

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `Space` | 播放 / 暂停 |
| `←` / `→` | 上一首 / 下一首 |
| `↑` / `↓` | 音量调节 |
| `R` | 录制自定义动作预设 |
| `V` | 切换语音监听开关 |
| `ESC` / `Q` | 退出 |

### 添加本地音乐

将你的 `.mp3` / `.wav` / `.ogg` / `.flac` 文件放入 `music/` 目录下的对应文件夹即可：

```
music/
├── calm/          # 舒缓音乐
├── relaxed/       # 轻松音乐
├── energetic/     # 动感音乐
├── party/         # 派对音乐
├── happy/         # 快乐音乐
├── focus/         # 专注音乐
└── ...
```

---

## 🏗️ 项目结构

```
SoulPlayer/
├── main.py                  # 主程序入口，整合所有模块
├── motion_capture.py        # 动作捕捉模块（MediaPipe + 特征提取）
├── gesture_presets.py       # 预设定义与匹配引擎
├── music_player.py          # 音频播放引擎
├── voice_match.py            # 语音模板比对触发引擎（MFCC+DTW）
├── custom_preset.py         # 自定义预设录制管理
├── ui_theme.py              # UI 主题与渲染器
├── ui_animations.py         # 动画系统
├── run.py                   # 启动器（带依赖检查与错误捕获）
├── run.bat                  # Windows 一键启动脚本
├── requirements.txt         # Python 依赖清单
├── presets/
│   └── default_presets.json # 默认动作预设配置
├── voice_templates.json      # 语音模板清单
├── pose_landmarker_full.task # MediaPipe 姿态估计模型
├── ARCHITECTURE.md          # 项目架构设计文档
└── README.md                # 本文件
```

---

## 🧠 技术亮点

### 自适应归一化算法

不同用户的动作幅度差异很大。SoulPlayer 采用自适应归一化策略，动态学习用户的"最大动作幅度"，无需手动校准：

```python
self._max_energy = max(self._max_energy * 0.995, raw_energy, 0.001)
motion_energy = raw_energy / self._max_energy  # 自动归一化到 [0, 1]
```

### 7 维动作特征向量

| 特征 | 说明 |
|------|------|
| `motion_energy` | 整体运动能量 |
| `body_speed` | 身体质心移动速度 |
| `arm_amplitude` | 手臂摆动幅度 |
| `body_rhythm` | 身体起伏节奏性 |
| `posture` | 坐 / 站 / 躺 |
| `hand_position` | 手在上 / 下 / 中间 |
| `is_active` | 是否处于活动状态 |

---

## 🛠️ 技术栈

| 模块 | 技术 |
|------|------|
| 姿态估计 | [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |
| 语音识别 | [MFCC+DTW 模板比对](voice_match.py) |
| 音频播放 | pygame.mixer |
| GUI 渲染 | pygame |
| 摄像头 | OpenCV |
| 数值计算 | NumPy |

---

## 📝 自定义预设

按 `R` 键即可录制你的专属动作：

1. 按 `R` → 开始录制（约 3 秒）
2. 做出你想绑定的动作
3. 在弹窗中选择关联的歌曲
4. 完成！下次做出该动作即可触发播放

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

> 💡 **提示**：本项目处于活跃开发中，欢迎提交 Issue 和 PR！
