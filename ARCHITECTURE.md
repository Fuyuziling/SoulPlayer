# SoulPlayer - 项目架构设计

## 目录结构

```
SoulPlayer/
├── main.py                 # 主程序入口，整合所有模块，UI+主循环
├── motion_capture.py       # 动作捕捉模块（MediaPipe + 特征提取）
├── gesture_presets.py      # 动作预设系统（预设定义 + 匹配引擎）
├── music_player.py         # 音乐播放引擎（播放控制 + 切换 + 过渡）
├── requirements.txt        # 依赖清单
└── presets/
    └── default_presets.json  # 默认动作预设配置
```

## 模块间接口契约

### 1. motion_capture.py → 对外暴露

```python
class MotionCapture:
    def __init__(self, camera_id=0, model_complexity=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化动作捕捉器
        model_complexity: 0=Lite(快), 1=Full(平衡), 2=Heavy(精)
        """
        pass
    
    def start(self):
        """启动摄像头和推理线程"""
        pass
    
    def stop(self):
        """停止摄像头和推理线程"""
        pass
    
    def get_latest_motion_features(self) -> dict:
        """
        获取最新的动作特征向量（线程安全）
        返回: {
            'timestamp': float,           # 时间戳
            'motion_energy': float,       # 0.0~1.0, 整体运动能量
            'body_speed': float,          # 0.0~1.0, 身体质心移动速度
            'arm_amplitude': float,       # 0.0~1.0, 手臂摆动幅度
            'body_rhythm': float,         # 0.0~1.0, 身体起伏/节奏性
            'posture': str,               # 'standing', 'sitting', 'lying', 'unknown'
            'hand_position': str,         # 'up', 'down', 'center', 'unknown'
            'is_active': bool,            # 是否在活动（区别于静止）
            'landmarks': list,            # 33个关键点坐标（可选，用于调试/可视化）
        }
        """
        pass
    
    def get_camera_frame(self) -> np.ndarray:
        """获取当前摄像头帧（用于UI预览）"""
        pass
    
    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        pass
```

### 2. gesture_presets.py → 对外暴露

```python
class GesturePreset:
    """单个动作预设定义"""
    def __init__(self, name: str, description: str, 
                 conditions: dict, action: str, priority: int = 0):
        """
        name: 预设名称
        description: 描述
        conditions: 触发条件
            {
                'motion_energy': {'min': float, 'max': float},
                'body_speed': {'min': float, 'max': float},
                'arm_amplitude': {'min': float, 'max': float},
                'body_rhythm': {'min': float, 'max': float},
                'posture': [str],           # 允许的posture列表
                'hand_position': [str],     # 允许的hand_position列表
                'is_active': bool,          # 是否要求处于活动状态
                'duration': float,          # 需要持续满足条件的秒数
            }
        action: 触发动作类型
        priority: 优先级（数值越大优先级越高）
        """
        pass


class PresetEngine:
    """预设匹配引擎"""
    def __init__(self):
        self.presets = []
        self.active_preset = None
        self.hold_timers = {}  # 记录每个预设的持续满足时间
    
    def load_presets(self, filepath: str):
        """从JSON文件加载预设"""
        pass
    
    def save_presets(self, filepath: str):
        """保存预设到JSON文件"""
        pass
    
    def add_preset(self, preset: GesturePreset):
        """添加预设"""
        pass
    
    def remove_preset(self, name: str):
        """删除预设"""
        pass
    
    def match(self, motion_features: dict) -> str | None:
        """
        根据动作特征匹配预设，返回触发的action
        如果没有任何预设匹配，返回None
        如果有多个匹配，按priority最高的返回
        匹配需要考虑duration阈值（需要持续满足条件一段时间才触发）
        """
        pass
    
    def get_preset_list(self) -> list:
        """获取所有预设列表（用于UI展示）"""
        pass
```

### 3. music_player.py → 对外暴露

```python
class MusicPlayer:
    def __init__(self):
        """初始化播放器，支持多种后端"""
        pass
    
    def load_playlist(self, folder_path: str):
        """从文件夹加载音乐播放列表"""
        pass
    
    def load_playlist_from_spotify(self, playlist_id: str, token: str):
        """从Spotify加载播放列表（可选）"""
        pass
    
    def play(self, track_index=None):
        """播放/继续播放，可选指定曲目"""
        pass
    
    def pause(self):
        """暂停"""
        pass
    
    def stop(self):
        """停止"""
        pass
    
    def next_track(self):
        """下一首"""
        pass
    
    def previous_track(self):
        """上一首"""
        pass
    
    def set_volume(self, volume: float):
        """设置音量 0.0~1.0"""
        pass
    
    def crossfade_to(self, track_index: int, duration_ms: int = 2000):
        """
        淡入淡出切换到指定曲目
        duration_ms: 过渡时间（毫秒）
        """
        pass
    
    def set_playlist_by_mood(self, mood: str):
        """
        根据情绪/模式切换播放列表
        mood: 'calm', 'energetic', 'happy', 'focus', 'party'等
        """
        pass
    
    @property
    def is_playing(self) -> bool:
        pass
    
    @property
    def current_track(self) -> dict:
        """返回当前曲目信息"""
        pass
```

### 4. 预设配置JSON格式 (default_presets.json)

```json
{
    "presets": [
        {
            "name": "安静模式",
            "description": "身体几乎不动时播放舒缓音乐",
            "conditions": {
                "motion_energy": {"min": 0.0, "max": 0.15},
                "body_speed": {"min": 0.0, "max": 0.1},
                "is_active": false,
                "duration": 3.0
            },
            "action": "mood_calm",
            "priority": 1
        },
        {
            "name": "轻度活动",
            "description": "轻微动作时播放轻松音乐",
            "conditions": {
                "motion_energy": {"min": 0.15, "max": 0.4},
                "body_speed": {"min": 0.1, "max": 0.35},
                "is_active": true,
                "duration": 2.0
            },
            "action": "mood_relaxed",
            "priority": 2
        },
        {
            "name": "活跃模式",
            "description": "中等动作活跃时播放流行/动感音乐",
            "conditions": {
                "motion_energy": {"min": 0.4, "max": 0.7},
                "body_speed": {"min": 0.35, "max": 0.6},
                "is_active": true,
                "duration": 2.0
            },
            "action": "mood_energetic",
            "priority": 3
        },
        {
            "name": "跳动模式",
            "description": "大幅跳动/舞动时播放高强度电子乐",
            "conditions": {
                "motion_energy": {"min": 0.7, "max": 1.0},
                "body_speed": {"min": 0.6, "max": 1.0},
                "body_rhythm": {"min": 0.3, "max": 1.0},
                "is_active": true,
                "duration": 1.5
            },
            "action": "mood_party",
            "priority": 4
        },
        {
            "name": "举手暂停",
            "description": "双手举起时暂停播放",
            "conditions": {
                "hand_position": ["up"],
                "motion_energy": {"min": 0.0, "max": 0.5},
                "duration": 1.0
            },
            "action": "player_pause",
            "priority": 10
        },
        {
            "name": "挥手切歌",
            "description": "单手快速挥动时切换到下一首",
            "conditions": {
                "motion_energy": {"min": 0.3, "max": 0.8},
                "arm_amplitude": {"min": 0.4, "max": 1.0},
                "duration": 0.5
            },
            "action": "player_next",
            "priority": 10
        }
    ]
}
```

## 主循环逻辑

```python
# 每帧（或每N帧）执行：
motion_features = motion_capture.get_latest_motion_features()
action = preset_engine.match(motion_features)

if action:
    handle_action(action, music_player)

# action 处理：
def handle_action(action, player):
    if action.startswith("mood_"):
        player.set_playlist_by_mood(action.replace("mood_", ""))
    elif action == "player_pause":
        player.pause()
    elif action == "player_play":
        player.play()
    elif action == "player_next":
        player.next_track()
    elif action == "player_prev":
        player.previous_track()
    # ... etc
```

## 依赖

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pygame>=2.5.0  # 用于音乐播放和简单UI
```
