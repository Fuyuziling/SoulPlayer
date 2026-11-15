"""
motion_capture.py —— 动作捕捉模块

基于 MediaPipe Pose 实现人体关键点检测与运动特征提取，
为「SoulPlayer——基于肢体动作捕捉的智能音乐播放器」提供实时动作数据。

功能：
- 摄像头视频流捕获（OpenCV）
- 人体 33 关键点检测（MediaPipe）
- 运动特征提取（能量、速度、幅度、节奏、姿态、手位置、活动状态）
- 线程安全的特征读取接口

对外暴露类：MotionCapture
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from typing import Optional, List, Dict, Any

# 导入挥手动作学习系统
from wave_learning import WaveLearningManager, create_wave_learning_manager
from wave_learning_rules import validate_and_filter_trigger


# ──────────────────────── 关键点索引常量 ────────────────────────
# MediaPipe Pose 关键点编号（33 点）
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


# ──────────────────────── 滤波工具（挥手/举手共用） ────────────────────────
def _median3(vals):
    """3 点中值滤波：彻底消除单帧脉冲尖峰（估计值偶发跳变）。"""
    if len(vals) < 3:
        return list(vals)
    out = [vals[0]]
    for i in range(1, len(vals) - 1):
        out.append(sorted((vals[i - 1], vals[i], vals[i + 1]))[1])
    out.append(vals[-1])
    return out


def _mean3(vals):
    """3 点均值滤波（端点也与邻点平均）：抑制持续性的周期抖动——
    中值滤波无法滤掉连续来回摆动（每帧都是"真实"极值），
    均值滤波将 ±A 的持续摆动压到 ±A/3；对单方向真实挥手，
    仅端点内收约 1 帧步长，跨度基本保留。"""
    n = len(vals)
    if n < 2:
        return list(vals)
    out = [(vals[0] + vals[1]) / 2.0]
    for i in range(1, n - 1):
        out.append((vals[i - 1] + vals[i] + vals[i + 1]) / 3.0)
    out.append((vals[n - 2] + vals[n - 1]) / 2.0)
    return out


class MotionCapture:
    """
    动作捕捉器：封装摄像头读取、MediaPipe 推理与运动特征提取。

    所有对外接口均为线程安全，内部推理在独立后台线程运行，
    主线程通过 get_latest_motion_features() 等非阻塞方式读取数据。
    """

    def __init__(
        self,
        camera_id: int = 0,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        初始化动作捕捉器。

        Args:
            camera_id: 摄像头设备索引（默认 0，通常为内置摄像头）。
            model_complexity: MediaPipe Pose 模型复杂度
                0 = Lite（最快，精度最低）
                1 = Full（平衡，推荐）
                2 = Heavy（最慢，精度最高）
            min_detection_confidence: 人体检测置信度阈值。
            min_tracking_confidence: 关键点跟踪置信度阈值。
        """
        # ── 基本参数 ──
        self.camera_id = camera_id
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # ── MediaPipe 版本兼容性检测与 API 选择 ──
        self._use_dummy = False
        self._use_tasks_api = False
        self._timestamp_ms = 0
        self.detector = None
        self.hand_detector = None  # HandLandmarker（每只手 21 点，精细手型识别）
        self.face_detector = None  # FaceLandmarker（面部 478 点，表情/头部姿态）
        self._temp_model_path: Optional[str] = None  # 临时复制的模型路径，需在 stop 时清理
        self._temp_hand_model_path: Optional[str] = None  # 手部模型临时 ASCII 路径
        self._temp_face_model_path: Optional[str] = None  # 面部模型临时 ASCII 路径

        # 优先尝试新 API (MediaPipe 0.10.30+ Tasks API)
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # 模型文件路径：优先使用纯 ASCII 路径（避免中文路径问题）
            # MediaPipe 底层 C++ 库可能无法正确处理包含中文的路径
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            project_model_path = os.path.join(_script_dir, 'pose_landmarker_full.task')
            
            # 尝试多种可能的纯 ASCII 备选路径
            ascii_candidates = [
                os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Temp', 'pose_landmarker_full.task'),
                os.path.join(tempfile.gettempdir(), 'pose_landmarker_full.task'),
            ]

            model_path = None
            # 先检查 ASCII 备选路径
            for candidate in ascii_candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            
            # 再检查项目目录
            if model_path is None and os.path.exists(project_model_path):
                # 如果项目路径包含中文，复制到临时 ASCII 路径
                if not all(ord(c) < 128 for c in project_model_path):
                    import shutil
                    temp_ascii_path = ascii_candidates[1]  # 使用 tempfile 路径
                    try:
                        shutil.copy2(project_model_path, temp_ascii_path)
                        model_path = temp_ascii_path
                        self._temp_model_path = temp_ascii_path
                        print(f"[MotionCapture] 已将模型复制到临时 ASCII 路径: {model_path}")
                    except Exception as copy_err:
                        print(f"[MotionCapture] 复制模型到临时路径失败: {copy_err}")
                        model_path = project_model_path  # 回退尝试原路径
                else:
                    model_path = project_model_path

            if model_path is None:
                print(f"[MotionCapture] 警告: 模型文件未找到，将使用模拟模式。")
                print(f"[MotionCapture] 搜索路径: {project_model_path}")
                self._use_dummy = True
            elif not os.path.exists(model_path):
                print(f"[MotionCapture] 警告: 模型路径不存在，将使用模拟模式。")
                self._use_dummy = True
            else:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=min_detection_confidence,
                    min_pose_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.detector = vision.PoseLandmarker.create_from_options(options)
                self._use_tasks_api = True
                self._use_dummy = False
                print(f"[MotionCapture] 使用新 API (MediaPipe Tasks) 加载模型。")

                # ── 手部检测器（HandLandmarker，每只手 21 点，精细手型识别） ──
                # 手部模型可选：加载失败仅降级（手型识别不可用，不影响 Pose 主流程）
                try:
                    _hand_project = os.path.join(_script_dir, 'hand_landmarker.task')
                    _hand_ascii = os.path.join(tempfile.gettempdir(), 'hand_landmarker.task')
                    _hand_path = None
                    for _cand in (_hand_ascii, _hand_project):
                        if os.path.exists(_cand):
                            _hand_path = _cand
                            break
                    if _hand_path is None:
                        print("[MotionCapture] 手部模型 hand_landmarker.task 未找到，手型识别不可用")
                    else:
                        if not all(ord(c) < 128 for c in _hand_path):
                            import shutil
                            shutil.copy2(_hand_path, _hand_ascii)
                            _hand_path = _hand_ascii
                            self._temp_hand_model_path = _hand_ascii
                        _hand_base = python.BaseOptions(model_asset_path=_hand_path)
                        _hand_options = vision.HandLandmarkerOptions(
                            base_options=_hand_base,
                            running_mode=vision.RunningMode.VIDEO,
                            num_hands=2,
                            min_hand_detection_confidence=0.5,
                            min_hand_presence_confidence=0.5,
                            min_tracking_confidence=0.5,
                        )
                        self.hand_detector = vision.HandLandmarker.create_from_options(_hand_options)
                        print("[MotionCapture] 手部检测器已加载 (HandLandmarker 21 点/手)")
                except Exception as _he:
                    self.hand_detector = None
                    print(f"[MotionCapture] 手部检测器加载失败，手型识别不可用: {_he}")

                # ── 面部检测器（FaceLandmarker，478 点，表情/头部姿态识别） ──
                # 面部模型可选：加载失败仅降级（面部识别不可用，不影响 Pose 主流程）
                try:
                    _face_project = os.path.join(_script_dir, 'face_landmarker.task')
                    _face_ascii = os.path.join(tempfile.gettempdir(), 'face_landmarker.task')
                    _face_path = None
                    for _cand in (_face_ascii, _face_project):
                        if os.path.exists(_cand):
                            _face_path = _cand
                            break
                    if _face_path is None:
                        print("[MotionCapture] 面部模型 face_landmarker.task 未找到，面部识别不可用")
                    else:
                        if not all(ord(c) < 128 for c in _face_path):
                            import shutil
                            shutil.copy2(_face_path, _face_ascii)
                            _face_path = _face_ascii
                            self._temp_face_model_path = _face_ascii
                        _face_base = python.BaseOptions(model_asset_path=_face_path)
                        _face_options = vision.FaceLandmarkerOptions(
                            base_options=_face_base,
                            running_mode=vision.RunningMode.VIDEO,
                            num_faces=1,
                            min_face_detection_confidence=0.5,
                            min_face_presence_confidence=0.5,
                            min_tracking_confidence=0.5,
                            output_face_blendshapes=True,
                        )
                        self.face_detector = vision.FaceLandmarker.create_from_options(_face_options)
                        print("[MotionCapture] 面部检测器已加载 (FaceLandmarker 478 点)")
                except Exception as _fe:
                    self.face_detector = None
                    print(f"[MotionCapture] 面部检测器加载失败，面部识别不可用: {_fe}")
        except Exception as e:
            # 新 API 加载失败，尝试旧 API 或降级到模拟模式
            print(f"[MotionCapture] 新 API 加载失败 ({e})，尝试旧 API...")
            if hasattr(mp, 'solutions'):
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                print(f"[MotionCapture] 使用旧 API (MediaPipe Solutions)。")
            else:
                print("[MotionCapture] 警告: MediaPipe 版本不兼容，将使用模拟模式。")
                self._use_dummy = True

        # ── 摄像头对象 ──
        self._cap: Optional[cv2.VideoCapture] = None

        # ── 线程与同步 ──
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running_flag = False

        # ── 最新数据缓存（受 _lock 保护） ──
        self._latest_features: Dict[str, Any] = self._get_default_features()
        self._latest_frame: Optional[np.ndarray] = None

        # ── 历史数据（用于计算时序特征） ──
        # 上一帧关键点（33 个点的 ndarray，shape=(33, 4) 对应 x,y,z,visibility）
        self._prev_landmarks: Optional[np.ndarray] = None
        # 上一帧时间戳
        self._prev_time: Optional[float] = None
        # 质心（躯干中心）y 坐标历史，用于节奏检测
        self._centroid_y_history: deque = deque(maxlen=60)
        # 运动能量历史，用于自适应归一化
        self._energy_history: deque = deque(maxlen=30)
        # 手臂幅度历史，用于自适应归一化
        self._arm_amp_history: deque = deque(maxlen=30)
        # 手臂独立运动能量历史（与头部/身体解耦，用于举手/挥手识别）
        self._arm_motion_history: deque = deque(maxlen=30)
        # 手腕水平运动历史（只算x轴，用于挥手识别）
        self._arm_hmotion_history: deque = deque(maxlen=30)
        # 左右手独立的历史队列（用于挥手/举手检测：记录最近约0.67秒的x,y坐标）。
        # 必须独立——共用同一队列时左右手条目交替挤占窗口，单侧仅保留约半数
        # 帧，挥手回程会被截断，导致往返翻转统计失效。
        # 窗口 20 帧：覆盖正常挥手（约1.5Hz）一次完整"去-回"周期，
        # 慢挥手（1Hz）也能截到 1 次往返，避免"正常挥手不达标"
        self._wrist_xy_history: dict = {
            'L': deque(maxlen=20),
            'R': deque(maxlen=20),
        }
        # 肘部坐标历史（用于挥手检测）：
        # 手指动作时 MediaPipe 对手腕(15/16)的估计会被指根污染而漂移，
        # 但肘部(13/14)远离手部，手指动作绝不会影响肘部坐标。
        # 挥手是整条手臂（含肘部）的横向摆动，肘部位移足够大；
        # 因此挥手检测追踪肘部，彻底免疫手指动作。左右手独立队列。
        self._elbow_xy_history: dict = {
            'L': deque(maxlen=20),
            'R': deque(maxlen=20),
        }
        # 挥手方向（用户视角，用户明确要求）：+1 = 向左挥手（下一首），
        # -1 = 向右挥手（上一首），0 = 无有效挥手。注意：MediaPipe 在原始
        # 图像坐标上推理，原始 x 增大的画面经镜像显示后是"手向左"（用户
        # 向左挥手）→ +1；原始 x 减小 → 用户向右挥手 → -1。
        # 方向：事件级锁存。挥手事件起点锁存"窗口内首段显著位移"方向
        #（挥出方向），事件期间恒定输出——回程/窗口滑动不改写方向，
        # 配合 PresetEngine 的 _wave_event_fired 锁存保证"一次挥手事件
        # 只触发一次切歌、同一方向恒触发同一首"（修复同向挥手串触发）。
        # 事件状态：_wave_event_active 事件进行中；_wave_event_dir 锁定方向；
        # _wave_still_frames 手臂连续静止帧计数（事件结束判定）。
        self._wave_direction: float = 0.0
        self._wave_event_active: bool = False
        self._wave_event_dir: float = 0.0
        self._wave_still_frames: int = 0
        # 视觉特征数据（用于白名单学习和数据库记录）
        self._latest_wave_x_span: float = 0.0
        self._latest_wave_ang_span: float = 0.0
        # 身体速度历史，用于自适应归一化
        self._speed_history: deque = deque(maxlen=30)

        # ── 自适应归一化基准（历史最大值） ──
        self._max_energy = 0.01
        self._max_speed = 0.01
        self._max_arm_amp = 0.01
        self._max_arm_motion = 0.01
        self._max_arm_hmotion = 0.01

        # ── 调试标志 ──
        self._debug = False

        # ── 挥手动作学习系统（P1+P2） ──
        # 解决单手模式下某个方向触发率低的问题
        self._wave_learning_enabled = True
        self._wave_learning_manager: Optional[WaveLearningManager] = None
        self._visual_whitelist = None  # 视觉特征白名单
        self._init_wave_learning()

    # ═══════════════════════════════════════════════════════
    # 挥手动作学习系统
    # ═══════════════════════════════════════════════════════

    def _init_wave_learning(self):
        """初始化挥手动作学习系统"""
        if not self._wave_learning_enabled:
            print("[MotionCapture] 学习系统未启用（_wave_learning_enabled=False）")
            return

        try:
            # 数据目录：与预设/日志同根（由 app_paths 决定开发态/便携/安装版）。
            # 旧版写在 ~/.soulplayer/learning_data，首次运行自动迁移。
            try:
                import app_paths
                data_dir = str(app_paths.migrate_legacy_home_data("learning_data"))
            except Exception:
                data_dir = os.path.join(os.path.expanduser('~'), '.soulplayer', 'learning_data')
            self._wave_learning_manager = create_wave_learning_manager({
                "data_dir": data_dir,
                "enable_p1": True,
                "enable_p2": True
            })
            # 始终输出学习系统状态（不依赖 _debug 标志）
            print("[MotionCapture] 挥手动作学习系统已启用（P1+P2）")
            if self._debug:
                print(f"[学习系统] 数据目录: {data_dir}")
                print(f"[学习系统] P1 自适应阈值: 启用")
                print(f"[学习系统] P2 幅度归一化: 启用")

            # 初始化视觉特征白名单
            from wave_learning_rules import get_visual_whitelist
            self._visual_whitelist = get_visual_whitelist()
            print("[MotionCapture] 视觉特征白名单已启用")
        except Exception as e:
            print(f"[MotionCapture] 挥手学习系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self._wave_learning_enabled = False

    # ═══════════════════════════════════════════════════════
    # 公共接口
    # ═══════════════════════════════════════════════════════

    def start(self) -> None:
        """
        启动摄像头和推理线程。

        如果摄像头无法打开，会抛出 RuntimeError。
        """
        if self._running_flag:
            return

        # ── 尝试打开摄像头（模拟模式下也需要摄像头画面） ──
        self._cap = cv2.VideoCapture(self.camera_id)
        camera_available = self._cap.isOpened()

        if not camera_available:
            # 摄像头不可用：如果是模拟模式，降级到纯模拟（无摄像头画面）
            if self._use_dummy:
                self._cap = None
                self._running_flag = True
                self._thread = threading.Thread(target=self._dummy_loop, daemon=True)
                self._thread.start()
                print("[MotionCapture] 模拟模式已启动（无摄像头）。")
                return
            # 非模拟模式且摄像头不可用：报错
            raise RuntimeError(
                f"无法打开摄像头设备 (camera_id={self.camera_id})，"
                "请检查摄像头连接或更换 camera_id。"
            )

        # 摄像头已打开：设置分辨率（提高清晰度）
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        # 启动推理线程
        self._running_flag = True
        if self._use_dummy:
            # 模拟模式：用摄像头画面 + 模拟数据
            self._thread = threading.Thread(target=self._dummy_loop, daemon=True)
            print("[MotionCapture] 模拟模式已启动（摄像头画面 + 模拟数据）。")
        else:
            self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def _dummy_loop(self) -> None:
        """
        模拟模式后台循环：读取摄像头画面（如有）+ 生成正弦波模拟数据。
        """
        t = 0.0
        while self._running_flag:
            current_time = time.time()

            # ── 尝试读取摄像头帧 ──
            frame = None
            if self._cap is not None and self._cap.isOpened():
                ret, raw_frame = self._cap.read()
                if ret and raw_frame is not None:
                    frame = cv2.flip(raw_frame, 1)

            # ── 正弦波模拟各特征 ──
            motion_energy = (np.sin(t * 2.0) + 1.0) / 2.0
            body_speed = (np.sin(t * 1.5 + 1.0) + 1.0) / 2.0
            arm_amplitude = (np.sin(t * 1.2 + 2.0) + 1.0) / 2.0
            body_rhythm = (np.sin(t * 3.0 + 0.5) + 1.0) / 2.0

            # 姿态和手位置模拟
            posture_cycle = int(t / 5) % 3
            posture = ['standing', 'sitting', 'lying'][posture_cycle]
            hand_cycle = int(t / 3) % 3
            hand_position = ['up', 'down', 'center'][hand_cycle]
            is_active = motion_energy > 0.3

            # 模拟 landmarks（33个空点）
            landmarks = []
            for i in range(33):
                landmarks.append({
                    'x': 0.5 + 0.1 * np.sin(t + i * 0.2),
                    'y': 0.5 + 0.1 * np.cos(t + i * 0.2),
                    'z': 0.0,
                    'visibility': 0.8,
                })

            features = {
                'timestamp': current_time,
                'motion_energy': float(motion_energy),
                'arm_motion_energy': float(motion_energy),  # 模拟模式下复用
                'arm_horizontal_motion': float(arm_amplitude),  # 模拟模式下复用
                'body_speed': float(body_speed),
                'arm_amplitude': float(arm_amplitude),
                'body_rhythm': float(body_rhythm),
                'posture': posture,
                'hand_position': hand_position,
                'is_active': bool(is_active),
                'landmarks': landmarks,
            }

            # ── 帧处理 ──
            if frame is None:
                # 无摄像头：生成模拟帧
                frame = np.zeros((600, 800, 3), dtype=np.uint8)
                stripe_color = int((motion_energy * 255))
                frame[:, ::20] = [stripe_color, 255 - stripe_color, 128]
            # 有摄像头：叠加模拟模式提示
            cv2.putText(frame, "SIMULATION MODE (no MediaPipe)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"motion_energy: {motion_energy:.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.putText(frame, f"posture: {posture}  |  hand: {hand_position}", (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            with self._lock:
                self._latest_features = features
                self._latest_frame = frame

            t += 0.033
            time.sleep(0.033)  # 约 30 FPS

    @staticmethod
    def _close_detector(detector, name: str, timeout: float = 0.5) -> None:
        """
        带超时地关闭 MediaPipe detector。

        断网/网络受限环境下，MediaPipe 关闭时会触发 telemetry（clearcut）
        上报，连接失败要重试很久——实测每个 detector.close() 都会阻塞 25 秒以上，
        三个 detector 累计 75 秒，直接导致"点了关闭却关不掉"。
        这里改为后台线程限时关闭：正常网络下瞬间完成，异常时到点即放弃
        （进程退出后由操作系统回收，不会有实际影响）。
        """
        if detector is None:
            return

        def _do_close():
            try:
                detector.close()
            except Exception:  # noqa: BLE001
                pass

        th = threading.Thread(target=_do_close, name=f"close-{name}", daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            print(f"[MotionCapture] {name} detector 关闭超时（{timeout}s），放弃等待")

    def stop(self) -> None:
        """
        停止摄像头和推理线程，释放资源。
        """
        self._running_flag = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # 释放 detector 资源（全部走带超时的关闭，见 _close_detector 说明）
        if self._use_tasks_api and self.detector is not None:
            self._close_detector(self.detector, "pose")

        # 释放手部检测器资源
        if self._use_tasks_api and self.hand_detector is not None:
            self._close_detector(self.hand_detector, "hand")
            self.hand_detector = None

        # 释放面部检测器资源
        if self._use_tasks_api and self.face_detector is not None:
            self._close_detector(self.face_detector, "face")
            self.face_detector = None

        # 释放旧 API Pose 资源
        if not self._use_tasks_api and hasattr(self, 'pose') and self.pose is not None:
            self._close_detector(self.pose, "legacy-pose")

        # 重置历史数据
        self._prev_landmarks = None
        self._prev_time = None
        self._centroid_y_history.clear()
        self._energy_history.clear()
        self._arm_amp_history.clear()
        self._arm_motion_history.clear()
        self._arm_hmotion_history.clear()
        for _h in self._wrist_xy_history.values():
            _h.clear()
        for _h in self._elbow_xy_history.values():
            _h.clear()
        self._speed_history.clear()
        # 重置挥手方向与事件状态（防止下次 start 携带旧方向/旧事件）
        self._wave_direction = 0.0
        self._wave_event_active = False
        self._wave_event_dir = 0.0
        self._wave_still_frames = 0

        with self._lock:
            self._latest_features = self._get_default_features()
            self._latest_frame = None

        # 清理临时复制的模型文件
        if self._temp_model_path is not None:
            try:
                if os.path.exists(self._temp_model_path):
                    os.remove(self._temp_model_path)
                    print(f"[MotionCapture] 已清理临时模型文件: {self._temp_model_path}")
            except Exception:
                pass
            self._temp_model_path = None

        # 清理临时复制的面部模型文件
        if self._temp_face_model_path is not None:
            try:
                if os.path.exists(self._temp_face_model_path):
                    os.remove(self._temp_face_model_path)
                    print(f"[MotionCapture] 已清理临时面部模型文件: {self._temp_face_model_path}")
            except Exception:
                pass
            self._temp_face_model_path = None

    @property
    def is_running(self) -> bool:
        """是否正在运行（摄像头已打开且推理线程存活）。"""
        return self._running_flag and self._thread is not None and self._thread.is_alive()

    def get_latest_motion_features(self) -> Dict[str, Any]:
        """
        获取最新的动作特征向量（线程安全）。

        返回字典结构如下：
        {
            'timestamp': float,           # 时间戳（秒，time.time()）
            'motion_energy': float,       # 0.0~1.0, 整体运动能量（头部解耦）
            'arm_motion_energy': float,   # 0.0~1.0, 手臂独立运动能量（与头部/身体解耦）
            'body_speed': float,          # 0.0~1.0, 身体质心移动速度
            'arm_amplitude': float,       # 0.0~1.0, 手臂摆动/伸展幅度
            'body_rhythm': float,         # 0.0~1.0, 身体起伏/节奏性
            'posture': str,               # 'standing', 'sitting', 'lying', 'unknown'
            'hand_position': str,         # 'up', 'down', 'center', 'unknown'
            'is_active': bool,            # 是否在活动（区别于静止）
            'landmarks': list,            # 33 个关键点坐标（用于调试/可视化）
        }
        """
        with self._lock:
            # 返回深拷贝，防止外部修改内部状态
            return self._deep_copy_features(self._latest_features)

    def get_camera_frame(self) -> Optional[np.ndarray]:
        """
        获取当前摄像头帧（用于 UI 预览）。

        Returns:
            最新一帧 BGR 图像（numpy ndarray），如果尚未捕获到帧则返回 None。
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    # ═══════════════════════════════════════════════════════
    # 挥手动作学习系统接口
    # ═══════════════════════════════════════════════════════

    def get_wave_learning_threshold(self, hand: str, direction: float):
        """获取挥手触发阈值（供gesture_presets.py查询）

        Args:
            hand: 手别 'L' 或 'R'
            direction: 方向 +1.0 或 -1.0

        Returns:
            (x_threshold, ang_threshold) - 触发阈值
        """
        if self._wave_learning_enabled and self._wave_learning_manager is not None:
            x_th, ang_th, _ = self._wave_learning_manager.get_threshold(hand, direction)
            return x_th, ang_th
        return 0.06, 0.20  # 默认值

    def get_wave_learning_summary(self) -> Dict:
        """获取学习摘要（用于UI显示/调试）

        Returns:
            学习统计摘要字典
        """
        if self._wave_learning_enabled and self._wave_learning_manager is not None:
            return self._wave_learning_manager.get_learning_summary()
        return {}

    def record_wave_result(self, hand: str, direction: float, triggered: bool,
                          confidence: float = 1.0, action: str = "",
                          wave_direction: float = 0.0,
                          arm_motion: float = 0.0,
                          x_span: float = 0.0,
                          ang_span: float = 0.0):
        """记录挥手触发结果（供gesture_presets.py调用）

        Args:
            hand: 手别 'L' 或 'R'
            direction: 方向 +1.0 或 -1.0
            triggered: 是否成功触发动作
            confidence: 触发置信度（0~1），低置信度的触发可能被当作"失败"处理
            action: 触发的动作名称（如 'player_next', 'player_prev'），用于白名单验证
            wave_direction: 实际检测到的方向幅度（用于视觉特征白名单）
            arm_motion: 手臂横向运动幅度（用于视觉特征白名单）
            x_span: x方向跨度（可选，用于视觉特征白名单）
            ang_span: 角度跨度（可选，用于视觉特征白名单）
        """
        if self._wave_learning_enabled and self._wave_learning_manager is not None:
            # ── 第1层：方向白名单验证（优先级最高） ──
            from wave_learning_rules import is_legal_trigger
            if action and not is_legal_trigger(hand, direction, action):
                # 不在白名单中：不记录到学习系统
                action_cn = "下一首" if direction > 0 else "上一首"
                print(f"\n{'='*60}")
                print(f"[白名单过滤] ⚠️ 检测到非法触发！")
                print(f"  方向: {'向左' if direction > 0 else '向右'} ({direction:+.1f})")
                print(f"  动作: {action}")
                print(f"  预期: {'下一首 (player_next)' if direction > 0 else '上一首 (player_prev)'}")
                print(f"  结果: 已拦截，不记录到学习系统")
                print(f"{'='*60}\n")
                return

            # ── 第2层：置信度过滤 ──
            effective_triggered = triggered and confidence >= 0.5

            # ── 第3层：视觉特征白名单验证（增强） ──
            if self._visual_whitelist is not None and action:
                from wave_learning_rules import validate_trigger_with_visual_features
                is_valid, reason = validate_trigger_with_visual_features(
                    hand=hand,
                    direction=direction,
                    triggered_action=action,
                    confidence=confidence,
                    wave_direction=wave_direction,
                    arm_motion=arm_motion,
                    x_span=x_span,
                    ang_span=ang_span,
                    visual_whitelist=self._visual_whitelist
                )
                if not is_valid:
                    print(f"\n{'='*60}")
                    print(f"[视觉特征白名单过滤] ⚠️ {reason}")
                    print(f"  手别: {hand}手")
                    print(f"  方向: {'向左' if direction > 0 else '向右'} ({direction:+.1f})")
                    print(f"  动作: {action}")
                    print(f"  视觉特征: wave_dir={wave_direction:.2f}, arm_motion={arm_motion:.2f}")
                    if x_span > 0:
                        print(f"            x_span={x_span:.3f}")
                    if ang_span > 0:
                        print(f"            ang_span={ang_span:.1f}°")
                    print(f"{'='*60}\n")
                    return

            # ── 传递给P1（记录触发结果） ──
            self._wave_learning_manager.record_wave_attempt(
                hand=hand,
                direction=direction,
                x_span=x_span,  # 传递实际幅度数据
                y_span=0.0,
                ang_span=ang_span,
                triggered=effective_triggered,
                confidence=confidence if effective_triggered else 0.0
            )

            # ── 记录视觉特征数据到数据库（用于学习） ──
            self._wave_learning_manager.record_visual_features(
                hand=hand,
                direction=direction,
                wave_direction=wave_direction,
                arm_motion=arm_motion,
                x_span=x_span,
                ang_span=ang_span,
                triggered=effective_triggered
            )

            # ── 调试输出 ──
            if self._debug:
                action_str = "下一首" if direction > 0 else "上一首"
                if effective_triggered:
                    print(f"[学习记录OK] {hand}手向{'左' if direction > 0 else '右'} -> {action} (置信度={confidence:.2f})")
                elif confidence < 0.5:
                    print(f"[置信度过滤] {hand}手向{'左' if direction > 0 else '右'} -> {action} (置信度={confidence:.2f}<0.5)")

    def save_wave_learning_data(self):
        """保存学习数据（供程序退出时调用）"""
        if self._wave_learning_enabled and self._wave_learning_manager is not None:
            self._wave_learning_manager.save()

    def reset_wave_learning(self):
        """重置学习数据（调试用）"""
        if self._wave_learning_enabled and self._wave_learning_manager is not None:
            self._wave_learning_manager.reset()

    # ═══════════════════════════════════════════════════════
    # 内部推理循环
    # ═══════════════════════════════════════════════════════

    def _inference_loop(self) -> None:
        """
        后台推理线程主循环：读取帧 → MediaPipe 推理 → 特征提取 → 更新缓存。

        关键设计：MediaPipe 在**原始图像**上推理（不镜像），避免左右混淆和精度下降；
        显示帧单独做镜像翻转，让用户看到像镜子一样的画面。
        """
        while self._running_flag:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.05)
                continue

            # ── 读取原始帧（不在此处镜像，保持原始坐标用于推理） ──
            ret, raw_frame = self._cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue

            # 显示帧：镜像翻转，让用户看到镜子般的画面
            display_frame = cv2.flip(raw_frame, 1)

            try:
                if self._use_tasks_api:
                    # ── 新 API (MediaPipe 0.10.30+) ──
                    # 在原始图像上推理，不镜像，避免左右混淆
                    rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    # 确保数组是 C-连续内存布局
                    if not rgb_frame.flags['C_CONTIGUOUS']:
                        rgb_frame = np.ascontiguousarray(rgb_frame)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    self._timestamp_ms += 33
                    result = self.detector.detect_for_video(mp_image, self._timestamp_ms)

                    current_time = time.time()
                    if result.pose_landmarks and len(result.pose_landmarks) > 0:
                        features = self._extract_features(result.pose_landmarks[0], current_time)
                    else:
                        features = self._get_default_features(current_time)

                    # ── 手部检测（HandLandmarker 21 点/手，精细手型识别） ──
                    # 在原始帧上推理（与 Pose 同一坐标系），记录 21 点与手指弯曲度
                    _hands = self._detect_hands(mp_image)
                    features['hand_landmarks'] = [h['points'] for h in _hands]
                    features.update(self._extract_hand_features(_hands))

                    # ── 面部检测（FaceLandmarker 478 点，表情/头部姿态） ──
                    # 在原始帧上推理（与 Pose 同一坐标系），记录 478 点与表情分数
                    _faces = self._detect_faces(mp_image)
                    features.update(self._extract_face_features(_faces))
                else:
                    # ── 旧 API (MediaPipe 0.9.x) ──
                    rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb_frame)
                    current_time = time.time()
                    features = self._extract_features(results, current_time)
                    features['hand_landmarks'] = []
                    features.update(self._extract_hand_features([]))
                    features.update(self._extract_face_features([]))

                # ── 更新缓存（加锁） ──
                # features 中的 landmarks 基于原始图像坐标
                # _latest_frame 存储镜像后的显示帧
                with self._lock:
                    self._latest_features = features
                    self._latest_frame = display_frame

            except Exception as e:
                # 推理异常时不崩溃，记录错误并继续下一帧
                print(f"[MotionCapture] 推理异常: {e}")
                if self._debug:
                    import traceback
                    traceback.print_exc()

            # 控制帧率，避免 CPU 过高（约 30 FPS）
            time.sleep(0.001)


    def _detect_hands(self, mp_image) -> list:
        """
        对原始帧运行 HandLandmarker，返回手部 21 点列表。

        Returns:
            list: 每只手一个 dict：{'side': 'L'/'R', 'points': [21 点 dict]}。
                  未加载手部模型或无手时返回空列表。
        """
        if self.hand_detector is None:
            return []
        try:
            result = self.hand_detector.detect_for_video(mp_image, self._timestamp_ms)
            hands = []
            if result.hand_landmarks:
                handedness = getattr(result, 'handedness', None)
                for i, hlm in enumerate(result.hand_landmarks):
                    pts = []
                    for lm in hlm:
                        pts.append({
                            'x': float(lm.x),
                            'y': float(lm.y),
                            'z': float(getattr(lm, 'z', 0.0)),
                            'visibility': 1.0,
                        })
                    # 左右手：优先用 handedness 分类名
                    side = 'R'
                    try:
                        if handedness and i < len(handedness) and handedness[i]:
                            _name = handedness[i][0].category_name
                            side = 'L' if 'Left' in _name else 'R'
                    except Exception:
                        pass
                    hands.append({'side': side, 'points': pts})
            return hands
        except Exception as _he:
            if self._debug:
                print(f"[MotionCapture] 手部检测异常: {_he}")
            return []

    def _detect_faces(self, mp_image) -> list:
        """
        对原始帧运行 FaceLandmarker，返回面部 478 点列表（含表情 blendshape 分数）。

        Returns:
            list: 每张脸一个 dict：{'points': [478 点 dict], 'blendshapes': {名称: 分数}}。
                  未加载面部模型或无脸时返回空列表。
        """
        if self.face_detector is None:
            return []
        try:
            result = self.face_detector.detect_for_video(mp_image, self._timestamp_ms)
            faces = []
            if result.face_landmarks:
                blends = getattr(result, 'face_blendshapes', None)
                for i, flm in enumerate(result.face_landmarks):
                    pts = []
                    for lm in flm:
                        pts.append({
                            'x': float(lm.x),
                            'y': float(lm.y),
                            'z': float(getattr(lm, 'z', 0.0)),
                            'visibility': 1.0,
                        })
                    bs = {}
                    try:
                        if blends and i < len(blends) and blends[i]:
                            for cat in blends[i]:
                                bs[cat.category_name] = float(cat.score)
                    except Exception:
                        pass
                    faces.append({'points': pts, 'blendshapes': bs})
            return faces
        except Exception as _fe:
            if self._debug:
                print(f"[MotionCapture] 面部检测异常: {_fe}")
            return []

    # 手指关节索引（HandLandmarker 21 点）：腕 0 / 拇指 1-4 / 食指 5-8 / 中指 9-12 / 无名指 13-16 / 小指 17-20
    # 每根手指取 (根节, 中节, 指尖)，用于计算弯曲度
    _FINGER_JOINTS = {
        'thumb':  (1, 2, 4),
        'index':  (5, 6, 8),
        'middle': (9, 10, 12),
        'ring':   (13, 14, 16),
        'pinky':  (17, 18, 20),
    }

    def _finger_curl(self, pts, base, mid, tip) -> float:
        """单指弯曲度 0=伸直 ~1=握拳（尺度无关，免疫距离/体型）。"""
        def _d(a, b):
            return ((pts[a]['x'] - pts[b]['x']) ** 2
                    + (pts[a]['y'] - pts[b]['y']) ** 2) ** 0.5
        l1 = _d(base, mid)
        l2 = _d(mid, tip)
        l3 = _d(base, tip)
        if l1 + l2 < 1e-6:
            return 0.0
        return max(0.0, min(1.0, 1.0 - l3 / (l1 + l2)))

    def _extract_hand_features(self, hands: list) -> Dict[str, Any]:
        """
        从手部 21 点计算精细特征：
        - hand_landmarks_L/R：左右手 21 点原始坐标（供 UI 绘制）
        - finger_curls_L/R：左右手 5 指弯曲度 [拇指, 食指, 中指, 无名指, 小指]
        - hand_gesture_L/R：手型分类（open/fist/victory/ok/thumbs_up 等）
        """
        feats: Dict[str, Any] = {
            'hand_landmarks_L': [],
            'hand_landmarks_R': [],
            'finger_curls_L': None,
            'finger_curls_R': None,
            'hand_gesture_L': None,
            'hand_gesture_R': None,
        }
        for h in hands:
            side = h.get('side', 'R')
            pts = h.get('points', [])
            if len(pts) < 21:
                continue
            feats[f'hand_landmarks_{side}'] = pts
            curls = [self._finger_curl(pts, *self._FINGER_JOINTS[name])
                     for name in ('thumb', 'index', 'middle', 'ring', 'pinky')]
            feats[f'finger_curls_{side}'] = curls
            feats[f'hand_gesture_{side}'] = self._classify_hand_gesture(pts, curls)
        return feats

    def _extract_shoulder_features(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        从 Pose 33 点中提取肩部特征（与手部 21 点同样按"独立部位"方式暴露）：
        - shoulder_landmarks_L/R：左右肩（11/12）原始坐标（供 UI 绘制/姿态参考）
        - shoulder_mid_x/y：肩中点（归一化坐标）
        - shoulder_width：肩宽（左右肩归一化距离，体型尺度参考）
        - shoulder_tilt：肩部侧倾角（左右肩连线相对水平线的角度，度）
        """
        feats: Dict[str, Any] = {
            'shoulder_landmarks_L': None,
            'shoulder_landmarks_R': None,
            'shoulder_mid_x': None,
            'shoulder_mid_y': None,
            'shoulder_width': None,
            'shoulder_tilt': None,
        }
        if landmarks is None or landmarks.shape[0] < 13:
            return feats
        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]
        if ls[3] <= 0.3 or rs[3] <= 0.3:
            return feats
        feats['shoulder_landmarks_L'] = {
            'x': float(ls[0]), 'y': float(ls[1]),
            'z': float(ls[2]), 'visibility': float(ls[3]),
        }
        feats['shoulder_landmarks_R'] = {
            'x': float(rs[0]), 'y': float(rs[1]),
            'z': float(rs[2]), 'visibility': float(rs[3]),
        }
        feats['shoulder_mid_x'] = float((ls[0] + rs[0]) / 2.0)
        feats['shoulder_mid_y'] = float((ls[1] + rs[1]) / 2.0)
        feats['shoulder_width'] = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        if abs(dx) > 1e-6:
            feats['shoulder_tilt'] = float(np.degrees(np.arctan2(dy, dx)))
        return feats

    def _extract_face_features(self, faces: list) -> Dict[str, Any]:
        """
        从面部 478 点 + blendshape 表情分数计算精细特征：
        - face_landmarks：第一张脸的 478 点原始坐标（供 UI 绘制）
        - face_present：是否检测到人脸
        - face_mouth_open：张嘴程度 0~1（blendshape jawOpen，缺失时用上下唇几何比）
        - face_smile：微笑程度 0~1（blendshape mouthSmile，缺失时用嘴角几何抬升）
        - face_head_tilt：头部侧倾角（左右眼外角连线相对水平线的角度，度）
        """
        feats: Dict[str, Any] = {
            'face_landmarks': [],
            'face_present': False,
            'face_mouth_open': None,
            'face_smile': None,
            'face_head_tilt': None,
        }
        if not faces:
            return feats
        face = faces[0]
        pts = face.get('points', [])
        if len(pts) < 468:
            return feats
        feats['face_landmarks'] = pts
        feats['face_present'] = True
        bs = face.get('blendshapes') or {}

        def _d(a, b):
            return ((pts[a]['x'] - pts[b]['x']) ** 2
                    + (pts[a]['y'] - pts[b]['y']) ** 2) ** 0.5

        # ── 张嘴：优先 blendshape jawOpen，缺失回退几何（上下唇 13/14 间距 / 眼距 33/263） ──
        jaw = bs.get('jawOpen')
        if jaw is not None:
            feats['face_mouth_open'] = max(0.0, min(1.0, jaw))
        else:
            eye_d = _d(33, 263)
            lip_d = _d(13, 14)
            if eye_d > 1e-6:
                feats['face_mouth_open'] = max(0.0, min(1.0, lip_d / eye_d * 2.0))

        # ── 微笑：优先 blendshape mouthSmile，缺失回退嘴角（61/291）相对鼻尖(1)的抬升 ──
        smile = max(bs.get('mouthSmileLeft', 0.0), bs.get('mouthSmileRight', 0.0))
        if smile > 0.0:
            feats['face_smile'] = max(0.0, min(1.0, smile))
        else:
            nose_y = pts[1]['y']
            corner_y = (pts[61]['y'] + pts[291]['y']) / 2.0
            eye_d = _d(33, 263)
            if eye_d > 1e-6:
                lift = (nose_y - corner_y) / eye_d
                feats['face_smile'] = max(0.0, min(1.0, lift * 2.0 + 0.5))

        # ── 头部侧倾角：左右眼外角连线与水平线的夹角（度） ──
        dx = pts[263]['x'] - pts[33]['x']
        dy = pts[263]['y'] - pts[33]['y']
        if abs(dx) > 1e-6:
            feats['face_head_tilt'] = float(np.degrees(np.arctan2(dy, dx)))
        return feats

    def _classify_hand_gesture(self, pts: list, curls: list) -> str:
        """
        基于 21 点 + 5 指弯曲度分类手型。

        弯曲度 0=伸直 ~1=握拳；结合拇指-食指指尖距离判 OK。

        Returns:
            手型名: open / fist / victory / ok / thumbs_up / pointing / half_fist / unknown
        """
        if not pts or len(pts) < 21 or not curls or len(curls) < 5:
            return 'unknown'
        thumb, index, middle, ring, pinky = curls

        def _d(a, b):
            return ((pts[a]['x'] - pts[b]['x']) ** 2
                    + (pts[a]['y'] - pts[b]['y']) ** 2) ** 0.5

        # OK 手势：拇指指尖(4) 与食指指尖(8) 距离 < 手掌长度的 35%，
        # 且中指未全弯（OK 时中指通常伸直/放松；握拳时中指全弯 → 不误判为 OK）
        palm_ref = max(_d(0, 9), 1e-3)  # 腕→中指根 = 掌长，防 0
        if _d(4, 8) < 0.35 * palm_ref and middle < 0.5:
            return 'ok'
        # 张开：四指全部伸直
        if index < 0.3 and middle < 0.3 and ring < 0.3 and pinky < 0.3:
            return 'open'
        # 点赞：拇指伸直 + 中/无名/小指弯曲（与 fist 互斥：fist 要求拇指也弯）
        if thumb < 0.3 and middle > 0.5 and ring > 0.5 and pinky > 0.5:
            return 'thumbs_up'
        # 指人：仅食指伸直，其余弯曲
        if index < 0.3 and middle > 0.5 and ring > 0.5 and pinky > 0.5:
            return 'pointing'
        # 比耶：食指+中指伸直，无名指+小指弯曲
        if index < 0.3 and middle < 0.3 and ring > 0.5 and pinky > 0.5:
            return 'victory'
        # 双指（食指+中指伸直，无名指/小指状态不限）
        if index < 0.3 and middle < 0.3:
            return 'two_fingers'
        # 握拳：拇指+四指全部弯曲
        if thumb > 0.5 and index > 0.5 and middle > 0.5 and ring > 0.5 and pinky > 0.5:
            return 'fist'
        return 'half_fist'

    def _extract_features(
        self, results: Any, current_time: float
    ) -> Dict[str, Any]:
        """
        从 MediaPipe 推理结果中提取所有运动特征。

        兼容旧 API (PoseLandmarkerResult) 和新 API (NormalizedLandmarks 可迭代对象)。

        Args:
            results: MediaPipe Pose 处理结果（新 API 为 pose_landmarks[0]，旧 API 为完整结果）。
            current_time: 当前时间戳（time.time()）。

        Returns:
            完整的特征字典。
        """
        # 如果未检测到人体，返回默认特征
        if results is None:
            return self._get_default_features(current_time)

        landmarks = None

        # 兼容新 API (MediaPipe Tasks): result.pose_landmarks[0] 是可迭代的 NormalizedLandmarks
        # 直接尝试解析，成功即可使用
        try:
            landmarks = self._parse_landmarks(results)
        except (TypeError, AttributeError):
            pass

        # 兼容旧 API (MediaPipe Solutions): results.pose_landmarks.landmark
        if landmarks is None:
            if hasattr(results, 'pose_landmarks') and results.pose_landmarks is not None:
                try:
                    landmarks = self._parse_landmarks(results.pose_landmarks.landmark)
                except (TypeError, AttributeError):
                    pass

        # 所有解析方式都失败，返回默认特征
        if landmarks is None:
            return self._get_default_features(current_time)

        # 基础几何：计算关键中间点
        shoulder_mid = self._midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_mid = self._midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
        ankle_mid = self._midpoint(landmarks, LEFT_ANKLE, RIGHT_ANKLE)
        knee_mid = self._midpoint(landmarks, LEFT_KNEE, RIGHT_KNEE)

        # 时间差
        dt = current_time - self._prev_time if self._prev_time is not None else 0.0

        # 计算各项特征
        motion_energy = self._calc_motion_energy(landmarks, dt)
        arm_motion_energy = self._calc_arm_motion_energy(landmarks, dt)
        arm_horizontal_motion = self._calc_arm_horizontal_motion(landmarks, dt)
        palm_hover = self._calc_palm_hover(landmarks)
        body_speed = self._calc_body_speed(shoulder_mid, hip_mid, dt)
        arm_amplitude = self._calc_arm_amplitude(landmarks, shoulder_mid, hip_mid)
        body_rhythm = self._calc_body_rhythm(hip_mid)
        posture = self._classify_posture(landmarks, shoulder_mid, hip_mid, ankle_mid, knee_mid)
        hand_position = self._classify_hand_position(landmarks, shoulder_mid, hip_mid)
        is_active = self._classify_activity(motion_energy, body_speed)

        # 更新历史数据
        self._prev_landmarks = landmarks.copy()
        self._prev_time = current_time

        # 静态构型特征（肘角/腕偏移/spread），供静态动作模板匹配
        pose_config = self._calc_pose_config(landmarks)

        # 构建特征字典
        features = {
            'timestamp': current_time,
            'motion_energy': motion_energy,
            'arm_motion_energy': arm_motion_energy,
            'arm_horizontal_motion': arm_horizontal_motion,
            'palm_hover': palm_hover,
            'wave_direction': float(self._wave_direction),
            'body_speed': body_speed,
            'arm_amplitude': arm_amplitude,
            'body_rhythm': body_rhythm,
            'posture': posture,
            'hand_position': hand_position,
            'is_active': is_active,
            'landmarks': self._landmarks_to_list(landmarks),
            # 肘部运动轨迹（当前窗口），供 DTW 模板匹配使用：
            # 取左右肘部历史中横向跨度较大的一侧，作为"当前动作轨迹快照"
            'arm_trajectory': self._current_arm_trajectory(),
            # 视觉特征白名单增强数据（从 _calc_arm_horizontal_motion 获取）
            'wave_x_span': self._latest_wave_x_span,
            'wave_ang_span': self._latest_wave_ang_span,
        }
        features.update(pose_config)  # pose_elbow_angle_*/pose_wrist_*/pose_spread_*
        features.update(self._extract_shoulder_features(landmarks))
        return features

    def _current_arm_trajectory(self) -> list:
        """返回当前窗口的肘部运动轨迹 [(x, y), ...]，供 DTW 模板匹配。
        取左右肘部历史中 x+y 综合跨度较大的一侧（纵向动作也能选中）；
        无数据返回空列表。"""
        best = []
        best_span = -1.0
        for side in ('L', 'R'):
            pts = list(getattr(self, '_elbow_xy_history', {}).get(side, []))
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                span = (max(xs) - min(xs)) + (max(ys) - min(ys))
                if span > best_span:
                    best_span = span
                    best = [(float(p[0]), float(p[1])) for p in pts]
        return best

    def _calc_motion_energy(self, landmarks: np.ndarray, dt: float) -> float:
        """
        计算整体运动能量（头部解耦：排除 0-10 号面部关键点）。

        算法：
        1. 对所有可见身体关键点（11-32，visibility > 0.5），计算与上一帧的欧氏位移。
        2. 取所有位移的平均值，除以时间 dt 得到速度。
        3. 使用自适应最大值归一化到 [0, 1]。

        Args:
            landmarks: 当前帧 33 关键点，shape=(33, 4)。
            dt: 与上一帧的时间差（秒）。

        Returns:
            归一化后的运动能量，范围 [0, 1]。
        """
        if self._prev_landmarks is None or dt <= 0:
            self._energy_history.append(0.0)
            return 0.0

        # 只取可见关键点，排除头部（0-10）和手臂（11-16），只算躯干+腿部（17-32）
        visible = (landmarks[:, 3] > 0.5) & (self._prev_landmarks[:, 3] > 0.5)
        # 躯干+腿部索引：髋部(23-24)、膝(25-26)、踝(27-28)、脚跟(29-30)、脚尖(31-32)
        torso_leg_indices = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
                             LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL,
                             LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]
        body_mask = np.zeros(33, dtype=bool)
        body_mask[torso_leg_indices] = True
        visible = visible & body_mask

        if not visible.any():
            self._energy_history.append(0.0)
            return 0.0

        # 计算每点位移（x, y 平面，z 暂时忽略）
        diff = landmarks[visible, :2] - self._prev_landmarks[visible, :2]
        distances = np.linalg.norm(diff, axis=1)
        avg_displacement = float(np.mean(distances))

        # 计算速度（位移/时间），并缩放
        raw_energy = avg_displacement / max(dt, 0.001)

        # 自适应归一化：维护历史最大值，但允许较快衰减
        # 衰减系数 0.995 表示约 140 帧（~4.6s）后 max 降到原来的 50%
        self._max_energy = max(self._max_energy * 0.995, raw_energy, 0.001)
        motion_energy = raw_energy / self._max_energy
        motion_energy = min(motion_energy, 1.0)
        self._energy_history.append(motion_energy)
        return motion_energy

    def _calc_arm_motion_energy(self, landmarks: np.ndarray, dt: float) -> float:
        """
        计算手部独立运动能量（与躯干/头部完全解耦，不含肩膀）。

        只计算纯手部关键点（肘部 13,14、手腕 15,16）的位移，
        肩膀属于躯干，不纳入计算。这样身体扭动、摇头时肩膀的连带运动
        不会影响手部运动能量的值，实现手部与躯干的完全独立识别。

        Args:
            landmarks: 当前帧 33 关键点，shape=(33, 4)。
            dt: 与上一帧的时间差（秒）。

        Returns:
            归一化后的手部运动能量，范围 [0, 1]。
        """
        if self._prev_landmarks is None or dt <= 0:
            self._arm_motion_history.append(0.0)
            return 0.0

        # 纯手部关键点索引（不含肩膀，肩膀属于躯干）
        HAND_INDICES = [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
        hand_mask = np.zeros(33, dtype=bool)
        hand_mask[HAND_INDICES] = True

        # 只取可见的手部关键点
        visible = (landmarks[:, 3] > 0.5) & (self._prev_landmarks[:, 3] > 0.5) & hand_mask

        if not visible.any():
            self._arm_motion_history.append(0.0)
            return 0.0

        # 计算手部点位移
        diff = landmarks[visible, :2] - self._prev_landmarks[visible, :2]
        distances = np.linalg.norm(diff, axis=1)
        avg_displacement = float(np.mean(distances))

        raw_arm_motion = avg_displacement / max(dt, 0.001)

        # 自适应归一化
        self._max_arm_motion = max(self._max_arm_motion * 0.995, raw_arm_motion, 0.001)
        arm_motion_energy = raw_arm_motion / self._max_arm_motion
        arm_motion_energy = min(arm_motion_energy, 1.0)
        self._arm_motion_history.append(arm_motion_energy)
        return arm_motion_energy


    def _calc_pose_config(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        静态构型特征（供静态动作模板匹配，动态动作不依赖此组）。
        统一采样 头部/胸腔/大臂/小臂/手部 五个部位（全部相对坐标，
        免疫体型/距离差异），判别时五段全部命中才触发：

        - 头部   pose_head_dx/dy：NOSE 相对肩中点的 (x,y) 偏移
        - 胸腔   pose_torso_dx/dy：肩中点相对髋中点的 (x,y) 偏移
        - 大臂   pose_upper_dx/dy_L/R：肘部相对同侧肩的 (x,y) 偏移
        - 小臂   pose_fore_dx/dy_L/R：腕部相对同侧肘的 (x,y) 偏移
        - 手部   pose_hand_dx/dy_L/R：腕部相对同侧肩的 (x,y) 偏移
                 pose_spread_L/R：手型张开度（拇指-食指根+小指根横向展开）

        关节不可见时返回 None，采样端（_extract_sample 的 isinstance 过滤）
        自动跳过，不影响其余特征。
        """
        VIS = 0.3
        cfg: Dict[str, Any] = {}

        def _off(a, b):
            """两点相对偏移 (dx, dy)，任一点不可见返回 None。"""
            if a[3] <= VIS or b[3] <= VIS:
                return None, None
            return float(a[0] - b[0]), float(a[1] - b[1])

        # ── 头部：NOSE 相对肩中点 ──
        shm = self._midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
        if landmarks[NOSE, 3] > VIS:
            cfg['pose_head_dx'] = float(landmarks[NOSE, 0] - shm[0])
            cfg['pose_head_dy'] = float(landmarks[NOSE, 1] - shm[1])

        # ── 胸腔：肩中点相对髋中点（可见度用两侧肩/髋关节原始值判定） ──
        hip_mid = self._midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
        if (landmarks[LEFT_SHOULDER, 3] > VIS and landmarks[RIGHT_SHOULDER, 3] > VIS
                and landmarks[LEFT_HIP, 3] > VIS and landmarks[RIGHT_HIP, 3] > VIS):
            cfg['pose_torso_dx'] = float(shm[0] - hip_mid[0])
            cfg['pose_torso_dy'] = float(shm[1] - hip_mid[1])

        for side, (sh, el, wr, ix, pk, th) in (
            ('L', (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
                   LEFT_INDEX, LEFT_PINKY, LEFT_THUMB)),
            ('R', (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
                   RIGHT_INDEX, RIGHT_PINKY, RIGHT_THUMB)),
        ):
            # ── 大臂：肘相对肩 ──
            dx, dy = _off(landmarks[el], landmarks[sh])
            if dx is not None:
                cfg[f'pose_upper_dx_{side}'] = dx
                cfg[f'pose_upper_dy_{side}'] = dy
            # ── 小臂：腕相对肘 ──
            dx, dy = _off(landmarks[wr], landmarks[el])
            if dx is not None:
                cfg[f'pose_fore_dx_{side}'] = dx
                cfg[f'pose_fore_dy_{side}'] = dy
            # ── 手部：腕相对肩 + spread ──
            dx, dy = _off(landmarks[wr], landmarks[sh])
            if dx is not None:
                cfg[f'pose_hand_dx_{side}'] = dx
                cfg[f'pose_hand_dy_{side}'] = dy
            if (landmarks[ix, 3] > VIS and landmarks[th, 3] > VIS
                    and landmarks[pk, 3] > VIS):
                cfg[f'pose_spread_{side}'] = float(
                    abs(landmarks[ix, 0] - landmarks[th, 0])
                    + abs(landmarks[pk, 0] - landmarks[th, 0]))
        return cfg

    def _calc_arm_horizontal_motion(self, landmarks: np.ndarray, dt: float) -> float:
        """
        挥手跨度检测：分别追踪左右手"肘部"的横向运动，取较大 span。

        追踪点 = 肘部坐标（LEFT_ELBOW / RIGHT_ELBOW）。不用手腕/指根：
        MediaPipe Pose 的手腕(15/16)是手部区域粗粒度估计点，手指动作时
        会被指根污染而明显漂移（实测跨x可达 0.13，导致手指动作误触切歌）；
        肘部(13/14)远离手部，手指动作绝不会影响其坐标。真实挥手是整条
        手臂（含肘部）的横向摆动，肘部位移足够大，因此追踪肘部彻底
        剔除手指噪声（手指动作不参与位移计算）。腕部坐标仍单独记录，
        供举手悬停（palm_hover）判定使用。

        严格横向判定（只看横向位移，纵向仅用于排除干扰手势）：
        1. 将肘部追踪点坐标（含手型张开度 spread）存入左右手独立历史队列
           （20 帧窗口）
        2. 对 x/y 序列做 均值+中值 双重滤波（顺序关键）：均值先把持续性
           周期抖动压到 1/3 幅度，中值再消除残余尖峰；真实挥手是持续
           同向位移，仅端点内收约 1 帧步长，跨度基本保留
        3. 分别追踪左右手，计算滤波后 x 的 max - min = 横向跨度，
           跨度须 ≥ 6% 屏宽（参考：挥手位移约占屏宽 15~30%，肘部为支点
           位移约折半即 7.5~15%，6% 贴近下限留足触发余量——实机挥手
           肘部常只有 7~8% 位移，8% 恰好卡在参考下限上导致难触发；
           仅排除轻微摆手/小幅度动作）
        4. 区分"挥手"与"其他手势"：
           a. 肘部高度门：肘部不得明显高于肩部（窗口内相对肩部 y 的
              中位数 ≥ -0.12）——举手/挠头时手臂抬起、肘部高于肩
              通常远超 12% 屏高，排除；正常挥手肘部绕肩弧线略高于肩
              不误杀，与 palm_hover"手腕在肩上方=举手"互斥
           b. 横向主导：纵向跨度不得超过横向跨度的 160%——
              允许明显斜向/弧线挥手（实机挥手肘部绕肩弧线，纵向分量
              常达横向 1.2~1.6 倍）；y/x>1.6 的纵向主导手势仍排除
           c. 肘-肩矢量角度跨度：≥0.20 rad（排除手指带动的手腕扇形
              移动——肘-肩矢量角度几乎不变）且 ≤2.5 rad（排除锤桌/
              大幅绕圈——肘部绕肩几乎转一整圈，实测角度跨度 2.8~4.9 rad；
              正常弧线挥手仅 1.2~1.5 rad）
        5. 方向取窗口内相对起点偏移最大的点（单程挥与往返挥都能定向）
        6. 返回值 = min(跨度 * 2.5, 1.0)

        Args:
            landmarks: 当前帧 33 关键点，shape=(33, 4)。
            dt: 与上一帧的时间差（秒，保留参数兼容）。

        Returns:
            挥手强度 [0, 1]，跨越大半个屏幕时接近 1.0。
        """
        # 重置本次计算的视觉特征数据
        self._latest_wave_x_span = 0.0
        self._latest_wave_ang_span = 0.0

        VIS_THRESHOLD = 0.3       # 肘部可见性阈值（作为整条手臂的存在判据）
        WAVE_X_THRESHOLD = 0.06   # 横向跨度阈值：6% 屏幕宽度（参考：挥手位移约占屏宽
                                  # 15~30%，肘部为支点约折半即 7.5~15%；6% 贴近下限
                                  # 留足触发余量——实机挥手肘部常只有 7~8% 位移，
                                  # 旧值 8% 恰好卡在参考下限上导致难触发；
                                  # 低于 6% 的轻微摆手/小幅度动作仍不触发）
        MIN_FRAMES = 4            # 最少帧数
        WRIST_VIS_MIN = 0.10      # 腕部高度门的腕部可见性记录阈值：远低于 VIS_THRESHOLD
                                  #（0.3）。戴耳机/挠头/调摄像头时手贴近头部，MediaPipe
                                  #  对腕部的可见性评分普遍很低（实机 0.1~0.5，触发帧
                                  #  常 0.09~0.28），若沿用 0.3 则腕部高度数据几乎全为
                                  #  None → 高度门空转失效（"实机戴耳机仍误触切歌"
                                  #  根因）。0.10 以下坐标基本不可信仍跳过；0.10~0.3
                                  #  的坐标噪声由 _median3+中位数 滤波抑制。
        WRIST_ABOVE_MAX = -0.08   # 腕部高度门：腕部相对肩部 y 的中位数不得小于 -0.08
                                  #（即腕部不得高于肩部超过 8% 屏高）。真实挥手时
                                  # 手在体侧横向摆动，腕部与肩部齐平或略低（相对 y
                                  # 约 -0.05~+0.05）；戴耳机/挠头/手动调摄像头时
                                  # 手抬到耳/头顶高度，腕部相对肩 y 常 < -0.10 甚至
                                  # -0.20 —— 用腕部高度结构性排除"手在头部附近"
                                  # 的动作（此前只查肘部高度，手臂弯曲抬起时肘部
                                  # 可在肩下方、腕部却远超肩部，导致误触切歌；
                                  # 实机戴耳机腕部在耳边 wry≈-0.10~-0.13，-0.08
                                  # 留足余量，挥手手略抬起亦不误杀）。
                                  # 与 palm_hover 的"腕部在肩上方=举手"互斥一致：
                                  # 举手/悬停腕部更高，同样被此门排除。
        MAX_Y_X_RATIO = 1.5       # 纵向跨度上限：不得超过横向跨度的 150%
                                  #（允许明显斜向/弧线挥手——实机挥手肘部绕肩
                                  #  弧线运动，纵向分量常达横向的 1.2~1.6 倍，
                                  #  1.5 留足余量但排除纯纵向/大幅弧线动作。
                                  #  旧值 2.0 过宽：用户日志显示大量"跨 y 明显
                                  #  大于跨 x"的动作（走路摆臂/身体调整）被
                                  #  误判为挥手切歌，收紧到 1.5 后这些帧的
                                  #  y/x（1.5~1.8）被排除，真实挥手仍放行）
        MIN_DIR_FLIPS = 1         # 最少横向方向翻转次数：≥1 即有"去-回"往返迹象，
                                  #（指向/摊手/伸臂等单程位移为 0 次翻转被排除。
                                  #  真实挥手一次"去-回"（0.33~0.67s，20 帧窗口
                                  #  恰好覆盖）仅 1 次翻转，要求 ≥2 会卡掉正常
                                  #  挥手导致"基本触发不灵敏"；慢速摆臂/身体
                                  #  调整等误触由 MAX_Y_X_RATIO 纵向主导排除
                                  #  与肘部/腕部高度门承担，不再靠翻转次数）
        FLIP_DEADBAND = 0.01      # 翻转计数死区（归一化坐标）：单步位移绝对值小于此
                                  # 值不视为方向变化（MediaPipe 静止抖动约 0.001，
                                  # 臂部运动估计噪声通常 <0.01）。防止纵向估计噪声
                                  # 刷出大量假 y 翻转，把真实挥手误判为"纵向主导"。
                                  # 真实纵向手势（上下摆手/锤桌）单步位移 ≥0.02，
                                  # 仍会被正常计入。
        WAVE_ANGLE_MIN = 0.20     # 肘-肩矢量最小角度跨度（弧度，≈11°）：
                                  #（真实挥手肘部绕肩摆动，角度变化明显；
                                  #  手指带动的手腕扇形移动以肘部为支点，
                                  #  肘-肩矢量角度几乎不变 → 被此门槛排除）
        WAVE_ANGLE_MAX = 2.5      # 肘-肩矢量最大角度跨度（弧度，≈143°）：
                                  #（锤桌/大幅绕圈时肘部绕肩几乎转一整圈，
                                  #  角度跨度实测 2.8~4.9 rad；正常挥手弧线
                                  #  角度跨度仅 1.2~1.5 rad。用角度上限把
                                  #  锤桌类大幅转动排除——比翻转平手规则
                                  #  更本质，允许真实弧线挥手平手翻转通过）
        HAND_SHAPE_SPAN_MAX = 0.12  # 窗口内手指张开度变化上限（滤波后）：
                                  #（挥手时手掌保持张开、spread 稳定；
                                  #  弹指/挠手等手指动作 spread 变化大 →
                                  #  判定为"手指动作"而非"挥手"，排除。
                                  #  实机挥手 spread 原始波动可达 0.05~0.14，
                                  #  滤波压噪声后放宽到 0.12，避免误杀正常挥手）

        def _median3(vals):
            """3 点中值滤波：彻底消除单帧脉冲尖峰（估计值偶发跳变）。"""
            if len(vals) < 3:
                return list(vals)
            out = [vals[0]]
            for i in range(1, len(vals) - 1):
                out.append(sorted((vals[i - 1], vals[i], vals[i + 1]))[1])
            out.append(vals[-1])
            return out

        def _mean3(vals):
            """3 点均值滤波（端点也与邻点平均）：抑制持续性的周期抖动——
            中值滤波无法滤掉连续来回摆动（每帧都是"真实"极值），
            均值滤波将 ±A 的持续摆动压到 ±A/3；对单方向真实挥手，
            仅端点内收约 1 帧步长，跨度基本保留。"""
            n = len(vals)
            if n < 2:
                return list(vals)
            out = [(vals[0] + vals[1]) / 2.0]
            for i in range(1, n - 1):
                out.append((vals[i - 1] + vals[i] + vals[i + 1]) / 3.0)
            out.append((vals[n - 2] + vals[n - 1]) / 2.0)
            return out

        # 以上滤波函数已提升为模块级（挥手/举手共用），见文件顶部 _median3/_mean3

        def hand_point(e_idx, s_idx, w_idx, ix_idx, pk_idx, th_idx) -> Optional[tuple]:
            """肘部追踪点 + 手型张开度 + 腕部相对肩部高度：返回 (肘相对肩x, 肘相对肩y, 手型spread, 腕相对肩y)。

            挥手检测追踪"肘部相对同侧肩部"的相对坐标（LEFT_ELBOW/RIGHT_ELBOW
            相对 LEFT_SHOULDER/RIGHT_SHOULDER）而非绝对坐标：
            1. 手指动作时手臂整体不动，肘部相对肩部几乎不变——即使 MediaPipe
               对手部区域的估计噪声放大（手离屏幕近时更明显），相对坐标也
               稳定，结构性杜绝"动手指切歌"；
            2. 真实挥手是整条手臂（含肘部）相对肩部的横向摆动，相对位移足够；
            3. 手近屏时肩/肘同时放大，相对坐标比值不变——抵消近屏放大效应。

            spread = |食指根.x−拇指根.x| + |小指根.x−拇指根.x|，即手指
            张开度：挥手时手掌保持张开、spread 稳定；弹指/挠手/指人等
            手指动作时 spread 剧烈变化 → 用于"手型稳定性"排除手指动作。
            肘部或肩部不可见时跳过本帧。

            wry（腕部相对肩部 y）：真实挥手手在体侧横向摆动，腕部与肩齐平
            或略低；戴耳机/挠头/调摄像头时腕部抬到耳/头顶高度，wry 显著为负。
            腕部不可见时返回 None，判定侧跳过该帧的高度门（不强求）。"""
            if (landmarks[e_idx, 3] > VIS_THRESHOLD
                    and landmarks[s_idx, 3] > VIS_THRESHOLD):
                ex = float(landmarks[e_idx, 0]) - float(landmarks[s_idx, 0])
                ey = float(landmarks[e_idx, 1]) - float(landmarks[s_idx, 1])
                spread = (abs(float(landmarks[ix_idx, 0]) - float(landmarks[th_idx, 0]))
                          + abs(float(landmarks[pk_idx, 0]) - float(landmarks[th_idx, 0])))
                wry = None
                # 腕部高度用 WRIST_VIS_MIN（0.10）而非 VIS_THRESHOLD（0.3）：
                # 戴耳机/挠头/调摄像头时手贴近头部，MediaPipe 腕部可见性普遍
                # 只有 0.1~0.5，若按 0.3 过滤则 wry 几乎全为 None，腕部高度门
                # 空转失效（实机误触根因）。0.10 以上坐标已包含高度信息，
                # 噪声由判定侧 _median3+中位数 滤波抑制。
                if landmarks[w_idx, 3] > WRIST_VIS_MIN:
                    wry = float(landmarks[w_idx, 1]) - float(landmarks[s_idx, 1])
                return (ex, ey, spread, wry)
            return None

        left_pt = hand_point(LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST,
                             LEFT_INDEX, LEFT_PINKY, LEFT_THUMB)
        right_pt = hand_point(RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST,
                              RIGHT_INDEX, RIGHT_PINKY, RIGHT_THUMB)

        # 双手均不可见：清空历史防止残留
        if left_pt is None and right_pt is None:
            self._wrist_xy_history['L'].clear()
            self._wrist_xy_history['R'].clear()
            self._elbow_xy_history['L'].clear()
            self._elbow_xy_history['R'].clear()
            self._wave_direction = 0.0
            self._wave_event_active = False
            self._wave_event_dir = 0.0
            self._wave_still_frames = 0
            return 0.0

        # 肘部坐标入挥手队列；同时记录腕部坐标供举手悬停判定
        if left_pt is not None:
            self._elbow_xy_history['L'].append(left_pt)
            if landmarks[LEFT_WRIST, 3] > VIS_THRESHOLD:
                self._wrist_xy_history['L'].append(
                    (float(landmarks[LEFT_WRIST, 0]), float(landmarks[LEFT_WRIST, 1])))
        if right_pt is not None:
            self._elbow_xy_history['R'].append(right_pt)
            if landmarks[RIGHT_WRIST, 3] > VIS_THRESHOLD:
                self._wrist_xy_history['R'].append(
                    (float(landmarks[RIGHT_WRIST, 0]), float(landmarks[RIGHT_WRIST, 1])))

        # 分别计算左右手的横向/纵向跨度与方向，取最大有效 span
        best_span = 0.0
        best_dir = 0.0
        dbg = []

        for side in ('L', 'R'):
            side_vals = list(self._elbow_xy_history[side])
            if len(side_vals) < MIN_FRAMES:
                continue
            xs = _median3(_mean3([p[0] for p in side_vals]))
            ys = _median3(_mean3([p[1] for p in side_vals]))
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            dbg.append(f"{side}跨x={x_span:.3f}跨y={y_span:.3f}")
            # 肘部高度门：挥手时手臂在体侧横向摆动，肘部与肩齐平或在肩下
            #（相对肩部 y ≥ 0）；举手/挠头时手臂抬起，肘部明显高于肩部
            #（相对肩部 y < 0）。用窗口内肘部相对肩部高度的中位数判定，
            # 高于肩部超过 12% 屏高 → 视为举手/挠头类手势，排除。
            #（从 8% 放宽到 12%：挥手肘部绕肩弧线时也常略高于肩，8% 会
            #  误杀正常挥手；挠头/举手肘部高于肩通常远超 12%。）
            # 与 palm_hover 的"手腕在肩上方 = 举手"判定天然互斥。
            if sorted(ys)[len(ys) // 2] < -0.12:
                continue
            # 腕部高度门：腕部相对肩部 y 的中位数不得 < WRIST_ABOVE_MAX
            #（腕部不得高于肩部超过 10% 屏高）。真实挥手手在体侧，腕部与肩
            # 齐平或略低；戴耳机/挠头/手动调摄像头时手抬到耳/头顶高度，
            # 腕部远高于肩（相对 y 常 < -0.10 甚至 -0.20），此门结构性排除
            # ——此前只查肘部高度，手臂弯曲抬起时肘部可在肩下方、腕部却
            # 远超肩部，导致这些头部附近动作被误判为挥手切歌。
            wrys = [p[3] for p in side_vals if len(p) > 3 and p[3] is not None]
            if wrys:
                wry_f = _median3(_mean3(wrys))
                if sorted(wry_f)[len(wry_f) // 2] < WRIST_ABOVE_MAX:
                    continue
            # 横向主导：纵向跨度超过横向跨度的 120% 视为
            # 上下摆手/挠头等纵向主导手势，排除
            if x_span <= WAVE_X_THRESHOLD or y_span > x_span * MAX_Y_X_RATIO:
                continue
            # 肘-肩矢量角度跨度：真实挥手肘部绕肩摆动，角度大幅变化；
            # 手指带动的手腕扇形移动以肘部为支点，肘-肩矢量角度几乎不变，
            # 即使位移被 MediaPipe 噪声推过阈值，角度跨度仍很小 → 排除。
            # 角度先解卷绕（±π 跳变），再算跨度
            angs = [float(np.arctan2(p[1], p[0])) for p in side_vals]
            for i in range(1, len(angs)):
                while angs[i] - angs[i - 1] > np.pi:
                    angs[i] -= 2.0 * np.pi
                while angs[i] - angs[i - 1] < -np.pi:
                    angs[i] += 2.0 * np.pi
            angs = _median3(_mean3(angs))
            ang_span = max(angs) - min(angs)
            dbg[-1] += f"角度跨度={ang_span:.3f}"
            if ang_span < WAVE_ANGLE_MIN or ang_span > WAVE_ANGLE_MAX:
                continue
            # 手型稳定性：窗口内手指张开度（spread）变化须 ≤ 阈值。
            # 挥手时手掌保持张开、spread 稳定；弹指/挠手/指人等手指动作
            # 时 spread 剧烈变化——即使肘部被连带带动，手型变化也足以识别
            # 为"手指动作"而非"挥手"，结构性排除手指误触。
            # spread 原始值带单帧估计噪声，先与 x/y 同款滤波再算跨度
            spreads = [p[2] for p in side_vals if len(p) > 2]
            if spreads:
                spread_f = _median3(_mean3(spreads))
                spread_span = max(spread_f) - min(spread_f)
                dbg[-1] += f"手型变化={spread_span:.3f}"
                if spread_span > HAND_SHAPE_SPAN_MAX:
                    continue
            # 有往返：窗口内横向方向至少翻转 1 次（"去-回"迹象），
            # 排除指向、摊手、伸臂等 x 只朝一个方向移动的手势。
            # 跳过零步长与死区内的抖动（滤波后的平台），但保留上一个
            # 非零方向继续比较
            flips = 0
            last_sign = 0
            for i in range(1, len(xs)):
                d = xs[i] - xs[i - 1]
                if abs(d) < FLIP_DEADBAND:
                    continue
                sign = 1 if d > 0 else -1
                if last_sign != 0 and sign != last_sign:
                    flips += 1
                last_sign = sign
            if flips < MIN_DIR_FLIPS:
                continue
            # 横向往复主导：x 方向翻转次数须**严格大于** y 方向翻转次数。
            # 挥手是横向往复（x 翻转为主）；锤桌/上下摆手是纵向往复
            #（y 翻转为主）。锤桌排除现由角度跨度上限（WAVE_ANGLE_MAX）
            # 承担——锤桌肘部绕肩几乎转一整圈、角度跨度实测 2.8~4.9 rad；
            # 正常弧线挥手角度跨度仅 1.2~1.5 rad。因此此处仅排除
            # y 翻转严格多于 x 的纵向主导手势，平手（x=1/y=1）放行——
            # 真实弧线挥手的去-回弧线也常出现 x/y 翻转持平。
            y_flips = 0
            last_sign = 0
            for i in range(1, len(ys)):
                d = ys[i] - ys[i - 1]
                if abs(d) < FLIP_DEADBAND:
                    continue
                sign = 1 if d > 0 else -1
                if last_sign != 0 and sign != last_sign:
                    y_flips += 1
                last_sign = sign
            dbg[-1] += f"翻转x={flips}/y={y_flips}"
            if flips < y_flips:
                continue
            # 方向：取窗口内**第一个显著位移**的方向（挥出方向）。
            # 用户挥出的第一段即意图方向；收手回程（挥完把手收回来）
            # 是紧跟其后的反向横扫，若用"相对起点偏移最大的点"定向，
            # 回程偏移最大会把方向带反（"收手时最容易串成反向"根因）。
            # 只认首段方向：单程挥、往返挥都正确；回程不覆盖方向。
            # 符号约定（用户视角，用户明确要求）：向右挥手 = -1（上一首），
            # 向左挥手 = +1（下一首）。原始图像坐标 x 增大经镜像显示后是
            # "手向左"（用户向左挥手）→ +1；x 减小 → 用户向右挥手 → -1。
            first_dir = 0.0
            for i in range(1, len(xs)):
                d = xs[i] - xs[i - 1]
                if abs(d) >= FLIP_DEADBAND:
                    first_dir = 1.0 if d > 0 else -1.0
                    break
            if x_span > best_span and first_dir != 0.0:
                best_span = x_span
                best_dir = first_dir

        # 事件激活和方向锁定（在循环外，只执行一次）
        if best_span > 0.0 and not self._wave_event_active:
            # 首次检测到有效挥手：激活事件并锁定方向
            self._wave_event_active = True
            self._wave_event_dir = best_dir

            # 保存当前最佳挥手的幅度数据（用于视觉特征白名单）
            # 需要在循环内部保存 best_x_span 和 best_ang_span
            # 这里简化处理：从历史队列计算
            for side in ('L', 'R'):
                vals = list(self._elbow_xy_history[side])
                if len(vals) >= 2:
                    xs = [p[0] for p in vals]
                    ys = [p[1] for p in vals]
                    self._latest_wave_x_span = max(xs) - min(xs)
                    # 计算角度跨度
                    sh = LEFT_SHOULDER if side == 'L' else RIGHT_SHOULDER
                    el = LEFT_ELBOW if side == 'L' else RIGHT_ELBOW
                    if landmarks is not None and landmarks.shape[0] > max(sh, el):
                        shoulder_vec = np.array([landmarks[sh, 0], landmarks[sh, 1]])
                        elbow_vec = np.array([landmarks[el, 0], landmarks[el, 1]])
                        arm_vec = elbow_vec - shoulder_vec
                        if np.linalg.norm(arm_vec) > 1e-6:
                            angles = [float(np.arctan2(p[1] - shoulder_vec[1], p[0] - shoulder_vec[0])) for p in vals if len(p) >= 2]
                            if len(angles) >= 2:
                                angles_unwrapped = []
                                for i, ang in enumerate(angles):
                                    if i == 0:
                                        angles_unwrapped.append(ang)
                                    else:
                                        while ang - angles_unwrapped[-1] > np.pi:
                                            ang -= 2.0 * np.pi
                                        while ang - angles_unwrapped[-1] < -np.pi:
                                            ang += 2.0 * np.pi
                                        angles_unwrapped.append(ang)
                                if len(angles_unwrapped) >= 2:
                                    self._latest_wave_ang_span = max(angles_unwrapped) - min(angles_unwrapped)
                    break  # 只保存一侧

        # 方向输出：事件级方向锁存
        # 注意：wave_direction 用于 gesture_presets 学习记录
        # 符号约定（用户视角）：向右挥手 = -1（上一首），向左挥手 = +1（下一首）
        self._wave_direction = self._wave_event_dir if self._wave_event_active else 0.0

        # 挥手事件生命周期管理
        if best_span > 0:
            # 检测到有效挥手：保持事件活跃，输出强度
            intensity = min(best_span * 2.5, 1.0)
            self._wave_still_frames = 0  # 重置静止计数
            return intensity
        elif self._wave_event_active:
            # 未检测到挥手，但事件仍活跃：增加静止计数
            self._wave_still_frames += 1
            if self._wave_still_frames >= 10:  # 连续10帧无挥手 → 事件结束
                self._wave_event_active = False
                self._wave_direction = 0.0
                self._wave_event_dir = 0.0
                self._wave_still_frames = 0
            return 0.0

        return 0.0

    def _calc_palm_hover(self, landmarks: np.ndarray) -> float:
        """
        举手暂停（手掌悬停）检测：整个手掌正对摄像头并悬停不动。

        纯几何判定（不看指根可见度评分——掌心正对镜头时 MediaPipe
        对指根的可见度评分反而偏低，按可见度卡会把正常姿势全部卡掉，
        这是"出现手指反而判定失败"的根因）：

        1. 手腕可见度 > 0.3（主关节评分可靠，作为整只手存在判据）
        2. 手臂举起：手腕 y 坐标 < 肩部 y 坐标（手在肩部上方）——
           参考人体举手运动学判据（Roborregos@Home：wrist.y < shoulder.y），
           结构性排除锤桌/拍桌等"手在桌面低位"的手势
        3. 掌心张开朝镜头：食指根/小指根的估计坐标都在手腕上方
           （手指竖起），且两点横向展开 ≥ 0.015（侧对镜头时展开塌陷）
        4. 悬停不动：该侧掌心追踪点在最近 10 帧窗口内
           横/纵抖动均 ≤ 0.04（约 4% 屏幕）

        与挥手天然互斥：挥手时掌心在动，条件 4 不满足；
        悬停时横向跨度低于挥手阈值，不会误触切歌。

        Returns:
            1.0 = 手掌悬停中，0.0 = 否。
        """
        WRIST_VIS = 0.3     # 手腕可见性要求（整只手存在判据）
        FINGER_ABOVE = 0.005  # 指根须在手腕上方的最小距离（放宽：掌心正对镜头时
                              # 指根估计噪声大，0.01 会卡掉正常举手）
        SPREAD_MIN = 0.008  # 食指根-小指根最小横向展开（放宽：自然并拢的举手）
        STILL_SPAN = 0.06   # 悬停允许的最大掌心抖动（6% 屏宽，容忍手部微颤+指根噪声）
        MIN_FRAMES = 4

        for side, idxs in (('L', (LEFT_WRIST, LEFT_INDEX, LEFT_PINKY, LEFT_SHOULDER)),
                           ('R', (RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY, RIGHT_SHOULDER))):
            w, ix, pk, s = idxs
            if landmarks[w, 3] <= WRIST_VIS:
                continue
            wy = float(landmarks[w, 1])
            # 手臂举起：手腕须在肩部上方（y 向下增大）——排除锤桌等低位手势
            if wy >= float(landmarks[s, 1]):
                continue
            # 手指竖起：两指根估计坐标均在手腕上方
            if not (float(landmarks[ix, 1]) < wy - FINGER_ABOVE
                    and float(landmarks[pk, 1]) < wy - FINGER_ABOVE):
                continue
            spread = abs(float(landmarks[ix, 0]) - float(landmarks[pk, 0]))
            if spread < SPREAD_MIN:
                continue
            vals = list(self._wrist_xy_history[side])
            if len(vals) < MIN_FRAMES:
                continue
            # 与挥手同款滤波：均值+中值抑制指根估计噪声，
            # 避免"手势已静止但原始坐标抖动超过阈值"导致举手不达标
            xs = _median3(_mean3([p[0] for p in vals]))
            ys = _median3(_mean3([p[1] for p in vals]))
            if (max(xs) - min(xs)) > STILL_SPAN or (max(ys) - min(ys)) > STILL_SPAN:
                continue
            # 手指静止检查：手指动作时手腕不动（腕部历史抖动小），
            # 但手指张开度（spread）会变化——要求窗口内 spread 稳定才
            # 判定"悬停"，结构性排除"手指动作误触举手暂停"
            e_vals = list(self._elbow_xy_history[side])
            spreads = [p[2] for p in e_vals if len(p) > 2]
            if spreads:
                spread_f = _median3(_mean3(spreads))
                spread_span = max(spread_f) - min(spread_f)
                if spread_span > STILL_SPAN:
                    continue

            # 调试输出：每 0.5 秒打印一次（校准用）
            now = time.time()
            if now - getattr(self, '_dbg_palm_t', 0.0) > 0.5:
                self._dbg_palm_t = now
                print(f"[掌心调试] {side} 展开={spread:.3f} "
                      f"腕vis={landmarks[w, 3]:.2f} 指vis={landmarks[ix, 3]:.2f}/"
                      f"{landmarks[pk, 3]:.2f} → 悬停中")
            return 1.0

        # 调试：手腕可见但未通过几何时，打印原因（每 0.5 秒）
        now = time.time()
        if now - getattr(self, '_dbg_palm_t', 0.0) > 0.5:
            for side, idxs in (('L', (LEFT_WRIST, LEFT_INDEX, LEFT_PINKY, LEFT_SHOULDER)),
                               ('R', (RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY, RIGHT_SHOULDER))):
                w, ix, pk, s = idxs
                if landmarks[w, 3] > WRIST_VIS:
                    wy = float(landmarks[w, 1])
                    raised = wy - float(landmarks[s, 1])
                    above = min(wy - float(landmarks[ix, 1]),
                                wy - float(landmarks[pk, 1]))
                    spread = abs(float(landmarks[ix, 0]) - float(landmarks[pk, 0]))
                    self._dbg_palm_t = now
                    print(f"[掌心调试] {side} 未通过: 手腕-肩={raised:.3f}(需<0) "
                          f"指上抬={above:.3f}(需>{FINGER_ABOVE}) "
                          f"展开={spread:.3f}(需>{SPREAD_MIN}) "
                          f"腕vis={landmarks[w, 3]:.2f} 指vis={landmarks[ix, 3]:.2f}/{landmarks[pk, 3]:.2f}")
                    break
        return 0.0

    def _calc_body_speed(
        self, shoulder_mid: np.ndarray, hip_mid: np.ndarray, dt: float
    ) -> float:
        """
        计算身体质心移动速度。

        质心定义为：肩中点与髋中点的中点（即 torso 中心）。
        速度 = 质心位移 / 时间差。

        Args:
            shoulder_mid: 肩中点坐标 (x, y)。
            hip_mid: 髋中点坐标 (x, y)。
            dt: 时间差（秒）。

        Returns:
            归一化后的身体速度，范围 [0, 1]。
        """
        centroid = (shoulder_mid + hip_mid) / 2.0

        if self._prev_landmarks is None or dt <= 0:
            self._speed_history.append(0.0)
            return 0.0

        # 上一帧的肩中点、髋中点
        prev_shoulder_mid = self._midpoint(self._prev_landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
        prev_hip_mid = self._midpoint(self._prev_landmarks, LEFT_HIP, RIGHT_HIP)
        prev_centroid = (prev_shoulder_mid + prev_hip_mid) / 2.0

        displacement = np.linalg.norm(centroid - prev_centroid)
        raw_speed = displacement / max(dt, 0.001)

        # 自适应归一化（衰减系数 0.995，约 4.6s 后 max 降到 50%）
        self._max_speed = max(self._max_speed * 0.995, raw_speed, 0.001)
        body_speed = raw_speed / self._max_speed
        body_speed = min(body_speed, 1.0)

        self._speed_history.append(body_speed)
        return body_speed

    def _calc_arm_amplitude(
        self, landmarks: np.ndarray, shoulder_mid: np.ndarray, hip_mid: np.ndarray
    ) -> float:
        """
        计算手臂摆动/伸展幅度。

        算法：
        1. 分别计算左、右手腕到对应肩膀的距离。
        2. 用肩-髋距离作为参考长度（近似躯干长度）。
        3. 手臂幅度 = 手腕到肩距离 / 参考长度，取两侧最大值。

        Args:
            landmarks: 当前帧关键点。
            shoulder_mid: 肩中点（仅用于计算参考长度）。
            hip_mid: 髋中点。

        Returns:
            归一化后的手臂幅度，范围 [0, 1]。
        """
        ref_length = np.linalg.norm(shoulder_mid - hip_mid)
        if ref_length < 1e-6:
            return 0.0

        left_arm = self._dist(landmarks, LEFT_SHOULDER, LEFT_WRIST)
        right_arm = self._dist(landmarks, RIGHT_SHOULDER, RIGHT_WRIST)
        max_arm = max(left_arm, right_arm)

        raw_amp = max_arm / ref_length

        # 自适应归一化（衰减系数 0.995）
        self._max_arm_amp = max(self._max_arm_amp * 0.995, raw_amp, 0.001)
        arm_amp = raw_amp / self._max_arm_amp
        arm_amp = min(arm_amp, 1.0)

        self._arm_amp_history.append(arm_amp)
        return arm_amp

    def _calc_body_rhythm(self, hip_mid: np.ndarray) -> float:
        """
        计算身体起伏/节奏性。

        算法：
        1. 将髋部中点 y 坐标加入历史队列。
        2. 对 y 坐标序列做一阶差分，得到垂直速度。
        3. 统计垂直速度在窗口内的零交叉（ZCR）次数。
        4. 零交叉率越高，说明上下起伏越频繁，节奏性越强。
        5. 同时用 y 坐标的标准差作为辅助指标（起伏幅度）。

        Args:
            hip_mid: 髋中点坐标 (x, y)。

        Returns:
            归一化后的节奏性，范围 [0, 1]。
        """
        y = hip_mid[1]
        self._centroid_y_history.append(y)

        if len(self._centroid_y_history) < 10:
            return 0.0

        arr = np.array(self._centroid_y_history)

        # 一阶差分（垂直速度）
        diff = np.diff(arr)
        if len(diff) < 2:
            return 0.0

        # 零交叉率：符号变化的次数
        signs = np.sign(diff)
        zcr = np.sum(np.abs(np.diff(signs)) > 0) / (len(diff) - 1)

        # 垂直起伏幅度（标准差）
        amplitude = float(np.std(arr))
        # 归一化：假设典型起伏标准差约 0.05（画面高度的 5%）
        amplitude_norm = min(amplitude / 0.05, 1.0)

        # 综合：ZCR 越高 + 幅度越大 → 节奏性越强
        rhythm = (zcr * 0.6 + amplitude_norm * 0.4)
        rhythm = min(rhythm, 1.0)
        return rhythm

    def _classify_posture(
        self,
        landmarks: np.ndarray,
        shoulder_mid: np.ndarray,
        hip_mid: np.ndarray,
        ankle_mid: np.ndarray,
        knee_mid: np.ndarray,
    ) -> str:
        """
        判断人体姿态：standing / sitting / lying / unknown。

        判断规则：
        - standing: 髋部到脚踝的垂直距离显著（> 肩到髋距离的 0.7 倍），且身体垂直。
        - sitting: 髋部到脚踝距离较小（< 肩到髋距离的 0.5 倍），但肩到髋距离仍然显著。
        - lying: 身体整体高度很小（最高点-最低点 < 0.2），或身体接近水平。
        - unknown: 关键点不足或无法判断。
        """
        # 检查关键点是否足够可见
        key_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]
        if not all(landmarks[i, 3] > 0.5 for i in key_indices):
            return 'unknown'

        # 计算垂直距离（注意 MediaPipe y 坐标向下为正，0 在上，1 在下）
        shoulder_y = shoulder_mid[1]
        hip_y = hip_mid[1]
        knee_y = knee_mid[1]
        ankle_y = ankle_mid[1]

        torso_len = abs(hip_y - shoulder_y)  # 躯干长度（肩到髋）
        leg_len = abs(ankle_y - hip_y)       # 腿长（髋到踝）
        thigh_len = abs(knee_y - hip_y)     # 大腿长度（髋到膝）

        # 获取所有关键点的边界框
        visible_y = landmarks[landmarks[:, 3] > 0.5, 1]
        visible_x = landmarks[landmarks[:, 3] > 0.5, 0]
        if len(visible_y) < 10:
            return 'unknown'

        body_height = float(np.max(visible_y) - np.min(visible_y))
        body_width = float(np.max(visible_x) - np.min(visible_x))

        # 躺下的判断：身体高度很小，或者宽度大于高度
        if body_height < 0.2 or (body_width > 0 and body_height / body_width < 0.6):
            return 'lying'

        # 坐下的判断：腿长明显短于躯干，且髋部与膝盖接近
        if leg_len < torso_len * 0.6 and thigh_len < torso_len * 0.5:
            return 'sitting'

        # 站立的判断：腿长足够，且身体高度足够
        if leg_len > torso_len * 0.6 and body_height > 0.3:
            return 'standing'

        return 'unknown'

    def _classify_hand_position(
        self,
        landmarks: np.ndarray,
        shoulder_mid: np.ndarray,
        hip_mid: np.ndarray,
    ) -> str:
        """
        判断双手位置：up / down / center / unknown。

        判断规则（收紧阈值，避免头部晃动误触发）：
        - up:   两只手都显著高于肩膀（y < shoulder_y - 0.08，至少高出 8% 画面高度）。
        - down: 两只手都低于髋部（y > hip_y + 0.03）。
        - center: 其他情况。
        - unknown: 手腕关键点不可见或可见度不足。
        """
        # 收紧可见性阈值：只有高置信度才判断，避免模糊检测误触发
        VIS_THRESHOLD = 0.75
        left_vis = landmarks[LEFT_WRIST, 3] > VIS_THRESHOLD
        right_vis = landmarks[RIGHT_WRIST, 3] > VIS_THRESHOLD

        if not left_vis and not right_vis:
            return 'unknown'

        shoulder_y = shoulder_mid[1]
        hip_y = hip_mid[1]

        # 显著偏移阈值：手腕必须明显高于肩膀才算 "up"，避免头部微动导致误判
        UP_OFFSET = 0.08    # 手腕必须在肩膀上方 8% 画面高度以上
        DOWN_OFFSET = 0.03  # 手腕必须低于髋部 3% 画面高度以下

        left_y = landmarks[LEFT_WRIST, 1] if left_vis else None
        right_y = landmarks[RIGHT_WRIST, 1] if right_vis else None

        # 如果只有一只手可见，以可见手为准（同样应用显著阈值）
        if left_vis and not right_vis:
            if left_y < shoulder_y - UP_OFFSET:
                return 'up'
            elif left_y > hip_y + DOWN_OFFSET:
                return 'down'
            else:
                return 'center'

        if right_vis and not left_vis:
            if right_y < shoulder_y - UP_OFFSET:
                return 'up'
            elif right_y > hip_y + DOWN_OFFSET:
                return 'down'
            else:
                return 'center'

        # 两只手都可见：应用显著阈值
        both_up = (left_y < shoulder_y - UP_OFFSET) and (right_y < shoulder_y - UP_OFFSET)
        both_down = (left_y > hip_y + DOWN_OFFSET) and (right_y > hip_y + DOWN_OFFSET)

        if both_up:
            return 'up'
        elif both_down:
            return 'down'
        else:
            return 'center'

    def _classify_activity(self, motion_energy: float, body_speed: float) -> bool:
        """
        判断是否处于活动状态（区别于静止）。

        阈值基于经验设定：运动能量或身体速度超过一定值即认为在活动。
        """
        # 阈值：能量 > 0.15 或 速度 > 0.15 才认为在活动（避免坐姿微动误触发）
        return motion_energy > 0.15 or body_speed > 0.15

    # ═══════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════

    def _parse_landmarks(self, pose_landmarks) -> np.ndarray:
        """
        将 MediaPipe 的 landmarks 转换为 numpy 数组。

        兼容旧 API (NormalizedLandmarkList) 和新 API (list of NormalizedLandmark)。

        Returns:
            ndarray of shape (33, 4) 对应 [x, y, z, visibility]。
        """
        landmarks = np.zeros((33, 4), dtype=np.float32)
        # 兼容旧 API：pose_landmarks 有 .landmark 属性且不为 None
        if hasattr(pose_landmarks, 'landmark') and pose_landmarks.landmark is not None:
            iterable = pose_landmarks.landmark
        elif hasattr(pose_landmarks, '__iter__') and not isinstance(pose_landmarks, (str, bytes)):
            # 新 API：直接是可迭代对象（列表等）
            iterable = pose_landmarks
        else:
            # 无法解析，返回全零数组
            return landmarks
        for idx, lm in enumerate(iterable):
            if idx >= 33:
                break
            vis = getattr(lm, 'visibility', 1.0)
            landmarks[idx] = [lm.x, lm.y, lm.z, vis]
        return landmarks

    def _landmarks_to_list(self, landmarks: np.ndarray) -> List[Dict[str, float]]:
        """将 landmarks ndarray 转换为列表格式，用于外部接口。"""
        result = []
        for i in range(33):
            result.append({
                'x': float(landmarks[i, 0]),
                'y': float(landmarks[i, 1]),
                'z': float(landmarks[i, 2]),
                'visibility': float(landmarks[i, 3]),
            })
        return result

    def _midpoint(self, landmarks: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
        """计算两个关键点之间的中点。"""
        return (landmarks[idx_a, :2] + landmarks[idx_b, :2]) / 2.0

    def _dist(self, landmarks: np.ndarray, idx_a: int, idx_b: int) -> float:
        """计算两个关键点之间的欧氏距离。"""
        return float(np.linalg.norm(landmarks[idx_a, :2] - landmarks[idx_b, :2]))

    def _get_default_features(self, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        返回默认（未检测到人体时）的特征字典。

        Args:
            timestamp: 可选时间戳，默认使用当前时间。
        """
        return {
            'timestamp': timestamp if timestamp is not None else time.time(),
            'motion_energy': 0.0,
            'arm_motion_energy': 0.0,
            'arm_horizontal_motion': 0.0,
            'palm_hover': 0.0,
            'wave_direction': 0.0,
            'body_speed': 0.0,
            'arm_amplitude': 0.0,
            'body_rhythm': 0.0,
            'posture': 'unknown',
            'hand_position': 'unknown',
            'is_active': False,
            'landmarks': [],
            # 手部精细特征（HandLandmarker 21 点，默认空）
            'hand_landmarks': [],
            'hand_landmarks_L': [],
            'hand_landmarks_R': [],
            'finger_curls_L': None,
            'finger_curls_R': None,
            # 静态构型特征（默认 None：采样端 isinstance 过滤自动跳过）
            'pose_head_dx': None,
            'pose_head_dy': None,
            'pose_torso_dx': None,
            'pose_torso_dy': None,
            'pose_upper_dx_L': None,
            'pose_upper_dy_L': None,
            'pose_upper_dx_R': None,
            'pose_upper_dy_R': None,
            'pose_fore_dx_L': None,
            'pose_fore_dy_L': None,
            'pose_fore_dx_R': None,
            'pose_fore_dy_R': None,
            'pose_hand_dx_L': None,
            'pose_hand_dy_L': None,
            'pose_hand_dx_R': None,
            'pose_hand_dy_R': None,
            'pose_spread_L': None,
            'pose_spread_R': None,
            # 肩部特征（Pose 11/12 点，默认 None）
            'shoulder_landmarks_L': None,
            'shoulder_landmarks_R': None,
            'shoulder_mid_x': None,
            'shoulder_mid_y': None,
            'shoulder_width': None,
            'shoulder_tilt': None,
            # 面部精细特征（FaceLandmarker 478 点，默认空）
            'face_landmarks': [],
            'face_present': False,
            'face_mouth_open': None,
            'face_smile': None,
            'face_head_tilt': None,
        }

    def _deep_copy_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """深拷贝特征字典（landmarks / hand_landmarks 列表需要独立复制）。"""
        copied = dict(features)
        if 'landmarks' in copied and isinstance(copied['landmarks'], list):
            copied['landmarks'] = [dict(pt) for pt in copied['landmarks']]
        # hand_landmarks：双层（每只手一个 21 点列表）
        if 'hand_landmarks' in copied and isinstance(copied['hand_landmarks'], list):
            copied['hand_landmarks'] = [
                [dict(pt) for pt in hand] for hand in copied['hand_landmarks']]
        # hand_landmarks_L/R：单层 dict 列表（21 个点）
        for key in ('hand_landmarks_L', 'hand_landmarks_R'):
            if key in copied and isinstance(copied[key], list):
                copied[key] = [dict(pt) for pt in copied[key]]
        # face_landmarks：单层 dict 列表（478 个点）
        if 'face_landmarks' in copied and isinstance(copied['face_landmarks'], list):
            copied['face_landmarks'] = [dict(pt) for pt in copied['face_landmarks']]
        return copied


# ──────────────────────── 模块自测入口 ────────────────────────
if __name__ == '__main__':
    """
    当直接运行本文件时，进入调试模式：
    打开摄像头，实时显示运动特征，并在画面上绘制 MediaPipe 骨架。
    按 'q' 键退出。
    """
    import sys

    print("=" * 50)
    print("MotionCapture 模块自测程序")
    print("=" * 50)
    print("按 'q' 键退出调试窗口。\n")

    cap = MotionCapture(camera_id=0, model_complexity=1)
    cap._debug = True

    try:
        cap.start()
    except RuntimeError as e:
        print(f"[错误] {e}")
        print("\n提示：如果没有摄像头，本模块仍然可以正常导入使用，")
        print("      只需在有摄像头的环境中调用 start() 即可。")
        sys.exit(1)

    # 用于绘制的 MediaPipe 工具
    if not cap._use_dummy:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    else:
        mp_drawing = None
        mp_drawing_styles = None

    try:
        while True:
            frame = cap.get_camera_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            features = cap.get_latest_motion_features()

            # 在画面上叠加特征信息
            y_offset = 20
            line_height = 22
            mode_label = "SIMULATION MODE" if cap._use_dummy else "REAL MODE"
            info_texts = [
                f"[{mode_label}]",
                f"motion_energy:      {features['motion_energy']:.3f}",
                f"arm_motion_energy:  {features['arm_motion_energy']:.3f}",
                f"body_speed:         {features['body_speed']:.3f}",
                f"arm_amplitude:      {features['arm_amplitude']:.3f}",
                f"body_rhythm:        {features['body_rhythm']:.3f}",
                f"posture:            {features['posture']}",
                f"hand_position:      {features['hand_position']}",
                f"is_active:          {features['is_active']}",
            ]
            for text in info_texts:
                cv2.putText(
                    frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                y_offset += line_height

            # 骨架绘制已禁用（landmarks 坐标与显示帧存在同步问题，暂时隐藏）
            # 如需重新启用，需要解决 _latest_frame 与 _latest_features 的原子同步问题
            # landmarks = features.get('landmarks', [])
            # h, w, _ = frame.shape
            # if len(landmarks) == 33:
            #     ...
            # 骨架绘制已禁用（landmarks 坐标与显示帧存在同步问题，暂时隐藏）
            # 如需重新启用，需要解决 _latest_frame 与 _latest_features 的原子同步问题
            # landmarks = features.get('landmarks', [])
            # h, w, _ = frame.shape
            # if len(landmarks) == 33:
            #     POSE_CONNECTIONS = [...]
            #     绘制连接线...
            #     绘制关键点...
            # else:
            #     cv2.putText(frame, "NO PERSON DETECTED", ...)

            cv2.imshow('SoulPlayer MotionCapture Debug', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        print("\n调试窗口已关闭，模块自测结束。")
