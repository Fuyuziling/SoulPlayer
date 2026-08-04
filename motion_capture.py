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
                r'D:\tmp\pose_landmarker_full.task',
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
                    temp_ascii_path = ascii_candidates[2]  # 使用 tempfile 路径
                    try:
                        shutil.copy2(project_model_path, temp_ascii_path)
                        model_path = temp_ascii_path
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
        # 手腕坐标历史（用于挥手跨度检测：记录最近0.5秒的x,y坐标）
        self._wrist_xy_history: deque = deque(maxlen=10)
        # 挥手方向（原始图像坐标）：+1 = 原始x增大（显像框中从右向左划），
        # -1 = 原始x减小（显像框中从左向右划），0 = 无有效挥手
        self._wave_direction: float = 0.0
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

        # 摄像头已打开：设置分辨率
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
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

        # 释放 detector 资源
        if self._use_tasks_api and self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass

        # 重置历史数据
        self._prev_landmarks = None
        self._prev_time = None
        self._centroid_y_history.clear()
        self._energy_history.clear()
        self._arm_amp_history.clear()
        self._arm_motion_history.clear()
        self._arm_hmotion_history.clear()
        self._wrist_xy_history.clear()
        self._speed_history.clear()

        with self._lock:
            self._latest_features = self._get_default_features()
            self._latest_frame = None

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
                else:
                    # ── 旧 API (MediaPipe 0.9.x) ──
                    rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb_frame)
                    current_time = time.time()
                    features = self._extract_features(results, current_time)

                # ── 更新缓存（加锁） ──
                # features 中的 landmarks 基于原始图像坐标
                # _latest_frame 存储镜像后的显示帧
                with self._lock:
                    self._latest_features = features
                    self._latest_frame = display_frame

            except Exception as e:
                # 推理异常时不崩溃，记录错误并继续下一帧
                if self._debug:
                    print(f"[MotionCapture] 推理异常: {e}")

            # 控制帧率，避免 CPU 过高（约 30 FPS）
            time.sleep(0.001)


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
        }
        return features

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


    def _calc_arm_horizontal_motion(self, landmarks: np.ndarray, dt: float) -> float:
        """
        挥手跨度检测：分别追踪左右手腕部的横向运动，取较大 span。

        追踪点只用腕部（最稳定的大关节）。手腕不可见时跳过该帧，
        绝不用其他点顶替——不同物理点间的固定偏移会在可见性抖动时
        造成追踪点跳变，凭空产生水平位移导致误触。

        严格横向判定（只看横向位移，纵向完全不参与）：
        1. 将腕部追踪点坐标存入历史队列（10 帧窗口）
        2. 对 x 序列做 3 点中值滤波——周期性的估计抖动（上下移动手时
           腕部 x 在真实值附近来回摆动）会被滤掉，而真实挥手是持续
           多帧的同向位移，中值滤波不改变其跨度
        3. 分别追踪左右手，计算滤波后 x 的 max - min = 横向跨度，
           跨度须 ≥ 15% 屏宽（日常抬手/上下移动的横向分量一般 < 10%，
           正常挥手跨度在 30% 以上，15% 能可靠区分两者）
        4. 方向取窗口内相对起点偏移最大的点（单程挥与往返挥都能定向）
        5. 返回值 = min(跨度 * 2.5, 1.0)

        Args:
            landmarks: 当前帧 33 关键点，shape=(33, 4)。
            dt: 与上一帧的时间差（秒，保留参数兼容）。

        Returns:
            挥手强度 [0, 1]，跨越大半个屏幕时接近 1.0。
        """
        VIS_THRESHOLD = 0.3       # 可见性阈值，容忍侧手/遮挡
        WAVE_X_THRESHOLD = 0.15   # 横向跨度阈值：15% 屏幕宽度
        MIN_FRAMES = 4            # 最少帧数

        def _median3(vals):
            """3 点中值滤波：抑制单帧/双帧周期的位置抖动，保留真实位移。"""
            if len(vals) < 3:
                return list(vals)
            out = [vals[0]]
            for i in range(1, len(vals) - 1):
                out.append(sorted((vals[i - 1], vals[i], vals[i + 1]))[1])
            out.append(vals[-1])
            return out

        def hand_point(idxs) -> Optional[tuple]:
            """挥手追踪点：只用腕部（idxs 第一个点）。

            手腕不可见时直接跳过本帧，绝不用手指根等其他点顶替——
            不同物理点之间有固定偏移，可见性抖动时追踪点会在两点之间
            来回跳，凭空产生水平位移（"掌心上下移动误触切歌"的根因）。"""
            w = idxs[0]
            if landmarks[w, 3] > VIS_THRESHOLD:
                return (float(landmarks[w, 0]), float(landmarks[w, 1]))
            return None

        left_pt = hand_point([LEFT_WRIST, LEFT_INDEX, LEFT_PINKY])
        right_pt = hand_point([RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY])

        # 双手均不可见：清空历史防止残留
        if left_pt is None and right_pt is None:
            self._wrist_xy_history.clear()
            self._wave_direction = 0.0
            return 0.0

        # 为每只可见手记录腕部追踪点坐标（带标签 'L'/'R'）
        if left_pt is not None:
            self._wrist_xy_history.append(('L', left_pt[0], left_pt[1]))
        if right_pt is not None:
            self._wrist_xy_history.append(('R', right_pt[0], right_pt[1]))

        if len(self._wrist_xy_history) < MIN_FRAMES:
            self._wave_direction = 0.0
            return 0.0

        # 分别计算左右手的横向跨度与方向，取最大有效 span
        best_span = 0.0
        best_dir = 0.0

        for side in ('L', 'R'):
            side_vals = [p for p in self._wrist_xy_history if p[0] == side]
            if len(side_vals) < MIN_FRAMES:
                continue
            xs = _median3([p[1] for p in side_vals])
            x_span = max(xs) - min(xs)
            if x_span > WAVE_X_THRESHOLD:
                # 方向：取窗口内相对起点偏移最大的点，
                # 单程挥与"甩出去再收回"的往返挥都能正确定向
                x0 = xs[0]
                x_peak = max(xs[1:], key=lambda v: abs(v - x0))
                dx = x_peak - x0
                if x_span > best_span:
                    best_span = x_span
                    best_dir = 1.0 if dx > 0 else -1.0

        self._wave_direction = best_dir

        if best_span > 0:
            intensity = min(best_span * 2.5, 1.0)
            return intensity

        return 0.0

    def _calc_palm_hover(self, landmarks: np.ndarray) -> float:
        """
        举手暂停（手掌悬停）检测：整个手掌正对摄像头并悬停不动。

        严格判定（三个条件同时满足，任一手满足即可）：
        1. 掌心朝镜头：手腕可见度 > 0.4，食指根/小指根可见度 > 0.25
           （姿态模型对指根的可见度评分天然偏低，0.25 即可区分
           "掌心正对"与"掌背/侧对/完全不可见"）
        2. 手掌张开正对：食指根与小指根的横向展开 ≥ 0.02
           （侧对镜头时两点投影重叠，展开度塌陷）
        3. 悬停不动：该侧腕部追踪点在最近 10 帧窗口内
           横/纵抖动均 ≤ 0.04（约 4% 屏幕）

        与挥手天然互斥：挥手时腕部在动，条件 3 不满足；
        悬停时横向跨度低于挥手阈值，不会误触切歌。

        Returns:
            1.0 = 手掌悬停中，0.0 = 否。
        """
        WRIST_VIS = 0.4     # 手腕可见性要求
        FINGER_VIS = 0.25   # 指根可见性要求（姿态模型对指根评分偏低，不能按常规 0.5 卡）
        SPREAD_MIN = 0.02   # 食指根-小指根最小横向展开（归一化坐标）
        STILL_SPAN = 0.04   # 悬停允许的最大腕部抖动
        MIN_FRAMES = 4

        for side, idxs in (('L', (LEFT_WRIST, LEFT_INDEX, LEFT_PINKY)),
                           ('R', (RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY))):
            w, ix, pk = idxs
            if not (landmarks[w, 3] > WRIST_VIS
                    and landmarks[ix, 3] > FINGER_VIS
                    and landmarks[pk, 3] > FINGER_VIS):
                continue
            spread = abs(float(landmarks[ix, 0]) - float(landmarks[pk, 0]))
            if spread < SPREAD_MIN:
                continue
            vals = [p for p in self._wrist_xy_history if p[0] == side]
            if len(vals) < MIN_FRAMES:
                continue
            xs = [p[1] for p in vals]
            ys = [p[2] for p in vals]
            if (max(xs) - min(xs)) > STILL_SPAN or (max(ys) - min(ys)) > STILL_SPAN:
                continue
            return 1.0
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
        }

    def _deep_copy_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """深拷贝特征字典（landmarks 列表需要独立复制）。"""
        copied = dict(features)
        if 'landmarks' in copied and isinstance(copied['landmarks'], list):
            copied['landmarks'] = [dict(pt) for pt in copied['landmarks']]
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
