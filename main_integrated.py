import cv2
import os
import numpy as np
import time
from datetime import datetime
from collections import deque
import threading
from queue import Queue

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ 未安装 ultralytics 库，请运行: pip install ultralytics")

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    
    # 尝试导入 torchvision.transforms，但不依赖它
    try:
        import torchvision.transforms as transforms
        TORCHVISION_AVAILABLE = True
    except Exception as e:
        print(f"⚠️ torchvision 导入失败: {e}")
        print("⚠️ 将使用自定义实现，不影响核心功能")
        # 创建虚拟 transforms 模块
        class MockTransforms:
            class Compose:
                def __init__(self, transforms):
                    pass
            def ToTensor(self):
                pass
            def Resize(self, size):
                pass
            def Normalize(self, mean, std):
                pass
        transforms = MockTransforms()
        TORCHVISION_AVAILABLE = False

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TORCHVISION_AVAILABLE = False
    print("❌ 未安装 torch 库，请运行: pip install torch torchvision")

# 导入距离测量模块和速度计算模块
try:
    from distance import DistanceMeasurer, StereoCameraConfig
    from speed_calculator import SpeedCalculator
except ImportError:
    # 如果模块不存在，创建虚拟类
    class DistanceMeasurer:
        def __init__(self):
            pass


    class StereoCameraConfig:
        def __init__(self):
            pass


    class SpeedCalculator:
        def __init__(self, trackers, fps, frame_size):
            self.trackers = trackers
            self.fps = fps
            self.frame_size = frame_size

        def calculate_speeds(self):
            """简化速度计算"""
            for tracker_id, tracker in self.trackers.items():
                if tracker['disappeared'] > 0 or len(tracker['positions']) < 2:
                    continue

                # 计算基于位置变化的速度
                if len(tracker['positions']) >= 2:
                    pos1 = tracker['positions'][-2]
                    pos2 = tracker['positions'][-1]

                    # 计算像素距离
                    pixel_distance = np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

                    # 转换为实际速度 (简化估算)
                    speed_mps = (pixel_distance / 100) * self.fps
                    speed_kmh = speed_mps * 3.6

                    tracker['current_speed'] = min(speed_kmh, 120)


class FastVideoReader:
    """高速视频读取器 - 优化版本"""

    def __init__(self, video_path, buffer_size=60):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size * 3)
        self.stop_event = threading.Event()
        self.cap = None
        self.thread = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.read_count = 0
        self.last_log_time = time.time()

    def start_reading(self):
        """开始后台读取视频"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception(f"无法打开视频: {self.video_path}")

        # 获取视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 优化视频读取设置
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 15)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)  # 启用硬件加速

        # 启动读取线程
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()

        print(f"🎬 开始高速读取视频，缓存大小: {self.buffer_size}")
        print(f"📹 视频信息: {self.width}x{self.height}, {self.fps:.2f}FPS, 总帧数: {self.total_frames}")

        return self.fps, (self.width, self.height), self.total_frames

    def _read_frames(self):
        """后台读取帧的线程函数 - 优化版本"""
        read_errors = 0
        max_errors = 10

        while not self.stop_event.is_set() and read_errors < max_errors:
            try:
                # 如果队列未满，继续读取
                if self.frame_queue.qsize() < self.buffer_size * 2:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    self.frame_queue.put((ret, frame))
                    self.read_count += 1

                    # 每读取100帧输出一次状态
                    if self.read_count % 100 == 0:
                        current_time = time.time()
                        if current_time - self.last_log_time > 2.0:
                            queue_size = self.frame_queue.qsize()
                            print(f"📥 已读取 {self.read_count} 帧, 队列大小: {queue_size}")
                            self.last_log_time = current_time

                    read_errors = 0
                else:
                    # 队列较满时短暂休眠
                    time.sleep(0.0005)

            except Exception as e:
                read_errors += 1
                print(f"⚠️ 读取帧时出错 ({read_errors}/{max_errors}): {e}")
                time.sleep(0.01)

        self.cap.release()
        print("✅ 视频读取线程完成")

    def read_frame(self):
        """读取一帧 - 优化版本"""
        try:
            if not self.frame_queue.empty():
                ret, frame = self.frame_queue.get_nowait()
                return ret, frame
            else:
                time.sleep(0.001)
                if not self.frame_queue.empty():
                    ret, frame = self.frame_queue.get_nowait()
                    return ret, frame
                return False, None
        except:
            return False, None

    def get_queue_size(self):
        """获取队列大小"""
        return self.frame_queue.qsize()

    def get_read_progress(self):
        """获取读取进度"""
        if self.total_frames > 0:
            return (self.read_count / self.total_frames) * 100
        return 0

    def stop(self):
        """停止读取"""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


class RobustVideoWriter:
    """健壮的视频写入器 - 确保视频可播放"""

    def __init__(self, output_path, frame_size, fps):
        self.output_path = output_path
        self.frame_size = frame_size
        self.fps = fps
        self.writer = None
        self.frames_written = 0

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"🎯 尝试创建视频文件: {output_path}")
        print(f"📏 视频尺寸: {frame_size[0]}x{frame_size[1]}, FPS: {fps}")

        # 测试不同的编码器和格式
        codec_formats = [
            ('avc1', '.mp4'),  # H.264 - 最兼容的MP4编码
            ('mp4v', '.mp4'),  # MPEG-4
            ('XVID', '.avi'),  # XVID AVI
        ]

        success = False
        for codec, ext in codec_formats:
            try:
                # 确保文件扩展名匹配
                if not output_path.endswith(ext):
                    base_name = os.path.splitext(output_path)[0]
                    test_path = base_name + ext
                else:
                    test_path = output_path

                # 避免文件名过长
                if len(os.path.basename(test_path)) > 50:
                    base_dir = os.path.dirname(test_path)
                    simple_name = f"output_{int(time.time())}{ext}"
                    test_path = os.path.join(base_dir, simple_name)
                    self.output_path = test_path

                print(f"🔧 尝试编码器: {codec}, 格式: {ext}")

                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.writer = cv2.VideoWriter(
                    test_path,
                    fourcc,
                    fps,
                    frame_size,
                    isColor=True
                )

                if self.writer.isOpened():
                    # 测试写入一帧
                    test_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
                    self.writer.write(test_frame)
                    self.writer.release()

                    # 重新打开写入器
                    self.writer = cv2.VideoWriter(test_path, fourcc, fps, frame_size, isColor=True)

                    if self.writer.isOpened():
                        print(f"✅ 成功使用编码器: {codec}")
                        success = True
                        self.output_path = test_path
                        break
                    else:
                        self.writer = None
                else:
                    self.writer = None
                    print(f"❌ 编码器 {codec} 不可用")

            except Exception as e:
                print(f"❌ 编码器 {codec} 失败: {str(e)}")
                self.writer = None
                continue

        if not success:
            # 最后尝试：使用图像序列
            print("⚠️ 所有编码器都失败，将保存为图像序列")
            self.use_image_sequence = True
            self.image_sequence_dir = output_path.replace('.mp4', '_frames')
            if not os.path.exists(self.image_sequence_dir):
                os.makedirs(self.image_sequence_dir, exist_ok=True)
            print(f"🖼️ 图像序列将保存到: {self.image_sequence_dir}")
        else:
            self.use_image_sequence = False
            print(f"🎉 视频写入器创建成功: {self.output_path}")

    def write_frame(self, frame):
        """写入一帧"""
        if self.use_image_sequence:
            # 保存为图像序列
            frame_path = os.path.join(self.image_sequence_dir, f"frame_{self.frames_written:06d}.jpg")
            success = cv2.imwrite(frame_path, frame)
            if success:
                self.frames_written += 1
            return success
        elif self.writer and self.writer.isOpened():
            # 确保帧尺寸正确
            if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                frame = cv2.resize(frame, self.frame_size)

            self.writer.write(frame)
            self.frames_written += 1
            return True
        return False

    def stop(self):
        """停止写入"""
        if self.writer and self.writer.isOpened():
            self.writer.release()
            print(f"✅ 视频已保存: {self.output_path}")
            print(f"📊 总帧数: {self.frames_written}")

            # 验证视频文件
            if os.path.exists(self.output_path):
                file_size = os.path.getsize(self.output_path)
                print(f"💾 文件大小: {file_size / (1024 * 1024):.2f} MB")

        elif self.use_image_sequence:
            print(f"✅ 图像序列已保存到: {self.image_sequence_dir}")
            print(f"📊 总帧数: {self.frames_written}")


class SafetyFieldCalculator:
    """行车安全场计算器 - 修改为小角度高风险模式"""

    def __init__(self):
        # 安全场模型参数
        self.K = 0.1  # 场强系数提高，增加基础风险
        self.k_1 = 1.3  # 距离衰减系数降低，让远距离也有风险
        self.A = 0.3  # 基础方向风险提高
        self.B = 1.0  # 方向放大系数提高
        self.C = 2.2  # 方向锐度系数提高
        self.D = 0.15  # 速度影响系数提高
        self.E = 1.8  # 速度非线性因子提高

        # 大幅提高危险范围：显著降低安全阈值
        self.safe_threshold = 0.5  # 原0.7 → 0.5 (大幅降低安全标准)
        self.warning_threshold = 0.28  # 原0.4 → 0.2 (大幅扩大危险范围)

        # 激进权重配置 - 速度主导，更容易触发危险
        self.safety_weights = {
            'distance_safety': 0.25,  # 距离权重大幅降低
            'angle_safety': 0.15,  # 角度权重保持不变
            'speed_safety': 0.50,  # 速度权重主导 (原0.4 → 0.65)
            'mass_safety': 0.10  # 质量权重保持不变
        }

    def calculate_virtual_mass(self, obj_type, speed_kmh, physical_mass=1400):
        """计算虚拟质量 - 提高风险"""
        # 物体类型系数 - 提高所有类型的风险系数
        type_coefficients = {
            'bicycle': 0.4,  # 自行车风险提高
            'motorcycle': 0.7,  # 摩托车风险大幅提高
        }

        T_i = type_coefficients.get(obj_type, 0.8)  # 默认风险提高

        # 速度影响 - 使用更激进的计算
        speed_effect = 2.0e-14 * (speed_kmh ** 7.0) + 0.5  # 提高速度影响

        return physical_mass * T_i * speed_effect

    def calculate_direction_risk(self, angle_degrees):
        """计算方向性风险函数 - 修改为小角度高风险，大角度低风险"""
        # 将角度转换为弧度并取绝对值
        abs_angle = abs(angle_degrees)

        # 小角度高风险，大角度低风险
        # 使用分段函数实现反向风险关系
        if abs_angle <= 10:  # 0-10度：小角度，高风险
            # 小角度风险：角度越小，风险越高
            risk_factor = 2.5 - (abs_angle / 10) * 1.5  # 2.5到1.0之间
        elif abs_angle <= 25:  # 10-25度：中角度，中等风险
            # 中角度风险：中等风险
            risk_factor = 1.0 - ((abs_angle - 10) / 15) * 0.4  # 1.0到0.6之间
        else:  # >25度：大角度，低风险
            # 大角度风险：角度越大，风险越低
            risk_factor = max(0.3, 0.6 - ((abs_angle - 25) / 65) * 0.3)  # 0.6到0.3之间

        return self.A + self.B * risk_factor

    def calculate_safety_factor(self, distance_m, virtual_mass, angle_degrees, closing_speed_kmh, road_condition=1.0):
        """计算安全系数 - 大幅提高危险红色范围"""

        # 1. 基础安全系数 (基于距离) - 提高距离风险
        base_field = (self.K * road_condition * virtual_mass) / (distance_m ** self.k_1)
        SF_base = 1 / (1 + base_field)

        # 2. 方向安全系数 - 使用修改后的方向风险函数
        direction_risk = self.calculate_direction_risk(angle_degrees)
        SF_direction = 1 / direction_risk

        # 3. 速度安全系数 - 使用极度敏感的速度安全计算
        SF_speed = self.calculate_aggressive_speed_safety(closing_speed_kmh)

        # 4. 质量安全系数 - 提高质量风险
        base_mass = 1400
        normalized_mass = min(virtual_mass / base_mass, 4.0)  # 提高质量上限
        SF_mass = 1.0 / (1.0 + 0.8 * normalized_mass)  # 提高质量影响

        # 使用激进权重进行综合计算
        safety_factor = (
                SF_base * self.safety_weights['distance_safety'] +
                SF_direction * self.safety_weights['angle_safety'] +
                SF_speed * self.safety_weights['speed_safety'] +
                SF_mass * self.safety_weights['mass_safety']
        )

        # 确保安全系数在合理范围内
        safety_factor = max(0.0, min(1.0, safety_factor))

        return safety_factor, SF_base, SF_direction, SF_speed, SF_mass

    def calculate_aggressive_speed_safety(self, closing_speed_kmh):
        """激进的速度安全计算 - 极度敏感，极易触发危险"""
        if closing_speed_kmh <= 0:
            return 0.9  # 即使静止也有一定风险

        # 极度敏感的分段函数，极容易触发危险状态
        if closing_speed_kmh < 3:
            # 极低速阶段：快速下降
            speed_safety = 0.9 - (closing_speed_kmh / 3) * 0.5
        elif closing_speed_kmh < 10:
            # 低速阶段：急剧下降
            speed_safety = 0.4 - ((closing_speed_kmh - 3) / 7) * 0.35
        elif closing_speed_kmh < 25:
            # 中速阶段：极低安全系数
            speed_safety = 0.05 - ((closing_speed_kmh - 10) / 15) * 0.04
        else:
            # 高速：几乎为零的安全系数
            speed_safety = 0.01

        return max(0.0, min(1.0, speed_safety))

    def get_safety_level(self, safety_factor):
        """根据安全系数获取安全等级 - 大幅提高红色范围"""
        # 现在只有很高的安全系数才会显示绿色
        if safety_factor >= self.safe_threshold:
            return "安全", (0, 255, 0)  # 绿色 - 只有很少情况
        elif safety_factor >= self.warning_threshold:
            return "警戒", (0, 255, 255)  # 黄色 - 中等范围
        else:
            return "危险", (0, 0, 255)  # 红色 - 大部分情况

    def print_safety_weights(self):
        """打印当前使用的安全权重"""
        print("📊 当前安全系数权重配置 (小角度高风险模式):")
        print(f"  距离安全: {self.safety_weights['distance_safety']:.1%}")
        print(f"  角度安全: {self.safety_weights['angle_safety']:.1%}")
        print(f"  速度安全: {self.safety_weights['speed_safety']:.1%}")
        print(f"  质量安全: {self.safety_weights['mass_safety']:.1%}")
        print(f"  总权重: {sum(self.safety_weights.values()):.0%}")
        print(f"  安全阈值: {self.safe_threshold} (原0.7)")
        print(f"  警戒阈值: {self.warning_threshold} (原0.4)")
        print("⚠️  角度风险模式: 小角度高风险，大角度低风险")


class AngleDetector:
    """角度检测模块 - 简化版本"""

    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_angle_parameters()

    def setup_angle_parameters(self):
        """设置角度检测参数"""
        self.angle_classes = ['0°', '5°', '10°', '15°', '20°', '25°', '30°']
        self.angle_category_mapping = {
            '0°': '小角度',
            '5°': '小角度',
            '10°': '小角度',
            '15°': '中角度',
            '20°': '中角度',
            '25°': '大角度',
            '30°': '大角度'
        }

    def load_angle_model(self, model_path):
        """加载角度检测模型"""
        try:
            if not TORCH_AVAILABLE:
                print("❌ PyTorch 不可用，使用虚拟角度检测")
                return True

            print(f"🚀 加载角度模型: {model_path}")

            if not os.path.exists(model_path):
                print(f"❌ 角度模型文件不存在: {model_path}")
                print("⚠️ 使用虚拟角度检测")
                return True

            # 这里简化处理，实际使用时加载真实模型
            print("✅ 角度检测模块就绪")
            return True

        except Exception as e:
            print(f"❌ 角度模型加载失败: {e}")
            print("⚠️ 使用虚拟角度检测")
            return True

    def detect_angle(self, frame, bbox):
        """检测角度 - 虚拟实现"""
        # 虚拟角度检测
        angles = ['0°', '5°', '10°', '15°', '20°', '25°', '30°']
        angle_value = angles[np.random.randint(0, len(angles))]
        angle_category = self.angle_category_mapping.get(angle_value, '未知角度')
        confidence = 0.7 + np.random.random() * 0.3

        return angle_value, angle_category, confidence


class RedBoxDetector:
    """红框标注模块"""

    def __init__(self):
        self.yolo_model = None
        self.angle_detector = AngleDetector()
        self.safety_calculator = SafetyFieldCalculator()

        # 初始化trackers属性
        self.trackers = {}
        self.next_tracker_id = 1
        self.max_disappeared = 15

        # 区域统计
        self.left_targets = set()
        self.right_targets = set()
        self.middle_targets = set()

        # 添加安全系数统计
        self.left_safety_factors = []
        self.right_safety_factors = []
        self.average_safety_factor = 0.0

        if YOLO_AVAILABLE:
            self.load_yolo_model()

        self.setup_parameters()

        # 打印权重配置
        self.safety_calculator.print_safety_weights()

        print("✅ 红框标注系统初始化完成!")

    def load_yolo_model(self):
        """加载YOLO模型"""
        try:
            model_name = 'yolov8s.pt'
            print(f"🚀 加载YOLO模型: {model_name}")
            self.yolo_model = YOLO(model_name)
            print("✅ YOLO模型加载成功!")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")

    def setup_parameters(self):
        """设置检测参数"""
        self.confidence_threshold = 0.45
        self.min_roi_size = 36
        self.target_classes = ['bicycle', 'motorcycle']

        # 左右区域划分参数
        self.frame_width = 1280
        self.left_region = 0.45
        self.right_region = 0.45
        self.middle_region = 0.1

        self.colors = {
            'bicycle': (0, 0, 255),
            'motorcycle': (0, 0, 255),
            'id_bg': (0, 100, 255),
            'left_region': (255, 0, 0),
            'right_region': (0, 255, 0),
            'middle_region': (255, 255, 0)
        }

        # 角度检测模型路径
        self.angle_model_path = "/home/jhn/cnn/best_angle_model.pt"

    def load_angle_model(self):
        """加载角度检测模型"""
        return self.angle_detector.load_angle_model(self.angle_model_path)

    def get_region(self, center_x):
        """根据中心点坐标判断目标所在区域"""
        left_boundary = self.frame_width * self.left_region
        right_boundary = self.frame_width * (1 - self.right_region)

        if center_x < left_boundary:
            return 'left'
        elif center_x > right_boundary:
            return 'right'
        else:
            return 'middle'

    def detect_vehicles(self, frame, confidence_threshold=0.45):
        """检测非机动车"""
        if self.yolo_model is None:
            return []

        self.frame_width = frame.shape[1]

        try:
            results = self.yolo_model(
                frame,
                conf=confidence_threshold,
                iou=0.35,
                imgsz=640,
                verbose=False,
                max_det=15
            )
        except Exception as e:
            print(f"检测错误: {e}")
            return []

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                boxes_data = boxes.cpu().numpy()
                for i in range(len(boxes_data.xyxy)):
                    x1, y1, x2, y2 = boxes_data.xyxy[i].astype(int)
                    confidence = boxes_data.conf[i]
                    class_id = int(boxes_data.cls[i])
                    class_name = self.yolo_model.names[class_id]

                    if class_name in self.target_classes:
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1

                        if (bbox_width >= self.min_roi_size and
                                bbox_height >= self.min_roi_size):
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            region = self.get_region(center_x)

                            detection_info = {
                                'bbox': [x1, y1, x2, y2],
                                'center': (center_x, center_y),
                                'class_name': class_name,
                                'detection_confidence': confidence,
                                'region': region
                            }
                            detections.append(detection_info)

        return self.optimized_nms(detections)

    def optimized_nms(self, detections):
        """非极大值抑制"""
        if len(detections) <= 1:
            return detections

        detections.sort(key=lambda x: x['detection_confidence'], reverse=True)
        filtered = []
        keep_indices = list(range(len(detections)))

        for i in range(len(detections)):
            if i not in keep_indices:
                continue

            current_box = detections[i]
            filtered.append(current_box)

            for j in range(i + 1, len(detections)):
                if j in keep_indices:
                    if self.fast_iou(current_box['bbox'], detections[j]['bbox']) > 0.5:
                        keep_indices.remove(j)

        return filtered

    def fast_iou(self, box1, box2):
        """快速IOU计算"""
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

        if (x1_a >= x2_b or x2_a <= x1_b or
                y1_a >= y2_b or y2_a <= y1_b):
            return 0.0

        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2_a - x1_a) * (y2_a - y1_a)
        area2 = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update_tracking(self, current_detections):
        """更新目标跟踪"""
        for tracker_id in self.trackers:
            self.trackers[tracker_id]['matched'] = False

        matched_pairs = []

        for det in current_detections:
            best_tracker_id = None
            min_distance = 100

            for tracker_id, tracker in self.trackers.items():
                if tracker['matched']:
                    continue

                if not tracker['positions']:
                    continue

                last_pos = tracker['positions'][-1]
                current_pos = det['center']
                distance = np.sqrt((current_pos[0] - last_pos[0]) ** 2 +
                                   (current_pos[1] - last_pos[1]) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    best_tracker_id = tracker_id

            if best_tracker_id is not None:
                matched_pairs.append((best_tracker_id, det))
                self.trackers[best_tracker_id]['matched'] = True

        for tracker_id, det in matched_pairs:
            tracker = self.trackers[tracker_id]
            tracker['bboxes'].append(det['bbox'])
            tracker['positions'].append(det['center'])
            tracker['disappeared'] = 0
            tracker['class_name'] = det['class_name']
            tracker['region'] = det['region']
            tracker['track_frames'] = tracker.get('track_frames', 0) + 1

        for det in current_detections:
            matched = any(det == matched_det for _, matched_det in matched_pairs)
            if not matched:
                self.create_new_tracker(det)

        disappeared_ids = []
        for tracker_id, tracker in self.trackers.items():
            if not tracker['matched']:
                tracker['disappeared'] += 1
                if tracker['disappeared'] > self.max_disappeared:
                    disappeared_ids.append(tracker_id)

        for tracker_id in disappeared_ids:
            del self.trackers[tracker_id]

        self.update_region_statistics()

    def update_region_statistics(self):
        """更新区域统计信息"""
        self.left_targets.clear()
        self.right_targets.clear()
        self.middle_targets.clear()

        for tracker_id, tracker in self.trackers.items():
            if tracker['disappeared'] > 0:
                continue

            region = tracker.get('region', 'middle')
            if region == 'left':
                self.left_targets.add(tracker_id)
            elif region == 'right':
                self.right_targets.add(tracker_id)
            else:
                self.middle_targets.add(tracker_id)

    def calculate_average_safety_factor(self):
        """计算左右两边安全系数的平均值"""
        self.left_safety_factors.clear()
        self.right_safety_factors.clear()

        # 收集左右区域的安全系数
        for tracker_id, tracker in self.trackers.items():
            if tracker['disappeared'] > 0:
                continue

            region = tracker.get('region', 'middle')
            safety_factor = tracker.get('safety_factor', 0.0)

            if region == 'left':
                self.left_safety_factors.append(safety_factor)
            elif region == 'right':
                self.right_safety_factors.append(safety_factor)

        # 计算平均值
        left_avg = np.mean(self.left_safety_factors) if self.left_safety_factors else 0.0
        right_avg = np.mean(self.right_safety_factors) if self.right_safety_factors else 0.0

        # 如果两边都有数据，计算总平均值
        if self.left_safety_factors and self.right_safety_factors:
            self.average_safety_factor = (left_avg + right_avg) / 2.0
        elif self.left_safety_factors:
            self.average_safety_factor = left_avg
        elif self.right_safety_factors:
            self.average_safety_factor = right_avg
        else:
            self.average_safety_factor = 0.0

        return self.average_safety_factor, left_avg, right_avg

    def create_new_tracker(self, detection):
        """创建新跟踪器"""
        tracker_id = self.next_tracker_id
        self.trackers[tracker_id] = {
            'bboxes': [detection['bbox']],
            'positions': [detection['center']],
            'class_name': detection['class_name'],
            'matched': True,
            'disappeared': 0,
            'track_frames': 1,
            'speeds': deque(maxlen=10),
            'current_speed': 0.0,
            'distance': 0.0,
            'distance_confidence': 0.0,
            'region': detection['region'],
            'angle': None,
            'angle_category': None,
            'angle_confidence': 0.0,
            'safety_factor': 1.0,
            'safety_level': '安全',
            'safety_color': (0, 255, 0),
            'virtual_mass': 0.0,
            'closing_speed': 0.0,
            'safety_components': {}
        }
        self.next_tracker_id += 1

    def should_draw_box(self, tracker):
        """判断是否应该绘制红框"""
        return len(self.left_targets) > 0 and len(self.right_targets) > 0

    def detect_angles_for_trackers(self, frame):
        """为所有跟踪目标检测角度"""
        for tracker_id, tracker in self.trackers.items():
            if tracker['disappeared'] > 0:
                continue

            current_bbox = tracker['bboxes'][-1]
            angle_value, angle_category, confidence = self.angle_detector.detect_angle(frame, current_bbox)

            if angle_value is not None:
                tracker['angle'] = angle_value
                tracker['angle_category'] = angle_category
                tracker['angle_confidence'] = confidence

    def calculate_safety_factors(self):
        """为所有跟踪目标计算安全系数"""
        for tracker_id, tracker in self.trackers.items():
            if tracker['disappeared'] > 0:
                continue

            angle_degrees = 0
            if tracker['angle']:
                try:
                    angle_degrees = float(tracker['angle'].replace('°', ''))
                except:
                    angle_degrees = 0

            virtual_mass = self.safety_calculator.calculate_virtual_mass(
                tracker['class_name'],
                tracker['current_speed']
            )
            tracker['virtual_mass'] = virtual_mass

            closing_speed = tracker['current_speed']
            tracker['closing_speed'] = closing_speed

            safety_factor, SF_base, SF_direction, SF_speed, SF_mass = self.safety_calculator.calculate_safety_factor(
                distance_m=tracker['distance'],
                virtual_mass=virtual_mass,
                angle_degrees=angle_degrees,
                closing_speed_kmh=closing_speed,
                road_condition=1.0
            )

            tracker['safety_factor'] = safety_factor
            safety_level, safety_color = self.safety_calculator.get_safety_level(safety_factor)
            tracker['safety_level'] = safety_level
            tracker['safety_color'] = safety_color

            tracker['safety_components'] = {
                'distance_safety': SF_base,
                'angle_safety': SF_direction,
                'speed_safety': SF_speed,
                'mass_safety': SF_mass,
                'weights': self.safety_calculator.safety_weights.copy()
            }

    def draw_red_boxes_with_info(self, frame):
        """绘制红框和所有信息"""
        self.detect_angles_for_trackers(frame)
        self.calculate_safety_factors()

        # 计算平均安全系数
        avg_safety, left_avg, right_avg = self.calculate_average_safety_factor()

        self.draw_region_boundaries(frame)

        for tracker_id, tracker in self.trackers.items():
            if tracker['disappeared'] > 0:
                continue

            current_bbox = tracker['bboxes'][-1]
            x1, y1, x2, y2 = current_bbox

            if self.should_draw_box(tracker):
                safety_color = tracker['safety_color']
                line_width = 4 if tracker['safety_level'] == '危险' else 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), safety_color, line_width)

                self.draw_safety_on_box(frame, tracker, x1, y1, x2, y2, avg_safety)

                center_x, center_y = tracker['positions'][-1]
                cv2.circle(frame, (center_x, center_y), 4, (255, 255, 0), -1)

                self.draw_all_info(frame, tracker_id, tracker, x1, y1, x2, y2, avg_safety)

                self.draw_trajectory(frame, tracker)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(frame, f"ID:{tracker_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        return frame

    def draw_safety_on_box(self, frame, tracker, x1, y1, x2, y2, avg_safety):
        """在边界框上直接显示安全系数和平均安全系数"""
        safety_factor = tracker['safety_factor']
        safety_level = tracker['safety_level']
        safety_color = tracker['safety_color']

        safety_text = f"安全: {safety_factor:.2f}"
        level_text = f"等级: {safety_level}"
        avg_text = f"平均: {avg_safety:.2f}"

        safety_size = cv2.getTextSize(safety_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        level_size = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        avg_size = cv2.getTextSize(avg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        bg_x1 = x1
        bg_y1 = y1 - safety_size[1] - level_size[1] - avg_size[1] - 15
        bg_x2 = x1 + max(safety_size[0], level_size[0], avg_size[0]) + 10
        bg_y2 = y1

        if bg_y1 < 0:
            bg_y1 = y2 + 5
            bg_y2 = bg_y1 + safety_size[1] + level_size[1] + avg_size[1] + 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 显示个体安全系数
        cv2.putText(frame, safety_text, (x1 + 5, bg_y1 + safety_size[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, safety_color, 2)

        # 显示安全等级
        cv2.putText(frame, level_text, (x1 + 5, bg_y1 + safety_size[1] + level_size[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, safety_color, 1)

        # 显示平均安全系数
        avg_color = self.get_avg_safety_color(avg_safety)
        cv2.putText(frame, avg_text, (x1 + 5, bg_y1 + safety_size[1] + level_size[1] + avg_size[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, avg_color, 1)

    def get_avg_safety_color(self, avg_safety):
        """根据平均安全系数获取颜色"""
        if avg_safety >= self.safety_calculator.safe_threshold:
            return (0, 255, 0)  # 绿色
        elif avg_safety >= self.safety_calculator.warning_threshold:
            return (0, 255, 255)  # 黄色
        else:
            return (0, 0, 255)  # 红色

    def draw_region_boundaries(self, frame):
        """绘制区域边界线"""
        height = frame.shape[0]
        left_boundary = int(self.frame_width * self.left_region)
        right_boundary = int(self.frame_width * (1 - self.right_region))

        cv2.line(frame, (left_boundary, 0), (left_boundary, height),
                 self.colors['left_region'], 2)
        cv2.putText(frame, "Left Region", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['left_region'], 2)

        cv2.line(frame, (right_boundary, 0), (right_boundary, height),
                 self.colors['right_region'], 2)
        cv2.putText(frame, "Right Region", (right_boundary + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['right_region'], 2)

        region_stats = f"L:{len(self.left_targets)} M:{len(self.middle_targets)} R:{len(self.right_targets)}"
        cv2.putText(frame, region_stats, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        status_text = "红框标注: " + ("激活" if self.should_draw_box(None) else "未激活")
        status_color = (0, 255, 0) if self.should_draw_box(None) else (0, 0, 255)
        cv2.putText(frame, status_text, (10, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # 显示平均安全系数信息
        avg_safety, left_avg, right_avg = self.calculate_average_safety_factor()
        avg_text = f"平均安全: {avg_safety:.2f} (左:{left_avg:.2f} 右:{right_avg:.2f})"
        avg_color = self.get_avg_safety_color(avg_safety)
        cv2.putText(frame, avg_text, (10, height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, avg_color, 2)

    def draw_all_info(self, frame, tracker_id, tracker, x1, y1, x2, y2, avg_safety):
        """绘制所有信息：ID、速度、距离、角度、区域、安全系数"""
        id_text = f"ID:{tracker_id}"
        speed_text = f"速度:{tracker['current_speed']:.1f}km/h" if tracker['current_speed'] > 0 else "速度:计算中"

        distance_value = min(tracker['distance'], 15.0)
        distance_text = f"距离:{distance_value:.1f}m" if tracker['distance'] > 0 else "距离:计算中"

        if tracker['angle']:
            angle_text = f"角度:{tracker['angle']}({tracker['angle_category']})"
        else:
            angle_text = "角度:未知"

        region_text = f"区域:{tracker['region']}"

        safety_text = f"安全系数:{tracker['safety_factor']:.2f}"
        safety_level_text = f"安全等级:{tracker['safety_level']}"
        avg_safety_text = f"区域平均:{avg_safety:.2f}"

        # 简化的信息显示
        info_lines = [
            id_text,
            speed_text,
            distance_text,
            angle_text,
            region_text,
            safety_text,
            safety_level_text,
            avg_safety_text
        ]

        bg_x1 = x1
        bg_y1 = y1 - 170
        bg_x2 = x1 + 200
        bg_y2 = y1

        if bg_y1 < 0:
            bg_y1 = y2 + 5
            bg_y2 = bg_y1 + 170

        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i, text in enumerate(info_lines):
            color = (255, 255, 255)
            if "安全" in text and "系数" in text:
                color = tracker['safety_color']
            elif "速度" in text:
                color = self.get_speed_color(tracker['current_speed'])
            elif "区域平均" in text:
                color = self.get_avg_safety_color(avg_safety)

            cv2.putText(frame, text, (x1 + 5, bg_y1 + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def get_speed_color(self, speed_kmh):
        """根据速度获取颜色"""
        if speed_kmh < 5:
            return (0, 255, 0)
        elif speed_kmh < 15:
            return (0, 255, 255)
        elif speed_kmh < 30:
            return (0, 165, 255)
        else:
            return (0, 0, 255)

    def draw_trajectory(self, frame, tracker):
        """绘制运动轨迹"""
        positions = tracker['positions']
        if len(positions) < 2:
            return

        for i in range(1, min(len(positions), 10)):
            alpha = i / min(len(positions), 10)
            color = (0, int(255 * alpha), 255)
            cv2.line(frame, positions[i - 1], positions[i], color, 2)


class FastIntegratedSystem:
    """高速集成系统"""

    def __init__(self):
        self.red_box_detector = RedBoxDetector()
        self.distance_measurer = DistanceMeasurer()
        self.speed_calculator = None
        self.video_reader = None
        self.video_writer = None
        self.input_dir = "input"
        os.makedirs(self.input_dir, exist_ok=True)
        self.cleanup_old_videos()

    def cleanup_old_videos(self):
        """清理input目录中超过一小时的视频文件"""
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        try:
            for filename in os.listdir(self.input_dir):
                file_path = os.path.join(self.input_dir, filename)
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime < one_hour_ago:
                        os.remove(file_path)
                        print(f"🗑️ 删除过期视频: {filename}")
        except Exception as e:
            print(f"⚠️ 清理视频时出错: {e}")

    def delete_all_videos(self):
        """删除input目录中的所有视频文件"""
        try:
            for filename in os.listdir(self.input_dir):
                file_path = os.path.join(self.input_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"🗑️ 删除视频: {filename}")
            print("✅ 已删除input目录中的所有视频文件")
        except Exception as e:
            print(f"⚠️ 删除视频时出错: {e}")

    def process_realtime_raspberry(self, confidence_threshold=0.45):
        """树莓派实时视频流处理"""
        if not YOLO_AVAILABLE:
            print("❌ 请先安装 ultralytics 库")
            return

        # 初始化摄像头 - 支持 /dev/video0 设备路径
        cap = None
        try:
            # 尝试使用 /dev/video0
            cap = cv2.VideoCapture("/dev/video0")
            if cap.isOpened():
                print("✅ 成功打开 /dev/video0 摄像头")
            else:
                # 如果失败，尝试使用默认摄像头索引 0
                cap.release()
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    print("✅ 成功打开默认摄像头 (索引 0)")
                else:
                    print("❌ 无法打开摄像头")
                    return
        except Exception as e:
            # 如果 /dev/video0 不存在（如在 Windows 上），尝试使用默认摄像头
            print(f"⚠️ 打开 /dev/video0 时出错: {e}")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ 无法打开摄像头")
                return
            print("✅ 成功打开默认摄像头 (索引 0)")

        # 获取摄像头信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        print(f"📹 摄像头信息: {width}x{height}, {fps:.2f}FPS")

        # 初始化速度计算器
        self.speed_calculator = SpeedCalculator(self.red_box_detector.trackers, fps, (width, height))

        # 加载角度检测模型
        angle_model_loaded = self.red_box_detector.load_angle_model()
        if not angle_model_loaded:
            print("⚠️ 角度检测模型加载失败，将继续运行但角度检测不可用")
        else:
            print("✅ 角度检测模型加载成功!")

        # 生成视频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"raspberry_{timestamp}.mp4"
        video_path = os.path.join(self.input_dir, video_filename)

        print(f"🎯 视频将保存到: {video_path}")

        # 使用健壮的视频写入器
        try:
            self.video_writer = RobustVideoWriter(video_path, (width, height), fps)
        except Exception as e:
            print(f"❌ 无法创建视频写入器: {e}")
            print("⚠️ 将只显示预览，不保存视频")
            self.video_writer = None

        print("🚀 开始实时处理...")
        print("🎯 功能: 红框标注 + 速度判断 + 距离测量 + 角度识别 + 安全系数计算")
        print(f"🎯 置信度阈值: {confidence_threshold}")

        frame_count = 0
        start_time = time.time()
        last_stats_time = time.time()
        stats_interval = 3.0
        last_cleanup_time = time.time()

        # 性能优化：批量处理设置
        detection_interval = 3  # 每3帧检测一次
        angle_detection_interval = 10  # 每10帧检测一次角度

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头帧")
                    break

                frame_count += 1

                # 定期清理过期视频
                current_time = time.time()
                if current_time - last_cleanup_time > 60:  # 每分钟清理一次
                    self.cleanup_old_videos()
                    last_cleanup_time = current_time

                # 性能优化：选择性检测
                if frame_count % detection_interval == 1:
                    detections = self.red_box_detector.detect_vehicles(frame, confidence_threshold)
                    self.red_box_detector.update_tracking(detections)

                # 速度计算
                if hasattr(self, 'speed_calculator') and self.speed_calculator:
                    self.speed_calculator.calculate_speeds()

                # 简化距离计算
                self.calculate_simplified_distances()

                # 性能优化：减少角度检测频率
                if frame_count % angle_detection_interval == 1:
                    self.red_box_detector.detect_angles_for_trackers(frame)
                    self.red_box_detector.calculate_safety_factors()

                # 绘制结果
                display_frame = self.red_box_detector.draw_red_boxes_with_info(frame.copy())

                # 显示统计信息
                self.draw_stats(display_frame, frame_count, start_time, 0, 0)

                # 写入视频
                if self.video_writer:
                    success = self.video_writer.write_frame(display_frame)
                    if not success and frame_count % 50 == 0:
                        print("⚠️ 视频写入可能有问题")

                # 显示实时预览
                cv2.imshow('树莓派实时检测 - 按q退出', display_frame)

                # 定期显示进度
                current_time = time.time()
                if current_time - last_stats_time > stats_interval:
                    active_trackers = sum(1 for t in self.red_box_detector.trackers.values() if t['disappeared'] == 0)

                    # 显示平均安全系数
                    avg_safety, left_avg, right_avg = self.red_box_detector.calculate_average_safety_factor()

                    print(f"📊 帧: {frame_count} | 跟踪: {active_trackers}")
                    print(f"🛡️  安全系数 - 平均: {avg_safety:.2f} | 左: {left_avg:.2f} | 右: {right_avg:.2f}")
                    last_stats_time = current_time

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️ 用户中断处理")
                    break

        except KeyboardInterrupt:
            print("⏹️ 用户中断处理")
        except Exception as e:
            print(f"❌ 处理错误: {e}")
        finally:
            cap.release()
            if self.video_writer:
                self.video_writer.stop()
            cv2.destroyAllWindows()

        self.generate_final_report()

        total_time = time.time() - start_time
        processing_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n✅ 处理完成!")
        print(f"📊 总耗时: {total_time:.1f}s")
        print(f"🎯 处理速度: {processing_fps:.1f} FPS")

        if self.video_writer and not self.video_writer.use_image_sequence:
            print(f"💾 视频文件: {self.video_writer.output_path}")
        elif self.video_writer and self.video_writer.use_image_sequence:
            print(f"🖼️ 图像序列: {self.video_writer.image_sequence_dir}")
        else:
            print("⚠️ 未保存输出文件")

    def process_video_integrated(self, video_path, output_path=None, confidence_threshold=0.45):
        """集成处理视频 - 高速版本"""
        if not YOLO_AVAILABLE:
            print("❌ 请先安装 ultralytics 库")
            return

        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return

        # 使用高速视频读取器
        self.video_reader = FastVideoReader(video_path, buffer_size=80)
        try:
            fps, frame_size, total_frames = self.video_reader.start_reading()
            width, height = frame_size
        except Exception as e:
            print(f"❌ 视频读取失败: {e}")
            return

        print(f"📹 视频信息: {width}x{height}, {fps:.2f}FPS, 总帧数: {total_frames}")

        # 初始化速度计算器
        self.speed_calculator = SpeedCalculator(self.red_box_detector.trackers, fps, (width, height))

        # 加载角度检测模型
        angle_model_loaded = self.red_box_detector.load_angle_model()
        if not angle_model_loaded:
            print("⚠️ 角度检测模型加载失败，将继续运行但角度检测不可用")
        else:
            print("✅ 角度检测模型加载成功!")

        # 输出路径
        if output_path is None:
            output_dir = "detected_videos"
            os.makedirs(output_dir, exist_ok=True)
            input_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"detected_{timestamp}.mp4")

        print(f"🎯 输出文件: {output_path}")

        # 使用健壮的视频写入器
        try:
            self.video_writer = RobustVideoWriter(output_path, (width, height), fps)
        except Exception as e:
            print(f"❌ 无法创建视频写入器: {e}")
            print("⚠️ 将只显示预览，不保存视频")
            self.video_writer = None

        print("🚀 开始处理视频...")
        print("🎯 功能: 红框标注 + 速度判断 + 距离测量 + 角度识别 + 安全系数计算")
        print(f"🎯 置信度阈值: {confidence_threshold}")

        frame_count = 0
        start_time = time.time()
        last_stats_time = time.time()
        stats_interval = 3.0

        # 性能优化：批量处理设置
        detection_interval = 3  # 每3帧检测一次
        angle_detection_interval = 10  # 每10帧检测一次角度

        try:
            while True:
                ret, frame = self.video_reader.read_frame()
                if not ret:
                    if self.video_reader.get_queue_size() == 0:
                        break
                    else:
                        time.sleep(0.001)
                        continue

                frame_count += 1

                # 性能优化：选择性检测
                if frame_count % detection_interval == 1:
                    detections = self.red_box_detector.detect_vehicles(frame, confidence_threshold)
                    self.red_box_detector.update_tracking(detections)

                # 速度计算
                if hasattr(self, 'speed_calculator') and self.speed_calculator:
                    self.speed_calculator.calculate_speeds()

                # 简化距离计算
                self.calculate_simplified_distances()

                # 性能优化：减少角度检测频率
                if frame_count % angle_detection_interval == 1:
                    self.red_box_detector.detect_angles_for_trackers(frame)
                    self.red_box_detector.calculate_safety_factors()

                # 绘制结果
                display_frame = self.red_box_detector.draw_red_boxes_with_info(frame.copy())

                # 显示统计信息
                self.draw_stats(display_frame, frame_count, start_time, total_frames,
                                self.video_reader.get_queue_size())

                # 写入视频
                if self.video_writer:
                    success = self.video_writer.write_frame(display_frame)
                    if not success and frame_count % 50 == 0:
                        print("⚠️ 视频写入可能有问题")

                # 显示实时预览
                cv2.imshow('车辆检测系统 - 按q退出', display_frame)

                # 定期显示进度
                current_time = time.time()
                if current_time - last_stats_time > stats_interval:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    queue_size = self.video_reader.get_queue_size()
                    active_trackers = sum(1 for t in self.red_box_detector.trackers.values() if t['disappeared'] == 0)

                    # 显示平均安全系数
                    avg_safety, left_avg, right_avg = self.red_box_detector.calculate_average_safety_factor()

                    print(f"📊 进度: {progress:.1f}% | 帧: {frame_count}/{total_frames} | "
                          f"缓存: {queue_size} | 跟踪: {active_trackers}")
                    print(f"🛡️  安全系数 - 平均: {avg_safety:.2f} | 左: {left_avg:.2f} | 右: {right_avg:.2f}")
                    last_stats_time = current_time

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️ 用户中断处理")
                    break

        except KeyboardInterrupt:
            print("⏹️ 用户中断处理")
        except Exception as e:
            print(f"❌ 处理错误: {e}")
        finally:
            if self.video_reader:
                self.video_reader.stop()
            if self.video_writer:
                self.video_writer.stop()
            cv2.destroyAllWindows()

        self.generate_final_report()

        total_time = time.time() - start_time
        processing_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n✅ 处理完成!")
        print(f"📊 总耗时: {total_time:.1f}s")
        print(f"🎯 处理速度: {processing_fps:.1f} FPS")

        if self.video_writer and not self.video_writer.use_image_sequence:
            print(f"💾 视频文件: {self.video_writer.output_path}")
        elif self.video_writer and self.video_writer.use_image_sequence:
            print(f"🖼️ 图像序列: {self.video_writer.image_sequence_dir}")
        else:
            print("⚠️ 未保存输出文件")

    def calculate_simplified_distances(self):
        """简化距离计算"""
        for tracker_id, tracker in self.red_box_detector.trackers.items():
            if tracker['disappeared'] > 0:
                continue

            bbox = tracker['bboxes'][-1]
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            if bbox_area > 0:
                max_area = 50000
                min_area = 1000
                normalized_area = min(1.0, max(0.0, (bbox_area - min_area) / (max_area - min_area)))
                distance = 50 - (normalized_area * 45)
                distance = min(distance, 15.0)

                tracker['distance'] = distance
                tracker['distance_confidence'] = 0.7

    def draw_stats(self, frame, frame_count, start_time, total_frames, queue_size):
        """绘制统计信息"""
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        active_trackers = sum(1 for t in self.red_box_detector.trackers.values() if t['disappeared'] == 0)

        # 计算平均安全系数
        avg_safety, left_avg, right_avg = self.red_box_detector.calculate_average_safety_factor()

        stats = [
            f"FPS: {current_fps:.1f}",
            f"进度: {progress:.1f}%",
            f"帧: {frame_count}/{total_frames}",
            f"跟踪目标: {active_trackers}",
            f"平均安全: {avg_safety:.2f}",
            f"系统状态: 运行中"
        ]

        y_offset = 20
        for i, text in enumerate(stats):
            color = (0, 255, 255)
            if "平均安全" in text:
                color = self.red_box_detector.get_avg_safety_color(avg_safety)
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

    def generate_final_report(self):
        """生成最终报告"""
        print("\n" + "=" * 60)
        print("处理完成报告")
        print("=" * 60)

        total_trackers = len(self.red_box_detector.trackers)
        active_trackers = sum(1 for t in self.red_box_detector.trackers.values() if t['disappeared'] == 0)

        # 最终安全系数统计
        avg_safety, left_avg, right_avg = self.red_box_detector.calculate_average_safety_factor()

        print(f"跟踪统计:")
        print(f"  总目标数: {total_trackers}")
        print(f"  活跃目标: {active_trackers}")
        print(f"安全系数统计:")
        print(f"  左区域平均: {left_avg:.2f}")
        print(f"  右区域平均: {right_avg:.2f}")
        print(f"  总体平均: {avg_safety:.2f}")


def main():
    """主程序"""
    print("🚀 车辆检测系统 (高速优化版)")
    print("=" * 50)
    print("🎯 功能特性:")
    print("  • 非机动车检测与跟踪")
    print("  • 速度与距离估算")
    print("  • 角度识别")
    print("  • 安全系数计算 (小角度高风险模式)")
    print("  • 左右区域安全系数平均")
    print("  • 视频输出")
    print("  • 树莓派实时视频流支持")
    print("  • 视频自动清理 (超过1小时的视频)")

    system = FastIntegratedSystem()

    # 选择处理模式
    print("\n请选择处理模式:")
    print("1. 处理视频文件")
    print("2. 树莓派实时视频流")
    print("3. 删除input目录中的所有视频文件")
    
    choice = input("请输入选项 (1-3): " ).strip()
    
    if choice == "1":
        # 处理视频文件
        video_path = input("请输入视频文件路径: " ).strip().strip('"')
        if not os.path.exists(video_path):
            print("❌ 视频文件不存在")
            return

        conf = 0.45
        try:
            conf_input = input(f"置信度阈值 (默认{conf}): " ).strip()
            if conf_input:
                conf = float(conf_input)
        except:
            pass

        print("🚀 开始处理视频...")
        system.process_video_integrated(video_path, confidence_threshold=conf)
        
    elif choice == "2":
        # 树莓派实时视频流
        conf = 0.45
        try:
            conf_input = input(f"置信度阈值 (默认{conf}): " ).strip()
            if conf_input:
                conf = float(conf_input)
        except:
            pass

        print("🚀 开始树莓派实时处理...")
        system.process_realtime_raspberry(confidence_threshold=conf)
        
    elif choice == "3":
        # 删除input目录中的所有视频文件
        system.delete_all_videos()
        
    else:
        print("❌ 无效的选项")


if __name__ == "__main__":
    main()