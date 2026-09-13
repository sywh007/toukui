"""
S5：基于红外热感应与区域判定的"鬼探头"风险预警系统
修改版：中间区域出现温度超过20度且距离≤5米的物体才警告
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os
import glob
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DetectionResult:
    """检测结果数据类"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    centroid: Tuple[int, int]  # 中心点坐标
    area: float  # 面积
    center_temperature: float = 0.0  # 中心点温度
    estimated_distance: float = 0.0  # 估计距离


class GhostProbeWarningSystem:
    """鬼探头风险预警系统"""

    def __init__(self,
                 model_path: str = 'yolov8n.pt',  # 默认模型路径
                 image_folder: str = r'D:\cnn1\red',  # 图片文件夹路径
                 temperature_threshold: float = 20.0,  # 温度阈值（摄氏度）
                 warning_distance: float = 5.0):  # 预警距离阈值（米）
        """
        初始化系统

        Args:
            model_path: YOLOv8模型路径（默认在当前目录）
            image_folder: 红外图像文件夹路径
            temperature_threshold: 温度阈值（摄氏度）
            warning_distance: 预警距离阈值（米）
        """
        # 文件路径设置
        self.image_folder = image_folder
        self.model_path = model_path

        # 图像尺寸初始化
        self.frame_width = 640
        self.frame_height = 480
        self.left_region = None
        self.middle_region = None
        self.right_region = None

        # 检测参数
        self.temperature_threshold = temperature_threshold  # 温度阈值20度
        self.warning_distance = warning_distance  # 距离阈值5米

        # 距离估计参数
        self.focal_length = 500  # 相机焦距（像素）
        self.real_height_person = 1.7  # 行人平均高度（米）
        self.real_height_bicycle = 1.2  # 自行车平均高度（米）
        self.real_height_motorcycle = 1.5  # 摩托车平均高度（米）
        self.real_height_car = 1.5  # 汽车近似高度
        self.real_height_bus = 2.5  # 公交车近似高度
        self.real_height_truck = 2.5  # 卡车近似高度

        # 温度-亮度映射参数
        self.temp_min = 15.0  # 最低温度对应的亮度值
        self.temp_max = 40.0  # 最高温度对应的亮度值
        self.brightness_min = 50  # 最低亮度
        self.brightness_max = 255  # 最高亮度

        # 加载模型
        self.model = self._load_model(model_path)
        self.class_names = {0: 'person', 1: 'bicycle', 2: 'motorcycle', 3: 'car', 4: 'bus', 5: 'truck'}

        # 状态变量
        self.is_warning = False
        self.frame_count = 0

        # 创建输出目录
        self.output_folder = r'D:\cnn1\red\results'
        os.makedirs(self.output_folder, exist_ok=True)

        print("=" * 60)
        print("鬼探头预警系统初始化完成")
        print(f"模型路径: {model_path}")
        print(f"图像文件夹: {image_folder}")
        print(f"输出文件夹: {self.output_folder}")
        print(f"预警条件: 中间区域出现温度>{temperature_threshold}°C且距离≤{warning_distance}米的物体")
        print("=" * 60)

    def _load_model(self, model_path: str) -> YOLO:
        """加载YOLO模型"""
        print(f"正在加载模型: {model_path}")
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在，将下载yolov8n模型")
                model = YOLO('yolov8n.pt')  # 自动下载
            else:
                model = YOLO(model_path)

            # 设置设备
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            print(f"✓ 模型加载成功，使用设备: {device}")
            return model
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("尝试使用默认模型...")
            try:
                model = YOLO('yolov8n.pt')
                print("✓ 默认模型加载成功")
                return model
            except:
                raise RuntimeError("无法加载YOLO模型")

    def _setup_regions(self, frame: np.ndarray):
        """划分左、中、右三个区域"""
        self.frame_height, self.frame_width = frame.shape[:2]

        # 计算三个等宽的垂直矩形区域
        region_width = self.frame_width // 3

        # 左区域
        self.left_region = (0, 0, region_width, self.frame_height)
        # 中区域
        self.middle_region = (region_width, 0, region_width * 2, self.frame_height)
        # 右区域
        self.right_region = (region_width * 2, 0, self.frame_width, self.frame_height)

        print(f"画面尺寸: {self.frame_width}x{self.frame_height}")
        print(f"中间区域范围: {self.middle_region}")

    def _brightness_to_temperature(self, brightness: float) -> float:
        """
        将亮度值转换为温度值
        这是一个简化的线性映射，实际应用中需要根据红外相机校准
        """
        # 线性映射：亮度 -> 温度
        brightness = np.clip(brightness, self.brightness_min, self.brightness_max)
        temperature = self.temp_min + (brightness - self.brightness_min) * \
                     (self.temp_max - self.temp_min) / (self.brightness_max - self.brightness_min)
        return temperature

    def _get_center_temperature(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        获取检测目标中心点的温度

        Args:
            frame: 原始图像
            bbox: 边界框 (x1, y1, x2, y2)

        Returns:
            float: 中心点温度（摄氏度）
        """
        x1, y1, x2, y2 = bbox

        # 确保边界框在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.frame_width, x2)
        y2 = min(self.frame_height, y2)

        # 计算中心点坐标
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 确保中心点在图像范围内
        center_x = np.clip(center_x, 0, self.frame_width - 1)
        center_y = np.clip(center_y, 0, self.frame_height - 1)

        # 获取中心点像素值
        if len(frame.shape) == 3:
            # 彩色图像，取灰度值
            pixel_value = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[center_y, center_x]
        else:
            # 灰度图像
            pixel_value = frame[center_y, center_x]

        # 将亮度转换为温度
        temperature = self._brightness_to_temperature(float(pixel_value))

        return temperature

    def _estimate_distance(self, bbox: Tuple[int, int, int, int], class_name: str) -> float:
        """
        估计目标距离

        Args:
            bbox: 边界框 (x1, y1, x2, y2)
            class_name: 目标类别

        Returns:
            float: 估计距离（米）
        """
        _, y1, _, y2 = bbox
        height_pixels = y2 - y1

        if height_pixels <= 0:
            return float('inf')

        # 根据目标类别选择真实高度
        if class_name == 'person':
            real_height = self.real_height_person
        elif class_name == 'bicycle':
            real_height = self.real_height_bicycle
        elif class_name == 'motorcycle':
            real_height = self.real_height_motorcycle
        elif class_name == 'car':
            real_height = self.real_height_car
        elif class_name == 'bus':
            real_height = self.real_height_bus
        elif class_name == 'truck':
            real_height = self.real_height_truck
        else:
            real_height = self.real_height_person  # 默认

        # 使用相似三角形原理估算距离
        # distance = (真实高度 * 焦距) / 像素高度
        distance = (real_height * self.focal_length) / height_pixels

        # 添加位置修正（目标在画面底部通常更近）
        centroid_y = (y1 + y2) / 2
        position_factor = 1.0 + (centroid_y / self.frame_height) * 0.3
        distance *= position_factor

        return distance

    def _process_frame(self, frame: np.ndarray) -> Dict:
        """
        处理单帧图像

        Args:
            frame: 输入图像

        Returns:
            Dict: 处理结果
        """
        # 首次运行时设置区域
        if self.middle_region is None:
            self._setup_regions(frame)

        result = {
            'detections': [],
            'middle_detections': [],
            'high_temp_targets': [],  # 高温目标列表
            'warning_targets': [],    # 预警目标列表（高温且距离≤5米）
            'warning_level': 0,
            'warning_message': '正常行驶'
        }

        # 提取中间区域
        x1, y1, x2, y2 = self.middle_region
        middle_frame = frame[y1:y2, x1:x2]

        # 使用YOLO检测中间区域
        try:
            detections = self.model(middle_frame, conf=0.25, verbose=False)

            # 处理检测结果
            if detections and len(detections) > 0:
                boxes = detections[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # 获取检测信息
                        x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.class_names.get(cls_id, f'class_{cls_id}')

                        # 转换为全局坐标
                        x1_global = x1_box + x1
                        y1_global = y1_box + y1
                        x2_global = x2_box + x1
                        y2_global = y2_box + y1

                        bbox = (x1_global, y1_global, x2_global, y2_global)
                        centroid = ((x1_global + x2_global) // 2, (y1_global + y2_global) // 2)
                        area = (x2_global - x1_global) * (y2_global - y1_global)

                        # 获取中心点温度
                        center_temp = self._get_center_temperature(frame, bbox)

                        # 估计距离
                        distance = self._estimate_distance(bbox, cls_name)

                        # 创建检测结果
                        detection = DetectionResult(
                            bbox=bbox,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                            centroid=centroid,
                            area=area,
                            center_temperature=center_temp,
                            estimated_distance=distance
                        )

                        result['detections'].append(detection)
                        result['middle_detections'].append(detection)

                        # 检查是否为高温目标
                        if center_temp > self.temperature_threshold:
                            result['high_temp_targets'].append({
                                'class': cls_name,
                                'temperature': center_temp,
                                'distance': distance,
                                'confidence': conf,
                                'bbox': bbox
                            })

                            # 检查是否同时满足距离条件
                            if distance <= self.warning_distance:
                                result['warning_targets'].append({
                                    'class': cls_name,
                                    'temperature': center_temp,
                                    'distance': distance,
                                    'confidence': conf,
                                    'bbox': bbox
                                })

                        print(f"  检测到{cls_name}: 置信度{conf:.2f}, 温度{center_temp:.1f}°C, 距离{distance:.1f}米")

        except Exception as e:
            print(f"检测过程中出错: {e}")

        # 确定预警级别
        # 新逻辑：温度>20度且距离≤5米才警告
        if len(result['warning_targets']) > 0:
            result['warning_level'] = 1
            # 找到最近的高温物体
            closest = min(result['warning_targets'], key=lambda x: x['distance'])
            result['warning_message'] = f"⚠️ 前方{closest['distance']:.1f}米有高温{closest['class']} ({closest['temperature']:.1f}°C)!"
            self.is_warning = True
        else:
            self.is_warning = False
            result['warning_message'] = "✅ 安全"

        return result

    def _draw_results(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """绘制检测结果和预警信息"""
        display_frame = frame.copy()

        # 绘制区域分割线
        if self.middle_region:
            # 左边界线
            cv2.line(display_frame,
                     (self.middle_region[0], 0),
                     (self.middle_region[0], self.frame_height),
                     (0, 255, 255), 2, cv2.LINE_AA)
            # 右边界线
            cv2.line(display_frame,
                     (self.middle_region[2], 0),
                     (self.middle_region[2], self.frame_height),
                     (0, 255, 255), 2, cv2.LINE_AA)

            # 标记中间区域
            cv2.rectangle(display_frame,
                          (self.middle_region[0], self.middle_region[1]),
                          (self.middle_region[2], self.middle_region[3]),
                          (0, 255, 255), 2, cv2.LINE_AA)

            # 添加区域标签
            cv2.putText(display_frame, "LEFT",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "MIDDLE",
                        (self.frame_width // 3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "RIGHT",
                        (self.frame_width // 3 * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 绘制检测框
        for detection in result['detections']:
            x1, y1, x2, y2 = detection.bbox

            # 判断目标状态
            is_warning_target = any(
                np.array_equal(detection.bbox, t['bbox'])
                for t in result['warning_targets']
            )

            is_high_temp_target = any(
                np.array_equal(detection.bbox, t['bbox'])
                for t in result['high_temp_targets']
            )

            # 根据状态选择颜色和厚度
            if is_warning_target:
                color = (0, 0, 255)  # 红色：预警目标（高温且距离≤5米）
                thickness = 3
            elif is_high_temp_target:
                color = (0, 165, 255)  # 橙色：高温目标但距离>5米
                thickness = 2
            else:
                color = (0, 255, 0)  # 绿色：低温目标
                thickness = 2

            # 绘制边界框
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签
            label = f"{detection.class_name} {detection.confidence:.2f} ({detection.center_temperature:.1f}°C, {detection.estimated_distance:.1f}m)"

            # 计算标签背景
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # 绘制标签背景
            cv2.rectangle(display_frame,
                          (x1, y1 - label_height - baseline - 5),
                          (x1 + label_width, y1),
                          color, -1)

            # 绘制标签文本
            cv2.putText(display_frame, label, (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 绘制中心点
            cx, cy = detection.centroid
            cv2.circle(display_frame, (cx, cy), 4, color, -1)

        # 显示预警信息
        if result['warning_level'] > 0:
            # 红色预警背景
            cv2.rectangle(display_frame, (0, 0), (self.frame_width, 80), (0, 0, 255), -1)
            cv2.rectangle(display_frame, (0, 0), (self.frame_width, 80), (255, 255, 255), 2)

            # 预警标题
            cv2.putText(display_frame, "⚠️ 鬼探头预警 ⚠️",
                        (self.frame_width // 2 - 100, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

            # 预警详情
            cv2.putText(display_frame, result['warning_message'],
                        (self.frame_width // 2 - 150, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 闪烁的红色圆点
            if int(time.time() * 3) % 2 == 0:
                cv2.circle(display_frame, (self.frame_width - 40, 40),
                           15, (0, 0, 255), -1)
                cv2.circle(display_frame, (self.frame_width - 40, 40),
                           15, (255, 255, 255), 2)
        else:
            # 绿色安全背景
            cv2.rectangle(display_frame, (0, 0), (self.frame_width, 50), (0, 180, 0), -1)
            cv2.putText(display_frame, "✅ 道路安全",
                        (self.frame_width // 2 - 70, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 显示统计信息
        stats_text = f"检测目标: {len(result['detections'])} | 高温目标: {len(result['high_temp_targets'])} | 预警目标: {len(result['warning_targets'])}"
        cv2.putText(display_frame, stats_text, (10, self.frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示参数信息
        param_text = f"温度阈值: {self.temperature_threshold}°C | 距离阈值: {self.warning_distance}m"
        cv2.putText(display_frame, param_text, (10, self.frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display_frame

    def process_single_image(self, image_path: str) -> Dict:
        """
        处理单张图像

        Args:
            image_path: 图像路径

        Returns:
            Dict: 处理结果
        """
        print(f"\n处理图像: {os.path.basename(image_path)}")

        # 读取图像
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"✗ 无法读取图像: {image_path}")
            return {}

        # 处理图像
        start_time = time.time()
        result = self._process_frame(frame)
        processing_time = time.time() - start_time

        # 绘制结果
        display_frame = self._draw_results(frame, result)

        # 保存结果
        filename = os.path.basename(image_path)
        output_path = os.path.join(self.output_folder, f"result_{filename}")
        cv2.imwrite(output_path, display_frame)

        # 显示结果
        print(f"✓ 处理完成 ({processing_time:.2f}秒)")
        print(f"  检测到目标: {len(result['detections'])}个")
        print(f"  高温目标(>{self.temperature_threshold}°C): {len(result['high_temp_targets'])}个")
        print(f"  预警目标(高温且距离≤{self.warning_distance}米): {len(result['warning_targets'])}个")

        if result['warning_level'] > 0:
            print(f"⚠️  预警信息: {result['warning_message']}")
        else:
            print(f"✅  安全状态: 无温度>{self.temperature_threshold}°C且距离≤{self.warning_distance}米的物体")

        print(f"✓ 结果已保存: {output_path}")

        # 显示图像
        cv2.imshow(f"Ghost Probe Detection: {filename}", display_frame)
        cv2.waitKey(1000)  # 显示1秒
        cv2.destroyAllWindows()

        return result

    def batch_process_images(self):
        """批量处理文件夹中的所有图像"""
        print("\n" + "=" * 60)
        print("开始批量处理红外图像")
        print("=" * 60)

        # 获取所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []

        for ext in image_extensions:
            pattern = os.path.join(self.image_folder, ext)
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(self.image_folder, ext.upper())
            image_files.extend(glob.glob(pattern))

        if not image_files:
            print(f"在 {self.image_folder} 中未找到图像文件")
            return

        print(f"找到 {len(image_files)} 张图像:")
        for img in image_files:
            print(f"  • {os.path.basename(img)}")

        print("\n开始处理...")
        print("-" * 60)

        # 处理每张图像
        results_summary = {
            'total_images': len(image_files),
            'processed_images': 0,
            'total_detections': 0,
            'total_high_temp': 0,
            'total_warnings': 0,
            'warning_images': [],
            'max_temperature': 0.0,
            'avg_temperature': 0.0
        }

        all_temperatures = []

        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] ", end="")

            try:
                result = self.process_single_image(image_path)

                # 更新统计
                results_summary['processed_images'] += 1
                results_summary['total_detections'] += len(result.get('detections', []))
                results_summary['total_high_temp'] += len(result.get('high_temp_targets', []))
                results_summary['total_warnings'] += len(result.get('warning_targets', []))

                if result.get('warning_level', 0) > 0:
                    results_summary['warning_images'].append(os.path.basename(image_path))

                # 收集温度数据
                for detection in result.get('detections', []):
                    all_temperatures.append(detection.center_temperature)

            except Exception as e:
                print(f"✗ 处理失败: {e}")
                continue

        # 计算温度统计
        if all_temperatures:
            results_summary['max_temperature'] = max(all_temperatures)
            results_summary['avg_temperature'] = sum(all_temperatures) / len(all_temperatures)

        # 显示统计摘要
        print("\n" + "=" * 60)
        print("批量处理完成！")
        print("=" * 60)
        print(f"处理图像总数: {results_summary['processed_images']}/{results_summary['total_images']}")
        print(f"总检测目标数: {results_summary['total_detections']}")
        print(f"总高温目标数(>{self.temperature_threshold}°C): {results_summary['total_high_temp']}")
        print(f"总预警目标数(高温且距离≤{self.warning_distance}米): {results_summary['total_warnings']}")
        if all_temperatures:
            print(f"最高温度: {results_summary['max_temperature']:.1f}°C")
            print(f"平均温度: {results_summary['avg_temperature']:.1f}°C")

        if results_summary['warning_images']:
            print(f"触发预警的图像: {len(results_summary['warning_images'])}张")
            for img in results_summary['warning_images']:
                print(f"  • {img}")

        # 保存统计结果
        summary_file = os.path.join(self.output_folder, "processing_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("鬼探头预警系统处理摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"图像文件夹: {self.image_folder}\n")
            f.write(f"温度阈值: {self.temperature_threshold}°C\n")
            f.write(f"距离阈值: {self.warning_distance}米\n")
            f.write(f"处理图像总数: {results_summary['processed_images']}/{results_summary['total_images']}\n")
            f.write(f"总检测目标数: {results_summary['total_detections']}\n")
            f.write(f"总高温目标数(>{self.temperature_threshold}°C): {results_summary['total_high_temp']}\n")
            f.write(f"总预警目标数(高温且距离≤{self.warning_distance}米): {results_summary['total_warnings']}\n")
            if all_temperatures:
                f.write(f"最高温度: {results_summary['max_temperature']:.1f}°C\n")
                f.write(f"平均温度: {results_summary['avg_temperature']:.1f}°C\n")
            if results_summary['warning_images']:
                f.write("\n触发预警的图像:\n")
                for img in results_summary['warning_images']:
                    f.write(f"  • {img}\n")

        print(f"\n✓ 处理摘要已保存: {summary_file}")
        print("=" * 60)

    def adjust_parameters(self):
        """交互式调整参数"""
        print("\n参数调整模式")
        print("当前参数:")
        print(f"  1. 温度阈值: {self.temperature_threshold}°C")
        print(f"  2. 距离阈值: {self.warning_distance}米")

        while True:
            print("\n选择要调整的参数 (1-2), 或输入 0 退出:")
            choice = input("> ").strip()

            if choice == '0':
                break
            elif choice == '1':
                new_value = input(f"请输入新的温度阈值 (当前: {self.temperature_threshold}°C): ").strip()
                try:
                    self.temperature_threshold = float(new_value)
                    print(f"✓ 温度阈值已更新为: {self.temperature_threshold}°C")
                    print(f"警告条件: 中间区域出现温度>{self.temperature_threshold}°C且距离≤{self.warning_distance}米的物体")
                except:
                    print("✗ 输入无效")
            elif choice == '2':
                new_value = input(f"请输入新的距离阈值 (当前: {self.warning_distance}米): ").strip()
                try:
                    self.warning_distance = float(new_value)
                    print(f"✓ 距离阈值已更新为: {self.warning_distance}米")
                    print(f"警告条件: 中间区域出现温度>{self.temperature_threshold}°C且距离≤{self.warning_distance}米的物体")
                except:
                    print("✗ 输入无效")
            else:
                print("✗ 无效选择")


# 主程序
if __name__ == "__main__":
    # 创建预警系统实例
    print("初始化鬼探头风险预警系统...")

    try:
        system = GhostProbeWarningSystem(
            model_path='yolov8n.pt',  # 模型在当前目录
            image_folder=r'D:\cnn1\red',  # 您的图片路径
            temperature_threshold=20.0,  # 温度阈值设为20度
            warning_distance=5.0  # 距离阈值设为5米
        )

        while True:
            print("\n" + "=" * 60)
            print("鬼探头风险预警系统 - 主菜单")
            print("=" * 60)
            print("1. 批量处理文件夹中的所有图像")
            print("2. 处理单张图像")
            print("3. 调整参数")
            print("4. 退出系统")
            print("-" * 60)

            choice = input("请选择操作 (1-4): ").strip()

            if choice == '1':
                # 批量处理
                system.batch_process_images()
                input("\n按Enter键继续...")

            elif choice == '2':
                # 处理单张图像
                print("\n可用的图像文件:")
                image_files = glob.glob(os.path.join(system.image_folder, "*.jpg"))
                image_files.extend(glob.glob(os.path.join(system.image_folder, "*.png")))
                image_files.extend(glob.glob(os.path.join(system.image_folder, "*.jpeg")))

                if not image_files:
                    print("未找到图像文件")
                    input("\n按Enter键继续...")
                    continue

                for i, img in enumerate(image_files, 1):
                    print(f"{i}. {os.path.basename(img)}")

                try:
                    img_choice = int(input(f"\n选择要处理的图像 (1-{len(image_files)}): ").strip())
                    if 1 <= img_choice <= len(image_files):
                        system.process_single_image(image_files[img_choice - 1])
                    else:
                        print("无效选择")
                except:
                    print("输入无效")

                input("\n按Enter键继续...")

            elif choice == '3':
                # 调整参数
                system.adjust_parameters()

            elif choice == '4':
                print("\n感谢使用鬼探头风险预警系统！")
                print("程序退出")
                break

            else:
                print("无效选择，请重新输入")

    except Exception as e:
        print(f"\n系统初始化失败: {e}")
        print("请检查:")
        print("1. 模型文件 'yolov8n.pt' 是否在当前目录")
        print("2. 图像文件夹 'D:\\cnn1\\red' 是否存在")
        print("3. Python环境是否安装了所需库 (ultralytics, opencv-python, torch)")
        input("\n按Enter键退出...")