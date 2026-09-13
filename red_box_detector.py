import cv2
import os
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from collections import deque


class RedBoxDetector:
    def __init__(self):
        """初始化红框标注系统"""
        # 模型配置
        self.yolo_model_type = 's'
        self.yolo_model = None
        self.load_yolo_model()

        # 配置参数
        self.setup_parameters()

        # 跟踪数据
        self.setup_tracking()

        print("✅ 红框标注系统初始化完成!")

    def load_yolo_model(self):
        """加载YOLO模型"""
        try:
            model_name = f'yolov8{self.yolo_model_type}.pt'
            print(f"🚀 加载YOLO模型: {model_name}")
            self.yolo_model = YOLO(model_name)
            print("✅ YOLO模型加载成功!")

        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")

    def setup_parameters(self):
        """设置检测参数"""
        self.confidence_threshold = 0.45
        self.iou_threshold = 0.4
        self.min_roi_size = 36

        # 目标类别
        self.target_classes = ['bicycle', 'motorcycle']

        # 颜色配置
        self.colors = {
            'bicycle': (0, 0, 255),
            'motorcycle': (0, 0, 255),
            'text_bg': (0, 0, 255),
            'text': (255, 255, 255),
            'id_bg': (0, 100, 255),
        }

        # 性能优化参数
        self.processing_times = []

    def setup_tracking(self):
        """设置目标跟踪"""
        self.trackers = {}  # {tracker_id: {'bboxes': [], 'positions': [], 'class_name': ''}}
        self.next_tracker_id = 1
        self.max_disappeared = 15

    def detect_vehicles(self, frame, confidence_threshold=0.45):
        """检测非机动车"""
        if self.yolo_model is None:
            return []

        try:
            results = self.yolo_model(
                frame,
                conf=confidence_threshold,
                iou=0.35,
                imgsz=640,
                verbose=False,
                max_det=15,
                agnostic_nms=False,
                half=False
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
                                bbox_height >= self.min_roi_size and
                                bbox_width < frame.shape[1] * 0.8 and
                                bbox_height < frame.shape[0] * 0.8):
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            detection_info = {
                                'bbox': [x1, y1, x2, y2],
                                'center': (center_x, center_y),
                                'class_name': class_name,
                                'detection_confidence': confidence,
                                'area': bbox_width * bbox_height
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
        # 重置匹配状态
        for tracker_id in self.trackers:
            self.trackers[tracker_id]['matched'] = False

        # 匹配当前检测与现有跟踪器
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

        # 更新匹配的跟踪器
        for tracker_id, det in matched_pairs:
            tracker = self.trackers[tracker_id]
            tracker['bboxes'].append(det['bbox'])
            tracker['positions'].append(det['center'])
            tracker['disappeared'] = 0
            tracker['class_name'] = det['class_name']

        # 为未匹配的检测创建新跟踪器
        for det in current_detections:
            matched = any(det == matched_det for _, matched_det in matched_pairs)
            if not matched:
                self.create_new_tracker(det)

        # 处理消失的跟踪器
        disappeared_ids = []
        for tracker_id, tracker in self.trackers.items():
            if not tracker['matched']:
                tracker['disappeared'] += 1
                if tracker['disappeared'] > self.max_disappeared:
                    disappeared_ids.append(tracker_id)

        for tracker_id in disappeared_ids:
            del self.trackers[tracker_id]

    def create_new_tracker(self, detection):
        """创建新跟踪器"""
        tracker_id = self.next_tracker_id
        self.trackers[tracker_id] = {
            'bboxes': [detection['bbox']],
            'positions': [detection['center']],
            'class_name': detection['class_name'],
            'matched': True,
            'disappeared': 0,
            'track_frames': 1
        }
        self.next_tracker_id += 1

    def draw_red_boxes_with_ids(self, frame):
        """绘制红框和ID"""
        for tracker_id, tracker in self.trackers.items():
            if tracker['disappeared'] > 0:
                continue

            # 获取当前边界框
            current_bbox = tracker['bboxes'][-1]
            x1, y1, x2, y2 = current_bbox

            # 绘制红色边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['bicycle'], 3)

            # 绘制中心点
            center_x, center_y = tracker['positions'][-1]
            cv2.circle(frame, (center_x, center_y), 4, (255, 255, 0), -1)

            # 绘制ID标签
            self.draw_id_label(frame, tracker_id, x1, y1, x2, y2)

            # 绘制运动轨迹
            self.draw_trajectory(frame, tracker)

        return frame

    def draw_id_label(self, frame, tracker_id, x1, y1, x2, y2):
        """绘制ID标签"""
        # ID文本
        id_text = f"ID:{tracker_id}"

        # 计算文本大小
        text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # 标签位置（框的左上角）
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10

        # 绘制标签背景
        bg_x1 = label_x - 5
        bg_y1 = label_y - text_size[1] - 5
        bg_x2 = label_x + text_size[0] + 5
        bg_y2 = label_y + 5

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), self.colors['id_bg'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制ID文本
        cv2.putText(frame, id_text, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_trajectory(self, frame, tracker):
        """绘制运动轨迹"""
        positions = tracker['positions']
        if len(positions) < 2:
            return

        # 绘制最近10个点的轨迹
        for i in range(max(1, len(positions) - 10), len(positions)):
            if i == 0:
                continue
            # 渐变色轨迹
            alpha = (i - max(0, len(positions) - 10)) / min(10, len(positions))
            color = (0, int(255 * alpha), 255)
            cv2.line(frame, positions[i - 1], positions[i], color, 2)

    def process_video(self, video_path, output_path=None, confidence_threshold=0.45):
        """处理视频并标注红框"""
        if self.yolo_model is None:
            print("❌ YOLO模型未加载")
            return

        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return

        # 获取视频信息
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 视频信息: {width}x{height}, {self.fps:.2f}FPS, 总帧数: {total_frames}")

        # 自动设置输出视频路径
        if output_path is None:
            output_dir = "red_box_results"
            os.makedirs(output_dir, exist_ok=True)
            input_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"{input_name}_redbox_{timestamp}.mp4")

        # 视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        print("🚀 开始红框标注...")
        print(f"🎯 置信度阈值: {confidence_threshold}")
        print(f"💾 输出文件: {output_path}")

        frame_count = 0
        start_time = time.time()
        self.processing_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 检测非机动车
            detect_start = time.time()
            detections = self.detect_vehicles(frame, confidence_threshold)
            detect_time = time.time() - detect_start
            self.processing_times.append(detect_time)

            # 更新跟踪
            self.update_tracking(detections)

            # 绘制红框和ID
            display_frame = self.draw_red_boxes_with_ids(frame.copy())

            # 显示统计信息
            self.draw_stats(display_frame, detections, detect_time, frame_count, start_time, total_frames)

            # 写入输出视频
            out.write(display_frame)

            # 显示实时预览
            cv2.imshow('红框标注系统', display_frame)

            # 进度显示
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                active_trackers = sum(1 for t in self.trackers.values() if t['disappeared'] == 0)
                print(f"📊 进度: {progress:.1f}% | 跟踪目标: {active_trackers}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ 用户中断处理")
                break

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 性能统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print(f"\n✅ 红框标注完成!")
        print(f"📊 统计:")
        print(f"  • 总帧数: {frame_count}")
        print(f"  • 平均FPS: {avg_fps:.1f}")
        print(f"  • 总耗时: {total_time:.1f}s")
        print(f"  • 输出文件: {output_path}")

    def draw_stats(self, frame, detections, detect_time, frame_count, start_time, total_frames):
        """绘制统计信息"""
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        active_trackers = sum(1 for t in self.trackers.values() if t['disappeared'] == 0)

        stats = [
            f"FPS: {fps:.1f}",
            f"进度: {progress:.1f}%",
            f"检测: {len(detections)}",
            f"跟踪: {active_trackers}",
            f"红框标注系统"
        ]

        y_offset = 20
        for i, text in enumerate(stats):
            color = (0, 255, 255) if i == 0 else (0, 255, 255)
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 18

    def get_tracking_data(self):
        """获取跟踪数据（供速度模块使用）"""
        return self.trackers.copy()


def main():
    """红框标注模块主程序"""
    print("🚀 红框标注系统")
    print("=" * 50)
    print("🎯 功能: 检测非机动车并标注红色框和ID")

    # 初始化检测器
    print("\n🔄 初始化检测器...")
    detector = RedBoxDetector()

    if detector.yolo_model is None:
        print("❌ 检测器初始化失败")
        return

    video_path = input("视频文件路径: ").strip().strip('"')
    if not os.path.exists(video_path):
        print("❌ 视频文件不存在")
        return

    conf = 0.45
    try:
        conf_input = input(f"置信度阈值 (默认{conf}): ").strip()
        if conf_input:
            conf = float(conf_input)
    except:
        pass

    print("🚀 开始红框标注...")
    detector.process_video(video_path, confidence_threshold=conf)


if __name__ == "__main__":
    main()