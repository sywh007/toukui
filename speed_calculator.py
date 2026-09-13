import cv2
import os
import numpy as np
import time
from datetime import datetime
from collections import deque


class SpeedCalculator:
    def __init__(self, tracking_data, fps, frame_size):
        """初始化速度判断系统"""
        self.tracking_data = tracking_data
        self.fps = fps
        self.width, self.height = frame_size

        # 速度计算参数
        self.setup_speed_parameters()

        # 速度数据
        self.speed_history = {}

        print("✅ 速度判断系统初始化完成!")

    def setup_speed_parameters(self):
        """设置速度计算参数"""
        self.pixel_to_meter = 0.001  # 像素到米的转换系数

        # 速度阈值 (km/h)
        self.speed_threshold_low = 10.0  # 低速阈值
        self.speed_threshold_high = 30.0  # 高速阈值

        # 颜色配置
        self.speed_colors = {
            'low': (0, 255, 0),  # 绿色 - 低速
            'medium': (0, 255, 255),  # 黄色 - 中速
            'high': (0, 0, 255)  # 红色 - 高速
        }

    def calculate_speeds(self):
        """计算所有跟踪目标的速度"""
        for tracker_id, tracker in self.tracking_data.items():
            if tracker['disappeared'] > 0:
                continue

            positions = tracker['positions']
            if len(positions) < 2:
                continue

            # 计算速度
            speed_kmh = self.calculate_instant_speed(positions)

            # 乘以9.5的系数
            speed_kmh *= 9.5

            # 存储速度历史
            if tracker_id not in self.speed_history:
                self.speed_history[tracker_id] = deque(maxlen=10)

            self.speed_history[tracker_id].append(speed_kmh)

            # 计算平滑速度
            smoothed_speed = self.get_smoothed_speed(tracker_id)
            tracker['current_speed'] = smoothed_speed

            # 判断速度等级
            tracker['speed_level'] = self.get_speed_level(smoothed_speed)

    def calculate_instant_speed(self, positions):
        """计算瞬时速度"""
        # 使用最近的两帧位置
        pos1 = positions[-2]
        pos2 = positions[-1]

        # 计算像素位移
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        pixel_distance = np.sqrt(dx ** 2 + dy ** 2)

        # 时间间隔
        time_interval = 1.0 / self.fps

        # 计算速度 (米/秒 -> 公里/小时)
        speed_mps = (pixel_distance * self.pixel_to_meter) / time_interval
        speed_kmh = speed_mps * 3.6

        return speed_kmh

    def get_smoothed_speed(self, tracker_id):
        """获取平滑后的速度"""
        if tracker_id not in self.speed_history or len(self.speed_history[tracker_id]) < 2:
            return 0.0

        # 使用加权平均平滑速度
        speeds = list(self.speed_history[tracker_id])
        weights = np.arange(1, len(speeds) + 1)
        smoothed_speed = np.average(speeds, weights=weights)

        return smoothed_speed

    def get_speed_level(self, speed_kmh):
        """判断速度等级"""
        if speed_kmh < self.speed_threshold_low:
            return 'low'
        elif speed_kmh < self.speed_threshold_high:
            return 'medium'
        else:
            return 'high'

    def get_speed_color(self, speed_level):
        """获取速度对应的颜色"""
        return self.speed_colors.get(speed_level, (255, 255, 255))

    def draw_speed_info(self, frame):
        """在帧上绘制速度信息"""
        for tracker_id, tracker in self.tracking_data.items():
            if tracker['disappeared'] > 0:
                continue

            if 'current_speed' not in tracker:
                continue

            # 获取当前边界框
            current_bbox = tracker['bboxes'][-1]
            x1, y1, x2, y2 = current_bbox

            # 速度信息
            speed_kmh = tracker['current_speed']
            speed_level = tracker.get('speed_level', 'low')
            speed_color = self.get_speed_color(speed_level)

            # 速度文本 - 显示乘以9.5后的速度
            speed_text = f"{speed_kmh:.1f} km/h"
            level_text = f"({speed_level})"

            # 绘制速度信息在红框上
            self.draw_speed_label(frame, x1, y1, x2, y2, speed_text, level_text, speed_color)

    def draw_speed_label(self, frame, x1, y1, x2, y2, speed_text, level_text, color):
        """绘制速度标签在红框上"""
        # 计算文本位置（在红框内部上方）
        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20

        # 速度文本大小
        speed_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # 背景框位置（在红框内部上方）
        bg_x1 = x1
        bg_y1 = text_y - speed_size[1] - 5
        bg_x2 = x1 + speed_size[0] + 10
        bg_y2 = text_y + 5

        # 确保背景框在图像范围内
        if bg_y1 < 0:
            bg_y1 = y2 + 5
            bg_y2 = bg_y1 + speed_size[1] + 10

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制速度文本
        cv2.putText(frame, speed_text, (x1 + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def generate_speed_report(self):
        """生成速度统计报告"""
        print("\n" + "=" * 60)
        print("速度统计报告")
        print("=" * 60)

        all_speeds = []
        speed_levels = {'low': 0, 'medium': 0, 'high': 0}

        for tracker_id, tracker in self.tracking_data.items():
            if 'current_speed' in tracker and tracker['current_speed'] > 0:
                speed_kmh = tracker['current_speed']
                speed_level = tracker.get('speed_level', 'low')

                all_speeds.append(speed_kmh)
                speed_levels[speed_level] += 1

                print(f"目标 {tracker_id}: {speed_kmh:.1f} km/h ({speed_level})")

        if all_speeds:
            print(f"\n全局统计:")
            print(f"  平均速度: {np.mean(all_speeds):.1f} km/h")
            print(f"  最高速度: {np.max(all_speeds):.1f} km/h")
            print(f"  最低速度: {np.min(all_speeds):.1f} km/h")
            print(f"  速度分布: 低速({speed_levels['low']}) 中速({speed_levels['medium']}) 高速({speed_levels['high']})")
        else:
            print("没有速度数据")

    def calibrate_pixel_ratio(self, pixel_distance, actual_distance):
        """校准像素到米的转换系数"""
        self.pixel_to_meter = actual_distance / pixel_distance
        print(f"✅ 校准完成: 1像素 = {self.pixel_to_meter * 100:.4f}厘米")


def main():
    """速度判断模块主程序"""
    print("🚀 速度判断系统")
    print("=" * 50)
    print("🎯 功能: 基于红框跟踪数据计算和显示速度")
    print("💡 请先运行红框标注模块获取跟踪数据")

    # 这里需要从红框标注模块获取数据
    # 在实际使用中，这两个模块可以通过数据接口连接
    print("\n📝 使用方法:")
    print("1. 先运行 red_box_detector.py 进行红框标注")
    print("2. 然后使用速度模块处理跟踪数据")
    print("3. 或者将两个模块集成到同一个系统中")


if __name__ == "__main__":
    main()