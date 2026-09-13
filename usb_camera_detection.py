import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import json
import time
import threading
import subprocess
import pygame

# ----------------- 警告管理类（同前，省略注释以节省篇幅）-----------------
class AlertManager:
    def __init__(self, red_angle_thresh=20, yellow_angle_thresh=10,
                 red_duration=1.0, yellow_duration=3.0):
        self.red_thresh = red_angle_thresh
        self.yellow_thresh = yellow_angle_thresh
        self.red_duration = red_duration
        self.yellow_duration = yellow_duration
        self.red_start_time = None
        self.yellow_start_time = None
        self.red_triggered = False
        self.yellow_triggered = False
        pygame.mixer.init()
        self.beep_sound = self.generate_beep_sound(440, 0.2)

    def generate_beep_sound(self, freq=440, duration=0.2, sample_rate=22050):
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        wave = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)

    def play_beep(self, repeat=3):
        def _play():
            for _ in range(repeat):
                self.beep_sound.play()
                time.sleep(0.3)
        threading.Thread(target=_play, daemon=True).start()

    def speak_warning(self, text="请注意当前驾驶可能有危险"):
        def _speak():
            subprocess.run(['espeak', '-v', 'zh', text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        threading.Thread(target=_speak, daemon=True).start()

    def update(self, angle_deg):
        current_time = time.time()
        is_red = angle_deg >= self.red_thresh
        is_yellow = self.yellow_thresh <= angle_deg < self.red_thresh

        if is_red:
            if self.red_start_time is None:
                self.red_start_time = current_time
                self.red_triggered = False
            else:
                elapsed = current_time - self.red_start_time
                if not self.red_triggered and elapsed >= self.red_duration:
                    print("🚨 红色警报触发！嘟嘟嘟...")
                    self.play_beep(repeat=3)
                    self.red_triggered = True
                    self.yellow_start_time = None
                    self.yellow_triggered = False
        else:
            self.red_start_time = None
            self.red_triggered = False

        if is_yellow and not is_red:
            if self.yellow_start_time is None:
                self.yellow_start_time = current_time
                self.yellow_triggered = False
            else:
                elapsed = current_time - self.yellow_start_time
                if not self.yellow_triggered and elapsed >= self.yellow_duration:
                    print("⚠️ 黄色警报触发！语音提示...")
                    self.speak_warning("请注意当前驾驶可能有危险")
                    self.yellow_triggered = True
        else:
            self.yellow_start_time = None
            self.yellow_triggered = False

        if is_red:
            return "RED"
        elif is_yellow:
            return "YELLOW"
        else:
            return "NORMAL"

# ----------------- 角度识别模型（保持不变）-----------------
class AngleCNN(torch.nn.Module):
    def __init__(self, conv_layers=3, start_channels=32, pool_kernel=2, pool_stride=2,
                 dropout_rate=0.3, use_batch_norm=True, input_size=64):
        super().__init__()
        self.conv_layers = conv_layers
        self.start_channels = start_channels
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        self.conv_blocks = torch.nn.ModuleList()
        current_channels = 3
        if conv_layers == 2:
            channel_growth = [start_channels, start_channels * 2]
        elif conv_layers == 3:
            channel_growth = [start_channels, start_channels * 2, start_channels * 3]
        else:
            channel_growth = [start_channels, start_channels * 2, start_channels * 3, start_channels * 3]

        for i in range(conv_layers):
            out_channels = channel_growth[i]
            layers = []
            layers.append(torch.nn.Conv2d(current_channels, out_channels, 3, padding=1, bias=False))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.MaxPool2d(pool_kernel, pool_stride))
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout2d(dropout_rate / 3))
            self.conv_blocks.append(torch.nn.Sequential(*layers))
            current_channels = out_channels

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = current_channels
        self.angle_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 192),
            torch.nn.BatchNorm1d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(192, 96),
            torch.nn.BatchNorm1d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(self.dropout_rate * 0.8),
            torch.nn.Linear(96, 48),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(self.dropout_rate * 0.6),
            torch.nn.Linear(48, 7)
        )

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.global_avg_pool(x)
        features = x.view(x.size(0), -1)
        angle_output = self.angle_classifier(features)
        return angle_output

# ----------------- 辅助函数 -----------------
def load_model(model_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_data = torch.load(model_path, map_location=device)
        model_config = model_data['model_config']
        model = AngleCNN(
            conv_layers=model_config['conv_layers'],
            start_channels=model_config['start_channels'],
            pool_kernel=model_config['pool_kernel'],
            pool_stride=model_config['pool_stride'],
            dropout_rate=model_config['dropout_rate'],
            use_batch_norm=model_config['use_batch_norm'],
            input_size=model_config['input_size']
        )
        model.load_state_dict(model_data['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"✅ 模型加载成功: {model_path}")
        return model, device
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def preprocess_image(image, input_size=64):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_angle_label(predicted_class):
    angle_mapping = {0: '0°', 1: '5°', 2: '10°', 3: '15°', 4: '20°', 5: '25°', 6: '30°'}
    return angle_mapping.get(predicted_class, '未知')

def save_detection_result(frame, angle, confidence, save_dir, timestamp):
    os.makedirs(save_dir, exist_ok=True)
    image_filename = f"detection_{timestamp}.jpg"
    image_path = os.path.join(save_dir, image_filename)
    annotated_frame = frame.copy()
    cv2.putText(annotated_frame, f"Angle: {angle}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Time: {timestamp}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(image_path, annotated_frame)
    json_filename = f"detection_{timestamp}.json"
    json_path = os.path.join(save_dir, json_filename)
    detection_data = {
        'timestamp': timestamp,
        'angle': angle,
        'confidence': float(confidence),
        'image_path': image_filename,
        'detection_time': datetime.now().isoformat()
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detection_data, f, ensure_ascii=False, indent=2)
    return image_path, json_path

# ----------------- 主函数（无 GUI 版本）-----------------
def main():
    print("🚀 USB摄像头实时检测系统（无头模式）")
    print("=" * 60)

    model_path = "best_angle_model.pt"
    save_dir = "usb_detections"
    os.makedirs(save_dir, exist_ok=True)

    model, device = load_model(model_path)
    if model is None:
        print("❌ 无法加载模型，退出程序")
        return

    alert_manager = AlertManager(red_angle_thresh=20, yellow_angle_thresh=10,
                                 red_duration=1.0, yellow_duration=3.0)

    print("📹 打开USB摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("✅ 摄像头打开成功")
    print(f"⚙️ 警告阈值: 红色≥{alert_manager.red_thresh}° | 黄色≥{alert_manager.yellow_thresh}°")
    print("💡 按 Ctrl+C 停止程序")

    frame_count = 0
    detection_interval = 5
    current_alert_level = "NORMAL"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 无法读取帧")
                continue

            frame_count += 1

            if frame_count % detection_interval == 0:
                try:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    input_tensor = preprocess_image(image).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted_class = torch.max(probabilities, 1)

                    angle_str = get_angle_label(predicted_class.item())
                    angle_val = int(angle_str.replace('°', ''))
                    confidence_val = confidence.item()

                    alert_level = alert_manager.update(angle_val)
                    if alert_level != current_alert_level:
                        current_alert_level = alert_level
                        print(f"🔔 警告级别变更: {alert_level} (角度={angle_val}°)")

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    image_path, json_path = save_detection_result(
                        frame, angle_str, confidence_val, save_dir, timestamp
                    )

                    print(f"📊 检测结果: Angle={angle_str}, Confidence={confidence_val:.2f}, Level={alert_level}")
                    print(f"💾 保存位置: {image_path}")
                    print("-" * 60)

                except Exception as e:
                    print(f"⚠️ 检测出错: {e}")

            # 无 GUI，仅通过键盘中断退出（Ctrl+C）
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断程序")
    finally:
        cap.release()
        print("✅ 资源释放完成")
        print(f"📁 检测结果保存目录: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()