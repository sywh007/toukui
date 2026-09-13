#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派客户端：角度检测 + 警报 + 视频录制上传 + WebSocket 控制
连接电脑中心服务器，接收远程命令（支持前端 start/stop）
默认不录制，等待网页命令
支持用户名，上传的视频和检测文件包含用户名
"""

import cv2
import os
import sys
import time
import json
import threading
import subprocess
import requests
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import pygame
import socketio
import re
import platform

# ================== 配置参数 ==================
# 服务器配置（电脑 IP）
SERVER_IP = "192.168.43.120"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_IP}:{SERVER_PORT}"
UPLOAD_URL = f"{SERVER_URL}/upload_video"
WEBSOCKET_URL = SERVER_URL  # SocketIO 连接地址

# 录制配置
VIDEO_DURATION = 30  # 每段视频时长（秒）
VIDEO_FPS = 20  # 录制帧率
VIDEO_WIDTH = 640  # 视频宽度
VIDEO_HEIGHT = 480  # 视频高度
LOCAL_TEMP_DIR = "temp_videos"  # 临时视频存储目录
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

# 检测配置
DETECTION_INTERVAL = 5  # 每5帧推理一次（降低CPU占用）
MODEL_PATH = "best_angle_model.pt"
SAVE_DIR = "usb_detections"  # 检测结果保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 警告阈值
RED_ANGLE_THRESH = 20  # 红色警报角度下限
YELLOW_ANGLE_THRESH = 10  # 黄色警报角度下限
RED_DURATION = 1.0  # 红色持续触发时间（秒）
YELLOW_DURATION = 3.0  # 黄色持续触发时间（秒）


# ================== 全局状态 ==================
class GlobalState:
    recording_enabled = True  # 连接后自动开始录制
    detection_enabled = True  # 是否允许检测
    alert_enabled = True  # 是否允许警报
    username = "unknown"  # 当前用户名（英文），由服务器下发
    camera_running = False  # 摄像头是否正在运行
    main_system = None  # 主系统实例
    cap = None  # 摄像头对象

    @staticmethod
    def safe_username():
        """返回安全的文件名友好用户名（只保留字母数字下划线）"""
        return re.sub(r'[^a-zA-Z0-9_]', '_', GlobalState.username)


# ================== 警告管理类 ==================
class AlertManager:
    def __init__(self, red_dur=1.0, yellow_dur=3.0):
        self.red_duration = red_dur
        self.yellow_duration = yellow_dur

        self.red_start_time = None
        self.yellow_start_time = None
        self.red_triggered = False
        self.yellow_triggered = False

        # 初始化 pygame.mixer
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.music.set_volume(1.0)

        self.beep_sound = self._generate_beep(440, 0.2)

    def _generate_beep(self, freq=440, duration=0.2, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        wave = (wave * 32767).astype(np.int16)
        stereo = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo)
        sound.set_volume(1.0)
        return sound

    def _play_beep(self, repeat=3):
        def _play():
            for _ in range(repeat):
                self.beep_sound.play()
                time.sleep(0.3)

        threading.Thread(target=_play, daemon=True).start()

    def _speak_warning(self, text="周围危险，请当心"):
        def _speak():
            subprocess.run(['espeak', '-v', 'zh', '-a', '200', text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        threading.Thread(target=_speak, daemon=True).start()

    def update(self, safety_level):
        now = time.time()
        is_red = safety_level == "危险"
        is_yellow = safety_level == "警戒"

        # 红色处理
        if is_red:
            if self.red_start_time is None:
                self.red_start_time = now
                self.red_triggered = False
            elif not self.red_triggered and (now - self.red_start_time) >= self.red_duration:
                print("🚨 红色警报触发！嘟嘟嘟...")
                self._play_beep(3)
                self.red_triggered = True
                self.yellow_start_time = None
                self.yellow_triggered = False
        else:
            self.red_start_time = None
            self.red_triggered = False

        # 黄色处理（红色优先）
        if is_yellow and not is_red:
            if self.yellow_start_time is None:
                self.yellow_start_time = now
                self.yellow_triggered = False
            elif not self.yellow_triggered and (now - self.yellow_start_time) >= self.yellow_duration:
                print("⚠️ 黄色警报触发！周围危险，请当心...")
                self._speak_warning("周围危险，请当心")
                self.yellow_triggered = True
        else:
            self.yellow_start_time = None
            self.yellow_triggered = False

        if is_red:
            return "RED"
        elif is_yellow:
            return "YELLOW"
        return "NORMAL"


# ================== 角度识别模型 ==================
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
            growth = [start_channels, start_channels * 2]
        elif conv_layers == 3:
            growth = [start_channels, start_channels * 2, start_channels * 3]
        else:
            growth = [start_channels, start_channels * 2, start_channels * 3, start_channels * 3]

        for i in range(conv_layers):
            out = growth[i]
            layers = [
                torch.nn.Conv2d(current_channels, out, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out) if use_batch_norm else torch.nn.Identity(),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out, out, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out) if use_batch_norm else torch.nn.Identity(),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(pool_kernel, pool_stride),
            ]
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout2d(dropout_rate / 3))
            self.conv_blocks.append(torch.nn.Sequential(*layers))
            current_channels = out

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # 使用 angle_classifier 以匹配权重文件键名
        self.angle_classifier = torch.nn.Sequential(
            torch.nn.Linear(current_channels, 192),
            torch.nn.BatchNorm1d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(192, 96),
            torch.nn.BatchNorm1d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout_rate * 0.8),
            torch.nn.Linear(96, 48),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout_rate * 0.6),
            torch.nn.Linear(48, 7)
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.angle_classifier(x)


def load_model(model_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.load(model_path, map_location=device)
        cfg = data['model_config']
        model = AngleCNN(
            conv_layers=cfg['conv_layers'],
            start_channels=cfg['start_channels'],
            pool_kernel=cfg['pool_kernel'],
            pool_stride=cfg['pool_stride'],
            dropout_rate=cfg['dropout_rate'],
            use_batch_norm=cfg['use_batch_norm'],
            input_size=cfg['input_size']
        )
        model.load_state_dict(data['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"✅ 模型加载成功: {model_path}")
        return model, device
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None


# ================== 视频录制上传管理类 ==================
class VideoRecorder:
    def __init__(self, cap, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=VIDEO_FPS):
        self.cap = cap
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        self.recording = False
        self.current_path = None
        self.frame_count = 0
        self.start_time = 0

    def start_new_video(self):
        """启动新视频，文件名包含当前用户名"""
        safe_name = GlobalState.safe_username()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{safe_name}_{timestamp}.mp4"
        self.current_path = os.path.join(LOCAL_TEMP_DIR, filename)

        codecs = [
            (cv2.VideoWriter_fourcc(*'mp4v'), 'mp4v'),
            (cv2.VideoWriter_fourcc(*'avc1'), 'avc1'),
            (cv2.VideoWriter_fourcc(*'X264'), 'X264'),
            (cv2.VideoWriter_fourcc(*'MJPG'), 'MJPG'),
        ]
        for fourcc, name in codecs:
            writer = cv2.VideoWriter(self.current_path, fourcc, self.fps, (self.width, self.height))
            if writer.isOpened():
                self.writer = writer
                print(f"🎥 开始录制: {filename} (编码: {name}) 用户: {GlobalState.username}")
                self.recording = True
                self.frame_count = 0
                self.start_time = time.time()
                return True

        self.current_path = self.current_path.replace('.mp4', '.avi')
        writer = cv2.VideoWriter(self.current_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                 self.fps, (self.width, self.height))
        if writer.isOpened():
            self.writer = writer
            print(f"🎥 开始录制: {os.path.basename(self.current_path)} (备用 AVI) 用户: {GlobalState.username}")
            self.recording = True
            self.frame_count = 0
            self.start_time = time.time()
            return True
        print("❌ 无法创建视频文件")
        return False

    def write_frame(self, frame):
        if not self.recording or self.writer is None:
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"User: {GlobalState.username}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        self.writer.write(frame)
        self.frame_count += 1

    def should_stop(self):
        if not self.recording:
            return True
        return (time.time() - self.start_time) >= VIDEO_DURATION

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False
        if self.current_path and os.path.exists(self.current_path):
            size_mb = os.path.getsize(self.current_path) / (1024 * 1024)
            print(f"💾 视频已保存: {os.path.basename(self.current_path)} ({size_mb:.2f} MB)")
            return self.current_path
        return None

    def upload(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return False
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📤 上传中: {os.path.basename(file_path)} ({size_mb:.2f} MB)")
        print(f"📡 上传地址: {UPLOAD_URL}")
        for attempt in range(3):
            try:
                mime = 'video/mp4' if file_path.endswith('.mp4') else 'video/x-msvideo'
                with open(file_path, 'rb') as f:
                    files = {'video': (os.path.basename(file_path), f, mime)}
                    resp = requests.post(UPLOAD_URL, files=files, timeout=60)
                if resp.status_code == 200:
                    print("✅ 上传成功")
                    print(f"📥 视频可访问地址: http://{SERVER_IP}:{SERVER_PORT}/play/{os.path.basename(file_path)}")
                    os.remove(file_path)
                    return True
                else:
                    print(f"⚠️ 上传失败 HTTP {resp.status_code}")
            except Exception as e:
                print(f"⚠️ 上传错误: {e}")
            if attempt < 2:
                time.sleep(5)
        return False


# ================== 摄像头打开函数 ==================
def open_camera():
    """跨平台安全打开摄像头（树莓派兼容）"""
    system = platform.system()

    if system == 'Linux':
        # 检查是否有 v4l2 设备
        if not os.path.exists('/dev/video0'):
            print("⚠️ /dev/video0 不存在，尝试加载 bcm2835-v4l2 模块...")
            subprocess.run(['sudo', 'modprobe', 'bcm2835-v4l2'], stderr=subprocess.DEVNULL)

        # 尝试多个索引
        for idx in range(3):
            # 使用 V4L2 后端
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"✅ 成功打开摄像头 /dev/video{idx}")
                return cap
            cap.release()

        # 如果 V4L2 失败，尝试默认后端
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"✅ 使用默认后端打开摄像头 {idx}")
                return cap
            cap.release()

        raise RuntimeError("❌ 所有摄像头尝试均失败，请检查连接和权限。")
    else:
        # Windows / macOS
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"✅ 打开摄像头 {idx}")
                return cap
            cap.release()
        raise RuntimeError("❌ 未找到可用摄像头")


def stop_camera():
    """停止并释放摄像头"""
    if GlobalState.cap:
        GlobalState.cap.release()
        GlobalState.cap = None
        GlobalState.camera_running = False
        print("📷 摄像头已关闭")


# ================== 主处理循环 ==================
def camera_processing_loop():
    """摄像头处理和录制主循环"""
    current_alert_level = "NORMAL"
    video_upload_count = 0
    alert_mgr = AlertManager(RED_DURATION, YELLOW_DURATION)
    recorder = None
    frame_idx = 0

    # 初始化主系统
    from main_integrated import FastIntegratedSystem
    main_system = FastIntegratedSystem()
    GlobalState.main_system = main_system

    # 加载角度检测模型
    angle_model_loaded = main_system.red_box_detector.load_angle_model()
    if not angle_model_loaded:
        print("⚠️ 角度检测模型加载失败，将继续运行但角度检测不可用")
    else:
        print("✅ 角度检测模型加载成功!")

    # 初始化速度计算器
    from speed_calculator import SpeedCalculator
    main_system.speed_calculator = SpeedCalculator(main_system.red_box_detector.trackers, VIDEO_FPS,
                                                   (VIDEO_WIDTH, VIDEO_HEIGHT))

    # 初始化录制器
    recorder = VideoRecorder(GlobalState.cap)

    print("🎬 摄像头已启动，等待录制命令...")

    while GlobalState.camera_running:
        ret, frame = GlobalState.cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_idx += 1

        # 检测逻辑（只有在启用检测且录制时才进行）
        if GlobalState.detection_enabled and GlobalState.recording_enabled and frame_idx % DETECTION_INTERVAL == 0:
            try:
                # 检测车辆
                detections = main_system.red_box_detector.detect_vehicles(frame, confidence_threshold=0.45)
                main_system.red_box_detector.update_tracking(detections)

                # 检查是否有检测到的目标
                active_trackers = sum(
                    1 for t in main_system.red_box_detector.trackers.values() if t['disappeared'] == 0)
                has_targets = active_trackers > 0

                # 计算速度
                if has_targets and hasattr(main_system, 'speed_calculator') and main_system.speed_calculator:
                    main_system.speed_calculator.calculate_speeds()

                # 简化距离计算
                if has_targets:
                    main_system.calculate_simplified_distances()

                # 检测角度
                if has_targets:
                    main_system.red_box_detector.detect_angles_for_trackers(frame)
                    main_system.red_box_detector.calculate_safety_factors()

                # 计算平均安全系数
                avg_safety, left_avg, right_avg = main_system.red_box_detector.calculate_average_safety_factor()

                # 确定安全等级
                if not has_targets:
                    safety_level = "安全"
                else:
                    # 检查是否有红色方框（危险目标）
                    has_red_box = any(
                        tracker['safety_level'] == "危险" for tracker in main_system.red_box_detector.trackers.values() if
                        tracker['disappeared'] == 0)

                    if has_red_box:
                        safety_level = "危险"
                    elif avg_safety < 0.5:
                        safety_level = "警戒"
                    else:
                        safety_level = "安全"

                if GlobalState.alert_enabled:
                    alert_level = alert_mgr.update(safety_level)
                else:
                    alert_level = "NORMAL"

                if alert_level != current_alert_level:
                    current_alert_level = alert_level
                    print(f"🔔 警报级别: {alert_level} (安全系数={avg_safety:.2f})")
                    sio.emit('pi_status_update', {'current_alert': alert_level})

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                # 绘制结果
                display_frame = main_system.red_box_detector.draw_red_boxes_with_info(frame.copy())

                # 删除左下角的红色文字
                height, width = display_frame.shape[:2]
                cv2.rectangle(display_frame, (0, height - 100), (300, height), (0, 0, 0), -1)

                # 保存检测结果
                safe_name = GlobalState.safe_username()
                img_name = f"detection_{safe_name}_{ts}.jpg"
                img_path = os.path.join(SAVE_DIR, img_name)
                cv2.imwrite(img_path, display_frame)

                # 保存 JSON 结果
                json_name = f"detection_{safe_name}_{ts}.json"
                json_path = os.path.join(SAVE_DIR, json_name)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'username': GlobalState.username,
                        'timestamp': ts,
                        'safety_level': safety_level,
                        'average_safety': float(avg_safety),
                        'left_safety': float(left_avg),
                        'right_safety': float(right_avg),
                        'image_path': img_name,
                        'detection_time': datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)

                print(f"📊 检测: {safety_level} (安全系数={avg_safety:.2f}) -> {img_path}")

            except Exception as e:
                print(f"⚠️ 检测错误: {e}")

        # 视频录制管理
        if GlobalState.recording_enabled:
            if not recorder.recording:
                if recorder.start_new_video():
                    print("🎥 开始录制视频")
                else:
                    time.sleep(1)
                    continue
            else:
                # 绘制结果
                display_frame = main_system.red_box_detector.draw_red_boxes_with_info(frame.copy())

                # 直接写入帧，不添加黑色遮挡
                recorder.write_frame(display_frame)

                # 检查是否达到录制时长
                if recorder.should_stop():
                    video_path = recorder.stop()
                    if video_path:
                        print("⏸️ 录制已完成，正在上传...")
                        if recorder.upload(video_path):
                            video_upload_count += 1
                            print(f"📊 已上传 {video_upload_count} 个视频")
                            sio.emit('pi_status_update', {'video_count': video_upload_count})
                        # 立即开始录制新视频
                        if recorder.start_new_video():
                            print("🎥 开始录制新视频")
                        else:
                            time.sleep(1)
        else:
            if recorder.recording:
                video_path = recorder.stop()
                if video_path:
                    print("⏸️ 录制已停止，正在上传...")
                    if recorder.upload(video_path):
                        video_upload_count += 1
                        print(f"📊 已上传 {video_upload_count} 个视频")
                        sio.emit('pi_status_update', {'video_count': video_upload_count})

        time.sleep(1.0 / VIDEO_FPS)

    # 循环结束，清理资源
    if recorder and recorder.recording:
        recorder.stop()
    print("📷 摄像头处理循环已停止")


# ================== WebSocket 客户端 ==================
sio = socketio.Client()
video_upload_count = 0


@sio.event
def connect():
    print("✅ 已连接到电脑服务器")
    sio.emit('pi_identify', {'client_type': 'pi'})

    # 先发送初始状态
    sio.emit('pi_status_update', {
        'recording_enabled': GlobalState.recording_enabled,
        'detection_enabled': GlobalState.detection_enabled,
        'alert_enabled': GlobalState.alert_enabled,
        'video_duration': VIDEO_DURATION,
        'current_alert': "NORMAL",
        'video_count': video_upload_count,
        'username': GlobalState.username,
        'camera_running': GlobalState.camera_running
    })

    # 延迟启动摄像头，避免连接过程中进行耗时操作
    def start_camera_task():
        time.sleep(2)  # 等待 2 秒，确保连接稳定
        if not GlobalState.camera_running:
            try:
                print("🚀 连接稳定，启动摄像头...")
                GlobalState.cap = open_camera()
                GlobalState.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
                GlobalState.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
                GlobalState.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
                GlobalState.camera_running = True
                # 明确设置为录制模式
                GlobalState.recording_enabled = True
                print(f"🔧 设置录制模式: {GlobalState.recording_enabled}")
                # 启动处理线程
                processing_thread = threading.Thread(target=camera_processing_loop, daemon=True)
                processing_thread.start()
                print("📷 摄像头已启动，开始录制和上传视频")
                # 发送更新后的状态
                sio.emit('pi_status_update', {
                    'recording_enabled': GlobalState.recording_enabled,
                    'camera_running': GlobalState.camera_running
                })
            except Exception as e:
                print(f"❌ 摄像头启动失败: {e}")
                sio.emit('pi_response', {'status': 'error', 'message': f'摄像头启动失败: {e}'})

    # 在后台线程中启动摄像头
    threading.Thread(target=start_camera_task, daemon=True).start()


@sio.event
def disconnect():
    print("❌ 与服务器断开连接，将尝试重连...")


@sio.on('server_command')
def on_server_command(data):
    global VIDEO_DURATION, video_upload_count
    cmd = data.get('command')
    params = data.get('params', {})
    print(f"📩 收到服务器命令: {cmd}")

    # 前端兼容命令：start / stop
    if cmd == 'start' or cmd == 'start_recording':
        # 启动摄像头（如果还没启动）
        if not GlobalState.camera_running:
            try:
                GlobalState.cap = open_camera()
                GlobalState.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
                GlobalState.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
                GlobalState.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
                GlobalState.camera_running = True
                # 启动处理线程
                processing_thread = threading.Thread(target=camera_processing_loop, daemon=True)
                processing_thread.start()
                print("📷 摄像头已启动")
            except Exception as e:
                print(f"❌ 摄像头启动失败: {e}")
                sio.emit('pi_response', {'status': 'error', 'message': f'摄像头启动失败: {e}'})
                return

        GlobalState.recording_enabled = True
        sio.emit('pi_response', {'status': 'ok', 'message': '录制已开启'})
        sio.emit('pi_status_update', {
            'recording_enabled': True,
            'camera_running': GlobalState.camera_running
        })

    elif cmd == 'stop' or cmd == 'stop_recording':
        GlobalState.recording_enabled = False
        sio.emit('pi_response', {'status': 'ok', 'message': '录制已停止'})
        sio.emit('pi_status_update', {'recording_enabled': False})

    elif cmd == 'stop_camera':
        # 完全停止摄像头
        GlobalState.recording_enabled = False
        GlobalState.camera_running = False
        if GlobalState.cap:
            GlobalState.cap.release()
            GlobalState.cap = None
        sio.emit('pi_response', {'status': 'ok', 'message': '摄像头已关闭'})
        sio.emit('pi_status_update', {'camera_running': False})

    elif cmd == 'enable_detection':
        GlobalState.detection_enabled = True
        sio.emit('pi_response', {'status': 'ok', 'message': '检测已开启'})
        sio.emit('pi_status_update', {'detection_enabled': True})

    elif cmd == 'disable_detection':
        GlobalState.detection_enabled = False
        sio.emit('pi_response', {'status': 'ok', 'message': '检测已停止'})
        sio.emit('pi_status_update', {'detection_enabled': False})

    elif cmd == 'enable_alert':
        GlobalState.alert_enabled = True
        sio.emit('pi_response', {'status': 'ok', 'message': '警报已开启'})
        sio.emit('pi_status_update', {'alert_enabled': True})

    elif cmd == 'disable_alert':
        GlobalState.alert_enabled = False
        sio.emit('pi_response', {'status': 'ok', 'message': '警报已静音'})
        sio.emit('pi_status_update', {'alert_enabled': False})

    elif cmd == 'set_video_duration':
        VIDEO_DURATION = params.get('duration', VIDEO_DURATION)
        sio.emit('pi_response', {'status': 'ok', 'message': f'录制时长已设为 {VIDEO_DURATION} 秒'})
        sio.emit('pi_status_update', {'video_duration': VIDEO_DURATION})

    elif cmd == 'set_username':
        # 接收用户名（英文）
        new_username = params.get('username', 'unknown')
        if re.match(r'^[a-zA-Z0-9_]{1,50}$', new_username):
            old = GlobalState.username
            GlobalState.username = new_username
            print(f"👤 用户名已更新: {old} -> {new_username}")
            sio.emit('pi_response', {'status': 'ok', 'message': f'用户名已设为 {new_username}'})
            sio.emit('pi_status_update', {'username': new_username})
        else:
            print(f"⚠️ 无效的用户名: {new_username}，仅允许英文、数字、下划线，长度1-50")
            sio.emit('pi_response', {'status': 'error', 'message': '用户名格式无效'})

    elif cmd == 'get_status':
        status = {
            'recording_enabled': GlobalState.recording_enabled,
            'detection_enabled': GlobalState.detection_enabled,
            'alert_enabled': GlobalState.alert_enabled,
            'video_duration': VIDEO_DURATION,
            'current_alert': "NORMAL",
            'video_count': video_upload_count,
            'username': GlobalState.username,
            'camera_running': GlobalState.camera_running
        }
        sio.emit('pi_response', {'status': 'ok', 'data': status})

    else:
        sio.emit('pi_response', {'status': 'error', 'message': f'未知命令: {cmd}'})


def connect_to_server():
    while True:
        try:
            sio.connect(WEBSOCKET_URL)
            sio.wait()
        except Exception as e:
            print(f"⚠️ 连接服务器失败: {e}，5秒后重试...")
            time.sleep(5)


# ================== 主程序 ==================
def main():
    print("=" * 60)
    print("🚀 树莓派客户端启动（自动录制模式）")
    print("📋 工作流程:")
    print("  1. 连接服务器")
    print("  2. 自动启动摄像头")
    print("  3. 开始实时识别和录制视频")
    print("  4. 根据检测结果发出警报")
    print("  5. 持续上传录制的视频")
    print(f"📡 服务器地址: {SERVER_URL}")
    print(f"🎥 录制时长: {VIDEO_DURATION}秒, 分辨率: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
    print(f"👤 当前用户名: {GlobalState.username} (可通过命令 set_username 修改)")
    print("=" * 60)
    print("💡 连接服务器后将自动启动摄像头和录制...\n")

    # 启动 WebSocket 连接线程
    ws_thread = threading.Thread(target=connect_to_server, daemon=True)
    ws_thread.start()

    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
        if GlobalState.camera_running:
            stop_camera()
        if sio.connected:
            sio.disconnect()
        print("✅ 系统停止")


if __name__ == "__main__":
    try:
        import cv2, torch, pygame, requests, PIL, socketio
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install opencv-python torch torchvision pillow pygame requests numpy python-socketio")
        sys.exit(1)

    main()