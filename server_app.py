from flask import Flask, request, jsonify, render_template
import os
from datetime import datetime

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploaded_videos'
TEMPLATES_FOLDER = 'templates'

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)

# 存储上传的视频信息
uploaded_videos = []

# 存储命令
current_command = None

@app.route('/')
def index():
    """主页"""
    return render_template('server_index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """接收树莓派上传的视频"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 生成唯一文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"raspberry_{timestamp}.mp4"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # 保存视频
    video.save(filepath)
    
    # 记录视频信息
    video_info = {
        'filename': filename,
        'path': filepath,
        'size': os.path.getsize(filepath),
        'upload_time': datetime.now().isoformat()
    }
    uploaded_videos.append(video_info)
    
    print(f"✅ 接收到视频: {filename} ({video_info['size']} bytes)")
    
    return jsonify({'success': True, 'filename': filename})

@app.route('/get_videos')
def get_videos():
    """获取已上传的视频列表"""
    return jsonify({'videos': uploaded_videos})

@app.route('/get_command')
def get_command():
    """获取发送给树莓派的命令"""
    global current_command
    command = current_command
    current_command = None  # 命令获取后清除
    return jsonify({'command': command})

@app.route('/command_raspberry', methods=['POST'])
def command_raspberry():
    """向树莓派发送命令"""
    global current_command
    data = request.json
    command = data.get('command')
    
    current_command = command
    print(f"📢 发送命令到树莓派: {command}")
    
    return jsonify({'success': True, 'command': command})

# 创建服务器网页模板
def create_server_template():
    """创建服务器网页模板"""
    server_index_html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>服务器控制中心</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            margin: 0 10px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            transition: background-color 0.3s;
            border: none;
            cursor: pointer;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .btn.stop {
            background-color: #f44336;
        }
        .btn.stop:hover {
            background-color: #da190b;
        }
        .video-list {
            margin-top: 20px;
        }
        .video-item {
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .video-item:hover {
            background-color: #f9f9f9;
        }
        .status {
            padding: 10px;
            background-color: #e8f5e8;
            border-radius: 4px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>服务器控制中心</h1>
        
        <div class="card">
            <h2>控制命令</h2>
            <div class="status">
                <p>状态: <span id="status">就绪</span></p>
            </div>
            <button class="btn" onclick="sendCommand('start')">开始拍摄</button>
            <button class="btn stop" onclick="sendCommand('stop')">停止拍摄</button>
            <button class="btn" onclick="refreshVideos()">刷新视频列表</button>
        </div>
        
        <div class="card">
            <h2>已上传视频</h2>
            <div class="video-list" id="videoList">
                <!-- 视频列表将通过JavaScript动态添加 -->
            </div>
        </div>
    </div>
    
    <script>
        // 发送命令到树莓派
        function sendCommand(command) {
            fetch('/command_raspberry', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({command: command})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('status').textContent = `命令已发送: ${command}`;
                }
            });
        }
        
        // 刷新视频列表
        function refreshVideos() {
            fetch('/get_videos')
                .then(response => response.json())
                .then(data => {
                    const videoList = document.getElementById('videoList');
                    videoList.innerHTML = '';
                    
                    data.videos.forEach(video => {
                        const item = document.createElement('div');
                        item.className = 'video-item';
                        item.innerHTML = `
                            <strong>${video.filename}</strong><br>
                            大小: ${(video.size / (1024 * 1024)).toFixed(2)} MB<br>
                            上传时间: ${new Date(video.upload_time).toLocaleString()}
                        `;
                        videoList.appendChild(item);
                    });
                });
        }
        
        // 页面加载时刷新视频列表
        window.onload = function() {
            refreshVideos();
        };
    </script>
</body>
</html>
    '''
    
    with open(os.path.join(TEMPLATES_FOLDER, 'server_index.html'), 'w', encoding='utf-8') as f:
        f.write(server_index_html)

if __name__ == "__main__":
    # 创建服务器网页模板
    create_server_template()
    print("🚀 启动服务器...")
    print("📡 服务地址: http://localhost:8000")
    print("🔧 功能: 接收树莓派视频上传 + 发送控制命令")
    app.run(host='0.0.0.0', port=8000, debug=False)