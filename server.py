# server.py - Flask 服务器（Vue3 项目支持）
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
import json
from datetime import datetime
from pathlib import Path
import sqlite3

# 获取当前文件所在目录
BASE_DIR = Path(__file__).parent.absolute()

app = Flask(__name__, static_folder=str(BASE_DIR / 'dist'), static_url_path='')

# 配置 Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', manage_session=False)

# 配置
UPLOAD_FOLDER = BASE_DIR / 'uploaded_videos'

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 数据库初始化
def init_db():
    db_path = BASE_DIR / 'users.db'
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()


# 初始化数据库
init_db()

# 存储上传的视频信息
uploaded_videos = []
command_queue = []

# 在线的树莓派客户端
pi_clients = set()
# 在线的 Web 客户端
web_clients = set()


# ================== API 接口 ==================

@app.route('/api/register', methods=['POST'])
def register():
    """用户注册"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': '用户名和密码不能为空'}), 400

        db_path = BASE_DIR / 'users.db'
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()

        # 检查用户名是否已存在
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        if c.fetchone():
            conn.close()
            return jsonify({'error': '用户名已存在'}), 400

        # 插入新用户
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': '注册成功'}), 200

    except Exception as e:
        print(f"注册错误：{e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': '用户名和密码不能为空'}), 400

        db_path = BASE_DIR / 'users.db'
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()

        # 检查用户是否存在
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            return jsonify({'success': True, 'username': username}), 200
        else:
            return jsonify({'error': '用户名或密码错误'}), 401

    except Exception as e:
        print(f"登录错误：{e}")
        return jsonify({'error': str(e)}), 500


# ================== 视频管理接口 ==================

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """接收树莓派上传的视频（支持用户名）"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400

        video = request.files['video']
        if video.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 获取用户名（从表单数据或查询参数）
        username = request.form.get('username', request.args.get('username', 'unknown'))

        # 生成包含用户名的新文件名
        original_name = video.filename
        name, ext = os.path.splitext(original_name)

        # 如果文件名中已有用户名，先移除
        if '_' in name:
            parts = name.split('_')
            if len(parts) > 1 and parts[0] not in ['video', 'recording', 'pi']:
                name = '_'.join(parts[1:])

        # 生成新文件名格式：{username}_{original_filename}_{timestamp}.ext
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"{username}_{name}_{timestamp}{ext}"

        filepath = os.path.join(UPLOAD_FOLDER, new_filename)

        # 如果文件已存在，添加唯一标识
        if os.path.exists(filepath):
            unique_id = datetime.now().strftime('%H%M%S%f')
            new_filename = f"{username}_{name}_{unique_id}{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, new_filename)

        video.save(filepath)

        # 记录视频信息
        video_info = {
            'filename': new_filename,
            'original_filename': original_name,
            'path': str(filepath),
            'size': os.path.getsize(filepath),
            'upload_time': datetime.now().isoformat(),
            'username': username,
            'custom_name': new_filename  # 可用于后续自定义改名
        }
        uploaded_videos.insert(0, video_info)

        # 只保留最近 100 个
        if len(uploaded_videos) > 100:
            old_video = uploaded_videos.pop()
            try:
                os.remove(old_video['path'])
            except:
                pass

        print(f"✅ 接收到视频：{new_filename} (用户：{username}, 大小：{video_info['size'] / (1024 * 1024):.2f} MB)")

        return jsonify({
            'success': True,
            'filename': new_filename,
            'username': username
        })

    except Exception as e:
        print(f"❌ 上传错误：{e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_videos')
def get_videos():
    """获取已上传的视频列表"""
    return jsonify({'videos': uploaded_videos})


@app.route('/command_raspberry', methods=['POST'])
def command_raspberry():
    """接收命令"""
    data = request.json
    command = data.get('command')
    command_queue.append(command)
    print(f"📢 收到命令：{command}")
    return jsonify({'success': True, 'command': command})


@app.route('/get_command', methods=['GET'])
def get_command():
    """树莓派获取命令"""
    if command_queue:
        command = command_queue.pop(0)
        return jsonify({'command': command})
    return jsonify({'command': None})


@app.route('/play/<filename>')
def play_video(filename):
    """播放视频"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/download/<filename>')
def download_video(filename):
    """下载视频"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/rename_video', methods=['POST'])
def rename_video():
    """重命名视频"""
    try:
        data = request.json
        old_filename = data.get('old_filename')
        new_filename = data.get('new_filename')
        username = data.get('username', 'unknown')

        if not old_filename or not new_filename:
            return jsonify({'error': 'Missing filename'}), 400

        old_path = os.path.join(UPLOAD_FOLDER, old_filename)

        if not os.path.exists(old_path):
            return jsonify({'error': 'File not found'}), 404

        # 确保新文件名有正确的扩展名
        if not new_filename.endswith('.mp4'):
            name, _ = os.path.splitext(old_filename)
            _, ext = os.path.splitext(new_filename)
            if not ext:
                new_filename = f"{new_filename}.mp4"

        new_path = os.path.join(UPLOAD_FOLDER, new_filename)

        # 如果新文件名已存在，添加时间戳
        if os.path.exists(new_path):
            name, ext = os.path.splitext(new_filename)
            timestamp = datetime.now().strftime('%H%M%S')
            new_filename = f"{name}_{timestamp}{ext}"
            new_path = os.path.join(UPLOAD_FOLDER, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)

        # 更新视频信息
        for video in uploaded_videos:
            if video['filename'] == old_filename:
                video['filename'] = new_filename
                video['path'] = new_path
                video['custom_name'] = new_filename
                video['renamed_by'] = username
                video['rename_time'] = datetime.now().isoformat()
                break

        print(f"✏️ 视频重命名：{old_filename} -> {new_filename}")

        return jsonify({
            'success': True,
            'old_filename': old_filename,
            'new_filename': new_filename
        })

    except Exception as e:
        print(f"❌ 重命名错误：{e}")
        return jsonify({'error': str(e)}), 500


# ================== Socket.IO 事件处理 ==================

@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    print(f"✅ Socket.IO 连接：{request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开"""
    print(f"❌ Socket.IO 断开：{request.sid}")
    # 从集合中移除
    pi_clients.discard(request.sid)
    web_clients.discard(request.sid)
    # 通知其他客户端
    if not pi_clients:
        # 所有树莓派都离线了
        emit('pi_status_update', {'online': False}, broadcast=True)


@socketio.on('pi_identify')
def handle_pi_identify(data):
    """树莓派客户端标识自己"""
    print(f"🥧 树莓派客户端连接：{request.sid}")
    pi_clients.add(request.sid)
    # 通知所有 Web 客户端
    emit('pi_status_update', {'online': True, 'pi_count': len(pi_clients)}, broadcast=True)


@socketio.on('web_identify')
def handle_web_identify(data):
    """Web 客户端标识自己"""
    print(f"🌐 Web 客户端连接：{request.sid}")
    web_clients.add(request.sid)
    # 发送当前树莓派状态
    emit('pi_status_update', {'online': len(pi_clients) > 0}, to=request.sid)


@socketio.on('pi_status_update')
def handle_pi_status(data):
    """接收树莓派状态更新"""
    print(f"📊 收到树莓派状态：{data}")
    pi_clients.add(request.sid)
    # 转发给所有 Web 客户端
    emit('pi_status_update', {
        'online': True,
        'pi_count': len(pi_clients),
        'status': data.get('status', 'online'),
        'recording': data.get('recording', False)
    }, broadcast=True)


@socketio.on('web_command')
def handle_web_command(data):
    """接收 Web 客户端的命令并转发给树莓派"""
    command = data.get('command')
    params = data.get('params', {})
    print(f"🎮 Web 发送命令：{command}，参数：{params}")
    # 转发给所有树莓派客户端
    if pi_clients:
        for sid in pi_clients:
            emit('server_command', {'command': command, 'params': params}, to=sid)
    else:
        print("⚠️  没有树莓派客户端在线")


@socketio.on('pi_response')
def handle_pi_response(data):
    """接收树莓派响应并转发给 Web 客户端"""
    print(f"📤 收到树莓派响应：{data}")
    # 转发给所有 Web 客户端
    if web_clients:
        for sid in web_clients:
            emit('web_response', {
                'status': 'success',
                'message': data.get('status', 'ok'),
                'data': data
            }, to=sid)
    else:
        print("⚠️  没有 Web 客户端在线")


# ================== 静态文件服务 ==================

@app.route('/')
def index():
    """主页 - 返回 Vue 应用"""
    return send_from_directory(BASE_DIR / 'dist', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """服务静态文件"""
    try:
        return send_from_directory(BASE_DIR / 'dist', filename)
    except Exception as e:
        return jsonify({'error': f'File not found: {filename}'}), 404


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 树莓派视频服务器 (Vue3 + Socket.IO)")
    print("=" * 60)
    print(f"📁 视频保存路径：{os.path.abspath(UPLOAD_FOLDER)}")
    print(f"🌐 Web 界面：http://localhost:8000")
    print(f"📡 上传接口：http://你的 IP:8000/upload_video")
    print(f"🔌 Socket.IO: ws://你的 IP:8000/socket.io/")
    print("=" * 60)
    print("使用前请先运行：npm install && npm run build")
    print("=" * 60)

    # 使用 socketio.run 启动服务器（支持 WebSocket）
    socketio.run(app, host='0.0.0.0', port=8000, debug=False, allow_unsafe_werkzeug=True)
