@echo off
echo ========================================
echo   树莓派视频监控系统 - 启动器
echo ========================================
echo.

echo [1/2] 检查依赖...
pip install flask opencv-python requests

echo.
echo [2/2] 正在启动 Flask 服务器...
echo.
python server_app.py

pause
