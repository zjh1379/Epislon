@echo off
chcp 65001 >nul
REM 一键（重新）启动：先停掉已在运行的大脑代理，再启动对话模型 + 嵌入模型 + 大脑代理。
REM 按 Ctrl+C 退出时一并关闭。git pull 之后直接双击本文件即可加载新代码。
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m her_brain stop
python -m her_brain up
pause
