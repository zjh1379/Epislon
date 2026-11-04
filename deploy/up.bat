@echo off
chcp 65001 >nul
REM 一键启动：对话模型 + 嵌入模型 + 大脑代理。按 Ctrl+C 退出时一并关闭。
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m her_brain up
pause
