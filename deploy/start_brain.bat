@echo off
chcp 65001 >nul
REM 大脑代理。请在项目根目录运行：deploy\start_brain.bat
cd /d %~dp0..
python -m her_brain serve
