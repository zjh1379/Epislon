@echo off
chcp 65001 >nul
REM 端到端自测：需要先用 deploy\up.bat 启动模型服务。使用临时数据，不影响正式记忆。
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m her_brain doctor
echo.
echo ===== 端到端自测 =====
python -m her_brain selftest
pause
