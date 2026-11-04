@echo off
chcp 65001 >nul
setlocal
REM 一键安装：虚拟环境 + 依赖 + 配置 + 配置助手 + 环境自检
REM 双击运行，或在项目根目录执行 deploy\setup_windows.bat
cd /d "%~dp0.."

where python >nul 2>nul
if errorlevel 1 (
  echo [失败] 找不到 Python。请安装 Python 3.10 或更新版本，安装时勾选 "Add python.exe to PATH"。
  goto :end
)

if not exist .venv\Scripts\python.exe (
  echo [1/5] 创建虚拟环境 .venv
  python -m venv .venv
  if errorlevel 1 goto :fail
) else (
  echo [1/5] 虚拟环境已存在
)
call .venv\Scripts\activate.bat

echo [2/5] 安装依赖（含 Mem0，首次需要几分钟）
python -m pip install --upgrade pip >nul
pip install -e ".[dev,mem0]"
if errorlevel 1 goto :fail

echo [3/5] 初始化配置与数据库
python -m her_brain init
if errorlevel 1 goto :fail

echo [4/5] 配置助手
python -m her_brain setup
if errorlevel 1 goto :fail

echo [5/5] 环境自检
python -m her_brain doctor

echo.
echo 安装完成。以后每次启动：双击 deploy\up.bat（或激活 .venv 后运行 python -m her_brain up）
echo 启动后可另开窗口运行 deploy\selftest.bat 做端到端自测。
goto :end

:fail
echo.
echo [失败] 上一步出错，请把上面的输出发给我。

:end
pause
endlocal
