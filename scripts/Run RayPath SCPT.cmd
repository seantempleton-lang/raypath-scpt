@echo off
setlocal
pushd "%~dp0"

set "RAYPATH_LAUNCH_PYTHON="

if defined RAYPATH_SCPT_PYTHON call :probe_python "%RAYPATH_SCPT_PYTHON%"

for /f "delims=" %%P in ('where python.exe 2^>nul') do if not defined RAYPATH_LAUNCH_PYTHON call :probe_python "%%P"

if not defined RAYPATH_LAUNCH_PYTHON call :probe_python "C:\ProgramData\anaconda3\python.exe"

if not defined RAYPATH_LAUNCH_PYTHON (
    echo RayPath SCPT could not find a Python interpreter.
    echo Set RAYPATH_SCPT_PYTHON to the full path of a Python 3.12 python.exe.
    pause
    popd
    endlocal
    exit /b 1
)

"%RAYPATH_LAUNCH_PYTHON%" "%~dp0raypath_scpt.py" %*
set "RAYPATH_LAUNCH_EXIT=%errorlevel%"

if not "%RAYPATH_LAUNCH_EXIT%"=="0" (
    echo.
    echo RayPath SCPT exited with an error. Review the message above.
    pause
)

popd
endlocal & exit /b %RAYPATH_LAUNCH_EXIT%

:probe_python
if defined RAYPATH_LAUNCH_PYTHON exit /b 0
if "%~1"=="" exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import numpy, scipy, matplotlib, PySide6, reportlab" >nul 2>&1
if not errorlevel 1 set "RAYPATH_LAUNCH_PYTHON=%~1"
exit /b 0
