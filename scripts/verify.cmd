@echo off
setlocal
pushd "%~dp0\.."

python -m unittest discover -s tests -v
if errorlevel 1 goto failure

python raypath_scpt.py --self-test
if errorlevel 1 goto failure

python -m compileall -q raypath_core raypath_reporting.py raypath_scpt.py tests
if errorlevel 1 goto failure

popd
endlocal
exit /b 0

:failure
set "RAYPATH_VERIFY_EXIT=%errorlevel%"
popd
echo RayPath SCPT verification failed.
endlocal & exit /b %RAYPATH_VERIFY_EXIT%
