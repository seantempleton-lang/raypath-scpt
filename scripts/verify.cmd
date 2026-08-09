@echo off
setlocal
pushd "%~dp0\.."

python -m unittest discover -s tests -v
if errorlevel 1 goto failure

python raypath_scpt.py --self-test
if errorlevel 1 goto failure

python -m py_compile raypath_scpt.py tests\test_core.py tests\test_project_state.py
if errorlevel 1 goto failure

popd
endlocal
exit /b 0

:failure
set "RAYPATH_VERIFY_EXIT=%errorlevel%"
popd
echo RayPath SCPT verification failed.
endlocal & exit /b %RAYPATH_VERIFY_EXIT%
