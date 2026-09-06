@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

where uv >nul 2>&1
if errorlevel 1 (
    echo Error: Install uv and reopen this launcher.
    goto failed
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev.ps1" launch
if errorlevel 1 goto failed

popd
exit /b 0

:failed
echo.
echo Workbench could not start. See the error above and .native\logs.
pause
popd
exit /b 1
