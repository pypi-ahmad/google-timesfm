@echo off
setlocal
pushd "%~dp0"
set "TIMESFM_APP=%~dp0streamlit_app.py"

where uv >nul 2>&1
if errorlevel 1 (
    echo Error: uv is not installed or not available on PATH.
    popd
    exit /b 1
)

powershell.exe -NoProfile -Command ^
    "$app = [IO.Path]::GetFullPath($env:TIMESFM_APP); $pids = @(Get-NetTCPConnection -LocalPort 9587 -State Listen -ErrorAction SilentlyContinue ^| Select-Object -ExpandProperty OwningProcess -Unique); foreach ($pidValue in $pids) { $process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $pidValue); $command = [string]$process.CommandLine; $isApp = $command.IndexOf($app, [StringComparison]::OrdinalIgnoreCase) -ge 0; $usesPort = $command -match '(?i)--server\.port(?:=|\s+)9587(?:\s|$)'; if (-not ($isApp -and $usesPort)) { Write-Error ('Port 9587 is owned by another process (PID ' + $pidValue + '). It was not stopped.'); exit 2 }; Stop-Process -Id $pidValue -Force -ErrorAction Stop }"
if errorlevel 1 (
    echo Error: could not clear port 9587.
    popd
    exit /b 1
)

uv run streamlit run "%TIMESFM_APP%" --server.port=9587
set "app_exit=%errorlevel%"

popd
exit /b %app_exit%
