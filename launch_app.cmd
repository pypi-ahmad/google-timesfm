:: Launches the standalone Streamlit explorer (streamlit_app.py) on a fixed
:: port (9587). Requires `uv` on PATH; uv resolves/installs the Python
:: environment itself, so no separate setup step is needed here. Unlike
:: launch_workbench.cmd, this does not start any native services (Postgres,
:: Memurai, the FastAPI backend, or the Next.js frontend) -- the model loads
:: in-process inside Streamlit.
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

:: Before starting, free port 9587 if a previous instance of THIS app is
:: still holding it (e.g. a prior launch that wasn't stopped cleanly). The
:: script only kills a process on that port if its command line both
:: references this exact streamlit_app.py path and the same --server.port
:: value; any other process on port 9587 is left alone and aborts the launch
:: instead, so this never kills an unrelated service that happens to be
:: using the same port.
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
