# Dev launcher for the native TimesFM-3 workbench (FastAPI backend +
# Postgres/Memurai/Next.js frontend under .native\ and web\), driven through
# src/timesfm_app/native.py for every command except 'launch' and 'setup',
# which this script implements directly.
#
# Commands:
#   setup   - install/build Python (incl. CUDA torch), native services, and
#             the Next.js frontend (web\). Run once, or whenever dependencies
#             change; 'launch' calls this automatically on first run.
#   start   - start the native services (Postgres, Memurai, API) via
#             timesfm_app.native, without waiting for readiness or opening a
#             browser.
#   stop    - stop the native services via timesfm_app.native.
#   doctor  - health-check the native services via timesfm_app.native.
#   dev     - start timesfm_app.native in dev mode (see --dev there for what
#             that changes).
#   launch  - the one-click path used by launch_workbench.cmd: ensures setup
#             has run, starts services, polls doctor until ready (90s
#             timeout), opens the browser to the frontend port, then tails
#             .native\logs until Ctrl+C (services keep running after that).
#
# Prerequisites this script assumes but does not verify up front: `uv` on
# PATH, and (for 'setup') `npm` on PATH for the web\ build. The frontend
# port defaults to 3000 and is overridable via $env:TIMESFM_FRONTEND_PORT;
# the API port is owned by timesfm_app.native/config.py, not this script.
param(
  [ValidateSet('setup', 'start', 'stop', 'doctor', 'dev', 'launch')]
  [string]$Command = 'start'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if ($Command -eq 'launch') {
  # Presence check, not a version/integrity check: any one of these missing
  # is treated as "never set up" and triggers a full 'setup' run below.
  $requiredPaths = @(
    '.venv\Scripts\python.exe',
    '.native\postgresql\pgsql\bin\pg_ctl.exe',
    '.native\postgres-data\PG_VERSION',
    '.native\memurai\memurai.exe',
    'web\node_modules\next\package.json',
    'web\.next\BUILD_ID'
  )
  $needsSetup = $false
  foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { $needsSetup = $true }
  }
  if ($needsSetup) {
    Write-Host 'Preparing the workbench for its first launch...'
    & $PSCommandPath setup
  }
  # Follow each service separately so one quiet log cannot block the others.
  $logOffsets = @{}
  Get-ChildItem -LiteralPath '.native\logs' -Filter '*.log' -ErrorAction SilentlyContinue | ForEach-Object {
    $logOffsets[$_.FullName] = $_.Length
  }
  function Show-WorkbenchLogs {
    Get-ChildItem -LiteralPath '.native\logs' -Filter '*.log' -ErrorAction SilentlyContinue | ForEach-Object {
      $logFile = $_
      $offset = if ($logOffsets.ContainsKey($logFile.FullName)) { $logOffsets[$logFile.FullName] } else { 0 }
      $stream = $null
      $reader = $null
      try {
        $stream = [IO.File]::Open($logFile.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
        if ($stream.Length -lt $offset) { $offset = 0 }
        [void]$stream.Seek($offset, [IO.SeekOrigin]::Begin)
        $reader = [IO.StreamReader]::new($stream)
        $newText = $reader.ReadToEnd()
        $logOffsets[$logFile.FullName] = $stream.Position
        foreach ($line in ($newText -split '\r?\n')) {
          if ($line.Length) { Write-Host "[$($logFile.BaseName)] $line" }
        }
      } catch [IO.IOException] {
        # A service can recreate its log during restart; retry on the next pass.
      } finally {
        if ($reader) { $reader.Dispose() }
        elseif ($stream) { $stream.Dispose() }
      }
    }
  }
  & $PSCommandPath start
  Write-Host 'Waiting for the workbench...'
  $ready = $false
  $deadline = (Get-Date).AddSeconds(90)
  while ((Get-Date) -lt $deadline) {
    uv run --no-sync python -m timesfm_app.native doctor *> $null
    $healthExit = $LASTEXITCODE
    Show-WorkbenchLogs
    if ($healthExit -eq 0) {
      $ready = $true
      break
    }
    Start-Sleep -Seconds 1
  }
  if (-not $ready) {
    throw 'Startup timed out. Run .\dev.ps1 doctor and inspect .native\logs.'
  }
  $workbenchPort = if ($env:TIMESFM_FRONTEND_PORT) { $env:TIMESFM_FRONTEND_PORT } else { '3000' }
  $workbenchUrl = "http://127.0.0.1:$workbenchPort"
  Start-Process $workbenchUrl
  Write-Host "Workbench ready: $workbenchUrl"
  Write-Host 'Live service logs below. Ctrl+C closes this log view; services keep running.'
  Write-Host 'To stop the app, run: .\dev.ps1 stop'
  while ($true) {
    Show-WorkbenchLogs
    Start-Sleep -Milliseconds 500
  }
} elseif ($Command -eq 'setup') {
  # Preserve a working CUDA build; uv's ordinary Windows index resolves CPU Torch.
  uv sync --extra server --extra app --group dev --inexact --no-install-package torch
  if ($LASTEXITCODE -ne 0) { throw 'Python dependency setup failed.' }
  # If uv's --no-install-package left no CUDA-capable torch installed (or
  # skipped it entirely), re-resolve torch with the backend auto-detected
  # for this machine instead of leaving a CPU-only install in place.
  uv run --no-sync python -c 'import torch; assert torch.cuda.is_available()'
  if ($LASTEXITCODE -ne 0) {
    uv pip install torch --torch-backend=auto
    if ($LASTEXITCODE -ne 0) { throw 'Torch installation failed.' }
  }
  uv run --no-sync python -m timesfm_app.native setup
  if ($LASTEXITCODE -ne 0) { throw 'Native service setup failed.' }
  npm --prefix web ci
  if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency setup failed.' }
  npm --prefix web run build
  if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} elseif ($Command -eq 'dev') {
  uv run --no-sync python -m timesfm_app.native start --dev
} else {
  uv run --no-sync python -m timesfm_app.native $Command
}
if ($LASTEXITCODE -ne 0) { throw "Workbench command failed: $Command" }
