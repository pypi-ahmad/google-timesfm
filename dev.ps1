param(
  [ValidateSet('setup', 'start', 'stop', 'doctor', 'dev', 'launch')]
  [string]$Command = 'start'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if ($Command -eq 'launch') {
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
