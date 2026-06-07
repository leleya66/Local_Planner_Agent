param(
    [ValidateSet("single", "collab")]
    [string]$Mode = "single",

    [int]$Port = 0,

    [string]$CondaEnv = "localmate_agent4",

    [switch]$Restart,

    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-LocalMatePython {
    param([string]$PythonPath)
    if (-not $PythonPath -or -not (Test-Path $PythonPath)) {
        return $false
    }
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        & $PythonPath -c "import fastapi, uvicorn" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previousErrorPreference
    }
}

function Find-LocalMatePython {
    $candidates = @()

    if ($env:CONDA_PREFIX) {
        $candidates += Join-Path $env:CONDA_PREFIX "python.exe"
    }

    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        try {
            $condaInfo = & $conda.Source env list 2>$null
            foreach ($line in $condaInfo) {
                if ($line -match "^\s*$([regex]::Escape($CondaEnv))\s+(.+)$") {
                    $candidates += Join-Path $matches[1].Trim() "python.exe"
                }
            }
        } catch {
            # Fall back to the explicit and PATH-based candidates below.
        }
    }

    if ($env:USERPROFILE) {
        $candidates += Join-Path $env:USERPROFILE ".conda\envs\$CondaEnv\python.exe"
    }

    try {
        $candidates += (Get-Command python -ErrorAction Stop).Source
    } catch {
        # Python is not on PATH.
    }

    return $candidates |
        Where-Object { $_ } |
        Select-Object -Unique |
        Where-Object { Test-LocalMatePython $_ } |
        Select-Object -First 1
}

function Stop-PortListener {
    param([int]$TargetPort)

    $listeners = Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue
    if (-not $listeners) {
        return
    }

    $pids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $pids) {
        Write-Host "Stopping existing listener on port $TargetPort (PID $processId) ..." -ForegroundColor Yellow
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
}

if ($Port -le 0) {
    if ($Mode -eq "collab") {
        $Port = 8042
    } else {
        $Port = 8041
    }
}

$env:PYTHONIOENCODING = "utf-8"
$env:ENABLE_LLM_STOP_NARRATIVES = "0"
$env:ENABLE_LLM_ROUTE_NARRATIVES = "1"
$env:LOCALMATE_PORT = "$Port"

if ($Restart) {
    Stop-PortListener -TargetPort $Port
    if ($Mode -eq "collab") {
        Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force
    }
}

if ($Mode -eq "collab") {
    Write-Host "Starting LocalMate collaboration demo on port $Port ..." -ForegroundColor Cyan
    Write-Host "A public ngrok URL will be printed after startup." -ForegroundColor Cyan
    & "$PSScriptRoot\start_ngrok_collaboration.ps1" -Port $Port
    exit $LASTEXITCODE
}

$pythonPath = Find-LocalMatePython
if (-not $pythonPath) {
    throw "No Python interpreter with FastAPI and Uvicorn was found. Create the environment with 'conda env create -f environment.yml' and retry."
}

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listeners) {
    Write-Host "LocalMate is already running on port $Port." -ForegroundColor Green
    $listeners | Select-Object LocalAddress, LocalPort, OwningProcess | Format-Table -AutoSize
    Write-Host "Open: http://127.0.0.1:$Port/v8" -ForegroundColor Green
    if (-not $NoBrowser) {
        Start-Process "http://127.0.0.1:$Port/v8"
    }
    return
}

Write-Host "Starting LocalMate single-user demo on port $Port ..." -ForegroundColor Cyan
Write-Host "Python: $pythonPath"
Write-Host "Open:   http://127.0.0.1:$Port/v8" -ForegroundColor Green

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port/v8"
}

& $pythonPath -m uvicorn app_api:app --host 127.0.0.1 --port $Port
