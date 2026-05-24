<#
.SYNOPSIS
    One-click setup for ChessTransformer.
    Run this script from inside the ChessTransformer folder.
    It creates an isolated .venv, installs all dependencies, and launches the app.

.NOTES
    Requirements:
      - Windows 10/11
      - Python 3.12 (must be on PATH)
      - NVIDIA GPU with CUDA support (RTX 20-series or newer)
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ChessTransformer - Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 0: Verify model files are present (not Git LFS stubs) ──
$modelDir = Join-Path $root "models"
$requiredModels = @("SSChess_12M.pt", "SSChess_78M_BF16.pt")
foreach ($model in $requiredModels) {
    $modelPath = Join-Path $modelDir $model
    if (-not (Test-Path $modelPath)) {
        Write-Host "[ERROR] Missing model file: $model" -ForegroundColor Red
        Write-Host "        The models/ folder must contain the actual .pt weight files." -ForegroundColor Red
        exit 1
    }
    $size = (Get-Item $modelPath).Length
    if ($size -lt 1MB) {
        Write-Host "[ERROR] $model appears to be a Git LFS pointer ($size bytes) instead of the real model." -ForegroundColor Red
        Write-Host "        Make sure you have the actual model weights, not LFS stubs." -ForegroundColor Red
        exit 1
    }
}
Write-Host "[OK] Model files verified." -ForegroundColor Green

# ── Step 1: Verify native_mcts.exe is present ──
$exePath = Join-Path $root "engines\native_mcts.exe"
if (-not (Test-Path $exePath)) {
    Write-Host "[ERROR] Missing engines\native_mcts.exe" -ForegroundColor Red
    Write-Host "        The Grandmaster and Impossible bots need this binary." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] native_mcts.exe found." -ForegroundColor Green

# ── Step 2: Find Python 3.12 ──
$pythonCmd = $null
foreach ($candidate in @("python3.12", "python3", "python", "py -3.12")) {
    try {
        $tokens = $candidate -split " "
        $ver = & $tokens[0] ($tokens[1..($tokens.Length)] + @("--version")) 2>&1
        if ($ver -match "3\.12") {
            $pythonCmd = $candidate
            break
        }
    } catch { }
}

if (-not $pythonCmd) {
    # Try the py launcher explicitly
    try {
        $ver = & py -3.12 --version 2>&1
        if ($ver -match "3\.12") { $pythonCmd = "py -3.12" }
    } catch { }
}

if (-not $pythonCmd) {
    Write-Host "[ERROR] Python 3.12 not found on PATH." -ForegroundColor Red
    Write-Host "        Install Python 3.12 from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "        Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Found Python: $pythonCmd" -ForegroundColor Green

# ── Step 3: Create venv (if missing or broken) ──
$venvDir = Join-Path $root ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$needsVenv = $true

if (Test-Path $venvPython) {
    try {
        $venvVer = & $venvPython --version 2>&1
        if ($venvVer -match "3\.12") {
            Write-Host "[OK] Existing .venv is valid." -ForegroundColor Green
            $needsVenv = $false
        } else {
            Write-Host "[WARN] Existing .venv uses wrong Python ($venvVer). Recreating..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $venvDir
        }
    } catch {
        Write-Host "[WARN] Existing .venv is broken. Recreating..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvDir
    }
}

if ($needsVenv) {
    Write-Host "[...] Creating virtual environment..." -ForegroundColor Yellow
    $tokens = $pythonCmd -split " "
    & $tokens[0] ($tokens[1..($tokens.Length)] + @("-m", "venv", $venvDir))
    if (-not (Test-Path $venvPython)) {
        Write-Host "[ERROR] Failed to create .venv" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

# ── Step 4: Install dependencies ──
Write-Host "[...] Installing dependencies (this may take a few minutes on first run)..." -ForegroundColor Yellow
$reqFile = Join-Path $root "requirements.txt"
& $venvPython -m pip install --upgrade pip --quiet 2>&1 | Out-Null
& $venvPython -m pip install -r $reqFile --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] pip install failed. Check your internet connection and try again." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] All dependencies installed." -ForegroundColor Green

# ── Step 5: Verify CUDA is available ──
Write-Host "[...] Checking CUDA availability..." -ForegroundColor Yellow
$cudaCheck = & $venvPython -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
$cudaLines = $cudaCheck -split "`n"
if ($cudaLines[0].Trim() -ne "True") {
    Write-Host "[ERROR] CUDA is not available. This app requires an NVIDIA GPU with CUDA drivers." -ForegroundColor Red
    Write-Host "        Install NVIDIA drivers from https://www.nvidia.com/drivers" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] CUDA available: $($cudaLines[1].Trim())" -ForegroundColor Green

# ── Done! ──
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup complete! Launching app..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Open your browser to: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""

& $venvPython (Join-Path $root "web_app.py")
