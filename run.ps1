# Cyber Threat Detection System - PowerShell Runner
Write-Host "🚀 CYBER THREAT DETECTION SYSTEM" -ForegroundColor Cyan
Write-Host "=" * 60

# Check if in correct directory
if (-not (Test-Path "src\detect_threats.py")) {
    Write-Host "❌ ERROR: detect_threats.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory" -ForegroundColor Yellow
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Gray
    pause
    exit 1
}

# Check Python
$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    $python = "python3"
    if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
        Write-Host "❌ Python not found!" -ForegroundColor Red
        Write-Host "Please install Python 3.8+ from python.org" -ForegroundColor Yellow
        pause
        exit 1
    }
}

# Show menu
Write-Host "`n📋 SELECT MODE:" -ForegroundColor Green
Write-Host "1. Real-time simulation (30 seconds)" -ForegroundColor White
Write-Host "2. Analyze existing data" -ForegroundColor White
Write-Host "3. Generate and analyze sample data" -ForegroundColor White
Write-Host "4. Quick test" -ForegroundColor White
Write-Host "5. Run training first" -ForegroundColor White
Write-Host "6. Exit" -ForegroundColor White

$choice = Read-Host "`nEnter choice (1-6)"

switch ($choice) {
    "1" {
        Write-Host "`nStarting real-time simulation..." -ForegroundColor Yellow
        & $python src\detect_threats.py
    }
    "2" {
        Write-Host "`nAnalyzing existing data..." -ForegroundColor Yellow
        & $python src\detect_threats.py --mode analyze
    }
    "3" {
        Write-Host "`nGenerating sample data..." -ForegroundColor Yellow
        & $python src\detect_threats.py --mode sample
    }
    "4" {
        Write-Host "`nRunning quick test..." -ForegroundColor Yellow
        & $python -c "
import sys
sys.path.append('src')
try:
    import detect_threats
    print('✅ Module loaded successfully')
    detect_threats.simulate_real_time_detection(10)
except Exception as e:
    print(f'❌ Error: {e}')
"
    }
    "5" {
        Write-Host "`nRunning training..." -ForegroundColor Yellow
        if (Test-Path "src\train_model.py") {
            & $python src\train_model.py
            Write-Host "`n✅ Training complete! Now run detection..." -ForegroundColor Green
            & $python src\detect_threats.py
        } else {
            Write-Host "❌ train_model.py not found!" -ForegroundColor Red
        }
    }
    default {
        Write-Host "`n👋 Goodbye!" -ForegroundColor Cyan
    }
}

Write-Host "`n" + "=" * 60
Write-Host "✅ Execution completed" -ForegroundColor Green
if ($choice -in @("1","2","3","4","5")) {
    Write-Host "Check 'reports\' directory for output files" -ForegroundColor Yellow
}
pause