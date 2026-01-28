@echo off
echo ========================================
echo    CYBER THREAT DETECTION SYSTEM
echo ========================================
echo.

REM Check if detect_threats.py exists
if not exist "src\detect_threats.py" (
    echo ERROR: detect_threats.py not found!
    echo Please run this script from project root
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

:menu
cls
echo ========================================
echo    SELECT OPTION
echo ========================================
echo.
echo 1. Real-time simulation (30 seconds)
echo 2. Analyze existing data
echo 3. Generate and analyze sample data
echo 4. Quick test
echo 5. Run training first
echo 6. Exit
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto realtime
if "%choice%"=="2" goto analyze
if "%choice%"=="3" goto sample
if "%choice%"=="4" goto test
if "%choice%"=="5" goto train
if "%choice%"=="6" goto exit
goto menu

:realtime
echo.
echo Starting real-time simulation...
python src\detect_threats.py
goto end

:analyze
echo.
echo Analyzing existing data...
python src\detect_threats.py
goto end

:sample
echo.
echo Generating sample data...
python src\detect_threats.py
goto end

:test
echo.
echo Running quick test...
python -c "import sys; sys.path.append('src'); import detect_threats; detect_threats.simulate_real_time_detection(10)"
goto end

:train
echo.
echo Running training...
if exist "src\train_model.py" (
    python src\train_model.py
    echo.
    echo Training complete! Running detection...
    python src\detect_threats.py
) else (
    echo ERROR: train_model.py not found!
)
goto end

:exit
echo.
echo Goodbye!
pause
exit /b 0

:end
echo.
echo ========================================
echo    EXECUTION COMPLETE
echo ========================================
echo Check 'reports\' directory for output
pause