@echo off
REM ============================================================
REM  GFP Designer v4.0 — 一键提交脚本
REM  2026 蛋白质设计竞赛
REM ============================================================

echo ============================================================
echo   GFP Designer v4.0 — Submission Pipeline
echo ============================================================
echo.

REM ---- 1. Check Python ----
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+.
    pause
    exit /b 1
)
echo [OK] Python found.

REM ---- 2. Check dependencies ----
echo Checking dependencies...
pip install -r requirements.txt --quiet 2>nul
echo [OK] Dependencies checked.

REM ---- 3. Check data files ----
if not exist "data\GFP_data.xlsx" (
    echo [ERROR] data\GFP_data.xlsx not found.
    echo Please copy from ..\2026Protein Design\
    pause
    exit /b 1
)
if not exist "data\Exclusion_List.csv" (
    echo [ERROR] data\Exclusion_List.csv not found.
    pause
    exit /b 1
)
echo [OK] Data files found.

REM ---- 4. Check ESMFold model ----
if exist "..\ESMFold\pytorch_model.bin" (
    echo [OK] ESMFold model found at ..\ESMFold\
) else (
    echo [WARN] ESMFold model not found. 3D validation will be skipped.
    echo To enable: download ESMFold to ..\ESMFold\
)

REM ---- 5. Create output directory ----
if not exist "output" mkdir output

REM ---- 6. Run pipeline ----
echo.
echo ============================================================
echo   Running v4.0 pipeline...
echo ============================================================
python main.py

REM ---- 7. Check results ----
if exist "output\submission.csv" (
    echo.
    echo ============================================================
    echo   SUCCESS! Submission generated.
    echo ============================================================
    echo   output\submission.csv        - 6 sequences for submission
    echo   output\submission_detailed.csv - with all scores
    echo   output\agent_logic_tree.md    - Agent decision tree
    echo   output\stress_test_scatter.png - Quality scatter plot
) else (
    echo.
    echo [ERROR] Pipeline failed. Check output above for details.
    pause
    exit /b 1
)

pause
