@echo off
REM ===================================================================
REM  run_all.bat - one double-click: clean up, run the evals headless,
REM  show the summary, commit and push.
REM
REM  Run it from the project folder. Jupyter does not need to be open;
REM  this executes the notebook without a browser and writes the
REM  outputs back into the .ipynb, which is what gets submitted.
REM ===================================================================
setlocal
cd /d "%~dp0"

echo.
echo ===== 1/6  virtual environment =====
if not exist ".venv\Scripts\activate.bat" (
  echo    .venv not found in %CD%
  echo    Run setup.bat first.
  goto :fail
)
call ".venv\Scripts\activate.bat"
python --version

echo.
echo ===== 2/6  removing junk files =====
if exist "0.3" del /q "0.3" && echo    deleted: 0.3
if exist "=1.0" del /q "=1.0" && echo    deleted: =1.0
echo    done

echo.
echo ===== 3/6  checking the setup =====
REM Informational only. An unset LOCAL_EXIT_MODEL before the fine-tuning run
REM is expected, and must not stop the evals.
python scripts\check_setup.py

echo.
echo ===== 4/6  running the evals (177 API calls, a few minutes) =====
python -m pip install -q nbconvert ipykernel
python -m jupyter nbconvert ^
  --to notebook ^
  --execute ^
  --inplace ^
  --ExecutePreprocessor.timeout=1800 ^
  --ExecutePreprocessor.kernel_name=python3 ^
  "tests\test_evals.ipynb"
if errorlevel 1 (
  echo.
  echo    The notebook raised an error. Open tests\test_evals.ipynb in
  echo    Jupyter to see which cell, and send the traceback.
  goto :fail
)

echo.
echo ===== 5/6  the numbers =====
if exist "tests\eval_summary.md" (
  type "tests\eval_summary.md"
) else (
  echo    tests\eval_summary.md was not written - check the last cell.
  goto :fail
)

echo.
echo ===== 6/6  commit and push =====
git add -A
git -c user.name="Oleg Rogazkin" commit ^
  -m "Fix normalize(), relax prompt rule 2, run evals end to end" ^
  -m "normalize() now returns 'invalid' instead of the raw model text, so a" ^
  -m "chatty answer counts as a mistake rather than entering the confusion" ^
  -m "matrix as a phantom label. Prompt rule 2 no longer requires the time to" ^
  -m "have been offered. Notebook writes tests/eval_summary.md." ^
  -m "" ^
  -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" ^
  -m "Claude-Session: https://claude.ai/code/session_01Ru1u6HbArZumU5w2WewfxA"
git push

echo.
echo ===== finished =====
echo Send tests\eval_summary.md and the numbers above.
goto :end

:fail
echo.
echo ===== stopped =====

:end
echo.
pause
endlocal
