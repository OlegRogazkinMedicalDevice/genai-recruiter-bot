@echo off
REM One-shot setup for Windows. Run from the project folder:  setup.bat
REM Creates the virtual environment, installs everything, and checks the setup.

setlocal

echo.
echo === 1/4  virtual environment ===
if exist .venv (
    echo .venv already exists, reusing it
) else (
    python -m venv .venv || goto :failed
)
call .venv\Scripts\activate.bat || goto :failed

echo.
echo === 2/4  dependencies (a few minutes) ===
python -m pip install --quiet --upgrade pip
pip install -r requirements.txt || goto :failed

echo.
echo === 3/4  .env ===
if exist .env (
    echo .env already exists, leaving it alone
) else (
    copy .env.example .env >nul
    echo Created .env from the template.
    echo.
    echo   OPEN .env NOW and fill in two values:
    echo     OPENAI_API_KEY=sk-...     from platform.openai.com
    echo     HF_TOKEN=hf_...           from genAI.txt
    echo.
    echo Then run setup.bat again.
    goto :end
)

echo.
echo === 4/4  checking ===
python scripts\check_setup.py

echo.
echo Next:
echo   python scripts\build_chroma.py       embeds the job description
echo   jupyter notebook tests\test_evals.ipynb
echo.
echo For fine-tuning later:
echo   pip install -r requirements-train.txt
echo   python scripts\finetune_exit_advisor.py
goto :end

:failed
echo.
echo Setup stopped on an error above.
exit /b 1

:end
endlocal
