@echo off
REM Commits the README and config changes that followed the eval run.
setlocal
cd /d "%~dp0"

echo.
echo ===== committing the measured numbers =====
git add -A
git commit ^
  -m "README: real eval numbers; default prompt v2, not v3" ^
  -m "v2 measured 83.05%% against v3's 76.27%% on the 59 labelled turns - four" ^
  -m "items apart, which is above this notebook's own noise threshold of two." ^
  -m "The few-shot examples in v3 are single candidate messages while the" ^
  -m "labels depend on the whole conversation, so they taught the model to" ^
  -m "decide from the last message alone. MAIN_AGENT_PROMPT_VERSION now" ^
  -m "defaults to v2." ^
  -m "" ^
  -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" ^
  -m "Claude-Session: https://claude.ai/code/session_01Ru1u6HbArZumU5w2WewfxA"
git push

echo.
echo ===== done =====
git log --oneline -3
echo.
pause
endlocal
