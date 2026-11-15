@echo off
rem Exports your saved HUMAN.RIG sprites as game packages (loadouts\<name>\game\).
rem   double-click          -> every saved sprite
rem   drag a sprite folder onto this file -> just that one
rem   To copy into the game too, run from a command window:  Export_To_Game.bat --game "C:\path\to\lifex-100"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" export_to_game.py %*
) else (
  python export_to_game.py %*
)
pause
