@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0.tools\Start-Speech.ps1" -Stop
if errorlevel 1 pause
