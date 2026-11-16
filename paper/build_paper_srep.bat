@echo off
rem Rebuilds the Scientific Reports manuscript and supplementary file from the result CSVs.
rem Double-click this file, or run it from any command window.
cd /d "%~dp0.."
set "PYTHONPATH=%CD%\deps;%CD%\paper\pydeps"
"C:\Users\hassa\anaconda3\python.exe" paper\build_paper_srep.py
pause
