@echo off
echo Running Satellite Anomaly Detection System Pipeline...
call scad_env\Scripts\activate.bat
scad_env\Scripts\python.exe main_pipeline.py
pause
