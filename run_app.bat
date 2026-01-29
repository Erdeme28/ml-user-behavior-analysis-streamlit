@echo off
echo ===============================
echo ML User Behavior Analysis
echo Opening the app...
echo ===============================

REM Go to project directory
cd /d %~dp0

REM Open the app
py -3.11 -m streamlit run app/1_ML_User_Behavior_Analysis.py

pause
