@echo off
title MLOps Dashboard Launcher
echo ==============================================
echo    MLOps Dashboard - Model & Data Drift
echo ==============================================
echo Iniciando Streamlit usando modulo de Python...
python -m streamlit run app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Ocurrio un error al iniciar la aplicacion.
    echo Asegurate de que Python y las dependencias esten instaladas.
    pause
)
