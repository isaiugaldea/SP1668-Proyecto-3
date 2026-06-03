# Launcher script for MLOps Dashboard
Clear-Host
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   MLOps Dashboard - Model & Data Drift      " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Iniciando Streamlit a través de Python..." -ForegroundColor Yellow

python -m streamlit run app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError al iniciar la aplicación. Asegúrate de tener instalado Python y las dependencias." -ForegroundColor Red
    Read-Host "Presiona Enter para salir..."
}
