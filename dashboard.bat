@echo off
cd /d "%~dp0volt_sim"
start http://localhost:8080/dashboard/dashboard.html
python -m http.server 8080
