# Pull Latest Changes from GitHub
# Run this on another computer or to get latest updates

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pulling Latest from GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Refresh Git path
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "[INFO] Pulling latest changes from GitHub..." -ForegroundColor Yellow
&"C:\Program Files\Git\bin\git.exe" pull origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "[SUCCESS] Project updated!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[ERROR] Pull failed. Check your internet connection." -ForegroundColor Red
}

Write-Host ""
pause

