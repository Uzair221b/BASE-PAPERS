# Automatic GitHub Sync Script
# Run this script whenever you make changes to automatically push to GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Auto-Sync for BASE-PAPERS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Refresh Git path
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Check for changes
Write-Host "Checking for changes..." -ForegroundColor Yellow
$status = &"C:\Program Files\Git\bin\git.exe" status --short

if ($status) {
    Write-Host "Changes detected:" -ForegroundColor Green
    Write-Host $status
    Write-Host ""
    
    # Get commit message from user or use default
    $commitMessage = Read-Host "Enter commit message (or press Enter for auto-message)"
    
    if ([string]::IsNullOrWhiteSpace($commitMessage)) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
        $commitMessage = "Auto-sync: Updates from $timestamp"
    }
    
    # Add all changes
    Write-Host "[INFO] Adding changes..." -ForegroundColor Yellow
    &"C:\Program Files\Git\bin\git.exe" add .
    
    # Commit
    Write-Host "[INFO] Creating commit..." -ForegroundColor Yellow
    &"C:\Program Files\Git\bin\git.exe" commit -m "$commitMessage"
    
    # Push to GitHub
    Write-Host "[INFO] Pushing to GitHub..." -ForegroundColor Yellow
    &"C:\Program Files\Git\bin\git.exe" push origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "[SUCCESS] Changes synced to GitHub!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "View at: https://github.com/Uzair221b/BASE-PAPERS" -ForegroundColor White
    } else {
        Write-Host ""
        Write-Host "[ERROR] Push failed. Check your internet connection." -ForegroundColor Red
    }
} else {
    Write-Host "[INFO] No changes detected. Everything is up to date!" -ForegroundColor Green
}

Write-Host ""
pause

