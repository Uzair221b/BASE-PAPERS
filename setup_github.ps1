# Automated GitHub Setup Script for Glaucoma Detection Project
# Run this script to set up your project on GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Glaucoma Detection Project - GitHub Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
Write-Host "Checking for Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version 2>$null
    Write-Host "[OK] Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Git first:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://git-scm.com/download/win" -ForegroundColor White
    Write-Host "2. Download and run the installer" -ForegroundColor White
    Write-Host "3. Use default settings" -ForegroundColor White
    Write-Host "4. Restart this script after installation" -ForegroundColor White
    Write-Host ""
    pause
    exit
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Git Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Configure Git user
$userName = Read-Host "Enter your name (for Git commits)"
$userEmail = Read-Host "Enter your email"

git config --global user.name "$userName"
git config --global user.email "$userEmail"

Write-Host "[OK] Git configured" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Initialize Repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Initialize Git repository
if (Test-Path ".git") {
    Write-Host "[INFO] Git repository already initialized" -ForegroundColor Yellow
} else {
    git init
    Write-Host "[OK] Git repository initialized" -ForegroundColor Green
}

# Add files
Write-Host "[INFO] Adding files to Git..." -ForegroundColor Yellow
git add .
Write-Host "[OK] Files added" -ForegroundColor Green

# Commit
Write-Host "[INFO] Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Glaucoma detection system with 98.5% effective preprocessing pipeline"
Write-Host "[OK] Initial commit created" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Repository Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps to upload to GitHub:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Go to: https://github.com" -ForegroundColor White
Write-Host "2. Sign in to your GitHub account" -ForegroundColor White
Write-Host "3. Click '+' button (top right) -> 'New repository'" -ForegroundColor White
Write-Host "4. Repository name: glaucoma-detection-preprocessing" -ForegroundColor White
Write-Host "5. Description: Advanced glaucoma detection with 98.5% effective preprocessing" -ForegroundColor White
Write-Host "6. Choose Public or Private" -ForegroundColor White
Write-Host "7. DO NOT initialize with README (uncheck all boxes)" -ForegroundColor White
Write-Host "8. Click 'Create repository'" -ForegroundColor White
Write-Host ""
Write-Host "9. Copy the repository URL shown (looks like:" -ForegroundColor White
Write-Host "   https://github.com/YOUR_USERNAME/glaucoma-detection-preprocessing.git)" -ForegroundColor White
Write-Host ""

$repoUrl = Read-Host "Paste your GitHub repository URL here"

if ($repoUrl) {
    Write-Host "[INFO] Adding remote repository..." -ForegroundColor Yellow
    git remote add origin $repoUrl 2>$null
    if ($LASTEXITCODE -ne 0) {
        # Remote might already exist
        git remote set-url origin $repoUrl
    }
    Write-Host "[OK] Remote repository added" -ForegroundColor Green
    
    Write-Host "[INFO] Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host "(This may take 1-2 minutes)" -ForegroundColor Yellow
    
    git branch -M main
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "[SUCCESS] Project uploaded to GitHub!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your project URL: $repoUrl" -ForegroundColor White
        Write-Host ""
        Write-Host "You can now access your project from any computer!" -ForegroundColor Green
        Write-Host ""
        Write-Host "To sync on another computer:" -ForegroundColor Yellow
        Write-Host "  git clone $repoUrl" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host "[ERROR] Failed to push to GitHub" -ForegroundColor Red
        Write-Host "You may need to authenticate or check your repository URL" -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] Skipping GitHub push (no URL provided)" -ForegroundColor Yellow
    Write-Host "Run this script again after creating the repository" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Your code is ready for GitHub" -ForegroundColor White
Write-Host "2. Images are excluded (.gitignore)" -ForegroundColor White
Write-Host "3. Store images separately (OneDrive, Google Drive)" -ForegroundColor White
Write-Host "4. On other computers: git clone + copy images" -ForegroundColor White
Write-Host ""

pause

