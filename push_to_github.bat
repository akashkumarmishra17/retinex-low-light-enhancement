@echo off
title Push Retinex AI to GitHub - Akash Kumar Mishra
echo ====================================================================
echo Pushing Retinex AI to GitHub:
echo https://github.com/akashkumarmishra17/retinex-low-light-enhancement
echo ====================================================================
cd /d "%~dp0"
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo ====================================================================
    echo [SUCCESS] Project successfully pushed to GitHub!
    echo Visit: https://github.com/akashkumarmishra17/retinex-low-light-enhancement
    echo ====================================================================
) else (
    echo ====================================================================
    echo [NOTE] If a browser window opened, please click "Authorize" or sign in.
    echo ====================================================================
)
echo.
pause
