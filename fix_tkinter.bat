@echo off
echo ===============================================
echo Tkinter Fix Helper
echo ===============================================
echo.
echo This script will help fix the tkinter/Tcl issue by installing the required components.
echo.
echo Options:
echo 1. Download and install ActiveTcl (recommended)
echo 2. Reinstall Python with tcl/tk components
echo 3. Install tkinter using pip
echo 4. Cancel
echo.

set /p choice=Enter your choice (1-4): 

if "%choice%"=="1" (
    echo.
    echo Opening ActiveTcl download page...
    start https://www.activestate.com/products/tcl/downloads/
    echo After installing ActiveTcl, restart your application.
) else if "%choice%"=="2" (
    echo.
    echo Opening Python download page...
    start https://www.python.org/downloads/
    echo When installing Python, make sure to check 'tcl/tk and IDLE' option.
) else if "%choice%"=="3" (
    echo.
    echo Attempting to install tkinter with pip...
    pip install tk
    echo.
    echo If this doesn't work, please try option 1 or 2.
) else if "%choice%"=="4" (
    echo Exiting...
    exit /b
) else (
    echo Invalid choice. Please run the script again.
)

echo.
pause
