@echo off
echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete!
echo Virtual environment created at: .venv
pause
