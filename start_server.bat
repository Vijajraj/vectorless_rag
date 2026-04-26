@echo off
echo ===================================================
echo   VectorlessRAG Server
echo ===================================================
echo.
echo Starting the FastAPI backend...
echo Keep this window open to keep the server running.
echo You can safely close Antigravity.
echo.
echo Running on: http://localhost:8000
echo.

set PYTHONIOENCODING=utf-8
python server.py

pause
