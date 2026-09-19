@echo off
setlocal

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

echo Starting CODESYSCompileService on the configured local address...
echo Run example_client.py separately after the server is ready.
python debug_server.py
