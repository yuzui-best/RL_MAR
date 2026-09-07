@echo off
if not defined CODESYS_PATH set "CODESYS_PATH=C:\Program Files\CODESYS\Common\CODESYS.exe"
"%CODESYS_PATH%" --profile="CODESYS V3.5 SP20 Patch 3" --noUI --runscript="%~dp0codesys_debugger.py"
