# CODESYSCompileService

This directory contains the optional CODESYS compilation service adapted from
https://github.com/cangkui/CODESYSCompileService (MIT).

Requirements: Windows, Python 3, CODESYS V3.5 SP20 with ScriptEngine, and a
configured `CODESYS_PATH`. Install Python packages with:

```bash
pip install -r requirements.txt
```

Start the service with `python debug_server.py` or `start.bat`. Safe defaults:

- `CODESYS_HOST=127.0.0.1` and `CODESYS_PORT=9000`.
- Set `CODESYS_API_KEY` before first start, or a random key is written to the
  ignored local file `api_keys.json`.
- The `/api/v1/script/execute` endpoint is disabled unless
  `CODESYS_ALLOW_SCRIPT_EXECUTION=1` is explicitly set. Enable it only on a
  trusted local machine.
- Do not expose this development service directly to an untrusted network.

The upstream template project is not redistributed here. Obtain and configure
it from the upstream repository if your CODESYS installation requires it.
