---
name: Python Engine in pnpm Monorepo
description: How a Python FastAPI service is hosted alongside Node.js in this pnpm workspace and routed via the shared proxy.
---

The Python engine is NOT a pnpm workspace package. It lives in `artifacts/python-engine/` as a plain Python directory.

**Proxy routing**: Added as a second `[[services]]` block in `artifacts/api-server/.replit-artifact/artifact.toml` with `localPort = 8000` and `paths = ["/engine"]`. This routes `/engine/*` through the shared proxy to the Python FastAPI process.

**Why:** The artifact system only supports specific artifact types (react-vite, expo, slides, etc.) — Python is not one of them. Adding it as a service under the api-server artifact is the correct workaround.

**Workflow name**: `artifacts/api-server: Python Engine` — auto-created by the platform when the second service block was added to artifact.toml. Cannot be overridden via configureWorkflow (PROHIBITED_ACTION). Use restart_workflow to start/restart it.

**Start command**: `cd /home/runner/workspace/artifacts/python-engine && python main.py` — must use absolute path; relative path fails with "No such file or directory".

**Port**: 8000 (supported port). Node.js API is on 8080.
