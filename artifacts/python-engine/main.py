"""
Latent Diffusion-HMM Trading Engine — FastAPI entry point.

Serves at /engine (routed via the shared proxy).
Port read from $PORT environment variable (default 8000).
"""
from __future__ import annotations

import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

# ─────────────────────────────────────── #
# Logging                                 #
# ─────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("engine")

# ─────────────────────────────────────── #
# App                                     #
# ─────────────────────────────────────── #
app = FastAPI(
    title="Latent Diffusion-HMM Trading Engine",
    description=(
        "Quantitative trading signal pipeline implementing the v3.0 architecture: "
        "Dollar-Volume Bars → 6D Feature Tensor → Kalman Filter → "
        "TVTP-HMM → Triple Gate Execution → Wasserstein Surveillance."
    ),
    version="3.0.0",
    docs_url="/engine/redoc",
    redoc_url="/engine/api-docs",
    openapi_url="/engine/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "engine": "Latent Diffusion-HMM v3.0",
        "docs": "/engine/redoc",
        "health": "/engine/healthz",
        "endpoints": [
            "POST /engine/analyze",
            "GET  /engine/regime/{ticker}",
            "POST /engine/signals",
            "POST /engine/features",
            "POST /engine/validate",
            "GET  /engine/docs-summary",
        ],
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Latent Diffusion-HMM engine on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
