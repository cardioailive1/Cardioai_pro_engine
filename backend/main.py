"""
CardioAI Pro — Main Engine Entrypoint
=========================================
Serves the API under /api/*, the live WebSocket at /ws/stream, and the
dashboard frontend as static files at /. One process, one Render service —
backend and frontend are connected by construction, not configuration.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import router as api_router, ws_router

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="CardioAI Pro — Main Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your hospital/payer domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
app.include_router(ws_router)  # /ws/stream — kept off the /api prefix so it isn't shadowed by the static mount


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# Serve the dashboard. Mounted last so /api and /healthz take precedence.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
