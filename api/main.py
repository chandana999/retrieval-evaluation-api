"""
FastAPI backend for Advanced Retrieval Evaluation UI.
Provides streaming execution logs and executive summary.

NOTE: Evaluation runs in a separate subprocess (not thread) to avoid asyncio
conflicts when Ctrl+C stops the server - Ragas uses async HTTP internally.
"""
import os
import json
from decimal import Decimal
from multiprocessing import Process, Queue as MPQueue
from queue import Empty
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio

# Add parent dir to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.evaluation_runner import run_evaluation


# Global state for evaluation run
evaluation_state = {"running": False, "last_result": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    evaluation_state["running"] = False


app = FastAPI(
    title="Advanced Retrieval Evaluation API",
    description="API for running retriever evaluation with execution log and executive summary",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_serializer(obj):
    """Convert Decimal and other non-JSON types for json.dumps."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


RESULTS_FILENAME = "retriever_evaluation_results.csv"

def get_results_path() -> Path:
    """Check data/ first, then temporary/ - so you can copy results to temporary for quick UI preview."""
    root = Path(__file__).resolve().parent.parent
    for folder in ("data", "temporary"):
        path = root / folder / RESULTS_FILENAME
        if path.exists():
            return path
    return root / "data" / RESULTS_FILENAME  # default path


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/results")
async def get_results():
    """Return existing evaluation results from CSV if available."""
    path = get_results_path()
    if not path.exists():
        return {"results": None, "message": "No evaluation results yet. Run evaluation to generate."}

    import pandas as pd
    df = pd.read_csv(path)
    results = df.to_dict(orient="records")
    return {"results": results, "message": "Loaded from previous run"}


@app.get("/api/executive-summary")
async def get_executive_summary():
    """Generate executive summary from saved results."""
    path = get_results_path()
    if not path.exists():
        return {"summary": "No evaluation results available. Run the evaluation to generate an executive summary."}

    import pandas as pd
    df = pd.read_csv(path)
    results = df.to_dict(orient="records")

    from api.evaluation_runner import _build_executive_summary
    summary = _build_executive_summary(results)
    return {"summary": summary, "results": results}


def _run_evaluation_worker(log_queue: MPQueue, result_queue: MPQueue) -> None:
    """Worker run in subprocess - has its own asyncio, no conflict with uvicorn."""
    try:
        result = run_evaluation(log_queue=log_queue)
        result_queue.put(result)
    except Exception as e:
        result_queue.put({"success": False, "error": str(e)})
    finally:
        result_queue.put(None)  # sentinel: done


@app.get("/api/run/stream")
async def run_evaluation_stream():
    """
    Start evaluation and stream execution logs via Server-Sent Events (SSE).
    Runs in a subprocess to avoid asyncio/thread conflicts (Ragas uses async internally).
    """
    if evaluation_state["running"]:
        return {"error": "Evaluation already in progress"}

    log_queue = MPQueue()
    result_queue = MPQueue()

    async def event_generator():
        evaluation_state["running"] = True
        proc = Process(target=_run_evaluation_worker, args=(log_queue, result_queue), daemon=True)
        proc.start()

        result = None
        try:
            while proc.is_alive() or not log_queue.empty():
                try:
                    entry = log_queue.get(timeout=0.3)
                    if isinstance(entry, dict) and entry.get("_event") == "retriever_result":
                        yield f"data: {json.dumps({'type': 'retriever_result', 'payload': entry['result']}, default=_json_serializer)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'log', 'payload': entry})}\n\n"
                except Empty:
                    await asyncio.sleep(0.05)
            while not log_queue.empty():
                try:
                    entry = log_queue.get_nowait()
                    if isinstance(entry, dict) and entry.get("_event") == "retriever_result":
                        yield f"data: {json.dumps({'type': 'retriever_result', 'payload': entry['result']}, default=_json_serializer)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'log', 'payload': entry})}\n\n"
                except Empty:
                    break
            proc.join(timeout=5)
            while True:
                v = result_queue.get(timeout=1)
                if v is None:
                    break
                result = v
        except Empty:
            pass
        finally:
            if proc.is_alive():
                proc.terminate()
            evaluation_state["running"] = False

        if result:
            evaluation_state["last_result"] = result
        yield f"data: {json.dumps({'type': 'complete', 'payload': result}, default=_json_serializer)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/run")
async def run_evaluation_sync(background_tasks: BackgroundTasks):
    """
    Run evaluation in background. Returns immediately.
    Poll /api/results or /api/executive-summary for results.
    """
    if evaluation_state["running"]:
        return {"status": "already_running", "message": "Evaluation in progress"}

    def run():
        evaluation_state["running"] = True
        try:
            result = run_evaluation(log_queue=None)
            evaluation_state["last_result"] = result
        finally:
            evaluation_state["running"] = False

    background_tasks.add_task(run)
    return {"status": "started", "message": "Evaluation started. Poll /api/results for updates."}


@app.get("/api/status")
async def get_status():
    """Check if evaluation is running and get last result."""
    return {
        "running": evaluation_state["running"],
        "has_last_result": evaluation_state["last_result"] is not None,
    }


# Serve frontend
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/assets", StaticFiles(directory=frontend_path / "assets"), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_path / "index.html")
else:
    @app.get("/")
    async def root():
        return {"message": "Frontend not found. Create frontend/index.html"}
