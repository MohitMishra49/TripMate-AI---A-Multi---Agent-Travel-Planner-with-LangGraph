"""
app_v2.py
=========
New FastAPI entry point that uses backend_v2 (async graph, pooled Postgres
connection) instead of backend.py. app.py is untouched.

Only three things actually changed vs. app.py:
  1. `nest_asyncio` is gone — nothing needs it anymore, since backend_v2
     never calls asyncio.run() from inside a running event loop.
  2. A `lifespan` context manager opens the connection pool once at startup
     and closes it once at shutdown, instead of backend.py's global
     connection that opened at import time and never closed.
  3. The two routes `await` the now-async `run_travel_agent` /
     `resume_travel_agent` instead of calling them as blocking sync calls.

Everything else (routes, request/response models, templates, static
mounting) is identical to app.py on purpose — this is a swap of the
execution model, not a rewrite of the API surface.

To run this instead of app.py:
    uvicorn app_v2:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from backend import init_travel_backend, close_travel_backend, run_travel_agent, resume_travel_agent

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Opens the AsyncConnectionPool + runs checkpointer.setup() once, here,
    # instead of at import time. This is also where you'd add any other
    # startup/shutdown resources later.
    await init_travel_backend()
    try:
        yield
    finally:
        await close_travel_backend()


app = FastAPI(
    title="TripMate AI",
    description=(
        "LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, "
        "Human-in-the-Loop, and FastAPI Frontend (async / pooled backend)"
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        result = await run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print("ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.post("/api/travel/approve")
async def approve_travel_plan(request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        result = await resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print("APPROVAL ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "TripMate AI API is running (async backend, pooled Postgres)",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
            "connection_pooling",
            "parallel_data_agents",
        ],
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run(
        "app_v2:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )