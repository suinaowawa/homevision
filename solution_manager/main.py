"""Declare and Start the FastAPI app"""
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates

from solution_manager.routers import api, solutions
from solution_manager.sm import SolutionManager

logging.basicConfig(level=logging.INFO)

middleware = [
    Middleware(SessionMiddleware, secret_key="super-secret")
]
app = FastAPI(middleware=middleware)
app.mount("/static", StaticFiles(directory="solution_manager/static"), name="static")
app.include_router(solutions.router)
app.include_router(api.router)
templates = Jinja2Templates(directory="solution_manager/templates")

sm = SolutionManager.instance() # pylint: disable=no-member

@app.on_event("startup")
async def startup_event():
    """Load available solutions and cameras on startup"""
    sm.load_solutions()
    sm.load_cameras()

@app.on_event("shutdown")
async def shutdown_event():
    """Kill all running subprocesses on shutdown"""
    sm.stop()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port = 5555, reload=False, log_level="debug")
