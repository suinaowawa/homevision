"""Route for HomeVision solution manager front-end"""
import json
from typing import Any

from fastapi import APIRouter, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette import status
from starlette.responses import RedirectResponse
from solution_manager.sm import SolutionDetail, SolutionManager

def flash(request: Request, message: Any, category: str = "primary") -> None:
    """Flash a message with category"""
    if "_messages" not in request.session:
        request.session["_messages"] = []
        request.session["_messages"].append({"message": message, "category": category})

def get_flashed_messages(request: Request):
    """Get all flashed messages"""
    return request.session.pop("_messages") if "_messages" in request.session else []

router = APIRouter(
    # prefix="/",
    tags=["solutions"],
    responses={404: {"description": "Not found"}}
)
router.mount("/static", StaticFiles(directory="solution_manager/static"), name="static")
templates = Jinja2Templates(directory="solution_manager/templates")
templates.env.globals['get_flashed_messages'] = get_flashed_messages

sm = SolutionManager.instance() #pylint: disable=no-member

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def all_solutions(request: Request):
    """Render HomeVision solution manager home page"""

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "solutions":  sm.available_solutions,
            "cameras": sm.available_cameras,
            "running": {
                (k.camera_src, k.solution_name, k.config.json()):
                v.port for k,v in sm.running_solutions.items()
            },
            "ports": [detail.port for detail in sm.running_solutions.values()]
        }
    )


@router.post("/add_camera", response_class=HTMLResponse, include_in_schema=False)
async def add_cameras(camera_name: str = Form(), camera_src: str = Form()):
    """Add a new camera to SolutionManager"""
    sm.add_camera(camera_name, camera_src)
    return RedirectResponse(url='/', status_code=status.HTTP_302_FOUND)

@router.post("/start", include_in_schema=False)
async def start_solution(
    camera_src: str = Form(), solution_name: str = Form(), config: str = Form()
):
    """Start a HomeVision solution based on camera_src, solution_name and solution_config"""
    try:
        solution_detail = SolutionDetail(
            camera_src=camera_src, solution_name=solution_name, config=json.loads(config)
        )
    except ValueError as e:
        return Response(
            content=str(e), status_code=422
        )

    _ = sm.start_solution(solution_detail)
    return RedirectResponse(
        url='/', status_code=status.HTTP_302_FOUND
    )

@router.post("/stop", include_in_schema=False)
async def stop_solution(
    request: Request, camera_src: str = Form(), solution_name: str = Form(), config: str = Form()
):
    """Stop a running HomeVision solution"""
    solution_detail = SolutionDetail(
        camera_src=camera_src, solution_name=solution_name, config=json.loads(config)
    )
    msg = sm.stop_solution(solution_detail)
    flash(request, msg, "primary")
    return RedirectResponse(
        url='/', status_code=status.HTTP_302_FOUND
    )
