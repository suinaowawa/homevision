"""Route for HomeVision API"""
from fastapi import APIRouter, HTTPException
from solution_manager.sm import SolutionDetail, SolutionManager

router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={404: {"description": "Not found"}}
)

sm = SolutionManager.instance() #pylint: disable=no-member

@router.get("/solutions")
async def available_solutions() -> dict:
    """List all available solutions"""
    return sm.available_solutions

@router.post("/start")
async def start_solution(solution_detail: SolutionDetail) -> str:
    """Start a subprocess running a HomeVision solution

    Args:
        solution_detail (SolutionDetail): contain solution name, camera src and solution config

    Returns:
        str: the url where the requested solution is running
    """
    url = sm.start_solution(solution_detail)
    return url

@router.post("/stop")
async def stop_solution(solution_detail: SolutionDetail) -> str:
    """Stop a subprocess that is running a HomeVision solution

    Args:
        solution_detail (SolutionDetail): contain solution name, camera src and solution config

    Raises:
        HTTPException: when try to stop a solution that is not running

    Returns:
        str: a message that the solution is stopped
    """
    if solution_detail not in sm.running_solutions.keys():
        raise HTTPException(
            status_code=404, detail='solution not running!'
        )
    msg = sm.stop_solution(solution_detail)
    return msg

@router.get("/running_solutions")
async def running_solutions() -> set:
    """List all solutions that are currently running"""
    return set(sm.running_solutions)
