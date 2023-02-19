"""Tests for Home Vision SolutionManager HTTP API"""
import pytest
from fastapi.testclient import TestClient
from solution_manager.main import app
from solution_manager.sm import ProcessDetail, SolutionDetail, SolutionManager


@pytest.mark.pipeline
def test_api_get_solutions():
    """Test API endpoint GET /api/solutions"""
    with TestClient(app) as client:
        response = client.get("/api/solutions")
    assert response.status_code == 200
    assert response.json() is not None

@pytest.mark.parametrize(
    "solution_name, camera_src, config, response_status, start_called_times",
    [
        ('object_detection_solution', 'tests/test.mp4', {}, 200, 1),
        ('raw_datachannel_solution', 'tests/test.mp4', {}, 200, 1),
        ('raw_stream_solution', 'tests/test.mp4' ,{}, 200, 1),
        ('raw_stream_solution', 'invalid_camera_src',{}, 422, 0),
        ('raw_stream_solution', 'tests/test.mp4', {'wrong_key': 'wrong_val'}, 422, 0),
        ('not_available_solution', 'tests/test.mp4', {}, 422, 0)
    ]
)
@pytest.mark.pipeline
def test_api_start_solution(
    mocker, solution_name, camera_src, config, response_status, start_called_times
):
    """Test API endpoint POST /api/start"""
    #pylint: disable=no-member
    spy_start = mocker.spy(SolutionManager.instance(), "start_solution")

    with TestClient(app) as client:
        response = client.post(
            "/api/start",
            json={
                "solution_name": solution_name,
                "camera_src": camera_src,
                "config": config
                },
        )
        assert response.status_code == response_status

    assert spy_start.call_count == start_called_times

@pytest.mark.parametrize(
    "solution_name, running, response_status",
    [
        ('raw_stream_solution', True, 200),
        ('raw_stream_solution', False, 404),
        ('not_available_solution', False, 422)
    ]
)
@pytest.mark.pipeline
def test_api_stop_solution(mocker, solution_name, running, response_status):
    """Test API endpoint POST /api/stop"""
    sm = SolutionManager.instance() #pylint: disable=invalid-name, no-member

    with TestClient(app) as client:
        proc = mocker.patch('subprocess.Popen')
        spy_kill = mocker.spy(proc, "kill")
        if running:
            solution_detail = SolutionDetail(
                    solution_name=solution_name,
                    camera_src='tests/test.mp4',
                    )
            process_detail = ProcessDetail(
                port=3456, pid=proc.pid, url="http://localhost:3456"
            )
            sm.running_solutions[solution_detail] = process_detail
            sm.running_process[solution_detail] = proc

        response = client.post(
            "/api/stop",
            json={
                "solution_name": solution_name,
                "camera_src": 'tests/test.mp4'
                },
        )
        assert response.status_code == response_status

        if running:
            spy_kill.assert_called_once()
        else:
            spy_kill.assert_not_called()

@pytest.mark.pipeline
def test_api_get_running_solutions():
    """Test API endpoint GET /api/running_solutions"""
    with TestClient(app) as client:
        response = client.get("/api/running_solutions")
    assert response.status_code == 200
    assert response.json() == []
