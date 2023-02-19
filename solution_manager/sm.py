"""Manager for HomeVision Solutions"""
import os
import json
import socket
from contextlib import closing
from subprocess import Popen
from typing import Dict, List, Optional

import av
import yaml
from pydantic import BaseModel, Field, root_validator  # pylint: disable=no-name-in-module
from home_vision.common.singleton import Singleton
from home_vision.solutions.solution_base import Solution



def find_free_port() -> int:
    """Return free port on localhost"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(('', 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


class SolutionDetail(BaseModel):
    """Detail for a HomeVision solution with camera_src, solution_name
    and solution config validations"""
    solution_name: str
    camera_src: str
    config: Optional[dict] = Field(None, description="input config to start solution")

    @root_validator
    def check_config(cls, values): #pylint: disable=no-self-argument
        """Validate solution_name, camera_src and solution_config"""
        sm = SolutionManager.instance() #pylint: disable=no-member, invalid-name
        solution_name = values.get('solution_name')
        camera_src = values.get('camera_src')
        config = values.get('config')
        if solution_name not in sm.available_solutions.keys():
            raise ValueError(f'{solution_name} not in {list(sm.available_solutions.keys())}')

        try:
            av.open(camera_src, timeout=3)
        except Exception as exc:
            raise ValueError(f'{camera_src} not valid') from exc

        config_type = Solution.by_name(solution_name).config_type
        try:
            if config is None: # use default config
                config = config_type()
                values['config'] = config
            else:
                values['config'] = config_type(**config)
            return values

        except Exception as exc:
            raise ValueError(exc) from exc

    def __hash__(self):
        return hash((self.solution_name, self.camera_src, self.config))

    def __eq__(self, other):
        return (self.solution_name, self.camera_src, self.config) == \
        (other.solution_name, other.camera_src, other.config)


class ProcessDetail(BaseModel):
    """Detail of a running solution's process"""
    pid: int
    port: int
    url: str

class HomeVisionSolutionBaseConfig(BaseModel):
    """Base config for HomeVision Solution"""

    name: str
    enable: bool = False

    class Config:
        """Forbid extra fields"""
        extra = "forbid"

class CameraSrcBaseConfig(BaseModel):
    """Base config for Camera Source"""
    name: str
    src: str

    class Config:
        """Forbid extra fields"""
        extra = "forbid"

class SolutionManagerConfig(BaseModel):
    """Config for SolutionManager"""
    host_ip: str
    solutions: List[HomeVisionSolutionBaseConfig]
    cameras: List[CameraSrcBaseConfig]

@Singleton
class SolutionManager:
    """Manager for HomeVision Solutions, including start/stop a solution,
    list all available/running solutions"""
    available_solutions: Dict[str, str] = {}
    available_cameras: Dict[str, str] = {}
    running_solutions: Dict[SolutionDetail, ProcessDetail] = {}
    host_ip = None
    config = None
    additional_cam_cnt = 0
    running_process: Dict[SolutionDetail, Popen] = {}

    def __init__(self, config_path: Optional[str]=None):
        default_config = (
            "data/configs/solution_manager.yaml"
            if not config_path
            else config_path
        )
        test_config = "tests/solution_manager.yaml"
        config_path = (
            default_config
            if os.path.exists(default_config)
            else test_config
        )

        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            self.config = SolutionManagerConfig(**config)
            self.host_ip = self.config.host_ip

    def load_solutions(self):
        """Load enabled solution in SolutionManager config, and its default SolutionConfig,
        store in self.available_solutions"""
        for solution in self.config.solutions:
            if solution.enable:
                self.available_solutions[solution.name] = \
                Solution.by_name(solution.name).config_type().dict()
        self.available_solutions['raw_stream_solution'] = \
            Solution.by_name('raw_stream_solution').config_type().dict()
        self.available_solutions['raw_datachannel_solution'] = \
            Solution.by_name('raw_datachannel_solution').config_type().dict()

    def load_cameras(self):
        """Load all camera_src in SolutionManager config,
        and store them in self.available_cameras"""
        for camera in self.config.cameras:
            self.available_cameras[camera.name] =camera.src

    def add_camera(self, camera_name: str, camera_src: str) -> str:
        """Add a new camera_src to self.available_cameras

        Args:
            camera_name (str): name tag for camera_src
            camera_src (str): the camera src

        Returns:
            str: message to indicate if camera was added
        """
        if camera_name in self.available_cameras:
            return f"'{camera_name}' already exist!"
        self.available_cameras[camera_name] = camera_src
        return "camera successfully added!!"

    def start_solution(self, solution_detail: SolutionDetail) -> str:
        """Start a HomeVision solution subprocess by camera_src, solution_name and solution_config

        Args:
            solution_detail (SolutionDetail): Detail of the solution

        Returns:
            str: url for the started and running HomeVision solution
        """
        codec = False
        if solution_detail in self.running_solutions:
            return self.running_solutions[solution_detail].url

        solution_name = solution_detail.solution_name
        solution_config = solution_detail.config
        solution_config_str = json.dumps(solution_config.dict())
        camera_src = solution_detail.camera_src

        if camera_src not in self.available_cameras.values():
            self.additional_cam_cnt += 1
            self.available_cameras[f'add_cam_{self.additional_cam_cnt}'] = camera_src

        if solution_name in ['room_projection_solution']: #mapping in config
            codec = solution_detail.config.codec
            connect_solution_detail = SolutionDetail(
                solution_name='person_tracking_solution',
                camera_src=camera_src,
                config=solution_config.tracking_solution
            )

            camera_src = self.start_solution(solution_detail=connect_solution_detail)

        port = find_free_port()
        url = f"http://{self.host_ip}:{port}"
        args = ['python3', 'demo/stream_solution.py', '--solution_name', solution_name, \
        '--solution_config', solution_config_str, '--port', str(port), \
        '--src', camera_src, '--codec', str(codec)]
        output_file = f"data/{port}_{solution_name}.txt"
        if not os.path.exists('data'):
            os.makedirs('data')
        with open(output_file, "w", 1, encoding="utf-8") as log:
            process = Popen(args, stdout = log, stderr = log, text = True)

        process_detail = ProcessDetail(port=port, pid=process.pid, url=url)
        self.running_process[solution_detail] = process
        self.running_solutions[solution_detail] = process_detail
        return url

    def stop_solution(self, solution_detail: SolutionDetail) -> str:
        """Stop a running HomeVision solution by camera_src, solution_name and solution_config

        Args:
            solution_detail (SolutionDetail): Detail of the solution

        Returns:
            str: message to indicate if solution was stopped
        """
        if solution_detail not in self.running_solutions:
            return "solution not running!"
        self.running_process[solution_detail].kill()
        del self.running_solutions[solution_detail]
        del self.running_process[solution_detail]
        return "solution stopped!"

    def stop(self):
        """Kill all running solutions"""
        for solution_detail in self.running_solutions:
            self.running_process[solution_detail].kill()
        self.running_process = {}
        self.running_solutions = {}
