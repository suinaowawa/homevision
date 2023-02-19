"""Run Home Vision Solution and re-stream the result thought webrtc server"""
import argparse
import logging

from home_vision.modules.module_base import Module
from home_vision.solutions.solution_base import Solution
from home_vision.utils.utils import load_solution_config_from_str, str2bool

print("available solutions:", Solution.list_available())
print("available modules:", Module.list_available())


def main(
    solution_name: str,
    solution_config_str: str,
    src: str,
    port: int,
    codec: bool=False,
    buffered: bool=False
):
    """Run HomeVision Solution through webrtc server

    Args:
        solution_name (str): name of HomeVision solution
        solution_config_str (str): string of solution config
        src (str): source of the camera/video
        port (int): port where the webrtc server running
        codec (bool, optional): re-stream compressed video. Defaults to False.
        buffered (bool, optional): buffered video streaming. Defaults to False.
    """
    solution_config = load_solution_config_from_str(solution_name, solution_config_str)

    rtc_dict = {
        'source': src,
        'port': port,
        'codec': codec,
        'buffered': buffered,
    }
    rtc_config_type = Module.by_name('rtc_server').config_type
    rtc_config = rtc_config_type(**rtc_dict)

    module_cls = Module.by_name('rtc_server')

    rtc_server = module_cls.from_config(rtc_config)
    rtc_server.run_server(solution_name, solution_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Home Vision -- Stream Solution')
    parser.add_argument(
        '--solution_name', type=str, default='object_detection_solution', help='solution name'
    )
    parser.add_argument('--solution_config', type=str, default=None, help='solution config string')
    parser.add_argument('--src', default="tests/test.mp4", type=str, help='input src')
    parser.add_argument('--port', default=5556, type=int, help='server running port')
    parser.add_argument('--codec', default=False, type=str2bool, help='compressed codec stream')
    parser.add_argument('--buffered', default=False, type=str2bool, help='buffer stream')
    parser.add_argument('--verbose', default=False, type=str2bool, help='show debug logging')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # run
    main(
        args.solution_name,
        args.solution_config,
        args.src,
        args.port,
        args.codec,
        args.buffered
    )
