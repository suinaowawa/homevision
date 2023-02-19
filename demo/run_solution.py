"""Run Home Vision Solution through cv2.capture"""
import argparse
import logging

from home_vision.modules.module_base import Module
from home_vision.solutions.solution_base import Solution
from home_vision.utils.utils import load_solution_config_from_file, str2bool

print("available solutions:", Solution.list_available())
print("available modules:", Module.list_available())

def main(
    solution_name: str,
    solution_config_path: str,
    src: str,
    stream_fps: int = 20,
    threading: bool = False,
    stream_w: int = 5000,
    stream_h: int = 5000,
    display: bool = True,
    save_raw: bool = False,
    save_processed: bool = False
):
    """Run Home Vision Solution using cv2.VideoCapture

    Args:
        solution_name (str): name of HomeVision solution
        solution_config_path (str): path to solution's yaml config file
        src (str): source of the camera/video
        stream_fps (int, optional): desired fps. Defaults to 20.
        threading (bool, optional): if use threading for read in frames. Defaults to False.
        stream_w (int, optional): stream width, value should be larger than frame width.
        Defaults to 5000.
        stream_h (int, optional): stream height, value should be larger than frame height.
        Defaults to 5000.
        display (bool, optional): if display the processed frames. Defaults to True.
        save_raw (bool, optional): save the raw video from camer_src. Defaults to False.
        save_processed (bool, optional): save the processed video as .avi file. Defaults to False.
    """
    solution_config = load_solution_config_from_file(solution_name, solution_config_path)
    # change capture module config
    cap_dict = {
        'source': src,
        'stream_fps': stream_fps,
        'threading': threading,
        'stream_w': stream_w,
        'stream_h': stream_h,
    }
    cap_config_type = Module.by_name('capture').config_type
    cap_config = cap_config_type(**cap_dict)

    module_cls = Module.by_name('capture')
    capture = module_cls.from_config(cap_config)
    capture.run_capture(solution_name, solution_config, display, save_raw, save_processed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Home Vision -- Run Solution')
    parser.add_argument(
        '--solution_name', type=str, default='object_detection_solution', help='solution name'
    )
    parser.add_argument('--solution_config', type=str, default='', help='solution config path')
    parser.add_argument('--src', default="tests/test.mp4", type=str, help='input src')
    parser.add_argument('--display', default=False, type=str2bool, help='display the result')
    parser.add_argument('--save_raw', default=False, type=str2bool, help='save raw video')
    parser.add_argument(
        '--save_processed', default=True, type=str2bool, help='save processed video result'
    )
    parser.add_argument('--threading', default=False, type=str2bool, help='read video threading')
    parser.add_argument('--stream_fps', default=15, type=int, help='threaded streaming fps')
    parser.add_argument('--stream_w', default=5000, type=int, help='stream width')
    parser.add_argument('--stream_h', default=5000, type=int, help='stream height')
    parser.add_argument('--verbose', default=True, type=str2bool, help='show debug logging')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    main(
        args.solution_name,
        args.solution_config,
        args.src,
        args.stream_fps,
        args.threading,
        args.stream_w,
        args.stream_h,
        args.display,
        args.save_raw,
        args.save_processed
    )
