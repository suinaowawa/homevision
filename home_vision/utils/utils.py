"""HomeVision utility functions"""
import argparse
import json
from collections import ChainMap
from typing import List

import cv2
import yaml
import numpy as np
# from yamlinclude import YamlIncludeConstructor
from home_vision.solutions.solution_base import Solution, SolutionConfig
# YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir='data/configs/')



def str2bool(val):
    """Helper function to convert string to bool for argument parsing"""
    if isinstance(val, bool):
        return val
    if val.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if val.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def draw_note(img_rd: np.ndarray, name: str, fps: float, frame_cnt: int):
    """Draw info on processed frame"""
    font = cv2.FONT_ITALIC
    # Add some info on windows
    cv2.putText(img_rd, name, (20, 60), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img_rd, "Frame:  " + str(frame_cnt), (20, 120), font, 1.8, (138,43,226), 4,
                cv2.LINE_AA)
    cv2.putText(img_rd, f"FPS:  {fps : .2f}", (20, 200), font, 1.8, (138,43,226), 4,
                cv2.LINE_AA)
    cv2.putText(img_rd, "Q: Quit", (20, 360), font, 1.8, (255, 255, 255), 4, cv2.LINE_AA)

def load_solution(solution_name: str, solution_config: SolutionConfig) -> Solution:
    """Load HomeVision Solution by name and SolutionConfig"""
    solution_cls = Solution.by_name(solution_name)
    print("solution_cls:", solution_cls)
    print("solution config:", solution_config)
    solution = solution_cls.from_config(solution_config)
    return solution

def load_solution_config(config_file: str) -> dict:
    """Load dictionary from yaml file"""
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    chain = ChainMap(*config)

    try:
        return dict(chain)
    except:
        return config

def load_solution_config_from_file(
    solution_name: str, solution_config_path: str
) -> SolutionConfig:
    """Load SolutionConfig from file"""
    config_type = Solution.by_name(solution_name).config_type
    if solution_config_path == '':
        print("config_type:", config_type)
        solution_config = config_type()
        print("solution_config:", solution_config)
    else:
        config = load_solution_config(solution_config_path)
        solution_config = config_type(**config)
    return solution_config

def load_solution_config_from_str(
    solution_name: str, solution_config_str: str
) -> SolutionConfig:
    """Load SolutionConfig from config string"""
    config_type = Solution.by_name(solution_name).config_type
    if solution_config_str is not None:
        config = json.loads(solution_config_str)
        solution_config = config_type(**config)
    else:
        solution_config = config_type()
    return solution_config

def load_solution_config_from_dict(
    solution_name: str, solution_config_dict: dict
) -> SolutionConfig:
    """Load SolutionConfig from config dictionary"""
    config_type = Solution.by_name(solution_name).config_type
    if solution_config_dict == {}:
        solution_config = config_type()
    else:
        solution_config = config_type(**solution_config_dict)
    return solution_config

def load_solution_from_dict(
    solution_name: str, solution_config_dict: dict
) -> Solution:
    """Load Solution from config dictionary"""
    solution_config = load_solution_config_from_dict(
        solution_name, solution_config_dict
    )
    solution_cls = Solution.by_name(solution_name)
    solution = solution_cls.from_config(solution_config)
    return solution

def get_iou(bb1: List, bb2: List) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bb1 (List): ['x1', 'x2', 'y1', 'y2']
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
        bb2 (List): ['x1', 'x2', 'y1', 'y2']
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns:
        float: IoU of two bboxes
    """
    xmin, ymin, xmax, ymax = bb1
    bb1 = {'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax}
    xmin, ymin, xmax, ymax = bb2
    bb2 = {'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_ioa(bb1: List, bb2: List) -> float:
    """Calculate the Intersection over Area (IoA) of two bounding boxes,
    area is the smaller bbox's area.

    Args:
        bb1 (List): ['x1', 'x2', 'y1', 'y2']
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
        bb2 (List): ['x1', 'x2', 'y1', 'y2']
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns:
        float: IoA of two bboxes
    """
    xmin, ymin, xmax, ymax = bb1
    bb1 = {'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax}
    xmin, ymin, xmax, ymax = bb2
    bb2 = {'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over area by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    ioa = intersection_area / float(min(bb1_area, bb2_area))
    assert ioa >= 0.0
    assert ioa <= 1.0
    return ioa


if __name__ == "__main__":
    print(load_solution_config('data/configs/tracking_solution.yaml'))
