"""Define Capture class"""
from __future__ import annotations

import copy
import logging
import os
import time
from typing import Optional

import cv2
from home_vision.modules.module_base import BaseConfig, Module
from home_vision.solutions.solution_base import SolutionConfig, SolutionInput
from home_vision.utils.utils import load_solution

from .cam_loader import CamLoader


class CaptureConfig(BaseConfig):
    """Config for Capture

    Attributes:
        source (str): video/camera source
        threading (bool): whether use threading to read in frames
        stream_fps (int): desired output fps
        stream_w (int): stream width should be larger than frame width
        stream_h (int): stream height should be larger than frame height
    """
    source: str
    threading: Optional[bool] = False
    stream_fps: Optional[int] = 20
    stream_w: Optional[int] = 5000
    stream_h: Optional[int] = 5000

@Module.register('capture')
class Capture(Module):
    """Capture object that runs HomeVision solution based on cv2.VideoCapture"""
    config_type = CaptureConfig
    cap = None
    module_name = 'Capture Module'

    def __init__(
        self,
        source: str,
        threading: bool,
        stream_w: int,
        stream_h: int,
        stream_fps: int
    ):
        if source.isdigit():
            cap_source = int(source)
        else:
            cap_source = source

        cap = cv2.VideoCapture(cap_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, stream_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, stream_h)
        if threading:
            self.cap = CamLoader(cap, stream_fps=stream_fps).start()
        else:
            self.cap = cap
        self.frame_cnt = 0
        self.fps = 0
        self.solution = None
        self.frame = None

    @classmethod
    def from_config(cls, config: CaptureConfig) -> Capture:
        return cls(config.source, config.threading, config.stream_w, \
            config.stream_h, config.stream_fps)

    def _process(self, **kwargs):
        pass

    def run_capture(
        self,
        solution_name: str,
        solution_config: SolutionConfig,
        display: bool=True,
        save_raw: bool=False,
        save_processed: bool=False
    ):
        """Start HomeVision solution through cv2.VideoCapture

        Args:
            solution_name (str): name of HomeVision solution
            solution_config (SolutionConfig): config of HomeVision solution
            display (bool, optional): display processed video. Defaults to True.
            save_raw (bool, optional): save raw video from source. Defaults to False.
            save_processed (bool, optional): save processed video. Defaults to False.
        """
        self.solution = load_solution(solution_name, solution_config)
        stream_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        stream_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if save_raw:
            raw_out = cv2.VideoWriter(
                f'data/videos/{solution_name}_{timestamp}.avi', fourcc, 20.0, (stream_w,stream_h)
            )
        if save_processed:
            if not os.path.exists('results'):
                os.makedirs('results')
            processed_out = cv2.VideoWriter(
                f'results/{solution_name}_{timestamp}.avi', fourcc, 20.0, (stream_w,stream_h)
            )

        total_frame_cnt = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.debug("TOTAL FRAME: %s", total_frame_cnt)

        while self.cap.isOpened():
            with open("sample.txt", "a+", encoding="utf-8") as outfile:
                frame_s = time.perf_counter()
                success, frame = self.cap.read()
                self.frame = copy.deepcopy(frame)
                if save_raw:
                    raw_out.write(self.frame)

                if not success:
                    break

                key = cv2.waitKey(1)
                if key == ord('q') or key == ord('Q'):
                    break
                self.frame_cnt += 1
                logging.debug('Frame Cnt: %s/%s', self.frame_cnt, total_frame_cnt)
                process_s = time.perf_counter()
                res = self.solution.process(SolutionInput(image=self.frame))
                process_e = time.perf_counter()

                self.frame = res.image
                outfile.write(res.json(exclude={'image'})+'\n')
                self.solution.draw_note(self.frame, self.fps, self.frame_cnt)

                if save_processed:
                    processed_out.write(self.frame)

                if display:
                    img_show = cv2.resize(self.frame,(1080,720))
                    cv2.imshow(solution_name, img_show)


                frame_e = time.perf_counter()
                self.fps = 1 / (frame_e - frame_s)
                logging.info(
                    "FPS: %s | capturing: %.2f ms | processing: %.2f ms | total: %.2f ms",
                    int(self.fps), (process_s - frame_s) * 1000, (process_e - process_s) * 1000,
                    (frame_e - frame_s) * 1000
                )

        self.cap.release()
        cv2.destroyAllWindows()
