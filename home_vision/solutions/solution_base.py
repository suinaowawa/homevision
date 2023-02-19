"""Base Class for HomeVision Solution
a solution can consist of modules and solution
e.g. sol0 = M0 -> M1 -> M2
     sol1 = M0 -> M1 -> M2 -> M3
     sol1 = sol0 -> M3
"""
from __future__ import annotations
from abc import abstractmethod
from typing import TypeVar
import cv2
import numpy as np
from pydantic import BaseModel #pylint: disable=no-name-in-module
from home_vision.common.registrable import Registrable
from home_vision.modules.module_base import Module, ModuleInput, ModuleOutput

class SolutionConfig(Registrable, BaseModel):
    """Abstraction for solution config.

    A custom solution config must inherit this class.
    """
    class Config:
        """Config for solution config pydantic model"""
        frozen=True
        extra='forbid'


InputT = TypeVar('InputT', bound='ModuleInput')
OutputT = TypeVar('OutputT', bound='ModuleOutput')
ConfigT = TypeVar('ConfigT', bound='SolutionConfig')


class Solution(Module[InputT, OutputT, ConfigT]):
    """Abstraction for solution.

    A custom solution must inherit this class.
    """
    @classmethod
    @abstractmethod
    def from_config(cls, config: ConfigT) -> Solution:
        pass

    def draw_note(
        self, img_rd: np.ndarray, fps: float, frame_cnt: int
    ) -> None:
        """Draw fps and solution name for the frame

        Args:
            img_rd (np.ndarray): input frame
            fps (float): current frame fps
            frame_cnt (int): current frame count
        """
        font = cv2.FONT_ITALIC
        # Add some info on windows
        cv2.putText(img_rd, self.solution_name, (20, 60), font, 2, (255, 255, 255), 4, cv2.LINE_AA) #pylint: disable=no-member
        cv2.putText(img_rd, "Frame:  " + str(frame_cnt), (20, 120), font, 1.8, (138,43,226), 4,
                    cv2.LINE_AA)
        cv2.putText(
            img_rd, f"FPS:    {fps : .2f}", (20, 200), font, 1.8, (138,43,226), 4, cv2.LINE_AA
        )
        cv2.putText(img_rd, "Q: Quit", (20, 360), font, 1.8, (255, 255, 255), 4, cv2.LINE_AA)

class SolutionInput(ModuleInput):
    """Common Solution Input"""
    image: np.ndarray
