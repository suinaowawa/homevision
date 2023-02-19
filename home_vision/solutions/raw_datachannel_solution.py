"""HomeVision Raw Datachannel Solution which will datachannel the local
saved txt file results instead of real-time processing the results"""
from __future__ import annotations

import json
from typing import Dict, Optional, Type

from pydantic import create_model #pylint:disable=no-name-in-module

import numpy as np
from home_vision.modules.module_base import Module, ModuleOutput

from .solution_base import Solution, SolutionConfig, SolutionInput


class RawDatachannelSolutionConfig(SolutionConfig):
    """Config for Raw Datachannel Solution

    Attributes:
        solution_name (str): name of solution that need to datachannel
        message_file (str): path to the processed results txt file
    """
    solution_name: Optional[str] = 'object_detection_solution'
    message_file: Optional[str] = 'tests/object_detection_solution.txt'

class RawDatachannelSolutionOutput(ModuleOutput):
    """Raw Datachannel Solution Output"""
    image: np.ndarray

@Solution.register('raw_datachannel_solution')
@Module.register('raw_datachannel_solution')
class RawDatachannelSolution(Solution[SolutionInput, Dict, RawDatachannelSolutionConfig]):
    """Raw Datachannel solution that read processed results from file"""
    input_types: Type[SolutionInput] = SolutionInput
    output_types: Type[RawDatachannelSolutionOutput] = RawDatachannelSolutionOutput
    config_type: Type[RawDatachannelSolutionConfig] = RawDatachannelSolutionConfig
    solution_name = "Raw Datachannel Solution"
    module_name = solution_name

    def __init__(self, message_file: str, solution_name: str):
        self.cnt = 0
        with open(message_file, "r", encoding="utf-8") as msg_file:
            self.messages = msg_file.read().splitlines()
        solution_output_types = Solution.by_name(solution_name).output_types
        NewRawDatachannelSolutionOutput = create_model(
            "NewRawDatachannelSolutionOutput", __base__=(
                solution_output_types, RawDatachannelSolutionOutput
            )
        )
        self.output_types: Type[RawDatachannelSolutionOutput] = NewRawDatachannelSolutionOutput

    @classmethod
    def from_config(cls, config: RawDatachannelSolutionConfig) -> RawDatachannelSolution:
        return cls(config.message_file, config.solution_name)


    def _process(self, inputs: SolutionInput) -> RawDatachannelSolutionOutput:
        """Return the results that read from file"""
        self.cnt += 1
        frame_cnt = (self.cnt % len(self.messages))
        msg = json.loads(self.messages[frame_cnt - 1])
        outputs_dict = {**msg, 'image': inputs.image}
        outputs = self.output_types(**outputs_dict)
        return outputs
