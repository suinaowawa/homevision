"""HomeVision Raw Stream Solution which will re-stream the video without any processing"""
from __future__ import annotations

from typing import Optional, Type

import numpy as np
from home_vision.modules.module_base import Module, ModuleOutput

from .solution_base import Solution, SolutionConfig, SolutionInput


class RawStreamSolutionConfig(SolutionConfig):
    """Config for Raw Stream Solution

    Attributes:
        dummy (str): dummy attributes
    """
    dummy: Optional[str] = None

class RawStreamSolutionOutput(ModuleOutput):
    """Raw Stream Solution Output"""
    image: np.ndarray

@Solution.register('raw_stream_solution')
@Module.register('raw_stream_solution')
class RawStreamSolution(Solution[SolutionInput, RawStreamSolutionOutput, RawStreamSolutionConfig]):
    """Raw Stream solution that only re-stream"""
    input_types: Type[SolutionInput] = SolutionInput
    output_types: Type[RawStreamSolutionOutput] = RawStreamSolutionOutput
    config_type: Type[RawStreamSolutionConfig] = RawStreamSolutionConfig
    solution_name = "Raw Stream Solution"
    module_name = solution_name

    def __init__(self):
        self.cnt = 0

    @classmethod
    def from_config(cls, config: RawStreamSolutionConfig) -> RawStreamSolution:
        return cls()


    def _process(self, inputs: SolutionInput) -> RawStreamSolutionOutput:
        self.cnt += 1
        outputs = RawStreamSolutionOutput(image=inputs.image)
        return outputs
