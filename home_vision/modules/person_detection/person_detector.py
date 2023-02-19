"""Base class for Person Detectors"""
from __future__ import annotations

import logging
from typing import List, Any, Literal, Type
import numpy as np

from home_vision.modules.module_base import Module, ModuleConfig, ModuleInput, ModuleOutput


class PersonDetectorConfig(ModuleConfig):
    """Config for Person Detector"""
    method: Literal['YOLOX']
    config: Any

class PersonDetectorInput(ModuleInput):
    """Person Detector Input"""
    image: np.ndarray

class PersonDetectorOutput(ModuleOutput):
    """Person Detector Output"""
    bbox: List[List[int]]
    scores: List[float]


@Module.register('person_detector')
class PersonDetector(Module[PersonDetectorInput, PersonDetectorOutput, PersonDetectorConfig]):
    """Person Detector factory"""
    input_types: Type[PersonDetectorInput] = PersonDetectorInput
    output_types: Type[PersonDetectorOutput] = PersonDetectorOutput
    config_type: Type[PersonDetectorConfig] = PersonDetectorConfig

    @classmethod
    def from_config(cls, config: PersonDetectorConfig) -> PersonDetector:
        method_name = config.method
        method = PersonDetector.by_name(method_name)
        config = config.config
        logging.info('loading Person Detector from config: %s', config)
        return method.from_config(config)
