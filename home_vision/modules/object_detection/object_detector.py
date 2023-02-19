"""Base class for Object Detectors"""
from __future__ import annotations

import logging
from typing import List, Any, Literal, Type
import numpy as np

from home_vision.modules.module_base import Module, ModuleConfig, ModuleInput, ModuleOutput


class ObjectDetectorConfig(ModuleConfig):
    """Config for Object Detector"""
    method: Literal['YOLOV8']
    config: Any

class ObjectDetectorInput(ModuleInput):
    """Object Detector Input"""
    image: np.ndarray

class ObjectDetectorOutput(ModuleOutput):
    """Object Detector Output"""
    bbox: List[List[int]]
    scores: List[float]
    class_names: List[str]


@Module.register('object_detector')
class ObjectDetector(Module[ObjectDetectorInput, ObjectDetectorOutput, ObjectDetectorConfig]):
    """Object Detector factory"""
    input_types: Type[ObjectDetectorInput] = ObjectDetectorInput
    output_types: Type[ObjectDetectorOutput] = ObjectDetectorOutput
    config_type: Type[ObjectDetectorConfig] = ObjectDetectorConfig

    @classmethod
    def from_config(cls, config: ObjectDetectorConfig) -> ObjectDetector:
        method_name = config.method
        method = ObjectDetector.by_name(method_name)
        config = config.config
        logging.info('loading Object Detector from config: %s', config)
        return method.from_config(config)
