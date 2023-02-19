"""HomeVision Object Detection Solution"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from home_vision.modules.module_base import Module, ModuleOutput
from home_vision.modules.object_detection.object_detector import \
    (ObjectDetector,
    ObjectDetectorInput,
    ObjectDetectorConfig)

from .solution_base import Solution, SolutionConfig, SolutionInput

class ObjectDetectionSolutionConfig(SolutionConfig):
    """Config for Object Detection Solution

    Attributes:
        object_detector (ObjectDetectorConfig): Object Detector module config
    """
    object_detector: ObjectDetectorConfig = ObjectDetectorConfig(
        method='YOLOV8', config={"gpu": True}
    )

class ObjectDetectionSolutionOutput(ModuleOutput):
    """Object Detection Solution Output"""
    class_names: List[str]
    bboxes: List[List[int]]
    image: np.ndarray

@Solution.register('object_detection_solution')
@Module.register('object_detection_solution')
class ObjectDetectionSolution(
    Solution[SolutionInput, ObjectDetectionSolutionOutput, ObjectDetectionSolutionConfig]
):
    """HomeVision Solution that detects object in a frame"""
    input_types: SolutionInput = SolutionInput
    output_types: ObjectDetectionSolutionOutput = ObjectDetectionSolutionOutput
    config_type:ObjectDetectionSolutionConfig = ObjectDetectionSolutionConfig
    solution_name = "Object Detection Solution"
    module_name = solution_name

    def __init__(self, object_detector: ObjectDetector):
        self.object_detector = object_detector
        self.cnt = 0

    @classmethod
    def from_config(cls, config: ObjectDetectionSolutionConfig) -> ObjectDetectionSolution:
        object_detector_cls: ObjectDetector = Module.by_name('object_detector')
        object_detector_config = config.object_detector
        object_detector = object_detector_cls.from_config(object_detector_config)
        return cls(object_detector)


    def _process(self, inputs: SolutionInput) -> ObjectDetectionSolutionOutput:
        """Detects all objects in a frame"""
        self.cnt += 1
        image = inputs.image
        raw_image = image.copy()
        object_detector_output = self.object_detector.process(ObjectDetectorInput(image=raw_image))
        object_bbox = object_detector_output.bbox
        object_scores = object_detector_output.scores
        class_names = object_detector_output.class_names

        for k, bbox in enumerate(object_bbox):
            bbox = list(map(int, bbox))
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            cv2.putText(
                image,class_names[k]+'_'+str(object_scores[k]),(xmin, ymin - 2),
                cv2.FONT_HERSHEY_SIMPLEX,0.75,[255, 0, 0],thickness=2
            )
        outputs = ObjectDetectionSolutionOutput(bboxes=object_bbox, image=image, class_names=class_names)
        return outputs
