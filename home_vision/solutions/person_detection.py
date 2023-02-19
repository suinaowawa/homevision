"""HomeVision Person Detection Solution"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from home_vision.modules.module_base import Module, ModuleOutput
from home_vision.modules.person_detection.person_detector import \
    (PersonDetector,
    PersonDetectorInput,
    PersonDetectorConfig)

from .solution_base import Solution, SolutionConfig, SolutionInput

class PersonDetectionSolutionConfig(SolutionConfig):
    """Config for Person Detection Solution

    Attributes:
        person_detector (PersonDetectorConfig): Person Detector module config
    """
    person_detector: PersonDetectorConfig = PersonDetectorConfig(
        method='YOLOX', config={"gpu": True}
    )

class PersonDetectionSolutionOutput(ModuleOutput):
    """Person Detection Solution Output"""
    bboxes: List[List[int]]
    image: np.ndarray

@Solution.register('person_detection_solution')
@Module.register('person_detection_solution')
class PersonDetectionSolution(
    Solution[SolutionInput, PersonDetectionSolutionOutput, PersonDetectionSolutionConfig]
):
    """HomeVision Solution that detects person in a frame"""
    input_types: SolutionInput = SolutionInput
    output_types: PersonDetectionSolutionOutput = PersonDetectionSolutionOutput
    config_type:PersonDetectionSolutionConfig = PersonDetectionSolutionConfig
    solution_name = "Person Detection Solution"
    module_name = solution_name

    def __init__(self, person_detector: PersonDetector):
        self.person_detector = person_detector
        self.cnt = 0

    @classmethod
    def from_config(cls, config: PersonDetectionSolutionConfig) -> PersonDetectionSolution:
        person_detector_cls: PersonDetector = Module.by_name('person_detector')
        person_detector_config = config.person_detector
        person_detector = person_detector_cls.from_config(person_detector_config)
        return cls(person_detector)


    def _process(self, inputs: SolutionInput) -> PersonDetectionSolutionOutput:
        """Detects all people in a frame"""
        self.cnt += 1
        image = inputs.image
        raw_image = image.copy()
        person_detector_output = self.person_detector.process(PersonDetectorInput(image=raw_image))
        person_bbox = person_detector_output.bbox
        person_scores = person_detector_output.scores
        if len(person_bbox) == 0:
            outputs = PersonDetectionSolutionOutput(image=raw_image, bboxes=[])
            return outputs
        for k, bbox in enumerate(person_bbox):
            bbox = list(map(int, bbox))
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            cv2.putText(
                image,str(person_scores[k]),(xmin, ymin - 2),
                cv2.FONT_HERSHEY_SIMPLEX,0.75,[255, 0, 0],thickness=2
            )
        outputs = PersonDetectionSolutionOutput(bboxes=person_bbox, image=image)
        return outputs
