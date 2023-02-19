"""Person detection method YOLOX ONNX"""
# https://github.com/ifzhang/ByteTrack
from __future__ import annotations

import logging
import os
import time
from typing import Literal, Optional

import cv2
import numpy as np
import onnxruntime
from home_vision.modules.module_base import BaseConfig
from home_vision.modules.person_detection.methods.yolox.utils import (
    demo_postprocess, multiclass_nms)
from home_vision.modules.person_detection.person_detector import (
    PersonDetector, PersonDetectorConfig, PersonDetectorInput, PersonDetectorOutput)

ROOT = os.path.dirname(os.path.abspath(__file__))

@PersonDetectorConfig.register('YOLOX')
class YOLOXConfig(BaseConfig):
    """YOLOX onnx person detector config

    Attributes:
        gpu (bool): use gpu or cpu to inference
        model_type (str): type of model will be used
        conf_threshold (float): confidence threshold for detection
        nms_threshold (float): non-maximum supression threshold for detection
    """
    gpu: bool
    model_type: Optional[str] = 'tiny'
    conf_threshold: Optional[float] = 0.5
    nms_threshold: Optional[float] = 0.5

@PersonDetector.register('YOLOX')
class YOLOX(PersonDetector):
    """YOLOX onnx person detector"""
    config_type = YOLOXConfig
    module_name = 'YOLOX person detector'

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        model_type: str,
        conf_threshold: float,
        nms_threshold: float
        ):
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.session = session
        if model_type in ['m', 'l']:
            self.input_shape = (800, 1440)
        else:
            self.input_shape = (608, 1088)
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold

    @classmethod
    def from_config(cls, config: YOLOXConfig) -> YOLOX:
        model_path = ROOT + ("/../../../../../models/"
        f"person_detection/yolox_{config.model_type}.onnx")
        if config.gpu:
            session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        else:
            session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return cls(session, config.model_type, config.conf_threshold, config.nms_threshold)

    def _process(self, inputs: PersonDetectorInput) -> PersonDetectorOutput:
        image = inputs.image
        blob = cv2.dnn.blobFromImage( #pylint: disable=no-member
            image, (1.0 / 255)/0.225, (self.input_shape[1], self.input_shape[0]),
            self.rgb_means, swapRB=True
        )
        ratio_w = self.input_shape[1] / image.shape[1]
        ratio_h = self.input_shape[0] / image.shape[0]
        ort_inputs = {self.session.get_inputs()[0].name: blob}
        time_s = time.perf_counter()
        output = self.session.run(None, ort_inputs)
        time_e = time.perf_counter()
        logging.debug("-yolox forward time: %s", time_e - time_s)
        predictions = demo_postprocess(output[0], self.input_shape, p6=False)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2]/2.) / ratio_w
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3]/2.) / ratio_h
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2]/2.) / ratio_w
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3]/2.) / ratio_h
        dets = multiclass_nms(
            boxes_xyxy, scores, nms_thr=self.nms_threshold, score_thr=self.conf_threshold
        )

        if dets is not None:
            dets = dets[:, :-1]
            scores = dets[:, 4].tolist()
            bboxes = dets[:, :4].astype(int).tolist()
        else:
            scores = []
            bboxes = []

        detector_output = PersonDetectorOutput(bbox=bboxes, scores=scores)
        return detector_output
