"""Object detection method YOLOv7 ONNX"""
# https://github.com/WongKinYiu/yolov7
from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np
import onnxruntime

from home_vision.modules.module_base import BaseConfig
from home_vision.modules.object_detection.methods.yolov8.utils import (
    nms, xywh2xyxy)
from home_vision.modules.object_detection.object_detector import (
    ObjectDetector, ObjectDetectorConfig, ObjectDetectorInput,
    ObjectDetectorOutput)

ROOT = os.path.dirname(os.path.abspath(__file__))

@ObjectDetectorConfig.register('YOLOV8')
class YOLOV8Config(BaseConfig):
    """YOLOv8 object detector config

    Attributes:
        gpu (bool): use gpu or cpu to inference
        conf_threshold (float): confidence threshold of detection
        nms_threshold (float): non maximum supression threshold of detection
    """
    gpu: bool
    conf_threshold: Optional[float] = 0.5
    nms_threshold: Optional[float] = 0.5

@ObjectDetector.register('YOLOV8')
class YOLOV8(ObjectDetector):
    """YOLOv8 onnx object detector"""
    config_type = YOLOV8Config
    module_name = 'YOLOV8 onnx object detector'
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        conf_threshold: float,
        nms_threshold: float
    ):
        self.session = session
        self.conf_threshold = conf_threshold
        self.iou_threshold = nms_threshold
        self.get_input_details()
        self.get_output_details()

    @classmethod
    def from_config(cls, config: YOLOV8Config) -> YOLOV8:
        model_path = "/home/yue/job/homevision/models/object_detection/yolov8n.onnx"
        if config.gpu:
            session = onnxruntime.InferenceSession(
                model_path, providers=['CUDAExecutionProvider']
            )
        else:
            session = onnxruntime.InferenceSession(
                model_path, providers=['CPUExecutionProvider']
            )
        return cls(session, config.conf_threshold, config.nms_threshold)


    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return np.array([]), np.array([]), []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)
        class_ids = class_ids[indices]
        class_names = [self.classes[class_id] for class_id in class_ids]

        return boxes[indices], scores[indices], class_names

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def _process(self, inputs: ObjectDetectorInput) -> ObjectDetectorOutput:
        image = inputs.image
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_names = self.process_output(outputs)
        output = ObjectDetectorOutput(bbox=self.boxes.tolist(), scores=self.scores.tolist(), class_names=self.class_names)

        return output
