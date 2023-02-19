"""Import all HomeVision Solutions"""
from home_vision.modules.module_base import Module
from .person_detection import PersonDetectionSolution
from .object_detection import ObjectDetectionSolution
from .raw_stream_solution import RawStreamSolution
from .raw_datachannel_solution import RawDatachannelSolution