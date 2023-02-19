"""Test HomeVision solutions loading and configs"""
import pytest
from home_vision.solutions.solution_base import Solution

from home_vision.utils.utils import load_solution_config_from_file, load_solution_from_dict


@pytest.mark.parametrize(
    "solution_name, solution_config, error_info",
    [
        ('person_detection_solution', {'wrong_key': 'wrong_val'}, 'validation error'),
        ('not_exist_solution',{}, 'not been registered'),
        (
            'object_detection_solution',
            {'object_detector':{'method': 'not exist', 'config':{}}},
            'not been registered'
        ),
        ('object_detection_solution', {'object_detector':{}}, 'define method'),
        ('object_detection_solution', {'object_detector':{'method': 'SCRFD_onnx'}}, 'define config'),
        (
            'object_detection_solution',
            {'object_detector':{'method': 'YOLOV8', 'config':{}}},
            'validation error'
        ),
        ('raw_stream_solution', {}, ''),
        ('object_detection_solution', {}, ''),
        ('person_detection_solution', {}, ''),
        ('raw_datachannel_solution', {}, '')
    ]
)
def test_load_solution(solution_name, solution_config, error_info):
    """Test solutions loading"""
    if error_info != '':
        with pytest.raises(Exception) as exc:
            load_solution_from_dict(solution_name, solution_config)
        assert error_info in str(exc.value)
    else:
        load_solution_from_dict(solution_name, solution_config)


@pytest.mark.parametrize(
    "solution_name, solution_config_path",
    [
        ('object_detection_solution', 'tests/default_object_detection.yaml'),
        ('person_detection_solution', 'tests/default_person_detection.yaml'),
    ]
)
@pytest.mark.pipeline
def test_default_solution_config(solution_name, solution_config_path):
    """Test default solution config is set right"""
    solution_config = load_solution_config_from_file(solution_name, solution_config_path)
    default_config = Solution.by_name(solution_name).config_type()
    assert solution_config == default_config

@pytest.mark.pipeline
def test_available_solution():
    """Test desired solutions are correctly imported"""
    need_solutions = [
        'raw_stream_solution', 'raw_datachannel_solution',
        'object_detection_solution', 'person_detection_solution'
    ]
    available_solutions = Solution.list_available()
    intersect_solutions = set(need_solutions).intersection(set(available_solutions))
    assert intersect_solutions == set(need_solutions)
