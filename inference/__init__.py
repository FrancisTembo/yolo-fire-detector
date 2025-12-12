"""YOLO Fire Detector Inference Package."""

from .config import DetectionConfig, DEFAULT_RECORD_FPS, DEFAULT_CONFIDENCE
from .source_manager import SourceManager
from .visualizer import DetectionVisualiser, VideoRecorder
from .logger import FPSCalculator, PredictionLogger
from .utils import (
    parse_arguments,
    parse_resolution,
    validate_model_path,
    process_detections,
    handle_keyboard_input
)

__all__ = [
    'DetectionConfig',
    'DEFAULT_RECORD_FPS',
    'DEFAULT_CONFIDENCE',
    'SourceManager',
    'DetectionVisualiser',
    'VideoRecorder',
    'FPSCalculator',
    'PredictionLogger',
    'parse_arguments',
    'parse_resolution',
    'validate_model_path',
    'process_detections',
    'handle_keyboard_input',
]
