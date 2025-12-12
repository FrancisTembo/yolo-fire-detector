from typing import Tuple, Optional, List
from dataclasses import dataclass


IMAGE_EXTENSIONS: List[str] = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
VIDEO_EXTENSIONS: List[str] = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

BBOX_COLOURS: List[Tuple[int, int, int]] = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
    (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
]

DEFAULT_CONFIDENCE: float = 0.5
FPS_BUFFER_LENGTH: int = 200
DEFAULT_RECORD_FPS: int = 30


@dataclass
class DetectionConfig:
    """
    Configuration for detection parameters.
    
    Attributes
    ----------
    model_path : str
        Path to the YOLO model file
    source : str
        Input source (image file, folder, video file, or USB camera)
    confidence_threshold : float
        Minimum confidence threshold for displaying detected objects
    resolution : Optional[Tuple[int, int]]
        Display resolution as (width, height), or None to match source resolution
    record : bool
        Whether to record annotated video output
    output : str
        Output filename for recorded video
    save_predictions : Optional[str]
        Path to save prediction CSV file, or None to disable logging
    """
    model_path: str
    source: str
    confidence_threshold: float
    resolution: Optional[Tuple[int, int]]
    record: bool
    output: str
    save_predictions: Optional[str]
