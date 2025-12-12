"""Utility functions for YOLO detection."""

import os
import argparse
from typing import Tuple, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_CONFIDENCE
from .visualizer import DetectionVisualiser


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing:
        - model: Path to YOLO model file
        - source: Input source path or identifier
        - thresh: Confidence threshold
        - resolution: Display resolution string
        - record: Whether to record video
        - output: Output video filename
        - save_predictions: Prediction CSV filename
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                       help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                       required=True)
    parser.add_argument('--source',
                       help='Image source: image file ("test.jpg"), image folder ("test_dir"), '
                            'video file ("testvid.mp4"), or USB camera ("usb0")',
                       required=True)
    parser.add_argument('--thresh',
                       help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                       type=float,
                       default=DEFAULT_CONFIDENCE)
    parser.add_argument('--resolution',
                       help='Resolution in WxH to display inference results at (example: "640x480"), '
                            'otherwise, match source resolution',
                       default=None)
    parser.add_argument('--record',
                       help='Record annotated results from video or webcam. Use --output to specify filename (default: "demo1.avi"). '
                            'Must specify --resolution argument to record.',
                       action='store_true')
    parser.add_argument('--output',
                       help='Output filename for recorded video (default: "demo1.avi")',
                       default='demo1.avi')
    parser.add_argument('--save-predictions',
                       help='Save prediction outputs to a CSV file (example: "predictions.csv")',
                       default=None)
    return parser.parse_args()


def parse_resolution(resolution_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse resolution string to tuple.
    
    Parameters
    ----------
    resolution_str : Optional[str]
        Resolution string in format "WIDTHxHEIGHT" (e.g., "640x480")
        
    Returns
    -------
    Optional[Tuple[int, int]]
        Resolution as (width, height) tuple, or None if input is None
    """
    if resolution_str is None:
        return None
    parts = resolution_str.split('x')
    return (int(parts[0]), int(parts[1]))


def validate_model_path(model_path: str) -> None:
    """
    Validate that model file exists.
    
    Parameters
    ----------
    model_path : str
        Path to the YOLO model file
        
    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the specified path
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model path is invalid or model was not found. '
                              'Make sure the model filename was entered correctly.')


def process_detections(
    detections: any, 
    visualizer: DetectionVisualiser, 
    frame: NDArray[np.uint8]
) -> Tuple[NDArray[np.uint8], int]:
    """
    Process detections and draw on frame.
    
    Parameters
    ----------
    detections : any
        Detection results from YOLO model (ultralytics Boxes object)
    visualizer : DetectionVisualiser
        Visualiser instance for drawing detections
    frame : NDArray[np.uint8]
        Input frame in BGR format
        
    Returns
    -------
    Tuple[NDArray[np.uint8], int]
        Tuple containing:
        - Annotated frame with detections drawn
        - Count of detected objects above confidence threshold
    """
    object_count = 0
    
    for detection in detections:
        # Extract bounding box coordinates
        xyxy_tensor = detection.xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        
        # Get class and confidence
        class_idx = int(detection.cls.item())
        confidence = detection.conf.item()
        
        # Draw detection if confidence is high enough
        if confidence > visualizer.confidence_threshold:
            frame = visualizer.draw_detection(frame, xyxy, class_idx, confidence)
            object_count += 1
    
    return frame, object_count


def handle_keyboard_input(key: int, frame: NDArray[np.uint8]) -> bool:
    """
    Handle keyboard input during inference.
    
    Parameters
    ----------
    key : int
        Key code from cv2.waitKey()
    frame : NDArray[np.uint8]
        Current frame (for saving on 'p' press)
        
    Returns
    -------
    bool
        True to continue inference loop, False to quit
        
    Notes
    -----
    Keyboard shortcuts:
    - 'q' or 'Q': Quit the program
    - 's' or 'S': Pause inference (press any key to resume)
    - 'p' or 'P': Save current frame as 'capture.png'
    """
    if key == ord('q') or key == ord('Q'):
        return False
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()  # Pause
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png', frame)
        print('Frame saved as capture.png')
    return True
