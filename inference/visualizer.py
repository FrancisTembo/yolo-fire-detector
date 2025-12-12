"""Visualization components for YOLO detection results."""

from typing import Tuple, Dict

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import BBOX_COLOURS, DEFAULT_CONFIDENCE, DEFAULT_RECORD_FPS


class DetectionVisualiser:
    """
    Handles visualization of detection results.
    
    Parameters
    ----------
    labels : Dict[int, str]
        Dictionary mapping class IDs to class names
    confidence_threshold : float, optional
        Minimum confidence threshold for displaying detections, by default DEFAULT_CONFIDENCE
        
    Attributes
    ----------
    labels : Dict[int, str]
        Dictionary mapping class IDs to class names
    confidence_threshold : float
        Minimum confidence threshold for displaying detections
    """
    
    def __init__(self, labels: Dict[int, str], confidence_threshold: float = DEFAULT_CONFIDENCE) -> None:
        self.labels: Dict[int, str] = labels
        self.confidence_threshold: float = confidence_threshold
    
    def draw_detection(
        self, 
        frame: NDArray[np.uint8], 
        bbox: NDArray[np.float64], 
        class_idx: int, 
        confidence: float
    ) -> NDArray[np.uint8]:
        """
        Draw bounding box and label on frame.
        
        Parameters
        ----------
        frame : NDArray[np.uint8]
            Input frame in BGR format
        bbox : NDArray[np.float64]
            Bounding box coordinates as [xmin, ymin, xmax, ymax]
        class_idx : int
            Class ID of the detected object
        confidence : float
            Confidence score of the detection
            
        Returns
        -------
        NDArray[np.uint8]
            Frame with detection drawn (if confidence > threshold)
        """
        if confidence <= self.confidence_threshold:
            return frame
        
        xmin, ymin, xmax, ymax = bbox.astype(int)
        class_name = self.labels[class_idx]
        color = BBOX_COLOURS[class_idx % len(BBOX_COLOURS)]
        
        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw label with background
        label = f'{class_name}: {int(confidence * 100)}%'
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                     (xmin + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def draw_stats(
        self, 
        frame: NDArray[np.uint8], 
        fps: float, 
        object_count: int, 
        show_fps: bool = True
    ) -> NDArray[np.uint8]:
        """
        Draw FPS and object count on frame.
        
        Parameters
        ----------
        frame : NDArray[np.uint8]
            Input frame in BGR format
        fps : float
            Current frames per second
        object_count : int
            Number of detected objects in the frame
        show_fps : bool, optional
            Whether to display FPS, by default True
            
        Returns
        -------
        NDArray[np.uint8]
            Frame with stats drawn
        """
        if show_fps:
            cv2.putText(frame, f'FPS: {fps:0.2f}', (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Number of objects: {object_count}', (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame


class VideoRecorder:
    """
    Manages video recording.
    
    Parameters
    ----------
    filename : str
        Output video filename
    fps : int
        Frames per second for output video
    resolution : Tuple[int, int]
        Video resolution as (width, height)
        
    Attributes
    ----------
    filename : str
        Output video filename
    writer : cv2.VideoWriter
        OpenCV video writer object
    """
    
    def __init__(self, filename: str, fps: int, resolution: Tuple[int, int]) -> None:
        self.filename: str = filename
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer: cv2.VideoWriter = cv2.VideoWriter(filename, fourcc, fps, resolution)
    
    def write(self, frame: NDArray[np.uint8]) -> None:
        """
        Write frame to video file.
        
        Parameters
        ----------
        frame : NDArray[np.uint8]
            Frame to write in BGR format
        """
        self.writer.write(frame)
    
    def release(self) -> None:
        """
        Release video writer.
        
        Closes the output video file.
        """
        self.writer.release()
