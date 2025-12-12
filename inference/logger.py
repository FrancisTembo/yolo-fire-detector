"""Logging and metrics tracking for YOLO detection."""

import csv
from typing import Dict, List, Any
import numpy as np

from .config import FPS_BUFFER_LENGTH


class FPSCalculator:
    """
    Calculates and tracks FPS using a rolling average.
    
    Parameters
    ----------
    buffer_length : int, optional
        Number of frames to average over, by default FPS_BUFFER_LENGTH
        
    Attributes
    ----------
    buffer_length : int
        Number of frames to average over
    frame_rate_buffer : List[float]
        Buffer storing recent frame rates
    """
    
    def __init__(self, buffer_length: int = FPS_BUFFER_LENGTH) -> None:
        self.buffer_length: int = buffer_length
        self.frame_rate_buffer: List[float] = []
    
    def update(self, frame_time: float) -> float:
        """
        Update FPS calculation with new frame time.
        
        Parameters
        ----------
        frame_time : float
            Time taken to process the frame in seconds
            
        Returns
        -------
        float
            Average FPS over the buffer length
        """
        frame_rate = 1.0 / frame_time if frame_time > 0 else 0
        
        if len(self.frame_rate_buffer) >= self.buffer_length:
            self.frame_rate_buffer.pop(0)
        self.frame_rate_buffer.append(frame_rate)
        
        return np.mean(self.frame_rate_buffer) # type: ignore


class PredictionLogger:
    """
    Logs prediction results to a CSV file.
    
    Parameters
    ----------
    filename : str
        Output CSV filename
    labels : Dict[int, str]
        Dictionary mapping class IDs to class names
        
    Attributes
    ----------
    filename : str
        Output CSV filename
    labels : Dict[int, str]
        Dictionary mapping class IDs to class names
    file : file object
        Open file handle for CSV output
    writer : csv.writer
        CSV writer object
    frame_count : int
        Current frame number being processed
    """
    
    def __init__(self, filename: str, labels: Dict[int, str]) -> None:
        self.filename: str = filename
        self.labels: Dict[int, str] = labels
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['frame', 'class_id', 'class_name', 'confidence', 
                             'xmin', 'ymin', 'xmax', 'ymax'])
        self.frame_count: int = 0
    
    def log_detections(self, detections: Any, confidence_threshold: float) -> None:
        """
        Log all detections for current frame.
        
        Parameters
        ----------
        detections : Any
            Detection results from YOLO model (ultralytics Boxes object)
        confidence_threshold : float
            Minimum confidence threshold for logging detections
        """
        for detection in detections:
            # Extract bounding box coordinates
            xyxy_tensor = detection.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            
            # Get class and confidence
            class_idx = int(detection.cls.item())
            confidence = detection.conf.item()
            
            # Only log if confidence is above threshold
            if confidence > confidence_threshold:
                class_name = self.labels[class_idx]
                self.writer.writerow([self.frame_count, class_idx, class_name, 
                                    f'{confidence:.4f}', xmin, ymin, xmax, ymax])
        
        self.frame_count += 1
    
    def close(self) -> None:
        """
        Close the CSV file.
        
        Flushes and closes the output file handle.
        """
        self.file.close()
