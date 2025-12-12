"""YOLO detection script with modular architecture."""

import sys
import time
from typing import Optional

import cv2
from ultralytics import YOLO

from inference.config import DetectionConfig, DEFAULT_RECORD_FPS
from inference.source_manager import SourceManager
from inference.visualizer import DetectionVisualiser, VideoRecorder
from inference.logger import FPSCalculator, PredictionLogger
from inference.utils import (
    parse_arguments,
    parse_resolution,
    validate_model_path,
    process_detections,
    handle_keyboard_input
)


def run_inference_loop(
    model: YOLO, 
    source_manager: SourceManager, 
    visualizer: DetectionVisualiser, 
    config: DetectionConfig,
    recorder: Optional[VideoRecorder] = None,
    prediction_logger: Optional[PredictionLogger] = None
) -> float:
    """
    Main inference loop for processing frames and running detection.
    
    Parameters
    ----------
    model : YOLO
        Loaded YOLO model for object detection
    source_manager : SourceManager
        Manager for handling input sources
    visualizer : DetectionVisualiser
        Visualiser for drawing detections on frames
    config : DetectionConfig
        Configuration parameters for detection
    recorder : Optional[VideoRecorder], optional
        Video recorder for saving annotated output, by default None
    prediction_logger : Optional[PredictionLogger], optional
        Logger for saving predictions to CSV, by default None
        
    Returns
    -------
    float
        Average FPS across all processed frames
    """
    fps_calculator = FPSCalculator()
    avg_fps = 0.0
    
    while True:
        t_start = time.perf_counter()
        
        # Read frame
        frame = source_manager.read_frame()
        if frame is None:
            break
        
        # Resize if needed
        if config.resolution:
            frame = cv2.resize(frame, config.resolution)
        
        # Run inference
        results = model(frame, verbose=False)
        detections = results[0].boxes
        
        # Log predictions if enabled
        if prediction_logger:
            prediction_logger.log_detections(detections, config.confidence_threshold)
        
        # Process detections
        frame, object_count = process_detections(detections, visualizer, frame)
        
        # Calculate FPS
        t_stop = time.perf_counter()
        avg_fps = fps_calculator.update(t_stop - t_start)
        
        # Draw stats
        frame = visualizer.draw_stats(frame, avg_fps, object_count, 
                                      show_fps=source_manager.is_realtime())
        
        # Display and record
        cv2.imshow('YOLO detection results', frame)
        if recorder:
            recorder.write(frame)
        
        # Handle keyboard input
        wait_time = 0 if source_manager.source_type in ['image', 'folder'] else 5
        key = cv2.waitKey(wait_time)
        if not handle_keyboard_input(key, frame):
            break
    
    return avg_fps


def main() -> None:
    """
    Main entry point for YOLO detection script.
    
    Parses command line arguments, initialises components, runs inference,
    and handles cleanup.
    
    Raises
    ------
    FileNotFoundError
        If the model file is not found
    ValueError
        If recording is requested with invalid configuration
    Exception
        For any other errors during execution
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate model
        validate_model_path(args.model)
        
        # Parse resolution
        resolution = parse_resolution(args.resolution)
        
        # Create configuration
        config = DetectionConfig(
            model_path=args.model,
            source=args.source,
            confidence_threshold=args.thresh,
            resolution=resolution,
            record=args.record,
            output=args.output,
            save_predictions=args.save_predictions
        )
        
        # Load model
        print(f'Loading model from {config.model_path}...')
        model = YOLO(config.model_path, task='detect')
        labels = model.names
        
        # Initialise source manager
        source_manager = SourceManager(config.source, config.resolution)
        
        # Validate recording setup
        recorder = None
        if config.record:
            if source_manager.source_type not in ['video', 'usb']:
                raise ValueError('Recording only works for video and camera sources.')
            if not config.resolution:
                raise ValueError('Please specify resolution to record video at.')
            recorder = VideoRecorder(config.output, DEFAULT_RECORD_FPS, config.resolution)
            print(f'Annotated video will be saved to {config.output}')
        
        # Initialise visualiser
        visualizer = DetectionVisualiser(labels, config.confidence_threshold)
        
        # Initialise prediction logger if enabled
        prediction_logger = None
        if config.save_predictions:
            prediction_logger = PredictionLogger(config.save_predictions, labels)
            print(f'Predictions will be saved to {config.save_predictions}')
        
        # Run inference
        print('Starting inference...')
        avg_fps = run_inference_loop(model, source_manager, visualizer, config, recorder, prediction_logger)
        
        # Cleanup
        print(f'Average pipeline FPS: {avg_fps:.2f}')
        source_manager.release()
        if recorder:
            recorder.release()
            print(f'Annotated video saved to {config.output}')
        if prediction_logger:
            prediction_logger.close()
            print(f'Predictions saved to {config.save_predictions}')
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f'ERROR: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()


