import os
import glob
import logging
from typing import Tuple, Optional, List, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


class SourceManager:
    """
    Manages different input sources (image, folder, video, USB camera).
    
    Parameters
    ----------
    source : str
        Input source path or identifier
    resolution : Optional[Tuple[int, int]], optional
        Desired resolution as (width, height), by default None
        
    Attributes
    ----------
    source : str
        Input source path or identifier
    resolution : Optional[Tuple[int, int]]
        Desired resolution as (width, height)
    source_type : str
        Type of source ('image', 'folder', 'video', or 'usb')
    cap : Optional[cv2.VideoCapture]
        Video capture object for video/camera sources
    imgs_list : List[str]
        List of image file paths for image/folder sources
    img_count : int
        Current image index for image/folder sources
    """
    
    def __init__(self, source: str, resolution: Optional[Tuple[int, int]] = None) -> None:
        self.source: str = source
        self.resolution: Optional[Tuple[int, int]] = resolution
        self.source_type: str = self._determine_source_type()
        self.cap: Optional[cv2.VideoCapture] = None
        self.imgs_list: List[str] = []
        self.img_count: int = 0
        self._initialise_source()
    
    def _determine_source_type(self) -> str:
        """
        Determine the type of input source.
        
        Returns
        -------
        str
            Source type ('image', 'folder', 'video', or 'usb')
            
        Raises
        ------
        ValueError
            If the source type cannot be determined or is not supported
        """
        if os.path.isdir(self.source):
            return 'folder'
        elif os.path.isfile(self.source):
            _, ext = os.path.splitext(self.source)
            if ext in IMAGE_EXTENSIONS:
                return 'image'
            elif ext in VIDEO_EXTENSIONS:
                return 'video'
            else:
                raise ValueError(f'File extension {ext} is not supported.')
        elif 'usb' in self.source:
            return 'usb'
        else:
            raise ValueError(f'Input {self.source} is invalid. Please try again.')
    
    def _initialise_source(self) -> None:
        """
        Initialise the appropriate source handler.
        
        Sets up the video capture or image list based on source type.
        
        Raises
        ------
        ValueError
            If USB camera identifier is invalid
        RuntimeError
            If video source fails to open
        """
        if self.source_type == 'image':
            self.imgs_list = [self.source]
        elif self.source_type == 'folder':
            filelist = glob.glob(os.path.join(self.source, '*'))
            self.imgs_list = sorted([f for f in filelist if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
        elif self.source_type in ['video', 'usb']:
            cap_arg: Union[str, int] = self.source
            if self.source_type == 'usb':
                try:
                    cap_arg = int(self.source.replace('usb', ''))
                except ValueError:
                    raise ValueError(f'Invalid USB camera identifier: {self.source}')
            self.cap = cv2.VideoCapture(cap_arg)
            if not self.cap.isOpened():
                raise RuntimeError(f'Failed to open video source: {self.source}')
            if self.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
    
    def read_frame(self) -> Optional[NDArray[np.uint8]]:
        """
        Read the next frame from the source.
        
        Returns
        -------
        Optional[NDArray[np.uint8]]
            Frame as a numpy array in BGR format, or None if no more frames
        """
        if self.source_type in ['image', 'folder']:
            if self.img_count >= len(self.imgs_list):
                logger.info('All images have been processed.')
                return None
            frame = cv2.imread(self.imgs_list[self.img_count])
            if frame is None:
                logger.warning(f'Failed to read image: {self.imgs_list[self.img_count]}')
                self.img_count += 1
                return self.read_frame()  # Try next image
            self.img_count += 1
            return frame
        elif self.source_type in ['video', 'usb']:
            if self.cap is None or not self.cap.isOpened():
                logger.error('Video capture is not initialised or has been released.')
                return None
            ret, frame = self.cap.read()
            if not ret or frame is None:
                msg = 'Reached end of video.' if self.source_type == 'video' else 'Camera read failed.'
                logger.info(msg)
                return None
            return frame
        return None
    
    def release(self) -> None:
        """
        Release resources.
        
        Closes video capture if it was opened.
        """
        if self.cap is not None:
            self.cap.release()
    
    def __enter__(self) -> 'SourceManager':
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        self.release()
    
    @property
    def total_frames(self) -> Optional[int]:
        """
        Return total frame count if known.
        
        Returns
        -------
        Optional[int]
            Total number of frames, or None for live camera sources
        """
        if self.source_type in ['image', 'folder']:
            return len(self.imgs_list)
        elif self.source_type == 'video' and self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return None
    
    def is_realtime(self) -> bool:
        """
        Check if source is realtime (video/camera).
        
        Returns
        -------
        bool
            True if source is video or USB camera, False otherwise
        """
        return self.source_type in ['video', 'usb']
