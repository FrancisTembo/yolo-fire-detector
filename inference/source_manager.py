import os
import glob
from typing import Tuple, Optional, List

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


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
        """
        if self.source_type == 'image':
            self.imgs_list = [self.source]
        elif self.source_type == 'folder':
            filelist = glob.glob(self.source + '/*')
            self.imgs_list = [f for f in filelist if os.path.splitext(f)[1] in IMAGE_EXTENSIONS]
        elif self.source_type in ['video', 'usb']:
            cap_arg = self.source if self.source_type == 'video' else int(self.source[3:])
            self.cap = cv2.VideoCapture(cap_arg)
            if self.resolution:
                self.cap.set(3, self.resolution[0])
                self.cap.set(4, self.resolution[1])
    
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
                print('All images have been processed. Exiting program.')
                return None
            frame = cv2.imread(self.imgs_list[self.img_count])
            self.img_count += 1
            return frame
        elif self.source_type in ['video', 'usb']:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                msg = 'Reached end of the video file.' if self.source_type == 'video' else \
                      'Unable to read frames from the camera. This indicates the camera is disconnected or not working.'
                print(msg + ' Exiting program.')
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
    
    def is_realtime(self) -> bool:
        """
        Check if source is realtime (video/camera).
        
        Returns
        -------
        bool
            True if source is video or USB camera, False otherwise
        """
        return self.source_type in ['video', 'usb']
