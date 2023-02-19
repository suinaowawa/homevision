"""Threading for read in frames from cv2.VideoCapture"""
import copy
import logging
import time
from threading import Lock, Thread

import cv2


class CamLoader:
    """Use threading to capture a frame from camera for faster frame load.
    Recommend for camera or webcam.

    Args:
        camera: (str) Source of camera or video.,
        stream_fps: (int) to desired read in fps.
    """
    def __init__(self, camera: str, stream_fps: str = 15):
        self.stream = camera
        assert self.stream.isOpened(), 'Cannot read camera source!'
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.stopped = False
        self.ret = False
        self.frame = None
        self.ori_frame = None
        self.read_lock = Lock()
        self.frame_duration = 1.0 / stream_fps if stream_fps else None
        self.thread = None


    def get(self, *args):
        """Equivalent to cv2.VideoCapture.get"""
        return self.stream.get(*args)

    def start(self):
        """Start the thread of read in frames"""
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        logging.info('Video reading thread has started!')
        cnt = 0
        while not self.ret:
            time.sleep(0.1)
            cnt += 1
            if cnt > 20:
                self.stop()
                raise TimeoutError('Can not get a frame from camera!!!')
        return self

    def update(self):
        """Update the most current frame"""
        while not self.stopped:
            read_start = time.perf_counter()
            ret, frame = self.stream.read()

            # How much need to sleep in order to match the fps.
            # Useful when reading from file and need to
            # simulate reading from the camera (e.g. to display the stream).
            with self.read_lock:
                if ret:
                    self.ori_frame = frame.copy()

                self.ret, self.frame = ret, copy.deepcopy(frame)
            read_end = time.perf_counter()

            if self.frame_duration:
                sleep_time = self.frame_duration - (read_end - read_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def isOpened(self): #pylint: disable=invalid-name
        """Equivalent to cv2.VideoCapture.isOpened"""
        return self.ret

    def read(self):
        """Equivalent to cv2.VideoCapture.read"""
        return self.ret, self.frame

    def release(self):
        """Equivalent to cv2.VideoCapture.release, stop the thread"""
        if self.stopped:
            return
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

    def __del__(self):
        if self.stream.isOpened():
            self.stream.release()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream.isOpened():
            self.stream.release()
