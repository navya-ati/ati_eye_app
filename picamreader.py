import cv2
import time
from picamera2 import Picamera2
from threading import Thread, Event
import os
import json

# Load configuration from eye_config.json
with open("eye_config.json") as f:
    config = json.load(f)

# Camera settings from config
CAMERA_RESOLUTION = tuple(config.get("camera_resolution", [1920, 1080]))
CAMERA_FPS = config.get("camera_fps", 10)

class PiCamReader(Thread):
    def __init__(self, input_size=CAMERA_RESOLUTION, color_format="RGB888"):
        Thread.__init__(self)
        self._stop_event = Event()
        self.frame = 0
        self.keep_running = True
        self.latest_frame = None
        self.is_new_data = False

        # Camera Field of View (FOV) from config
        self.h_fov = config.get("camera_h_fov", 62.2)
        self.v_fov = config.get("camera_v_fov", 48.8)

        # Initialize PiCamera2
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = input_size
        self.picam2.preview_configuration.main.format = color_format
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()

    def run(self):
        """Main loop for capturing frames"""
        print(f"[PiCamReader] Capturing at {CAMERA_RESOLUTION} @ {CAMERA_FPS} FPS")
        while self.keep_running:
            start_time = time.time()

            # Capture frame
            img = self.picam2.capture_array()
            timestamp = time.time()
            self.frame += 1

            # Store latest frame in RAM only
            self.latest_frame = (img, timestamp, self.frame)
            self.is_new_data = True

            # Maintain target FPS
            elapsed = time.time() - start_time
            time.sleep(max(0, (1.0 / CAMERA_FPS) - elapsed))

        # Stop camera preview
        self.picam2.stop_preview()
        print("[PiCamReader] Thread stopped")

    def get_data(self):
        """Return the latest captured frame"""
        self.is_new_data = False
        return self.latest_frame

    def stop(self):
        """Stop the camera capture loop"""
        self.keep_running = False
        self._stop_event.set()
