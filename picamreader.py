import cv2
import time
import json
from picamera2 import Picamera2
from threading import Thread, Event
import os

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "eye_config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Camera settings from config
CAMERA_RESOLUTION = tuple(config.get("camera_resolution", [640, 480]))
CAMERA_FPS = config.get("camera_fps", 10)


class PiCamReader(Thread):
    def __init__(self, input_size=CAMERA_RESOLUTION, output_size=None, color_format="RGB888"):
        Thread.__init__(self)
        self._stop_event = Event()
        self.frame = 0

        # Camera FOV
        self.h_fov = config.get("camera_h_fov", 62.2)
        self.v_fov = config.get("camera_v_fov", 48.8)
        self.h_angle_per_pixel = self.h_fov / CAMERA_RESOLUTION[0]
        self.v_angle_per_pixel = self.v_fov / CAMERA_RESOLUTION[1]

        # Initialize PiCamera2
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = input_size
        self.picam2.preview_configuration.main.format = color_format
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()

        self.output_size = output_size if output_size else input_size
        self.cam_data = None
        self.keep_running = True
        self.is_new_data = False

    def run(self):
        print(f"Starting PiCamReader {CAMERA_RESOLUTION} @ {CAMERA_FPS} FPS")
        while self.keep_running:
            start_time = time.time()

            # Capture frame
            img = self.picam2.capture_array()
            if self.output_size != self.picam2.preview_configuration.main.size:
                img = cv2.resize(img, self.output_size)

            timestamp = time.time()
            self.frame += 1
            self.cam_data = (img, timestamp, self.frame)
            self.is_new_data = True

            # Maintain FPS timing
            elapsed = time.time() - start_time
            time.sleep(max(0, (1.0 / CAMERA_FPS) - elapsed))

        self.picam2.stop_preview()
        print("Killing PiCamReader thread")

    def get_data(self):
        self.is_new_data = False
        return self.cam_data

    def stop(self):
        print("Stopping PiCamReader thread")
        self.keep_running = False
        self._stop_event.set()

