import cv2
import time
import json
from picamera2 import Picamera2
from threading import Thread, Event
import os

# Load settings from eye_config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "eye_config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Get resolution and FPS from config
CAMERA_RESOLUTION = tuple(config.get("camera_resolution", [640, 480]))
CAMERA_FPS = config.get("camera_fps", 10)


class PiCamReader(Thread):
    def __init__(self, input_size=CAMERA_RESOLUTION, output_size=None, color_format="RGB888", ssd_frame_path="frames"):
        Thread.__init__(self)
        self._stop_event = Event()
        self.frame_id = 0

        # Camera field of view (FOV)
        self.h_fov = config.get("camera_h_fov", 62.2)
        self.v_fov = config.get("camera_v_fov", 48.8)

        # Setup PiCamera2
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = input_size
        self.picam2.preview_configuration.main.format = color_format
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()

        # Set output size (default = input size)
        self.output_size = output_size if output_size else input_size
        self.keep_running = True
        self.is_new_data = False
        self.cam_data = None

        # SSD storage folder
        self.ssd_frame_path = ssd_frame_path
        os.makedirs(self.ssd_frame_path, exist_ok=True)

    def run(self):
        """Keep capturing frames in a loop"""
        print(f"[PiCamReader] Capturing at {CAMERA_RESOLUTION} @ {CAMERA_FPS} FPS")
        while self.keep_running:
            start_time = time.time()

            # Take one picture (frame)
            img = self.picam2.capture_array()
            timestamp = time.time()
            self.frame_id += 1

            # Save frame as image in SSD
            frame_filename = os.path.join(self.ssd_frame_path, f"{self.frame_id}.jpg")
            cv2.imwrite(frame_filename, img)

            # Save frame in memory for other use
            self.cam_data = (img, timestamp, self.frame_id)
            self.is_new_data = True

            # Sleep to maintain correct FPS
            elapsed = time.time() - start_time
            time.sleep(max(0, (1.0 / CAMERA_FPS) - elapsed))

        self.picam2.stop_preview()
        print("PiCamReader thread stopped")

    def get_data(self):
        """Get the latest frame"""
        self.is_new_data = False
        return self.cam_data

    def stop(self):
        """Stop the camera thread"""
        print("Stopping PiCamReader thread")
        self.keep_running = False
        self._stop_event.set()


class VideoRecorder:
    """Saves video in different resolutions"""
    def __init__(self, save_dir, config):
        self.save_dir = save_dir
        self.camera_resolution = tuple(config.get("camera_resolution", [1920, 1080]))
        self.camera_fps = config.get("camera_fps", 10)

        # Writer for mid-resolution AVI
        self.sav_mid_avi = cv2.VideoWriter(
            os.path.join(save_dir, "pi_cam_mid.avi"),
            cv2.VideoWriter_fourcc(*'XVID'),
            self.camera_fps,
            self.camera_resolution
        )

        # Writer for low-resolution AVI
        self.sav_low_avi = cv2.VideoWriter(
            os.path.join(save_dir, "pi_cam_low.avi"),
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.camera_fps,
            (640, 480)
        )

    def write_frame(self, frame):
        """Write both mid and low resolution frames"""
        try:
            # Save original frame
            self.sav_mid_avi.write(frame)

            # Save smaller (640x480) frame
            frame_low = cv2.resize(frame, (640, 480))
            self.sav_low_avi.write(frame_low)
        except Exception as e:
            print("[VideoRecorder ERROR]", e)

    def release(self):
        """Close video files"""
        self.sav_mid_avi.release()
        self.sav_low_avi.release()
