import cv2
import os
from datetime import datetime

class VideoRecorder:
    """
    Saves video in different resolutions.
    Works with frames provided by PiCamReader (or any other frame source).
    """

    def __init__(self, save_dir="/mnt/ssd/videos", config=None):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Use camera resolution & FPS from config
        self.camera_resolution = tuple(config.get("camera_resolution", [1920, 1080]))
        self.camera_fps = config.get("camera_fps", 10)
        self.video_format = config.get("video_format", "mp4")  # e.g., "mp4" or "avi"

        # Generate timestamped filenames to avoid overwrites
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mid_path = os.path.join(self.save_dir, f"pi_cam_mid_{ts}.{self.video_format}")
        self.low_path = os.path.join(self.save_dir, f"pi_cam_low_{ts}.{self.video_format}")

        # Select codec based on format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if self.video_format == "mp4" else cv2.VideoWriter_fourcc(*'XVID')

        # Initialize video writers
        self.sav_mid = cv2.VideoWriter(self.mid_path, fourcc, self.camera_fps, self.camera_resolution)
        self.sav_low = cv2.VideoWriter(self.low_path, fourcc, self.camera_fps, (640, 480))

    def write_frame(self, frame):
        """
        Write a frame to both mid- and low-resolution videos.
        """
        try:
            # Full resolution
            self.sav_mid.write(frame)

            # Low resolution
            frame_low = cv2.resize(frame, (640, 480))
            self.sav_low.write(frame_low)
        except Exception as e:
            print("[VideoRecorder ERROR]", e)

    def release(self):
        """
        Release video files properly.
        """
        if self.sav_mid.isOpened():
            self.sav_mid.release()
        if self.sav_low.isOpened():
            self.sav_low.release()
