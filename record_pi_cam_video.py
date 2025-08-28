import cv2
from picamera2 import Picamera2
from datetime import datetime
import signal
import sys
import os
import time
import json

# Load configuration
with open("eye_config.json", "r") as f:
    config = json.load(f)

camera_resolution = tuple(config.get("camera_resolution", [1920, 1080]))
camera_fps = config.get("camera_fps", 10)

print(f"[CAMERA INFO] Horizontal FOV: {config.get('camera_h_fov')}°, Vertical FOV: {config.get('camera_v_fov')}°")

# Exit handler
def exit_handler(sig, frame):
    print("\n[INFO] Exiting gracefully...")
    try:
        picam2.stop()
        sav_mid.release()
        sav_low.release()
        print("[INFO] Camera stopped and files closed")
    except Exception as e:
        print("[ERROR] During exit:", e)
    sys.exit(0)

signal.signal(signal.SIGINT, exit_handler)
signal.signal(signal.SIGTERM, exit_handler)

# Camera setup
picam2 = Picamera2()
picam2.preview_configuration.main.size = camera_resolution
picam2.preview_configuration.main.format = "RGB888"
picam2.video_configuration.controls.FrameRate = camera_fps
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Video writers
sav_low = cv2.VideoWriter(
    f"{formatted_date}/pi_cam_low.avi",
    cv2.VideoWriter_fourcc(*'MJPG'),
    camera_fps,
    (640, 480)
)

sav_mid = cv2.VideoWriter(
    f"{formatted_date}/pi_cam_mid.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    camera_fps,
    camera_resolution
)

# Recording loop
start_time = time.time()
while True:
    img_high = picam2.capture_array()
    sav_mid.write(img_high)

    img_low = cv2.resize(img_high, (640, 480))
    sav_low.write(img_low)

    print("total_duration =", round(time.time() - start_time, 2), "seconds")

