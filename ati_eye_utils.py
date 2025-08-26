import os
import logging
import json
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np

# -------------------------------
# Load config
# -------------------------------
def load_config():
    with open('eye_config.json') as user_file:
        config = json.load(user_file)
    return config

# -------------------------------
# Get current time string
# -------------------------------
def get_time_str():
    return time.strftime("%Y-%m-%d-%H-%M-%S")

# -------------------------------
# Logger setup
# -------------------------------
def setup_logger(log_file, level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def create_logger(app_version=""):
    start_time_str = get_time_str()
    log_dir = os.path.join("/home/ati/out", start_time_str)
    os.makedirs(log_dir, exist_ok=True)
    print(f"created dir {log_dir}")
    log_file_path = os.path.join(log_dir, "ati_eye.log")
    logger = setup_logger(log_file_path)
    logger.info(f"initializing ati_eye application - v{app_version}!")
    return logger, log_dir

# -------------------------------
# Tensor conversion
# -------------------------------
def tensor_to_int_float(tens, to_type="int"):
    response = []
    if to_type == "int":
        for i in tens:
            response.append(int(i))
    elif to_type == "float":
        for i in tens:
            response.append(float(i))
    return response

# -------------------------------
# YOLO result processing
# -------------------------------
def process_yolo_result(result):
    classes = []
    xyxy = []
    scores = []
    for r in result:
        for clas, xy, score in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf):
            classes.append(clas)
            xy_int = tensor_to_int_float(xy, "int")
            xyxy.append(xy_int)
            scores.append(score)
    scores = tensor_to_int_float(scores, "float")
    classes = tensor_to_int_float(classes, "int")
    return classes, xyxy, scores

# -------------------------------
# Image display
# -------------------------------
def im_show(image):
    plt.figure()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

# -------------------------------
# Camera capture settings
# -------------------------------
def apply_camera_settings(cap, config):
    resolution = config.get("camera_resolution", [640, 480])
    fps = config.get("camera_fps", 30)

    frame_width, frame_height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print(f"[CAMERA CONFIG] Set resolution: {frame_width}x{frame_height} at {fps} FPS")

# -------------------------------
# Get camera FOV
# -------------------------------
def get_camera_fov(config):
    h_fov = config.get("camera_h_fov", None)
    v_fov = config.get("camera_v_fov", None)
    return h_fov, v_fov

