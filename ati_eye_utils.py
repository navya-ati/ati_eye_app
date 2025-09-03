import os
import logging
import json
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt  # for im_show

def load_config():
    """Load configuration from JSON file"""
    with open('eye_config.json') as f:
        config = json.load(f)
    return config

def get_time_str():
    """Return timestamp string for filenames/logging"""
    return time.strftime("%Y-%m-%d-%H-%M-%S")

def setup_logger(log_file, level=logging.INFO):
    """Create and configure logger with file handler"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def create_logger(app_version=""):
    """
    Create log directory and initialize logger.
    Only SSD/out directory is used for logs.
    No project directory for frames/images.
    """
    start_time_str = get_time_str()

    # SSD/out directory for logs
    log_dir = os.path.join("/home/ati/out", start_time_str)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOGGER] Created dir {log_dir}")

    # Log file path
    log_file_path = os.path.join(log_dir, "ati_eye.log")

    # Main logger
    logger = setup_logger(log_file_path)
    logger.info(f"Initializing ati_eye application - v{app_version}!")

    # Return logger and SSD log directory
    return logger, log_dir, None  # project directory removed

def tensor_to_int_float(tens, to_type="int"):
    """Convert tensor values to int or float in a list"""
    response = []
    if to_type == "int":
        for i in tens:
            response.append(int(i))
    elif to_type == "float":
        for i in tens:
            response.append(float(i))
    return response

def process_yolo_result(result):
    """
    Extract classes, bounding boxes, and scores from YOLO results.
    Converts tensors to Python types for easier processing.
    """
    classes, xyxy, scores = [], [], []
    for r in result:
        for clas, xy, score in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf):
            classes.append(clas)
            xyxy.append(xy)  # keep raw tensor
            scores.append(score)

    # Convert all tensors to int/float
    classes = tensor_to_int_float(classes, "int")
    xyxy = [tensor_to_int_float(x, "int") for x in xyxy]
    scores = tensor_to_int_float(scores, "float")
    return classes, xyxy, scores

def get_camera_fov(config):
    """Get horizontal and vertical camera FOV from config"""
    h_fov = config.get("camera_h_fov", None)
    v_fov = config.get("camera_v_fov", None)
    return h_fov, v_fov

def im_show(image):
    """Display image using matplotlib (in-memory only, no saving)"""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
