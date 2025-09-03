import os
import logging
import json
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt  # Used only for im_show()

def load_config():
    """Load configuration values from eye_config.json"""
    with open('eye_config.json') as f:
        config = json.load(f)
    return config

def get_time_str():
    """Generate a timestamp string (used for filenames or logs)"""
    return time.strftime("%Y-%m-%d-%H-%M-%S")

def setup_logger(log_file, level=logging.INFO):
    """Set up a logger that writes messages into a file"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create file handler for logging
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)

    # Define log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Attach handler to logger
    logger.addHandler(fh)
    return logger

def create_logger(app_version=""):
    """
    Create a log directory on SSD and initialize logger.
    - Logs are stored in /home/ati/out/<timestamp>/
    - No separate directory for frames/images.
    """
    start_time_str = get_time_str()

    # Create log directory on SSD
    log_dir = os.path.join("/home/ati/out", start_time_str)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOGGER] Created dir {log_dir}")

    # Path to log file
    log_file_path = os.path.join(log_dir, "ati_eye.log")

    # Set up logger
    logger = setup_logger(log_file_path)
    logger.info(f"Initializing ati_eye application - v{app_version}!")

    # Return logger and directory path
    return logger, log_dir, None  # project directory intentionally removed

def tensor_to_int_float(tens, to_type="int"):
    """Convert tensor values to a list of ints or floats"""
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
    Extract classes, bounding boxes, and confidence scores
    from YOLO detection results.
    Converts tensors → Python int/float for easier handling.
    """
    classes, xyxy, scores = [], [], []
    for r in result:
        for clas, xy, score in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf):
            classes.append(clas)
            xyxy.append(xy)   # keep bounding box tensor for now
            scores.append(score)

    # Convert tensors into lists of normal numbers
    classes = tensor_to_int_float(classes, "int")
    xyxy = [tensor_to_int_float(x, "int") for x in xyxy]
    scores = tensor_to_int_float(scores, "float")
    return classes, xyxy, scores

def get_camera_fov(config):
    """Read camera horizontal/vertical FOV values from config file"""
    h_fov = config.get("camera_h_fov", None)
    v_fov = config.get("camera_v_fov", None)
    return h_fov, v_fov

def im_show(image):
    """Show an image using matplotlib (only displays, does not save)"""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide axes for clean view
    plt.show()
