import os
import logging
import json
import time
import cv2
import numpy as np

# Load the configuration from eye_config.json file
def load_config():
    with open('eye_config.json') as f:
        config = json.load(f)
    return config

# Get current time as a string in format YYYY-MM-DD-HH-MM-SS
def get_time_str():
    return time.strftime("%Y-%m-%d-%H-%M-%S")

# Setup a logger that writes logs to a file
def setup_logger(log_file, level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create file handler to save logs
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)

    # Define log format (time, name, level, message)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Attach file handler to logger
    logger.addHandler(fh)
    return logger

# Create a logger and directory for logs
def create_logger(app_version=""):
    start_time_str = get_time_str()

    # Create directory with timestamp name inside /home/ati/out
    log_dir = os.path.join("/home/ati/out", start_time_str)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOGGER] Created dir {log_dir}")

    # Path of log file
    log_file_path = os.path.join(log_dir, "ati_eye.log")

    # Setup logger
    logger = setup_logger(log_file_path)

    # First log entry
    logger.info(f"Initializing ati_eye application - v{app_version}!")
    return logger, log_dir

# Convert tensor values (from YOLO output) to int or float
def tensor_to_int_float(tens, to_type="int"):
    response = []
    if to_type == "int":
        response = [int(i) for i in tens]
    elif to_type == "float":
        response = [float(i) for i in tens]
    return response

# Process YOLO model results: extract classes, bounding boxes, and scores
def process_yolo_result(result):
    classes, xyxy, scores = [], [], []
    for r in result:
        # Extract class, coordinates (x1,y1,x2,y2), and confidence score
        for clas, xy, score in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf):
            classes.append(clas)
            xyxy.append(tensor_to_int_float(xy, "int"))
            scores.append(float(score))

    # Convert class numbers to integers
    classes = tensor_to_int_float(classes, "int")
    return classes, xyxy, scores

# Get camera field of view from config file
def get_camera_fov(config):
    h_fov = config.get("camera_h_fov", None)
    v_fov = config.get("camera_v_fov", None)
    return h_fov, v_fov
