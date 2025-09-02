import os
import logging
import json
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt  # for im_show

def load_config():
    # Load configuration from JSON file
    with open('eye_config.json') as f:
        config = json.load(f)
    return config

def get_time_str():
    # Return timestamp string for filenames/logging
    return time.strftime("%Y-%m-%d-%H-%M-%S")

def setup_logger(log_file, level=logging.INFO):
    # Create and configure logger with file handler
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def create_logger(app_version=""):
    # Create log directory and initialize logger
    start_time_str = get_time_str()
    log_dir = os.path.join("/home/ati/out", start_time_str)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[LOGGER] Created dir {log_dir}")
    log_file_path = os.path.join(log_dir, "ati_eye.log")
    logger = setup_logger(log_file_path)
    logger.info(f"Initializing ati_eye application - v{app_version}!")
    return logger, log_dir

def tensor_to_int_float(tens, to_type="int"):
    # Verbose loop to convert tensor values to int or float
    response = []
    if to_type == "int":
        for i in tens:
            response.append(int(i))
    elif to_type == "float":
        for i in tens:
            response.append(float(i))
    return response

def process_yolo_result(result):
    # Extract classes, boxes, and scores from YOLO results
    classes, xyxy, scores = [], [], []
    for r in result:
        for clas, xy, score in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf):
            classes.append(clas)
            xyxy.append(xy)     # keep raw tensor for now
            scores.append(score)

    # Convert tensors at the end
    classes = tensor_to_int_float(classes, "int")
    xyxy = [tensor_to_int_float(x, "int") for x in xyxy]
    scores = tensor_to_int_float(scores, "float")
    return classes, xyxy, scores

def get_camera_fov(config):
    # Get horizontal and vertical FOV from config
    h_fov = config.get("camera_h_fov", None)
    v_fov = config.get("camera_v_fov", None)
    return h_fov, v_fov

def im_show(image):
    # Display image using matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
