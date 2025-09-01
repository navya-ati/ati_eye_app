import cv2
import os
import time
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO
from picamreader import PiCamReader
import ati_eye_utils as au
from datetime import datetime
import signal
import sys
from record_pi_cam_video import VideoRecorder  # <-- video writer moved here

# Application version
APP_VERSION = "0.0 - 18th April"

# Data structure for storing YOLO results
@dataclass
class YoloResult:
    classes: np.ndarray       # detected class IDs
    xyxy: np.ndarray          # bounding box coordinates [x1,y1,x2,y2]
    scores: np.ndarray        # confidence scores
    input_image: np.ndarray   # original image frame
    detected_object: bool     # True if any object detected


class AtiEyeApp:
    def __init__(self):
        """Initialize the Ati-Eye application"""
        # Load config + logger
        self.config = au.load_config()
        self.logger, self.log_dir = au.create_logger(APP_VERSION)

        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        self.threshold = 0.25   # detection confidence threshold

        # Camera properties (FOV = field of view)
        self.camera_h_fov, self.camera_v_fov = au.get_camera_fov(self.config)
        self.logger.info(f"Camera FOV - H: {self.camera_h_fov}°, V: {self.camera_v_fov}°")

        # Create folder on SSD for storing videos/frames
        base_path = "/mnt/ssd/videos"
        os.makedirs(base_path, exist_ok=True)
        now = datetime.now()
        self.data_folder = os.path.join(base_path, "data_collection_" + now.strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(self.data_folder, exist_ok=True)
        print("[INFO] Videos and frames will be saved in:", self.data_folder)

        # Start Pi camera capture thread
        self.cam_capture = PiCamReader(save_dir=self.data_folder)
        self.cam_capture.start()
        time.sleep(1)  # allow camera to warm up
        self.logger.info("PiCamReader initialized")

        # Initialize video recorder (separate class)
        self.recorder = VideoRecorder(self.data_folder, self.config)

        # Graceful exit signals
        signal.signal(signal.SIGINT, self._exit_handler)
        signal.signal(signal.SIGTERM, self._exit_handler)

        # Inference history setup
        self.inf_window = self.config.get("inference_window", 1)  # sliding window size
        self.inf_hist = np.zeros(self.inf_window)                 # inference history buffer
        self.min_apply_score = self.config.get("min_apply_score", 1)

        print(f"AtiEye:v{APP_VERSION} is online!")

    # Handle Ctrl+C or kill signal
    def _exit_handler(self, sig, frame):
        print("\n[INFO] Exiting gracefully...")
        try:
            self.cam_capture.stop()   # stop camera
            self.recorder.release()   # release video writer
            cv2.destroyAllWindows()
        except Exception as e:
            print("[ERROR] During exit:", e)
        sys.exit(0)

    # Run YOLOv8 inference on a frame
    def run_inference(self, image) -> YoloResult:
        input_image = image.copy()

        # Run detection
        result = self.model.predict(input_image, conf=self.threshold)
        classes, xyxy, scores = au.process_yolo_result(result)
        detected_object = bool(len(classes))  # True if anything detected

        # Draw bounding boxes for preview
        preview = image.copy()
        for xy, score in zip(xyxy, scores):
            if score >= self.threshold:
                cv2.rectangle(preview, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 0), 2)

        # Resize preview for display window
        preview_resized = cv2.resize(preview, (640, 480))
        cv2.imshow("Ati Eye Preview", preview_resized)
        cv2.waitKey(1)

        # Save frame to video file (handled by VideoRecorder)
        self.recorder.write_frame(preview)

        # Save snapshot as image with detection info
        frame_id = int(time.time())
        self.log_image(YoloResult(classes, xyxy, scores, input_image, detected_object), frame_id)

        # Update detection history
        self.update_inference_history(detected_object)
        self.process_inference_history()

        return YoloResult(classes, xyxy, scores, input_image, detected_object)

    # Save image snapshot to SSD
    def log_image(self, yolo_result: YoloResult, frame_id: int):
        image = yolo_result.input_image.copy()
        add_name = "no_detection"
        for xy, score in zip(yolo_result.xyxy, yolo_result.scores):
            if score >= self.threshold:
                cv2.rectangle(image, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 0), 4)
                add_name = "detected"
        cv2.imwrite(os.path.join(self.data_folder, f"{frame_id}-{add_name}.jpg"), image)

    # Keep track of detection history
    def update_inference_history(self, detected_object: bool):
        self.inf_hist = np.append(self.inf_hist[1:], detected_object)
        self.logger.info(f"Updated inference history: {self.inf_hist}")

    # Process detection history to confirm events
    def process_inference_history(self):
        inf_score = np.sum(self.inf_hist)
        if inf_score >= self.min_apply_score:
            self.logger.info(f"Object detected - inf_score: {inf_score}")
        else:
            self.logger.info(f"No object detected - inf_score: {inf_score}")

    # Destructor: release resources
    def __del__(self):
        if hasattr(self, "cam_capture"):
            self.cam_capture.stop()
        if hasattr(self, "recorder"):
            self.recorder.release()
        cv2.destroyAllWindows()
