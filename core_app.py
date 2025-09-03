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
from record_pi_cam_video import VideoRecorder  # SSD recorder only

APP_VERSION = "0.0 - 18th April"

@dataclass
class YoloResult:
    classes: np.ndarray
    xyxy: np.ndarray
    scores: np.ndarray
    input_image: np.ndarray
    detected_object: bool

class AtiEyeApp:
    def __init__(self):
        # Load config and logger
        self.config = au.load_config()
        self.logger, self.log_dir, _ = au.create_logger(APP_VERSION)  # only SSD logging

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        self.threshold = 0.25

        # Get camera FOV
        self.camera_h_fov, self.camera_v_fov = au.get_camera_fov(self.config)
        self.logger.info(f"Camera FOV - H: {self.camera_h_fov}°, V: {self.camera_v_fov}°")

        # Create SSD data collection folder (for storing processed frames)
        base_path = "/mnt/ssd/videos"
        os.makedirs(base_path, exist_ok=True)
        now = datetime.now()
        folder_name = "data_collection_" + now.strftime("%Y_%m_%d_%H_%M_%S")
        self.data_folder = os.path.join(base_path, folder_name)
        os.makedirs(self.data_folder, exist_ok=True)

        print("[INFO] Videos and frames will be saved in SSD only:")
        print("   SSD :", self.data_folder)

        # Initialize camera (frames kept in RAM, SSD storage after processing)
        self.cam_capture = PiCamReader()  # no save_dir argument
        self.cam_capture.start()
        time.sleep(1)
        self.logger.info("PiCamReader initialized")

        # Initialize recorder for SSD (only)
        self.recorder_ssd = VideoRecorder(self.data_folder, self.config)

        # Signal handling for clean exit
        signal.signal(signal.SIGINT, self._exit_handler)
        signal.signal(signal.SIGTERM, self._exit_handler)

        # Inference history
        self.inf_window = self.config.get("inference_window", 1)
        self.inf_hist = np.zeros(self.inf_window)
        self.min_apply_score = self.config.get("min_apply_score", 1)

        print(f"AtiEye:v{APP_VERSION} is online!")

    def _exit_handler(self, sig, frame):
        """Handle exit signals (CTRL+C, kill)"""
        print("\n[INFO] Exiting gracefully...")
        try:
            self.cam_capture.stop()
            self.recorder_ssd.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print("[ERROR] During exit:", e)
        sys.exit(0)

    def run_inference(self, image) -> YoloResult:
        """Run YOLO inference on a frame"""
        input_image = image.copy()
        result = self.model.predict(input_image, conf=self.threshold)

        # Process YOLO output
        classes, xyxy, scores = au.process_yolo_result(result)
        detected_object = bool(len(classes))

        # Draw detections
        preview = image.copy()
        for xy, score in zip(xyxy, scores):
            if score >= self.threshold:
                cv2.rectangle(preview, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 0), 2)

        # Show preview
        preview_resized = cv2.resize(preview, (640, 480))
        cv2.imshow("Ati Eye Preview", preview_resized)
        cv2.waitKey(1)

        # Save video/frame to SSD after processing
        self.recorder_ssd.write_frame(preview)
        self.log_image(YoloResult(classes, xyxy, scores, input_image, detected_object))

        # Update inference history
        self.update_inference_history(detected_object)
        self.process_inference_history()

        return YoloResult(classes, xyxy, scores, input_image, detected_object)

    def log_image(self, yolo_result: YoloResult):
        """Save image with/without detections to SSD only"""
        image = yolo_result.input_image.copy()
        add_name = "no_detection"

        for xy, score in zip(yolo_result.xyxy, yolo_result.scores):
            if score >= self.threshold:
                cv2.rectangle(image, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 0), 4)
                add_name = "detected"

        # Save processed frame to SSD only
        frame_id = int(time.time())
        cv2.imwrite(os.path.join(self.data_folder, f"{frame_id}-{add_name}.jpg"), image)

    def update_inference_history(self, detected_object: bool):
        """Update rolling history of detections"""
        self.inf_hist = np.append(self.inf_hist[1:], detected_object)
        self.logger.info(f"Updated inference history: {self.inf_hist}")

    def process_inference_history(self):
        """Analyze detection history and log results"""
        inf_score = np.sum(self.inf_hist)
        if inf_score >= self.min_apply_score:
            self.logger.info(f"Object detected - inf_score: {inf_score}")
        else:
            self.logger.info(f"No object detected - inf_score: {inf_score}")

    def __del__(self):
        """Destructor for cleanup"""
        if hasattr(self, "cam_capture"):
            self.cam_capture.stop()
        if hasattr(self, "recorder_ssd"):
            self.recorder_ssd.release()
        cv2.destroyAllWindows()
