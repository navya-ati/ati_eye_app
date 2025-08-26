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
        self.config = au.load_config()
        self.logger, self.log_dir = au.create_logger(APP_VERSION)
        self.model = self._import_yolo_model()
        self.threshold = 0.25

        # Camera FOV
        self.camera_h_fov, self.camera_v_fov = au.get_camera_fov(self.config)
        self.logger.info(f"Camera FOV - H: {self.camera_h_fov}°, V: {self.camera_v_fov}°")

        # Initialize camera
        self._init_cam_capture()

        # Video recording
        self._setup_video_recording()

        self.inf_window = self.config.get("inference_window", 1)
        self.inf_hist = np.zeros(self.inf_window)
        self.min_apply_score = self.config.get("min_apply_score", 1)

        print(f"AtiEye:v{APP_VERSION} is online!")

    def _init_cam_capture(self):
        self.cam_capture = PiCamReader()
        self.cam_capture.start()
        time.sleep(1)
        self.logger.info("cam_capture initialized")

    def _import_yolo_model(self):
        model_path = 'yolov8n.pt'
        model = YOLO(model_path)
        self.logger.info(f"Imported YOLO model from {model_path}")
        return model

    def _setup_video_recording(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        now = datetime.now()
        self.video_folder = os.path.join(base_path, "data_collection_" + now.strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.video_folder, exist_ok=True)
        print("Videos will be saved in:", self.video_folder)

        self.camera_resolution = tuple(self.config.get("camera_resolution", [1920, 1080]))
        self.camera_fps = self.config.get("camera_fps", 10)

        self.sav_mid_avi = cv2.VideoWriter(
            os.path.join(self.video_folder, "pi_cam_mid.avi"),
            cv2.VideoWriter_fourcc(*'XVID'),
            self.camera_fps,
            self.camera_resolution
        )

        self.sav_low_avi = cv2.VideoWriter(
            os.path.join(self.video_folder, "pi_cam_low.avi"),
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.camera_fps,
            (640, 480)
        )

        signal.signal(signal.SIGINT, self._exit_handler)
        signal.signal(signal.SIGTERM, self._exit_handler)

    def _exit_handler(self, sig, frame):
        print("\n[INFO] Exiting gracefully...")
        try:
            self.cam_capture.stop()
            self.sav_mid_avi.release()
            self.sav_low_avi.release()
            cv2.destroyAllWindows()
            print("[INFO] Camera stopped and videos saved")
        except Exception as e:
            print("[ERROR] During exit:", e)
        sys.exit(0)

    def run_inference(self, image) -> YoloResult:
        input_image = image.copy()  # full frame
        result = self.model.predict(input_image, conf=self.threshold)
        classes, xyxy, scores = au.process_yolo_result(result)
        detected_object = bool(len(classes))

        # Preview with green rectangles only
        preview = image.copy()
        for xy, score in zip(xyxy, scores):
            if score >= self.threshold:
                cv2.rectangle(preview, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 0), 2)

        preview_resized = cv2.resize(preview, (640, 480))
        cv2.imshow("Ati Eye Preview (Full Frame)", preview_resized)
        cv2.waitKey(1)

        # Log image
        frame_id = int(time.time())
        self.log_image(YoloResult(classes, xyxy, scores, input_image, detected_object), frame_id=frame_id)

        # Record videos
        self._record_videos(image)

        # Inference history
        self.update_inference_history(detected_object)
        self.process_inference_history()

        return YoloResult(classes, xyxy, scores, input_image, detected_object)

    def _record_videos(self, frame):
        self.sav_mid_avi.write(frame)
        frame_low = cv2.resize(frame, (640, 480))
        self.sav_low_avi.write(frame_low)

    def log_image(self, yolo_result: YoloResult, frame_id: int):
        add_name = "no_detection"
        image = yolo_result.input_image.copy()
        for xy, score in zip(yolo_result.xyxy, yolo_result.scores):
            if score >= self.threshold:
                cv2.rectangle(image, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 0), 4)
                add_name = "detected"
        cv2.imwrite(os.path.join(self.log_dir, f"{frame_id}-{add_name}.jpg"), image)

    def update_inference_history(self, detected_object: bool):
        self.inf_hist = np.append(self.inf_hist[1:], detected_object)
        self.logger.info(f"Updated inference history: {self.inf_hist}")

    def process_inference_history(self):
        inf_score = np.sum(self.inf_hist)
        if inf_score >= self.min_apply_score:
            self.logger.info(f"Object detected - inf_score: {inf_score}")
        else:
            self.logger.info(f"No object detected - inf_score: {inf_score}")

    def __del__(self):
        if hasattr(self, "cam_capture"):
            self.cam_capture.stop()
        cv2.destroyAllWindows()
        self.sav_mid_avi.release()
        self.sav_low_avi.release()

