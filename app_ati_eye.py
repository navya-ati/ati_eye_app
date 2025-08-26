print("Importing libraries...")

import time
import cv2
from core_app import AtiEyeApp, APP_VERSION

print(f"Initializing global AtiEye application v{APP_VERSION}!")

def main():
    print("Starting the main AtiEye application...")

    # Create AtiEye app instance
    eye = AtiEyeApp()

    start = time.time()
    while True:
        if eye.cam_capture.is_new_data:
            start = time.time()
            image_src, timestamp, frame_id = eye.cam_capture.get_data()
            if timestamp is None:
                eye.logger.info(f"Invalid image in frame: {frame_id}! Exiting the app!")
                break

            # Run inference on full frame
            yolo_result = eye.run_inference(image_src)

            # Only green rectangles are drawn in core_app.py
            # Update inference history
            eye.update_inference_history(yolo_result.detected_object)
            eye.process_inference_history()
            eye.log_image(yolo_result, frame_id)

            # Log processing info
            proc_time = round(time.time() - start, 3)
            eye.logger.info(f"img: {frame_id}, process_time - {proc_time} secs")

        else:
            diff_time = round(time.time() - start, 3)
            eye.logger.info(f"No new data for {diff_time} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping Ati-Eye gracefully...")
        cv2.destroyAllWindows()

