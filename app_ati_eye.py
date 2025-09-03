import time
import cv2
from core_app import AtiEyeApp, APP_VERSION

# Inform the user that libraries are being imported
print("importing the libs! This could take a minute!")
print(f"Initializing global AtiEye application v{APP_VERSION}!")

def main():
    # Initialize AtiEye application
    eye = AtiEyeApp()
    start_time = time.time()

    while True:
        if eye.cam_capture.is_new_data:
            # Get latest frame from RAM
            frame_data = eye.cam_capture.get_data()
            if frame_data is None:
                continue

            image_src, timestamp, frame_id = frame_data

            # Stop the app if camera data is invalid
            if timestamp is None:
                eye.logger.info(f"Invalid image in frame: {frame_id}! KILLING the app!")
                break

            # Run inference → this also saves video/frames to SSD after processing
            yolo_result = eye.run_inference(image_src)

            # Update and process inference history
            eye.update_inference_history(yolo_result.detected_object)
            eye.process_inference_history()

            # Log processing time
            proc_time = round(time.time() - start_time, 3)
            eye.logger.info(f"img: {frame_id}, process_time - {proc_time} secs")

            # Reset timer for next frame
            start_time = time.time()
        else:
            # Log time since last new frame
            diff_time = round(time.time() - start_time, 3)
            eye.logger.info(f"no new data since {diff_time} seconds")
            time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping Ati-Eye gracefully...")
        cv2.destroyAllWindows()
