import time
import cv2
from core_app import AtiEyeApp, APP_VERSION

print(f"Initializing global AtiEye application v{APP_VERSION}!")

def main():
    eye = AtiEyeApp()
    start_time = time.time()

    while True:
        if eye.cam_capture.is_new_data:
            frame_data = eye.cam_capture.get_data()
            if frame_data is None:
                continue

            image_src, timestamp, frame_id = frame_data

            # Run inference
            yolo_result = eye.run_inference(image_src)

            proc_time = round(time.time() - start_time, 3)
            eye.logger.info(f"Frame {frame_id} processed in {proc_time} sec")
            start_time = time.time()
        else:
            time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping Ati-Eye gracefully...")
        cv2.destroyAllWindows()
