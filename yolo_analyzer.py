import cv2
import os
from ultralytics import YOLO
import logging
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global state for frame capture ---
# This is a simplified way to share the latest frame.
# In a more robust app, consider a thread-safe queue or other IPC.
_latest_frame = None
_frame_lock = threading.Lock()
_stop_event = threading.Event()
_video_processing_thread = None
# --- ---

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best.pt")
CAPTURED_FRAMES_FOLDER = os.getenv("CAPTURED_FRAMES_FOLDER", "captured_frames")

# Load the YOLOv8 model
try:
    if os.path.exists(YOLO_MODEL_PATH):
        model = YOLO(YOLO_MODEL_PATH)
        logger.info(f"YOLOv8 model loaded successfully from {YOLO_MODEL_PATH}")
    else:
        model = None
        logger.error(f"YOLOv8 model not found at {YOLO_MODEL_PATH}. Dress analysis will be disabled.")
except Exception as e:
    model = None
    logger.error(f"Error loading YOLOv8 model: {e}. Dress analysis will be disabled.")


def process_video_source(source):
    """
    Processes video from a given source (file path or stream URL).
    Displays the video with YOLOv8 detections in an OpenCV window.
    Updates _latest_frame for capture by other parts of the application.
    """
    global _latest_frame, _stop_event

    if model is None:
        logger.warning("YOLO model not loaded. Cannot process video.")
        # Simulate video display without analysis if model is not there
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Error: Could not open video source: {source}")
            return
        while not _stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            with _frame_lock:
                _latest_frame = frame.copy()
            cv2.imshow("VISION - Dress Analysis (Model Missing)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Allow 'q' to quit this window
                break
        cap.release()
        cv2.destroyWindow("VISION - Dress Analysis (Model Missing)")
        _stop_event.clear() # Reset for next run
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video source: {source}")
        return

    logger.info(f"Started processing video source: {source}")
    window_name = "VISION - Dress Analysis"

    while not _stop_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video source or error reading frame.")
            break

        # Perform inference
        results = model(frame, verbose=False)  # verbose=False to reduce console output
        annotated_frame = results[0].plot()  # plot() returns a BGR numpy array with annotations

        with _frame_lock:
            _latest_frame = frame.copy() # Store the original frame for capture

        cv2.imshow(window_name, annotated_frame)

        # Press 'q' to exit the OpenCV window (and stop this processing thread)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Quit signal received from OpenCV window.")
            break
    
    cap.release()
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1: # Check if window still exists
        cv2.destroyWindow(window_name)
    logger.info(f"Stopped processing video source: {source}")
    _stop_event.clear() # Reset for next run

def start_video_analysis_thread(source):
    """Starts video analysis in a separate thread."""
    global _video_processing_thread, _stop_event
    if _video_processing_thread and _video_processing_thread.is_alive():
        logger.warning("Video analysis is already running. Please stop it first.")
        return False

    _stop_event.clear()
    _video_processing_thread = threading.Thread(target=process_video_source, args=(source,), daemon=True)
    _video_processing_thread.start()
    logger.info(f"Video analysis thread started for source: {source}")
    return True

def stop_video_analysis_thread():
    """Stops the video analysis thread."""
    global _stop_event, _video_processing_thread
    if _video_processing_thread and _video_processing_thread.is_alive():
        logger.info("Attempting to stop video analysis thread...")
        _stop_event.set()
        _video_processing_thread.join(timeout=5) # Wait for thread to finish
        if _video_processing_thread.is_alive():
            logger.warning("Video analysis thread did not stop in time.")
        else:
            logger.info("Video analysis thread stopped.")
        _video_processing_thread = None
        # Ensure OpenCV windows are closed if thread termination was abrupt
        cv2.destroyAllWindows() # Attempt to close any orphaned CV windows
        return True
    logger.info("Video analysis thread not running or already stopped.")
    return False


def capture_current_frame():
    """
    Captures the latest frame from the ongoing video processing.
    Saves it to the CAPTURED_FRAMES_FOLDER.
    Returns:
        str: Path to the saved frame, or None if no frame is available or error.
    """
    global _latest_frame
    if not os.path.exists(CAPTURED_FRAMES_FOLDER):
        try:
            os.makedirs(CAPTURED_FRAMES_FOLDER, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create directory {CAPTURED_FRAMES_FOLDER}: {e}")
            return None

    with _frame_lock:
        if _latest_frame is None:
            logger.warning("No frame available to capture.")
            return None
        frame_to_save = _latest_frame.copy()

    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(CAPTURED_FRAMES_FOLDER, filename)
        cv2.imwrite(filepath, frame_to_save)
        logger.info(f"Frame captured and saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving captured frame: {e}")
        return None

if __name__ == '__main__':
    # This is for direct testing of this module.
    # In the Flask app, these functions will be called from routes.
    print("YOLO Analyzer Module Test")
    if model is None:
        print("YOLO model not loaded. Limited test.")
    
    # Test with a webcam (source=0) or a video file path
    test_source = 0 # Use 0 for webcam, or replace with "path/to/your/video.mp4"
    
    print(f"Starting video analysis for source: {test_source} (Press 'q' in OpenCV window to stop)")
    start_video_analysis_thread(test_source)
    
    try:
        # Let it run for a bit, then try to capture
        time.sleep(10) 
        if _video_processing_thread and _video_processing_thread.is_alive():
            print("Attempting to capture frame...")
            captured_file = capture_current_frame()
            if captured_file:
                print(f"Frame captured to: {captured_file}")
            else:
                print("Failed to capture frame.")
        else:
            print("Video processing thread not running, cannot capture.")
        
        time.sleep(5) # Keep running a bit longer

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Stopping video analysis...")
        stop_video_analysis_thread()
        print("Test finished.")
