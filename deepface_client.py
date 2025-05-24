import os
import requests
from dotenv import load_dotenv
import logging
import base64
import mimetypes # To determine image type for data URI

load_dotenv()
logger = logging.getLogger(__name__)

DEEPFACE_API_BASE_URL = os.getenv("DEEPFACE_API_BASE_URL", "http://127.0.0.1:5005")
# Set facial recognition model: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepId, ArcFace, Dlib, SFace, GhostFaceNet
DEFAULT_MODEL_NAME = "Facenet"
# DEFAULT_MODEL_NAME = "Facenet512"
# DEFAULT_MODEL_NAME = "VGG-Face"
API_TIMEOUT = 45 # Timeout in seconds

def image_to_base64_uri(image_path):
    """Converts an image file to a base64 data URI."""
    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            if image_path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            else: 
                mime_type = 'image/jpeg' 
                logger.warning(f"Could not determine MIME type for {image_path}, defaulting to {mime_type}.")

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error converting image {image_path} to base64 URI: {e}")
        return None

def get_face_embedding(image_path, model_name=DEFAULT_MODEL_NAME):
    """
    Gets face embedding from the DeepFace API using JSON payload with base64 image.
    It's assumed the image_path provided is already a cropped face, 
    so enforce_detection is set to False.
    Args:
        image_path (str): Path to the (ideally cropped) image file.
        model_name (str): Name of the model to use (e.g., "Facenet").
    Returns:
        list: The embedding vector if successful, None otherwise.
    """
    represent_url = f"{DEEPFACE_API_BASE_URL}/represent"
    
    base64_image_uri = image_to_base64_uri(image_path)
    if not base64_image_uri:
        return None

    payload = {
        "img": base64_image_uri,
        "model_name": model_name,
        "enforce_detection": False 
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(represent_url, json=payload, headers=headers, timeout=API_TIMEOUT)
        response.raise_for_status() 
            
        data = response.json()
        if "embedding" in data:
            return data["embedding"]
        elif "results" in data and isinstance(data["results"], list) and len(data["results"]) > 0:
            if "embedding" in data["results"][0]:
                return data["results"][0]["embedding"]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "embedding" in data[0]:
             return data[0]["embedding"]

        logger.warning(f"Could not find 'embedding' in DeepFace API response (/represent). Response: {data}")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling DeepFace API (/represent) for {image_path}: {e}")
        if e.response is not None:
            logger.error(f"Response status: {e.response.status_code}, content: {e.response.text}")
        return None
    except Exception as e: 
        logger.error(f"An unexpected error occurred in get_face_embedding: {e}")
        return None


def detect_faces(image_path, detector_backend=None, min_detection_confidence=0.0, actions=None):
    """
    Detects faces in an image using the DeepFace API and returns their bounding boxes,
    filtered by a minimum detection confidence.
    Args:
        image_path (str): Path to the image file.
        detector_backend (str, optional): Specific face detector backend to use (e.g., 'mtcnn', 'retinaface').
        min_detection_confidence (float, optional): Minimum confidence for a detected face to be included.
        actions (list, optional): List of actions for DeepFace analyze. Defaults to ["age"] if None or empty.
    Returns:
        list: A list of dictionaries, where each dict contains 'region': {'x', 'y', 'w', 'h'}
              for a detected face that meets the confidence threshold. Returns empty list for errors/no valid faces.
              Returns None if API response format is completely unexpected.
    """
    if not actions: 
        actions = ["age"] # Default to ["age"] to satisfy API if actions must be a non-empty list
    
    analyze_url = f"{DEEPFACE_API_BASE_URL}/analyze"

    base64_image_uri = image_to_base64_uri(image_path)
    if not base64_image_uri:
        return [] 

    payload = {
        "img": base64_image_uri,
        "actions": actions
    }
    if detector_backend:
        payload["detector_backend"] = detector_backend
        
    headers = {
        "Content-Type": "application/json"
    }
            
    try:
        response = requests.post(analyze_url, json=payload, headers=headers, timeout=API_TIMEOUT)
        response.raise_for_status()
            
        results_json = response.json()
        
        actual_face_list = []
        detected_faces_output = []

        if isinstance(results_json, list):
            actual_face_list = results_json
        elif isinstance(results_json, dict) and "results" in results_json and isinstance(results_json["results"], list):
            actual_face_list = results_json["results"]
        elif isinstance(results_json, dict) and "error" in results_json:
             logger.warning(f"DeepFace /analyze returned an error: {results_json['error']}")
             return [] 
        else:
            logger.warning(f"DeepFace /analyze did not return a list or the expected dictionary structure. Response: {results_json}")
            return None # Indicates an unparseable response format

        # Process the extracted list of faces
        for face_data in actual_face_list:
            confidence = face_data.get("face_confidence", 0.0) # Get confidence, default to 0.0 if not present
            
            if confidence >= min_detection_confidence:
                if "region" in face_data and isinstance(face_data["region"], dict) and \
                   all(k in face_data["region"] for k in ["x", "y", "w", "h"]):
                    detected_faces_output.append({
                        "region": face_data["region"],
                        "confidence": confidence # Optionally include confidence for debugging or later use
                    }) 
                else:
                    logger.warning(f"Detected face (confidence {confidence:.2f}) missing 'region' or complete region data: {face_data}")
            else:
                logger.info(f"Skipping detected face due to low confidence ({confidence:.2f} < {min_detection_confidence:.2f}): Region {face_data.get('region')}")
        
        return detected_faces_output

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling DeepFace API (/analyze) for {image_path}: {e}")
        if e.response is not None:
            logger.error(f"Response status: {e.response.status_code}, content: {e.response.text}")
        return [] 
    except Exception as e: 
        logger.error(f"An unexpected error occurred in detect_faces: {e}")
        return []

if __name__ == '__main__':
    dummy_image_path = "test_image.jpg" 
    if not os.path.exists(dummy_image_path):
        try:
            from PIL import Image
            img = Image.new('RGB', (200, 200), color = 'blue')
            img.save(dummy_image_path)
            print(f"Created dummy image: {dummy_image_path}")
        except ImportError:
            print("Pillow not installed. Cannot create dummy image.")
            exit()
        except Exception as e:
            print(f"Could not create dummy image: {e}")
            exit()

    print(f"Testing DeepFace client with API at: {DEEPFACE_API_BASE_URL} (Timeout: {API_TIMEOUT}s)")
    
    test_detector_backend = os.getenv("DEEPFACE_DETECTOR_BACKEND", "mtcnn")
    test_min_confidence = float(os.getenv("DEEPFACE_DETECTION_CONFIDENCE_THRESHOLD", 0.90))

    print(f"\nTesting get_face_embedding (with enforce_detection=False)...")
    embedding = get_face_embedding(dummy_image_path)
    if embedding:
        print(f"Successfully retrieved embedding (first 5 elements): {embedding[:5]}... Length: {len(embedding)}")
    else:
        print("Failed to retrieve embedding.")

    print(f"\nTesting detect_faces (using detector: {test_detector_backend}, min_confidence: {test_min_confidence})...")
    detections = detect_faces(dummy_image_path, detector_backend=test_detector_backend, min_detection_confidence=test_min_confidence) 
    if detections is None:
        print("Failed to detect faces due to unexpected API response format.")
    elif not detections: 
        print("No faces detected meeting the criteria or API returned an error/empty list.")
    else: 
        print(f"Successfully detected {len(detections)} face(s):")
        for i, face in enumerate(detections):
            print(f"  Face {i+1} region: {face['region']}, Confidence: {face.get('confidence', 'N/A'):.2f}")