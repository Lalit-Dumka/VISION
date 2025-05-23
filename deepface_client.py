import os
import requests
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

DEEPFACE_API_BASE_URL = os.getenv("DEEPFACE_API_BASE_URL", "http://127.0.0.1:5005")
DEFAULT_MODEL_NAME = "Facenet" # Or make this configurable

def get_face_embedding(image_path, model_name=DEFAULT_MODEL_NAME):
    """
    Gets face embedding from the DeepFace API.
    Args:
        image_path (str): Path to the image file.
        model_name (str): Name of the model to use (e.g., "Facenet").
    Returns:
        list: The embedding vector if successful, None otherwise.
    """
    represent_url = f"{DEEPFACE_API_BASE_URL}/represent"
    try:
        with open(image_path, 'rb') as img_file:
            files = {'img': (os.path.basename(image_path), img_file)}
            payload = {'model_name': model_name}
            response = requests.post(represent_url, files=files, data=payload, timeout=20) # Increased timeout
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            
            data = response.json()
            # Try to extract embedding from common response structures
            if "embedding" in data:
                return data["embedding"]
            elif "results" in data and isinstance(data["results"], list) and len(data["results"]) > 0:
                if "embedding" in data["results"][0]:
                    return data["results"][0]["embedding"]
            # Fallback for a structure like: [[{"embedding": [...], ...}]]
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "embedding" in data[0]:
                 return data[0]["embedding"]
            # Fallback for older DeepFace API structure where it might return a list of dicts, one per face
            elif isinstance(data, list) and len(data) > 0 and "embedding" in data[0]:
                 return data[0]["embedding"]


            logger.warning(f"Could not find 'embedding' in DeepFace API response: {data}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling DeepFace API (/represent) for {image_path}: {e}")
        if e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return None
    except (IOError, FileNotFoundError) as e:
        logger.error(f"File error for {image_path}: {e}")
        return None
    except Exception as e: # Catch other potential errors like JSONDecodeError
        logger.error(f"An unexpected error occurred in get_face_embedding: {e}")
        return None


def detect_faces(image_path, actions=None):
    """
    Detects faces in an image using the DeepFace API and returns their bounding boxes.
    Args:
        image_path (str): Path to the image file.
        actions (list, optional): List of actions for DeepFace analyze (e.g., ["age"]). 
                                  Defaults to an empty list just to get regions.
    Returns:
        list: A list of dictionaries, where each dict contains 'region': {'x', 'y', 'w', 'h'}
              for a detected face. Returns None if an error occurs or no faces are detected.
    """
    if actions is None:
        actions = [] # Minimal actions just to get detections
    
    analyze_url = f"{DEEPFACE_API_BASE_URL}/analyze"
    try:
        with open(image_path, 'rb') as img_file:
            files = {'img': (os.path.basename(image_path), img_file)}
            # DeepFace API expects actions as a JSON string in form-data
            payload = {'actions': str(actions)} # e.g. '["age", "gender"]' or '[]'
            
            response = requests.post(analyze_url, files=files, data=payload, timeout=20) # Increased timeout
            response.raise_for_status()
            
            results = response.json()
            
            # The /analyze endpoint usually returns a list of face objects.
            # Each object should contain a "region" key.
            detected_faces = []
            if isinstance(results, list):
                for face_data in results:
                    if "region" in face_data and all(k in face_data["region"] for k in ["x", "y", "w", "h"]):
                        detected_faces.append({"region": face_data["region"]})
                    else:
                        logger.warning(f"Detected face object missing 'region' or complete region data: {face_data}")
            else:
                logger.warning(f"DeepFace /analyze did not return a list as expected. Response: {results}")
                return None

            return detected_faces if detected_faces else [] # Return empty list if no valid faces found

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling DeepFace API (/analyze) for {image_path}: {e}")
        if e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return None
    except (IOError, FileNotFoundError) as e:
        logger.error(f"File error for {image_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in detect_faces: {e}")
        return None

if __name__ == '__main__':
    # Create a dummy image file for testing
    # Ensure you have a 'test_image.jpg' in the same directory or provide a valid path
    # For this example, let's assume 'test_image.jpg' exists
    dummy_image_path = "test_image.jpg" # Replace with an actual image path for testing
    if not os.path.exists(dummy_image_path):
        try:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color = 'red')
            img.save(dummy_image_path)
            print(f"Created dummy image: {dummy_image_path}")
        except ImportError:
            print("Pillow not installed. Cannot create dummy image. Please create 'test_image.jpg' manually for testing.")
            exit()
        except Exception as e:
            print(f"Could not create dummy image: {e}")
            exit()


    print(f"Testing DeepFace client with API at: {DEEPFACE_API_BASE_URL}")
    
    print("\nTesting get_face_embedding...")
    embedding = get_face_embedding(dummy_image_path)
    if embedding:
        print(f"Successfully retrieved embedding (first 5 elements): {embedding[:5]}... Length: {len(embedding)}")
    else:
        print("Failed to retrieve embedding.")

    print("\nTesting detect_faces...")
    detections = detect_faces(dummy_image_path)
    if detections is not None:
        if detections:
            print(f"Successfully detected {len(detections)} face(s):")
            for i, face in enumerate(detections):
                print(f"  Face {i+1} region: {face['region']}")
        else:
            print("No faces detected or response format issue.")
    else:
        print("Failed to detect faces.")