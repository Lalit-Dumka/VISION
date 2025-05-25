import numpy as np
import os
import cv2 # OpenCV for drawing
import uuid

# Load configuration from .env
from dotenv import load_dotenv
load_dotenv()
COSINE_SIMILARITY_THRESHOLD = float(os.getenv("COSINE_SIMILARITY_THRESHOLD", 0.6))
CAPTURED_FRAMES_FOLDER = os.getenv("CAPTURED_FRAMES_FOLDER", "captured_frames")


def cosine_similarity(embedding1, embedding2):
    """Computes cosine similarity between two embedding vectors."""
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)
    
    # Ensure embeddings are 1D arrays (vectors)
    if embedding1.ndim > 1:
        embedding1 = embedding1.flatten()
    if embedding2.ndim > 1:
        embedding2 = embedding2.flatten()

    if embedding1.shape != embedding2.shape:
        # This can happen if embeddings are from different models or processing errors
        # print(f"Warning: Embedding shapes differ: {embedding1.shape} vs {embedding2.shape}")
        return 0.0 # Or raise an error

    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    
    if norm_embedding1 == 0 or norm_embedding2 == 0:
        return 0.0 # Avoid division by zero

    similarity = dot_product / (norm_embedding1 * norm_embedding2)
    return similarity

def find_best_match(target_embedding, known_embeddings_data, threshold=COSINE_SIMILARITY_THRESHOLD):
    """
    Finds the best match for a target embedding from a list of known embeddings.
    Args:
        target_embedding (list or np.array): The embedding of the face to recognize.
        known_embeddings_data (list of dicts): Each dict should have 'embedding_vector', 
                                               'person_name', 'person_id'.
        threshold (float): Minimum similarity score to be considered a match.
    Returns:
        dict: Information of the best match {'person_name', 'person_id', 'similarity'} 
              or None if no match found above threshold.
    """
    best_match = None
    max_similarity = -1.0  # Cosine similarity ranges from -1 to 1

    if not known_embeddings_data:
        return None

    for known_face in known_embeddings_data:
        similarity = cosine_similarity(target_embedding, known_face["embedding_vector"])
        if similarity > max_similarity:
            max_similarity = similarity
            if similarity >= threshold:
                best_match = {
                    "person_name": known_face["person_name"],
                    "person_id": known_face["person_id"],
                    "similarity": similarity
                }
    
    return best_match


def draw_detections_on_image(image_path, recognized_faces_info):
    """
    Draws bounding boxes and labels on an image for recognized faces.
    Args:
        image_path (str): Path to the original image.
        recognized_faces_info (list of dicts): Each dict should have 'region' and 'name'.
                                              'region' is {'x', 'y', 'w', 'h'}.
                                              'name' is the recognized person's name or "Unknown".
    Returns:
        str: Path to the new image with detections drawn, or None if error.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        for face_info in recognized_faces_info:
            region = face_info.get("region")
            name = face_info.get("name", "Unknown")
            similarity_score = face_info.get("similarity")

            if not region:
                continue

            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare label text
            label = name
            if similarity_score is not None:
                label = f"{name} ({similarity_score:.2f})"
            
            # Put label above the rectangle
            label_y = y - 10 if y - 10 > 10 else y + 10
            cv2.putText(image, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the modified image
        if not os.path.exists(CAPTURED_FRAMES_FOLDER):
            os.makedirs(CAPTURED_FRAMES_FOLDER, exist_ok=True)
        
        original_filename = os.path.basename(image_path)
        name_part, ext_part = os.path.splitext(original_filename)
        output_filename = f"{name_part}_annotated_{uuid.uuid4().hex[:6]}{ext_part}"
        output_image_path = os.path.join(CAPTURED_FRAMES_FOLDER, output_filename)
        
        cv2.imwrite(output_image_path, image)
        return output_image_path
    except Exception as e:
        print(f"Error drawing detections on image: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    emb1 = np.array([0.1, 0.2, 0.3, 0.4])
    emb2 = np.array([0.1, 0.2, 0.3, 0.4]) # Perfect match
    emb3 = np.array([0.4, 0.3, 0.2, 0.1]) # Different
    emb4 = np.array([0.11, 0.22, 0.28, 0.39]) # Similar

    print(f"Similarity (emb1, emb2): {cosine_similarity(emb1, emb2)}")
    print(f"Similarity (emb1, emb3): {cosine_similarity(emb1, emb3)}")
    print(f"Similarity (emb1, emb4): {cosine_similarity(emb1, emb4)}")

    known_faces = [
        {"embedding_vector": [1, 0, 0, 0], "person_name": "Alice", "person_id": 1},
        {"embedding_vector": [0, 1, 0, 0], "person_name": "Bob", "person_id": 2},
        {"embedding_vector": [0.9, 0.1, 0, 0], "person_name": "Alice Twin", "person_id": 3}
    ]
    target = [0.95, 0.05, 0.01, 0.02] # Should match Alice or Alice Twin
    
    match = find_best_match(target, known_faces, threshold=0.7)
    if match:
        print(f"Best match for target: {match['person_name']} with similarity {match['similarity']:.4f}")
    else:
        print("No match found for target.")

    # To test draw_detections_on_image, you'd need an image and some dummy detection data
    # e.g. create a dummy 'test_draw.jpg'
    # test_img_path = "test_draw.jpg"
    # if not os.path.exists(test_img_path):
    #     img = np.zeros((200, 200, 3), dtype=np.uint8)
    #     cv2.imwrite(test_img_path, img)
    # recognized_info = [
    #     {"region": {"x": 50, "y": 50, "w": 80, "h": 80}, "name": "Test Person", "similarity": 0.92}
    # ]
    # annotated_path = draw_detections_on_image(test_img_path, recognized_info)
    # if annotated_path:
    #     print(f"Annotated image saved to: {annotated_path}")
    # hello