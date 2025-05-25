import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import cv2 # For image reading/cropping - ensure it's imported
import uuid # For unique filenames

# Project modules
import database as db
import deepface_client as dfc
import yolo_analyzer
import utils

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_secret_key')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads/faces')
app.config['CAPTURED_FRAMES_FOLDER'] = os.getenv('CAPTURED_FRAMES_FOLDER', 'captured_frames')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Register Blueprints ---
# Import and register the new blueprint for movement monitoring
from movement_monitoring.routes import movement_monitoring_bp, init_movement_monitoring_db
app.register_blueprint(movement_monitoring_bp)
# --- End Register Blueprints ---


# DeepFace specific configurations from .env
DEEPFACE_DETECTOR_BACKEND = os.getenv('DEEPFACE_DETECTOR_BACKEND', 'mtcnn')
DEEPFACE_DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv('DEEPFACE_DETECTION_CONFIDENCE_THRESHOLD', 0.90))


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CAPTURED_FRAMES_FOLDER'], exist_ok=True)

with app.app_context():
    db.init_db()
    # Initialize movement monitoring DB tables after main DB and blueprint registration
    init_movement_monitoring_db()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# --- Dress Analysis Routes ---
@app.route('/start_dress_analysis', methods=['POST'])
def start_dress_analysis():
    source_type = request.form.get('source_type')
    video_source = None

    if source_type == 'upload':
        if 'video_file' not in request.files:
            flash('No video file part', 'danger')
            return redirect(url_for('index'))
        file = request.files['video_file']
        if file.filename == '':
            flash('No selected video file', 'danger')
            return redirect(url_for('index'))
        if file:
            temp_video_path = os.path.join(app.config['CAPTURED_FRAMES_FOLDER'], secure_filename(file.filename))
            file.save(temp_video_path)
            video_source = temp_video_path
            flash(f'Video uploaded: {secure_filename(file.filename)}. Analysis started in OpenCV window.', 'info')
    elif source_type == 'stream':
        video_source = request.form.get('stream_url')
        if not video_source:
            flash('No stream URL provided', 'danger')
            return redirect(url_for('index'))
        flash(f'Video stream analysis started for: {video_source} in OpenCV window.', 'info')
    else:
        flash('Invalid source type', 'danger')
        return redirect(url_for('index'))

    if video_source:
        if yolo_analyzer.start_video_analysis_thread(video_source):
             flash('Video analysis started. Check the OpenCV window.', 'success')
        else:
             flash('Failed to start video analysis. Is it already running?', 'warning')
    return redirect(url_for('index'))

@app.route('/stop_dress_analysis', methods=['POST'])
def stop_dress_analysis():
    if yolo_analyzer.stop_video_analysis_thread():
        flash('Video analysis stopped.', 'success')
    else:
        flash('Video analysis was not running or could not be stopped.', 'info')
    return redirect(url_for('index'))

@app.route('/capture_frame_and_recognize', methods=['POST'])
def capture_frame_and_recognize():
    captured_image_path = yolo_analyzer.capture_current_frame()
    if not captured_image_path:
        flash('Failed to capture frame or video analysis not running.', 'danger')
        return redirect(url_for('index'))

    flash(f'Frame captured: {os.path.basename(captured_image_path)}. Processing for face recognition...', 'info')

    # 1. Detect faces in the captured frame, using configured backend and confidence
    detected_face_objects = dfc.detect_faces(
        captured_image_path,
        detector_backend=DEEPFACE_DETECTOR_BACKEND,
        min_detection_confidence=DEEPFACE_DETECTION_CONFIDENCE_THRESHOLD
    )
    
    if detected_face_objects is None: # API response format error
        flash('Error communicating with DeepFace API or unexpected response format for face detection.', 'danger')
        return redirect(url_for('index'))
    if not detected_face_objects: # Empty list, meaning no faces met criteria
        flash(f'No faces detected with confidence >= {DEEPFACE_DETECTION_CONFIDENCE_THRESHOLD} using {DEEPFACE_DETECTOR_BACKEND} backend.', 'info')
        # Show the frame without annotations or just redirect
        return render_template('recognition_result.html', original_image=os.path.basename(captured_image_path),
                               annotated_image=None, results=[])

    recognition_results = []
    known_embeddings_data = db.get_all_face_embeddings_for_recognition()

    original_image_cv = cv2.imread(captured_image_path)
    if original_image_cv is None:
        flash(f"Error: Could not read captured image at {captured_image_path}", "danger")
        return redirect(url_for('index'))

    for face_obj in detected_face_objects: # face_obj is now {'region': ..., 'confidence': ...}
        region = face_obj['region']
        detection_confidence = face_obj.get('confidence', 'N/A') # Get confidence for logging/display
        
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        
        cropped_face_cv = original_image_cv[y:y+h, x:x+w]
        
        if cropped_face_cv.size == 0:
            recognition_results.append({"region": region, "name": "Crop Error", "similarity": 0, "detection_confidence": detection_confidence})
            continue

        temp_crop_filename = f"temp_crop_{uuid.uuid4().hex[:6]}.jpg"
        temp_crop_path = os.path.join(app.config['CAPTURED_FRAMES_FOLDER'], temp_crop_filename)
        cv2.imwrite(temp_crop_path, cropped_face_cv)

        target_embedding = dfc.get_face_embedding(temp_crop_path)
        
        # Clean up temp crop immediately after use
        try:
            os.remove(temp_crop_path)
        except OSError as e:
            app.logger.warning(f"Could not remove temp crop file {temp_crop_path}: {e}")


        current_result = {"region": region, "name": "Error", "similarity": 0, "detection_confidence": detection_confidence}
        if target_embedding:
            match = utils.find_best_match(target_embedding, known_embeddings_data) # find_best_match uses COSINE_SIMILARITY_THRESHOLD from utils
            if match:
                current_result["name"] = match["person_name"]
                current_result["similarity"] = match["similarity"]
            else:
                current_result["name"] = "Unknown"
        else:
            current_result["name"] = "Embedding Error"
            flash(f"Could not get embedding for a detected face (confidence {detection_confidence:.2f}).", "warning")
        recognition_results.append(current_result)
            
    annotated_image_filename = utils.draw_detections_on_image(captured_image_path, recognition_results)

    if annotated_image_filename:
        flash('Face recognition complete.', 'success')
    else:
        flash('Face recognition complete, but failed to draw annotations.', 'warning')
        
    return render_template('recognition_result.html',
                           original_image=os.path.basename(captured_image_path),
                           annotated_image=os.path.basename(annotated_image_filename) if annotated_image_filename else None,
                           results=recognition_results)


# --- Face Database Management Routes (largely unchanged, ensure imports are fine) ---
@app.route('/manage_faces')
def manage_faces():
    persons_data = db.get_all_persons_with_embeddings()
    return render_template('manage_faces.html', persons_data=persons_data)

@app.route('/add_person', methods=['GET', 'POST'])
def add_person_route():
    if request.method == 'POST':
        name = request.form.get('name')
        if 'face_image' not in request.files:
            flash('No face image file part', 'danger')
            return redirect(request.url)
        
        file = request.files['face_image']
        if file.filename == '':
            flash('No selected face image file', 'danger')
            return redirect(request.url)

        if not name:
            flash('Person name is required', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{name.replace(' ','_')}_{uuid.uuid4().hex[:8]}_{filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(image_path)

            # For adding a person, we assume the uploaded image is good.
            # enforce_detection=False in get_face_embedding is crucial here.
            embedding_vector = dfc.get_face_embedding(image_path)

            if embedding_vector:
                person_id = db.add_person(name)
                if person_id:
                    embedding_id = db.add_face_embedding(person_id, embedding_vector, unique_filename)
                    if embedding_id:
                        flash(f'Person "{name}" and their face image added successfully!', 'success')
                    else:
                        flash(f'Failed to save face embedding for "{name}".', 'danger')
                        if os.path.exists(image_path): os.remove(image_path)
                else:
                    flash(f'Failed to add or find person "{name}" in database.', 'danger')
                    if os.path.exists(image_path): os.remove(image_path)
            else:
                flash('Failed to get face embedding. Ensure image contains a clear face and API is running. If this is a new person, the image should be a good quality face shot.', 'danger')
                if os.path.exists(image_path): os.remove(image_path)
            return redirect(url_for('manage_faces'))
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg', 'danger')
            return redirect(request.url)

    return render_template('add_person.html')

@app.route('/edit_person/<int:person_id>', methods=['POST'])
def edit_person_name(person_id):
    new_name = request.form.get('new_name')
    if not new_name:
        flash('New name cannot be empty.', 'danger')
    elif db.update_person_name(person_id, new_name):
        flash('Person\'s name updated successfully.', 'success')
    else:
        flash('Failed to update person\'s name. Does the new name already exist?', 'danger')
    return redirect(url_for('manage_faces'))

@app.route('/delete_person/<int:person_id>', methods=['POST'])
def delete_person_route(person_id):
    person_data = db.get_all_persons_with_embeddings() 
    person_to_delete = next((p for p in person_data if p['person_id'] == person_id), None)
    
    if person_to_delete:
        for emb_info in person_to_delete.get('embeddings', []):
            try:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], emb_info['image_filename'])
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                flash(f"Error deleting image file {emb_info['image_filename']}: {e}", "warning")

    if db.delete_person(person_id): 
        flash('Person and their associated face images deleted successfully.', 'success')
    else:
        flash('Failed to delete person.', 'danger')
    return redirect(url_for('manage_faces'))

@app.route('/delete_face_image/<int:embedding_id>', methods=['POST'])
def delete_face_image_route(embedding_id):
    all_embeddings = []
    persons_data = db.get_all_persons_with_embeddings()
    image_filename_to_delete = None
    for p_data in persons_data:
        for emb in p_data.get('embeddings', []):
            if emb['embedding_id'] == embedding_id:
                image_filename_to_delete = emb['image_filename']
                break
        if image_filename_to_delete:
            break
            
    if db.delete_face_embedding(embedding_id):
        if image_filename_to_delete:
            try:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename_to_delete)
                if os.path.exists(img_path):
                    os.remove(img_path)
                flash('Face image deleted successfully.', 'success')
            except Exception as e:
                flash(f'DB entry deleted, but error deleting image file {image_filename_to_delete}: {e}', 'warning')
        else:
            flash('Face image DB entry deleted, but couldn_t find filename to remove from disk.', 'warning')
    else:
        flash('Failed to delete face image.', 'danger')
    return redirect(url_for('manage_faces'))

# --- Serve Uploaded and Captured Files ---
@app.route('/uploads/faces/<filename>')
def uploaded_face_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/captured_frames/<filename>')
def captured_frame_file(filename):
    return send_from_directory(app.config['CAPTURED_FRAMES_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)