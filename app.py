import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Project modules
import database as db
import deepface_client as dfc
import yolo_analyzer
import utils

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_secret_key') # Essential for flash messages
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads/faces')
app.config['CAPTURED_FRAMES_FOLDER'] = os.getenv('CAPTURED_FRAMES_FOLDER', 'captured_frames')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure upload and captured frames directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CAPTURED_FRAMES_FOLDER'], exist_ok=True)

# Initialize database tables at startup
with app.app_context():
    db.init_db()

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
        if file: # Add extension check if needed
            # For simplicity, save it temporarily and pass path.
            # In a real app, manage temp files better.
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

    # 1. Detect faces in the captured frame
    detected_regions = dfc.detect_faces(captured_image_path) # Returns list of {'region': ...}
    if detected_regions is None: # API error
        flash('Error communicating with DeepFace API for face detection.', 'danger')
        return redirect(url_for('index'))
    if not detected_regions:
        flash('No faces detected in the captured frame.', 'info')
        # Optionally, show the frame without annotations or just redirect
        return render_template('recognition_result.html', original_image=os.path.basename(captured_image_path),
                               annotated_image=None, results=[])


    # 2. For each detected face, get embedding and recognize
    recognition_results = []
    known_embeddings_data = db.get_all_face_embeddings_for_recognition() # List of {'embedding_vector', 'person_name', ...}

    original_image = cv2.imread(captured_image_path) # Read with OpenCV for cropping

    for face_info in detected_regions:
        region = face_info['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        
        # Crop the face from the original image
        cropped_face_cv = original_image[y:y+h, x:x+w]
        
        if cropped_face_cv.size == 0: # Check if crop is valid
            recognition_results.append({"region": region, "name": "Crop Error", "similarity": 0})
            continue

        # Save cropped face temporarily to send to /represent API
        temp_crop_filename = f"temp_crop_{uuid.uuid4().hex[:6]}.jpg"
        temp_crop_path = os.path.join(app.config['CAPTURED_FRAMES_FOLDER'], temp_crop_filename)
        cv2.imwrite(temp_crop_path, cropped_face_cv)

        target_embedding = dfc.get_face_embedding(temp_crop_path)
        os.remove(temp_crop_path) # Clean up temp crop

        if target_embedding:
            match = utils.find_best_match(target_embedding, known_embeddings_data)
            if match:
                recognition_results.append({
                    "region": region,
                    "name": match["person_name"],
                    "similarity": match["similarity"]
                })
            else:
                recognition_results.append({"region": region, "name": "Unknown", "similarity": 0})
        else:
            recognition_results.append({"region": region, "name": "Embedding Error", "similarity": 0})
            flash(f"Could not get embedding for a detected face region {region}.", "warning")
            
    # 3. Draw detections on the image
    annotated_image_filename = utils.draw_detections_on_image(captured_image_path, recognition_results)

    if annotated_image_filename:
        flash('Face recognition complete.', 'success')
        return render_template('recognition_result.html',
                               original_image=os.path.basename(captured_image_path),
                               annotated_image=os.path.basename(annotated_image_filename),
                               results=recognition_results)
    else:
        flash('Face recognition complete, but failed to draw annotations.', 'warning')
        # Still show results, but maybe without the annotated image if drawing failed
        return render_template('recognition_result.html',
                               original_image=os.path.basename(captured_image_path),
                               annotated_image=None, # Or original_image if no annotation
                               results=recognition_results)


# --- Face Database Management Routes ---
@app.route('/manage_faces')
def manage_faces():
    persons_data = db.get_all_persons_with_embeddings() # [{'person_id', 'name', 'embeddings': [{'image_filename', ...}]}]
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
            # Ensure unique filename to avoid overwrites if multiple people upload 'face.jpg'
            unique_filename = f"{name.replace(' ','_')}_{uuid.uuid4().hex[:8]}_{filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(image_path)

            # Get embedding from DeepFace API
            embedding_vector = dfc.get_face_embedding(image_path)

            if embedding_vector:
                # Add person to DB (or get existing by name)
                person_id = db.add_person(name) # This handles existing names too
                if person_id:
                    # Add embedding to DB
                    embedding_id = db.add_face_embedding(person_id, embedding_vector, unique_filename)
                    if embedding_id:
                        flash(f'Person "{name}" and their face image added successfully!', 'success')
                    else:
                        flash(f'Failed to save face embedding for "{name}".', 'danger')
                        os.remove(image_path) # Clean up uploaded image if DB fails
                else:
                    flash(f'Failed to add or find person "{name}" in database.', 'danger')
                    os.remove(image_path) # Clean up
            else:
                flash('Failed to get face embedding from DeepFace API. Ensure image contains a clear face and API is running.', 'danger')
                os.remove(image_path) # Clean up
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
    # Before deleting person, delete their images from filesystem
    person_data = db.get_all_persons_with_embeddings() # Find the person to get their image filenames
    person_to_delete = next((p for p in person_data if p['person_id'] == person_id), None)
    
    if person_to_delete:
        for emb_info in person_to_delete.get('embeddings', []):
            try:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], emb_info['image_filename'])
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                flash(f"Error deleting image file {emb_info['image_filename']}: {e}", "warning")

    if db.delete_person(person_id): # This will also delete embeddings due to CASCADE
        flash('Person and their associated face images deleted successfully.', 'success')
    else:
        flash('Failed to delete person.', 'danger')
    return redirect(url_for('manage_faces'))

@app.route('/delete_face_image/<int:embedding_id>', methods=['POST'])
def delete_face_image_route(embedding_id):
    # Find the image filename before deleting from DB
    # This requires a way to get a single embedding's info or to iterate.
    # For simplicity, let's assume we can get it or we iterate through all.
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
        else: # Should not happen if DB delete was successful and data is consistent
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
    import cv2 # For app.py direct run test
    import numpy as np # For app.py direct run test
    import uuid # For app.py direct run test

    print(f"Flask UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
    print(f"Flask CAPTURED_FRAMES_FOLDER: {app.config['CAPTURED_FRAMES_FOLDER']}")
    app.run(debug=True, use_reloader=True) # use_reloader=False if yolo thread causes issues