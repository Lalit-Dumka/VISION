import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, make_response
import cv2
import time
import csv
import datetime
import psycopg2.extras
import threading
import json
import numpy as np
from ultralytics import YOLO # Import YOLO from ultralytics

# Use get_db_connection from the main database.py
from database import get_db_connection # Corrected import

movement_monitoring_bp = Blueprint('movement_monitoring', __name__,
                           template_folder='../templates',  # Point to main templates folder
                           static_folder='static',
                           url_prefix='/movement_monitoring')

# --- Database Helper Functions (specific to movement_monitoring) ---

def init_movement_monitoring_db():
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS se_cameras (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL, -- 'stream' or 'video_file'
                source_path TEXT NOT NULL, -- URL or file path
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS se_camera_zones (
                id SERIAL PRIMARY KEY,
                camera_id INTEGER NOT NULL REFERENCES se_cameras(id) ON DELETE CASCADE,
                zone_name TEXT NOT NULL,
                points TEXT NOT NULL, -- Store as JSON string of points e.g., "[[x1,y1],[x2,y2],...]"
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(camera_id, zone_name)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS se_detection_logs (
                id SERIAL PRIMARY KEY,
                camera_id INTEGER NOT NULL REFERENCES se_cameras(id) ON DELETE CASCADE,
                zone_id INTEGER REFERENCES se_camera_zones(id) ON DELETE SET NULL, -- Can be null if not zone-specific
                video_timestamp FLOAT, -- Timestamp from the video if available
                system_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                count INTEGER NOT NULL,
                details TEXT -- e.g., for future use, like object types
            );
        """)
        
        # New table for movement tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS se_movement_logs (
                id SERIAL PRIMARY KEY,
                camera_id INTEGER NOT NULL REFERENCES se_cameras(id) ON DELETE CASCADE,
                track_id INTEGER NOT NULL, -- YOLOv8 track ID
                from_zone_id INTEGER REFERENCES se_camera_zones(id) ON DELETE SET NULL, -- NULL means entered from outside camera view
                to_zone_id INTEGER REFERENCES se_camera_zones(id) ON DELETE SET NULL, -- NULL means exited camera view
                video_timestamp FLOAT, -- Timestamp from the video if available
                system_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                movement_type TEXT DEFAULT 'zone_transition', -- 'zone_transition', 'entry', 'exit'
                details TEXT -- Additional information about the movement
            );
        """)
        
        # Index for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_movement_logs_camera_timestamp 
            ON se_movement_logs(camera_id, system_timestamp DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_movement_logs_zones 
            ON se_movement_logs(from_zone_id, to_zone_id);
        """)
        
        db_conn.commit()
    finally:
        if db_conn:
            db_conn.close()


def add_camera_to_db(name, source_type, source_path):
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor()
        cursor.execute(
            "INSERT INTO se_cameras (name, source_type, source_path) VALUES (%s, %s, %s) RETURNING id",
            (name, source_type, source_path)
        )
        camera_id = cursor.fetchone()[0]
        db_conn.commit()
        return camera_id
    except Exception as e:
        db_conn.rollback()
        current_app.logger.error(f"Error adding camera {name}: {e}")
        return None
    finally:
        if db_conn:
            db_conn.close()

def get_all_cameras_from_db():
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT * FROM se_cameras ORDER BY name")
        cameras = cursor.fetchall()
        return cameras
    finally:
        if db_conn:
            db_conn.close()

def get_camera_by_id_from_db(camera_id):
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT * FROM se_cameras WHERE id = %s", (camera_id,))
        camera = cursor.fetchone()
        return camera
    finally:
        if db_conn:
            db_conn.close()

def add_zone_to_db(camera_id, zone_name, points_json_string):
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor()
        cursor.execute(
            "INSERT INTO se_camera_zones (camera_id, zone_name, points) VALUES (%s, %s, %s) RETURNING id",
            (camera_id, zone_name, points_json_string)
        )
        zone_id = cursor.fetchone()[0]
        db_conn.commit()
        return zone_id
    except Exception as e:
        db_conn.rollback()
        current_app.logger.error(f"Error adding zone {zone_name} to camera {camera_id}: {e}")
        return None
    finally:
        if db_conn:
            db_conn.close()

def get_zones_for_camera_from_db(camera_id):
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT * FROM se_camera_zones WHERE camera_id = %s ORDER BY zone_name", (camera_id,))
        zones = cursor.fetchall()
        return zones
    finally:
        if db_conn:
            db_conn.close()

def delete_zone_from_db(zone_id):
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor()
        cursor.execute("DELETE FROM se_camera_zones WHERE id = %s RETURNING zone_name", (zone_id,))
        result = cursor.fetchone()
        db_conn.commit()
        return result[0] if result else None
    except Exception as e:
        db_conn.rollback()
        current_app.logger.error(f"Error deleting zone {zone_id}: {e}")
        return None
    finally:
        if db_conn:
            db_conn.close()

def log_detection_to_db(camera_id, count, zone_id=None, video_timestamp=None, details=""):
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor()
        cursor.execute(
            """INSERT INTO se_detection_logs (camera_id, zone_id, video_timestamp, count, details)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (camera_id, zone_id, video_timestamp, count, details)
        )
        log_id = cursor.fetchone()[0]
        db_conn.commit()
        return log_id
    except Exception as e:
        db_conn.rollback()
        current_app.logger.error(f"Error logging detection for camera {camera_id}: {e}")
        return None
    finally:
        if db_conn:
            db_conn.close()

def log_movement_to_db(camera_id, track_id, from_zone_id, to_zone_id, video_timestamp=None, movement_type='zone_transition', details=""):
    """Log movement between zones"""
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor()
        cursor.execute(
            """INSERT INTO se_movement_logs (camera_id, track_id, from_zone_id, to_zone_id, video_timestamp, movement_type, details)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (camera_id, track_id, from_zone_id, to_zone_id, video_timestamp, movement_type, details)
        )
        log_id = cursor.fetchone()[0]
        db_conn.commit()
        return log_id
    except Exception as e:
        db_conn.rollback()
        current_app.logger.error(f"Error logging movement for camera {camera_id}, track {track_id}: {e}")
        return None
    finally:
        if db_conn:
            db_conn.close()

def get_recent_movements(camera_id, limit=50):
    """Get recent movements for a camera"""
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("""
            SELECT m.*, 
                   fz.zone_name as from_zone_name, 
                   tz.zone_name as to_zone_name
            FROM se_movement_logs m
            LEFT JOIN se_camera_zones fz ON m.from_zone_id = fz.id
            LEFT JOIN se_camera_zones tz ON m.to_zone_id = tz.id
            WHERE m.camera_id = %s
            ORDER BY m.system_timestamp DESC
            LIMIT %s
        """, (camera_id, limit))
        movements = cursor.fetchall()
        return movements
    finally:
        if db_conn:
            db_conn.close()

# --- CSV Logging ---
CSV_LOG_DIR = 'movement_monitoring_logs'
os.makedirs(CSV_LOG_DIR, exist_ok=True)

def log_detection_to_csv(camera_name, zone_name, count, video_timestamp=None):
    filename = os.path.join(CSV_LOG_DIR, f"detections_{camera_name.replace(' ', '_')}_{zone_name.replace(' ', '_')}.csv")
    file_exists = os.path.isfile(filename)
    now = datetime.datetime.now()
    system_timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    video_ts_str = str(video_timestamp) if video_timestamp is not None else "N/A"

    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['SystemTimestamp', 'VideoTimestamp', 'CameraName', 'ZoneName', 'Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'SystemTimestamp': system_timestamp_str,
            'VideoTimestamp': video_ts_str,
            'CameraName': camera_name,
            'ZoneName': zone_name,
            'Count': count
        })

def log_movement_to_csv(camera_name, track_id, from_zone_name, to_zone_name, video_timestamp=None, movement_type='zone_transition'):
    """Log movement between zones to CSV"""
    filename = os.path.join(CSV_LOG_DIR, f"movements_{camera_name.replace(' ', '_')}.csv")
    file_exists = os.path.isfile(filename)
    now = datetime.datetime.now()
    system_timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    video_ts_str = str(video_timestamp) if video_timestamp is not None else "N/A"

    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['SystemTimestamp', 'VideoTimestamp', 'CameraName', 'TrackID', 'FromZone', 'ToZone', 'MovementType']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'SystemTimestamp': system_timestamp_str,
            'VideoTimestamp': video_ts_str,
            'CameraName': camera_name,
            'TrackID': track_id,
            'FromZone': from_zone_name or 'Outside',
            'ToZone': to_zone_name or 'Outside',
            'MovementType': movement_type
        })

ACTIVE_STREAMS = {} # camera_id: { 'thread': threading.Thread, 'stop_event': threading.Event, 'latest_frame': None, 'lock': threading.Lock(), 'model': None }

# --- Routes ---

@movement_monitoring_bp.route('/')
def dashboard():
    cameras = get_all_cameras_from_db()
    return render_template('movement_monitoring_dashboard.html', cameras=cameras, active_streams=ACTIVE_STREAMS.keys())

@movement_monitoring_bp.route('/manage_cameras', methods=['GET', 'POST'])
def manage_cameras():
    if request.method == 'POST':
        name = request.form.get('name')
        source_type = request.form.get('source_type')
        source_path = request.form.get('source_path_video') if source_type == 'video_file' else request.form.get('source_path_stream')

        if not name or not source_type or not source_path:
            flash('All fields are required.', 'danger')
        else:
            camera_id = add_camera_to_db(name, source_type, source_path)
            if camera_id:
                flash(f'Camera "{name}" added successfully. You may need to define zones for it.', 'success')
                return redirect(url_for('movement_monitoring.configure_camera_zones', camera_id=camera_id))
            else:
                flash(f'Error adding camera "{name}". Does it already exist?', 'danger')
        return redirect(url_for('movement_monitoring.manage_cameras'))

    cameras = get_all_cameras_from_db()
    return render_template('movement_monitoring_manage_cameras.html', cameras=cameras)


@movement_monitoring_bp.route('/camera/<int:camera_id>/configure_zones', methods=['GET', 'POST'])
def configure_camera_zones(camera_id):
    camera = get_camera_by_id_from_db(camera_id)
    if not camera:
        flash('Camera not found.', 'danger')
        return redirect(url_for('movement_monitoring.manage_cameras'))

    if request.method == 'POST':
        zone_name = request.form.get('zone_name', '').strip()
        points_json_str = request.form.get('points_json', '').strip()

        if not zone_name:
            flash('Zone name is required.', 'danger')
            return redirect(url_for('movement_monitoring.configure_camera_zones', camera_id=camera_id))
            
        if not points_json_str:
            flash('Zone points are required. Please click on the image to define at least 3 points.', 'danger')
            return redirect(url_for('movement_monitoring.configure_camera_zones', camera_id=camera_id))

        try:
            points_data = json.loads(points_json_str)
            
            # Validate points format
            if not isinstance(points_data, list):
                raise ValueError("Points must be a list of coordinates.")
                
            if len(points_data) < 3:
                flash('A zone must have at least 3 points to form a valid polygon.', 'warning')
                return redirect(url_for('movement_monitoring.configure_camera_zones', camera_id=camera_id))
            
            # Validate each point
            for i, point in enumerate(points_data):
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError(f"Point {i+1} must be a list of exactly 2 coordinates [x, y].")
                if not all(isinstance(coord, (int, float)) for coord in point):
                    raise ValueError(f"Point {i+1} coordinates must be numbers.")
                # Convert to integers for storage
                points_data[i] = [int(point[0]), int(point[1])]
            
            # Convert back to JSON string for storage
            validated_points_json = json.dumps(points_data)
            
            zone_id = add_zone_to_db(camera_id, zone_name, validated_points_json)
            if zone_id:
                flash(f'Zone "{zone_name}" with {len(points_data)} points added successfully for camera "{camera["name"]}".', 'success')
                return redirect(url_for('movement_monitoring.configure_camera_zones', camera_id=camera_id))
            else:
                flash(f'Error adding zone "{zone_name}". A zone with this name may already exist for this camera.', 'danger')
                
        except json.JSONDecodeError as e:
            flash(f'Invalid JSON format for points: {str(e)}', 'danger')
        except ValueError as e:
            flash(f'Invalid points data: {str(e)}', 'danger')
        except Exception as e:
            current_app.logger.error(f"Unexpected error adding zone {zone_name} to camera {camera_id}: {e}")
            flash('An unexpected error occurred while saving the zone. Please try again.', 'danger')
            
        return redirect(url_for('movement_monitoring.configure_camera_zones', camera_id=camera_id))

    zones = get_zones_for_camera_from_db(camera_id)
    return render_template('movement_monitoring_configure_zones.html', camera=camera, zones=zones)

@movement_monitoring_bp.route('/zone/<int:zone_id>/delete', methods=['POST'])
def delete_zone(zone_id):
    # Get zone info before deleting for better flash message
    db_conn = get_db_connection()
    try:
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT zone_name, camera_id FROM se_camera_zones WHERE id = %s", (zone_id,))
        zone_info = cursor.fetchone()
    finally:
        if db_conn:
            db_conn.close()
    
    if not zone_info:
        flash('Zone not found.', 'danger')
        return redirect(url_for('movement_monitoring.manage_cameras'))
    
    deleted_zone_name = delete_zone_from_db(zone_id)
    if deleted_zone_name:
        flash(f'Zone "{deleted_zone_name}" deleted successfully.', 'success')
    else:
        flash('Failed to delete zone.', 'danger')
    
    return redirect(url_for('movement_monitoring.configure_camera_zones', camera_id=zone_info['camera_id']))

@movement_monitoring_bp.route('/camera/<int:camera_id>/movements')
def view_movements(camera_id):
    """View recent movements for a camera"""
    camera = get_camera_by_id_from_db(camera_id)
    if not camera:
        flash('Camera not found.', 'danger')
        return redirect(url_for('movement_monitoring.dashboard'))
    
    movements = get_recent_movements(camera_id, limit=100)
    return render_template('movement_monitoring_movements.html', camera=camera, movements=movements)

# Placeholder for actual video processing logic (similar to yolo_analyzer.py or traffic_eye)
def process_camera_stream(camera_id, source_path, source_type, stop_event):
    cap = None
    camera_db_info = get_camera_by_id_from_db(camera_id)
    if not camera_db_info:
        current_app.logger.error(f"Camera {camera_id} not found for processing.")
        return

    # Load the default YOLOv8n model
    try:
        model = YOLO('yolov8n.pt') 
        current_app.logger.info(f"YOLOv8n model loaded successfully for camera {camera_db_info['name']}.")
        if camera_id in ACTIVE_STREAMS: # Store model if needed, though less critical for default models
            ACTIVE_STREAMS[camera_id]['model'] = model
    except Exception as e:
        current_app.logger.error(f"Error loading YOLOv8n model for camera {camera_db_info['name']}: {e}")
        return

    try:
        current_app.logger.info(f"Starting movement tracking for camera: {camera_db_info['name']} ({source_path})")
        if source_type == 'video_file':
            if not os.path.exists(source_path):
                current_app.logger.error(f"Video file not found: {source_path}")
                return
            cap = cv2.VideoCapture(source_path)
        elif source_type == 'stream':
            cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
        else:
            current_app.logger.error(f"Unknown source type for camera {camera_db_info['name']}: {source_type}")
            return

        if not cap.isOpened():
            current_app.logger.error(f"Could not open video source for camera {camera_db_info['name']}: {source_path}")
            return

        # Get FPS for frame skipping calculation
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if can't determine
        target_fps = 5.0  # Process at 5 FPS
        frame_skip = max(1, int(source_fps / target_fps))  # Skip frames to achieve target FPS
        
        current_app.logger.info(f"Source FPS: {source_fps:.1f}, Target FPS: {target_fps}, Frame skip: {frame_skip}")

        # Load zones for this camera
        db_zones = get_zones_for_camera_from_db(camera_id)
        zones = []
        for db_zone in db_zones:
            try:
                points = np.array(json.loads(db_zone['points']), dtype=np.int32)
                zones.append({'id': db_zone['id'], 'name': db_zone['zone_name'], 'points': points})
            except (json.JSONDecodeError, ValueError) as e:
                current_app.logger.error(f"Error parsing points for zone {db_zone['zone_name']} (ID: {db_zone['id']}): {e}")

        # Movement tracking state
        person_zones = {}  # track_id -> current_zone_id (None if outside all zones)
        person_last_seen = {}  # track_id -> frame_number (for cleanup)
        
        frame_count = 0
        processed_frame_count = 0
        max_frames_missing = int(source_fps * 3)  # Remove tracks not seen for 3 seconds

        while not stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                current_app.logger.info(f"End of stream or error for camera {camera_db_info['name']}.")
                break

            frame_count += 1
            processed_frame = frame.copy()

            # Process only every nth frame for target FPS
            if frame_count % frame_skip == 0:
                processed_frame_count += 1
                video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds
                
                # Perform tracking using YOLOv8
                yolo_results = model.track(frame, verbose=False, persist=True)
                detections_for_frame = yolo_results[0] 
                
                current_active_tracks = set()
                
                # Check if tracking data is available
                if detections_for_frame.boxes is not None and detections_for_frame.boxes.data is not None:
                    # Extract data properly from YOLOv8 results
                    bbox_xyxys = detections_for_frame.boxes.xyxy.cpu().numpy().tolist()
                    bbox_confs = detections_for_frame.boxes.conf.cpu().numpy().tolist()
                    class_ids = detections_for_frame.boxes.cls.cpu().numpy().tolist()
                    
                    # Get track IDs if available
                    if detections_for_frame.boxes.id is None:
                        track_ids = []
                    else:
                        track_ids = detections_for_frame.boxes.id.cpu().numpy().astype(int).tolist()
                    
                    # Filter for person detections (class 0 in COCO) and create proper detections list
                    person_detections = []
                    person_track_ids = []
                    
                    for i, (bbox, conf, cls_id) in enumerate(zip(bbox_xyxys, bbox_confs, class_ids)):
                        if int(cls_id) == 0 and i < len(track_ids):  # Person class is 0 in COCO
                            person_detections.append(bbox)  # [x1, y1, x2, y2]
                            person_track_ids.append(track_ids[i])

                    # Process each detected person
                    if person_track_ids:
                        for i, (track_id, bbox) in enumerate(zip(person_track_ids, person_detections)):
                            current_active_tracks.add(track_id)
                            person_last_seen[track_id] = frame_count
                            
                            x1, y1, x2, y2 = bbox
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            
                            # Determine which zone the person is currently in
                            current_zone_id = None
                            current_zone_name = None
                            for zone in zones:
                                if cv2.pointPolygonTest(zone['points'], (float(x_center), float(y_center)), False) >= 0:
                                    current_zone_id = zone['id']
                                    current_zone_name = zone['name']
                                    break
                            
                            # Check if person has moved between zones
                            previous_zone_id = person_zones.get(track_id)
                            
                            if previous_zone_id != current_zone_id:
                                # Person has moved between zones or entered/exited
                                previous_zone_name = None
                                if previous_zone_id is not None:
                                    for zone in zones:
                                        if zone['id'] == previous_zone_id:
                                            previous_zone_name = zone['name']
                                            break
                                
                                # Determine movement type
                                if previous_zone_id is None and current_zone_id is not None:
                                    movement_type = 'entry'
                                elif previous_zone_id is not None and current_zone_id is None:
                                    movement_type = 'exit'
                                else:
                                    movement_type = 'zone_transition'
                                
                                # Log the movement
                                details = f"Track {track_id} moved from {previous_zone_name or 'Outside'} to {current_zone_name or 'Outside'}"
                                current_app.logger.info(f"Camera {camera_db_info['name']}: {details}")
                                
                                # Log to database and CSV
                                log_movement_to_db(
                                    camera_id=camera_id,
                                    track_id=track_id,
                                    from_zone_id=previous_zone_id,
                                    to_zone_id=current_zone_id,
                                    video_timestamp=video_timestamp,
                                    movement_type=movement_type,
                                    details=details
                                )
                                
                                log_movement_to_csv(
                                    camera_name=camera_db_info['name'],
                                    track_id=track_id,
                                    from_zone_name=previous_zone_name,
                                    to_zone_name=current_zone_name,
                                    video_timestamp=video_timestamp,
                                    movement_type=movement_type
                                )
                            
                            # Update person's current zone
                            person_zones[track_id] = current_zone_id
                            
                            # Draw custom bounding box
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Custom smaller bounding box
                            width = x2 - x1
                            height = y2 - y1
                            padding_x = int(width * 0.1)
                            padding_y = int(height * 0.1)
                            
                            custom_x1 = max(0, x1 + padding_x)
                            custom_y1 = max(0, y1 + padding_y)
                            custom_x2 = min(processed_frame.shape[1], x2 - padding_x)
                            custom_y2 = min(processed_frame.shape[0], y2 - padding_y)
                            
                            # Color code based on zone
                            if current_zone_id is not None:
                                color = (0, 255, 0)  # Green for in-zone
                            else:
                                color = (0, 255, 255)  # Yellow for outside zones
                            
                            cv2.rectangle(processed_frame, (custom_x1, custom_y1), (custom_x2, custom_y2), color, 2)
                            
                            # Add track ID and zone label
                            label = f'ID: {track_id}'
                            if current_zone_name:
                                label += f' ({current_zone_name})'
                            
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            cv2.rectangle(processed_frame, (custom_x1, custom_y1 - label_size[1] - 10), 
                                        (custom_x1 + label_size[0] + 4, custom_y1), color, -1)
                            cv2.putText(processed_frame, label, (custom_x1 + 2, custom_y1 - 6), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Clean up tracks that haven't been seen for a while
                tracks_to_remove = []
                for track_id, last_seen_frame in person_last_seen.items():
                    if frame_count - last_seen_frame > max_frames_missing:
                        tracks_to_remove.append(track_id)
                
                for track_id in tracks_to_remove:
                    # Log exit if person was in a zone
                    if track_id in person_zones and person_zones[track_id] is not None:
                        previous_zone_id = person_zones[track_id]
                        previous_zone_name = None
                        for zone in zones:
                            if zone['id'] == previous_zone_id:
                                previous_zone_name = zone['name']
                                break
                        
                        details = f"Track {track_id} lost/exited from {previous_zone_name}"
                        current_app.logger.info(f"Camera {camera_db_info['name']}: {details}")
                        
                        log_movement_to_db(
                            camera_id=camera_id,
                            track_id=track_id,
                            from_zone_id=previous_zone_id,
                            to_zone_id=None,
                            video_timestamp=video_timestamp,
                            movement_type='exit',
                            details=details
                        )
                        
                        log_movement_to_csv(
                            camera_name=camera_db_info['name'],
                            track_id=track_id,
                            from_zone_name=previous_zone_name,
                            to_zone_name=None,
                            video_timestamp=video_timestamp,
                            movement_type='exit'
                        )
                    
                    del person_zones[track_id]
                    del person_last_seen[track_id]

                # Draw zone polygons and current counts
                zone_counts = {}
                for zone in zones:
                    zone_counts[zone['id']] = 0
                
                # Count people currently in each zone
                for track_id, zone_id in person_zones.items():
                    if zone_id is not None:
                        zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1
                
                # Draw zones with current counts
                for zone in zones:
                    cv2.polylines(processed_frame, [zone['points']], isClosed=True, color=(0, 255, 255), thickness=2)
                    current_count = zone_counts.get(zone['id'], 0)
                    zone_label = f"{zone['name']}: {current_count}"
                    cv2.putText(processed_frame, zone_label, (zone['points'][0][0], zone['points'][0][1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                continue  # Skip processing this frame if not in the target FPS
            # Store processed frame for live feed
            if camera_id in ACTIVE_STREAMS:
                with ACTIVE_STREAMS[camera_id]['lock']:
                    ACTIVE_STREAMS[camera_id]['latest_frame'] = processed_frame.copy()
            
            # Skip video display for production
            cv2.resize(processed_frame, (720, 480))  # Resize for better visibility in UI
            cv2.imshow(f"Movement Tracking: {camera_db_info['name']}", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
            
            # Small sleep for streams to yield CPU
            if source_type == 'stream':
                time.sleep(0.01)

    except Exception as e:
        current_app.logger.error(f"Error processing camera {camera_db_info.get('name', camera_id)}: {e}", exc_info=True)
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        current_app.logger.info(f"Stopped movement tracking for camera: {camera_db_info.get('name', camera_id)}")
        if camera_id in ACTIVE_STREAMS:
            try:
                del ACTIVE_STREAMS[camera_id]
                current_app.logger.info(f"Removed camera {camera_id} from ACTIVE_STREAMS.")
            except KeyError:
                current_app.logger.warning(f"Camera {camera_id} was already removed from ACTIVE_STREAMS.")

@movement_monitoring_bp.route('/camera/<int:camera_id>/start', methods=['POST'])
def start_camera_stream(camera_id):
    if camera_id in ACTIVE_STREAMS:
        flash('Camera stream is already processing.', 'warning')
        return redirect(url_for('movement_monitoring.dashboard'))

    camera = get_camera_by_id_from_db(camera_id)
    if not camera:
        flash('Camera not found.', 'danger')
        return redirect(url_for('movement_monitoring.dashboard'))

    if not camera['is_active']:
        flash(f"Camera '{camera['name']}' is marked as inactive.", 'warning')
        return redirect(url_for('movement_monitoring.dashboard'))

    stop_event = threading.Event()
    # Pass the application context for logging and config access within the thread
    thread = threading.Thread(target=process_camera_stream_with_context, 
                              args=(current_app._get_current_object(), camera_id, camera['source_path'], camera['source_type'], stop_event))

    ACTIVE_STREAMS[camera_id] = {
        'thread': thread,
        'stop_event': stop_event,
        'latest_frame': None,
        'lock': threading.Lock(),
        'model': None # Model will be loaded by the thread
    }
    thread.start()
    flash(f"Started processing for camera: {camera['name']}", 'success')
    return redirect(url_for('movement_monitoring.dashboard'))

def process_camera_stream_with_context(app_context, camera_id, source_path, source_type, stop_event):
    with app_context.app_context():
        process_camera_stream(camera_id, source_path, source_type, stop_event)

@movement_monitoring_bp.route('/camera/<int:camera_id>/stop', methods=['POST'])
def stop_camera_stream(camera_id):
    if camera_id not in ACTIVE_STREAMS:
        flash('Camera stream is not running.', 'warning')
        return redirect(url_for('movement_monitoring.dashboard'))

    current_app.logger.info(f"Attempting to stop stream for camera ID {camera_id}")
    ACTIVE_STREAMS[camera_id]['stop_event'].set()
    # ACTIVE_STREAMS[camera_id]['thread'].join(timeout=15) # Increased timeout

    # It's better to let the thread clean up ACTIVE_STREAMS[camera_id] itself upon exiting.
    # Forcing a join here can make the UI unresponsive if the thread is stuck.
    # We can check its status after a short delay if needed, but the thread should handle its removal.

    # if ACTIVE_STREAMS[camera_id]['thread'].is_alive():
    #     flash(f"Camera stream for camera ID {camera_id} is stopping. It might take a moment to fully terminate.", 'warning')
    # else:
    #     flash(f"Stop signal sent to camera ID {camera_id}. It should stop shortly.", 'success')
    #     # Explicitly delete if thread is confirmed dead and hasn't cleaned up (should not be necessary)
    #     # if camera_id in ACTIVE_STREAMS:
    #     #     del ACTIVE_STREAMS[camera_id]

    flash(f"Stop signal sent to camera ID {camera_id}. It should stop shortly.", 'success')
    return redirect(url_for('movement_monitoring.dashboard'))


@movement_monitoring_bp.route('/camera/<int:camera_id>/snapshot_for_zone_editor')
def snapshot_for_zone_editor(camera_id):
    camera = get_camera_by_id_from_db(camera_id)
    if not camera:
        return jsonify({"error": "Camera not found"}), 404

    frame_to_serve = None

    if camera_id in ACTIVE_STREAMS and ACTIVE_STREAMS[camera_id].get('latest_frame') is not None:
        with ACTIVE_STREAMS[camera_id]['lock']:
            frame_to_serve = ACTIVE_STREAMS[camera_id]['latest_frame'].copy()
    else:
        # Attempt to grab a single frame if stream is not running actively for snapshots
        current_app.logger.info(f"No active stream for snapshot (cam {camera_id}). Attempting single frame grab.")
        cap = None
        try:
            if camera['source_type'] == 'video_file':
                if not os.path.exists(camera['source_path']):
                    return jsonify({"error": f"Video file not found: {camera['source_path']}"}), 404
                cap = cv2.VideoCapture(camera['source_path'])
            elif camera['source_type'] == 'stream':
                cap = cv2.VideoCapture(camera['source_path'], cv2.CAP_FFMPEG)
            
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_to_serve = frame
                else:
                    current_app.logger.warning(f"Could not read frame for snapshot from {camera['source_path']}")
            else:
                current_app.logger.warning(f"Could not open video source for snapshot: {camera['source_path']}")
        except Exception as e:
            current_app.logger.error(f"Error grabbing single frame for snapshot (cam {camera_id}): {e}")
        finally:
            if cap:
                cap.release()

    if frame_to_serve is None:
        # Create a dummy image if no frame is available
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "No Image", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        frame_to_serve = dummy_frame
    
    # Draw existing zones on the snapshot for reference
    zones = get_zones_for_camera_from_db(camera_id)
    if zones:
        # Generate different colors for each zone
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        for i, zone in enumerate(zones):
            try:
                points = json.loads(zone['points'])
                if points and len(points) >= 3:
                    # Convert points to numpy array
                    zone_points = np.array(points, dtype=np.int32)
                    
                    # Use different color for each zone (cycle through colors)
                    color = colors[i % len(colors)]
                    
                    # Draw the zone polygon with semi-transparent fill
                    overlay = frame_to_serve.copy()
                    cv2.fillPoly(overlay, [zone_points], color)
                    cv2.addWeighted(frame_to_serve, 0.7, overlay, 0.3, 0, frame_to_serve)
                    
                    # Draw the zone outline
                    cv2.polylines(frame_to_serve, [zone_points], isClosed=True, color=color, thickness=2)
                    
                    # Add zone name label
                    label_pos = (int(zone_points[0][0]), int(zone_points[0][1]) - 10)
                    cv2.putText(frame_to_serve, zone['zone_name'], label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
            except (json.JSONDecodeError, ValueError) as e:
                current_app.logger.warning(f"Error parsing zone {zone['zone_name']} points for display: {e}")

    _, buffer = cv2.imencode('.jpg', frame_to_serve)
    response = make_response(buffer.tobytes())
    response.mimetype = 'image/jpeg'
    return response

@movement_monitoring_bp.route('/camera/<int:camera_id>/live_feed')
def live_feed(camera_id):
    """Serve the live processed frame with annotations"""
    camera = get_camera_by_id_from_db(camera_id)
    if not camera:
        return jsonify({"error": "Camera not found"}), 404

    frame_to_serve = None

    # Try to get the latest processed frame from active stream
    if camera_id in ACTIVE_STREAMS and ACTIVE_STREAMS[camera_id].get('latest_frame') is not None:
        with ACTIVE_STREAMS[camera_id]['lock']:
            frame_to_serve = ACTIVE_STREAMS[camera_id]['latest_frame'].copy()
    
    # If no active stream, try to get a single frame (same as snapshot_for_zone_editor but without zone overlays)
    if frame_to_serve is None:
        current_app.logger.info(f"No active stream for live feed (cam {camera_id}). Attempting single frame grab.")
        cap = None
        try:
            if camera['source_type'] == 'video_file':
                if not os.path.exists(camera['source_path']):
                    return jsonify({"error": f"Video file not found: {camera['source_path']}"}), 404
                cap = cv2.VideoCapture(camera['source_path'])
            elif camera['source_type'] == 'stream':
                cap = cv2.VideoCapture(camera['source_path'], cv2.CAP_FFMPEG)
            
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_to_serve = frame
                else:
                    current_app.logger.warning(f"Could not read frame for live feed from {camera['source_path']}")
            else:
                current_app.logger.warning(f"Could not open video source for live feed: {camera['source_path']}")
        except Exception as e:
            current_app.logger.error(f"Error grabbing single frame for live feed (cam {camera_id}): {e}")
        finally:
            if cap:
                cap.release()

    if frame_to_serve is None:
        # Create a dummy image if no frame is available
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "No Video Feed", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(dummy_frame, f"Camera: {camera['name']}", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        frame_to_serve = dummy_frame

    _, buffer = cv2.imencode('.jpg', frame_to_serve)
    response = make_response(buffer.tobytes())
    response.mimetype = 'image/jpeg'
    
    # Add headers to prevent caching for live feed
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Last-Modified'] = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    response.headers['ETag'] = f'"{camera_id}-{int(time.time() * 1000)}"'  # Unique ETag with timestamp
    
    return response