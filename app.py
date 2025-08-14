import os
import logging
from datetime import timedelta
from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import uuid
import json
from database import (
    get_all_users, add_user, get_user_by_username,
    initialize_database, get_all_missing_persons,
    add_missing_person, get_missing_person, update_missing_person
)
from face_recognition_module import extract_face_embeddings, compare_embeddings
from video_processor import process_video_file, extract_frames

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Configure JWT
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "jwt-secret-key")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
jwt = JWTManager(app)

# Configure upload paths
UPLOAD_FOLDER = 'tmp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database on startup
initialize_database()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route: Home page
@app.route('/')
def index():
    return render_template('index.html')


# Route: Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        badge_number = request.form.get('badge_number')

        # Validation
        if not username or not password or not badge_number:
            flash('All fields are required', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')

        existing_user = get_user_by_username(username)
        if existing_user:
            flash('Username already exists', 'danger')
            return render_template('register.html')

        # Create new user
        password_hash = generate_password_hash(password)
        user_id = str(uuid.uuid4())

        user_data = {
            'id': user_id,
            'username': username,
            'password_hash': password_hash,
            'badge_number': badge_number,
            'role': 'officer'
        }

        add_user(user_data)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


# Route: Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = get_user_by_username(username)

        if not user or not check_password_hash(user['password_hash'], password):
            flash('Invalid username or password', 'danger')
            return render_template('login.html')

        # Login successful
        session['user_id'] = user['id']
        session['username'] = user['username']

        # Generate JWT token
        access_token = create_access_token(identity=user['id'])
        session['jwt_token'] = access_token

        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('login.html')


# Route: Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


# Route: Dashboard (requires login)
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard', 'warning')
        return redirect(url_for('login'))

    missing_persons = get_all_missing_persons()

    return render_template('dashboard.html',
                           username=session.get('username'),
                           missing_persons=missing_persons)


# Route: Add missing person
@app.route('/add_missing_person', methods=['GET', 'POST'])
def add_missing_person_route():
    if 'user_id' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        last_seen = request.form.get('last_seen')
        description = request.form.get('description')

        # Check if required fields are present
        if not name or not age:
            flash('Name and age are required', 'danger')
            return render_template('add_missing_person.html')

        # Check if an image was provided
        if 'photo' not in request.files:
            flash('No photo provided', 'danger')
            return render_template('add_missing_person.html')

        file = request.files['photo']

        if file.filename == '':
            flash('No photo selected', 'danger')
            return render_template('add_missing_person.html')

        if file and allowed_file(file.filename):
            # Create a unique filename
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{str(uuid.uuid4())}.{file_ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            # Save the file
            file.save(file_path)

            # Extract face embeddings from the photo
            try:
                logger.debug(f"Attempting to extract face embeddings from {file_path}")
                embeddings = extract_face_embeddings(file_path)
                if embeddings is None:
                    logger.warning(f"No face detected in the uploaded image: {file_path}")
                    flash('No face detected in the uploaded image. Please upload a clear face photo.', 'danger')
                    os.remove(file_path)  # Remove the file if no face detected
                    return render_template('add_missing_person.html')

                # Log successful detection
                logger.debug(f"Successfully extracted face embeddings with shape: {embeddings.shape}")

                # Create missing person entry
                person_id = str(uuid.uuid4())
                person_data = {
                    'id': person_id,
                    'name': name,
                    'age': age,
                    'gender': gender,
                    'last_seen': last_seen,
                    'description': description,
                    'photo_path': file_path,
                    'face_embedding': embeddings.tolist(),  # Convert numpy array to list for JSON storage
                    'added_by': session['user_id'],
                    'status': 'missing'
                }

                success = add_missing_person(person_data)
                if success:
                    flash('Missing person added successfully', 'success')
                    return redirect(url_for('view_missing_persons'))
                else:
                    logger.error("Failed to add missing person to database")
                    flash('Error saving to database. Please try again.', 'danger')
                    os.remove(file_path)  # Clean up the file
                    return render_template('add_missing_person.html')

            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                flash('Error processing the image. Please try another photo.', 'danger')

                # Clean up the file in case of error
                if os.path.exists(file_path):
                    os.remove(file_path)

                return render_template('add_missing_person.html')
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg', 'danger')
            return render_template('add_missing_person.html')

    return render_template('add_missing_person.html')


# Route: View all missing persons
@app.route('/view_missing_persons')
def view_missing_persons():
    if 'user_id' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))

    missing_persons = get_all_missing_persons()
    return render_template('view_missing_persons.html', missing_persons=missing_persons)


# Route: Clear detection results
# Route: Clear detection results
@app.route('/clear_detection_results')
def clear_detection_results():
    if 'user_id' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))

    # Clear detection results from session
    if 'detection_results' in session:
        session.pop('detection_results')

    # Clear any temporary files related to detection results
    import glob

    # Clear any temporary files related to detection results
    temp_files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'detection_*.*'))
    for file in temp_files:
        try:
            os.remove(file)
        except Exception as e:
            logger.error(f"Error removing temporary file {file}: {str(e)}")

    flash('All detection results have been cleared', 'success')
    return redirect(url_for('dashboard'))

# Route: Process video for face detection
@app.route('/process_video', methods=['GET', 'POST'])
def process_video():
    if 'user_id' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))

    # Clear previous detection results when accessing this page
    if 'detection_results' in session:
        session.pop('detection_results')

    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file provided', 'danger')
            return render_template('process_video.html')

        file = request.files['video']

        if file.filename == '':
            flash('No video selected', 'danger')
            return render_template('process_video.html')

        if file and allowed_file(file.filename):
            # Create a unique filename
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{str(uuid.uuid4())}.{file_ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            # Save the file
            file.save(file_path)

            try:
                # Process the video and find matches
                matches = process_video_file(file_path, get_all_missing_persons())

                # Store results in session for display
                session['detection_results'] = matches

                if matches:
                    flash(f'Found {len(matches)} potential matches!', 'success')
                else:
                    flash('No matches found in the video', 'info')

                return redirect(url_for('detection_results'))

            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                flash(f'Error processing video: {str(e)}', 'danger')
                return render_template('process_video.html')
        else:
            flash('Invalid file type. Allowed video types: mp4, avi, mov', 'danger')
            return render_template('process_video.html')

    return render_template('process_video.html')


# Route: Show detection results
# Route: Show detection results
@app.route('/detection_results')
def detection_results():
    if 'user_id' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))

    # Get results from session
    matches = session.get('detection_results', [])

    # If no matches found, redirect to process video page
    if not matches:
        flash('No detection results available. Please process a video first.', 'info')
        return redirect(url_for('process_video'))

    # Group matches by person_id and find the highest confidence match for each person
    best_matches = {}
    for match in matches:
        person_id = match['person_id']
        confidence = match['confidence']

        # Store the match with the highest confidence for each person
        if person_id not in best_matches or confidence > best_matches[person_id]['confidence']:
            best_matches[person_id] = match

    # Get the person with the highest overall confidence
    highest_confidence_match = None
    highest_confidence = 0

    for match in best_matches.values():
        if match['confidence'] > highest_confidence:
            highest_confidence = match['confidence']
            highest_confidence_match = match

    # Enhance match data with full person details
    enhanced_matches = []
    for match in matches:
        person_id = match['person_id']
        person = get_missing_person(person_id)

        if person:
            # Update status to "found" ONLY for the person with the highest confidence
            # and only if that confidence is high enough (>0.8)
            if (person['status'] == 'missing' and
                    highest_confidence_match and
                    person_id == highest_confidence_match['person_id'] and
                    highest_confidence >= 0.8):

                # Create a copy of the person data with updated status
                updated_person_data = person.copy()
                updated_person_data['status'] = 'found'

                # Update the person in the database
                update_success = update_missing_person(person_id, updated_person_data)

                if update_success:
                    logger.info(f"Updated status of {person['name']} (ID: {person_id}) to 'found'")
                    # Get the updated person data
                    person = get_missing_person(person_id)
                    flash(f"Status of {person['name']} updated to 'found'", 'success')
                else:
                    logger.error(f"Failed to update status of person {person_id}")

            # Add to enhanced matches
            enhanced_match = {
                'person': person,
                'confidence': match['confidence'],
                'frame_time': match['frame_time'],
                'location': match.get('location', 'Unknown')
            }
            enhanced_matches.append(enhanced_match)

    return render_template('detection_results.html', matches=enhanced_matches)

# Error handling for 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


# Error handling for 500
@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500
