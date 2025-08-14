import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
import os

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize face detection with higher confidence threshold for better accuracy
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# Initialize face mesh with static_image_mode=True for better accuracy with photos
# Set refine_landmarks=True for higher quality embeddings
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


def extract_face_embeddings(image_path: str) -> Optional[np.ndarray]:
    """
    Extract face embeddings from an image using MediaPipe Face Mesh

    Args:
        image_path: Path to the image file

    Returns:
        numpy array of face embeddings or None if no face is detected
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None

        # Convert to RGB (MediaPipe requires RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with Face Mesh
        results = face_mesh.process(image_rgb)

        # Check if any face landmarks were detected
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
            logger.warning(f"No face detected in {image_path}")
            return None

        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Extract landmark coordinates as embeddings
        # These points can be used as a simple form of face embedding
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        # Convert to numpy array
        embeddings = np.array(landmarks, dtype=np.float32).flatten()

        # Normalize the embeddings
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            embeddings = embeddings / norm

        return embeddings

    except Exception as e:
        logger.error(f"Error extracting face embeddings: {str(e)}")
        return None


def detect_faces_in_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect faces in a single frame and extract their embeddings

    Args:
        frame: OpenCV image (numpy array)

    Returns:
        List of dictionaries with face information (bbox, embeddings)
    """
    try:
        # Convert to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with Face Detection
        detection_results = face_detection.process(frame_rgb)

        faces = []

        if detection_results.detections:
            for detection in detection_results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape

                # Calculate absolute coordinates
                xmin = int(bbox.xmin * w)
                ymin = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Ensure coordinates are within frame
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                width = min(width, w - xmin)
                height = min(height, h - ymin)

                # Extract face ROI
                face_roi = frame[ymin:ymin + height, xmin:xmin + width]

                # Only proceed if we have a valid ROI
                if face_roi.size == 0:
                    continue

                # Process with Face Mesh
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(face_rgb)

                # Create face entry
                face_info = {
                    'bbox': (xmin, ymin, width, height),
                    'confidence': detection.score[0],
                    'embeddings': None
                }

                # Extract embeddings if landmarks were detected
                if mesh_results.multi_face_landmarks and len(mesh_results.multi_face_landmarks) > 0:
                    landmarks = []
                    for landmark in mesh_results.multi_face_landmarks[0].landmark:
                        # Convert relative coordinates to absolute
                        landmarks.append([landmark.x, landmark.y, landmark.z])

                    # Convert to numpy array
                    embeddings = np.array(landmarks, dtype=np.float32).flatten()

                    # Normalize the embeddings
                    norm = np.linalg.norm(embeddings)
                    if norm > 0:
                        embeddings = embeddings / norm
                        face_info['embeddings'] = embeddings

                faces.append(face_info)

        return faces

    except Exception as e:
        logger.error(f"Error detecting faces in frame: {str(e)}")
        return []


def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compare two face embeddings and return similarity score

    Args:
        embedding1: First face embedding (numpy array)
        embedding2: Second face embedding (numpy array)

    Returns:
        Similarity score (0-1, where 1 is perfect match)
    """
    try:
        # Handle different array types (list vs numpy array)
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1, dtype=np.float32)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2, dtype=np.float32)

        # Check if embeddings have the same shape
        if embedding1.shape != embedding2.shape:
            logger.warning(f"Embeddings have different shapes: {embedding1.shape} vs {embedding2.shape}")
            # Try to reshape if possible
            min_len = min(embedding1.size, embedding2.size)
            embedding1 = embedding1.flatten()[:min_len]
            embedding2 = embedding2.flatten()[:min_len]

        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        # Prevent division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate similarity using scalar values to avoid array truth ambiguity
        similarity = float(dot_product) / float(norm1 * norm2)

        # Clip to [0, 1] range and return
        return max(0.0, min(similarity, 1.0))

    except Exception as e:
        logger.error(f"Error comparing embeddings: {str(e)}")
        return 0


def draw_detection_results(frame: np.ndarray, face_info: Dict[str, Any],
                           person_info: Dict[str, Any], similarity: float) -> np.ndarray:
    """
    Draw detection results on a frame

    Args:
        frame: OpenCV image (numpy array)
        face_info: Face information (bbox, confidence, etc.)
        person_info: Missing person information (name, etc.)
        similarity: Matching similarity score

    Returns:
        Frame with detection annotations
    """
    try:
        # Get bounding box
        x, y, w, h = face_info['bbox']

        # Draw bounding box
        color = (0, 255, 0) if similarity > 0.8 else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw text background
        cv2.rectangle(frame, (x, y - 60), (x + w, y), color, -1)

        # Draw text
        name = person_info.get('name', 'Unknown')
        text = f"{name}: {similarity:.2f}"
        cv2.putText(frame, text, (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw additional info
        status = person_info.get('status', 'missing')
        cv2.putText(frame, status.upper(), (x + 5, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    except Exception as e:
        logger.error(f"Error drawing detection results: {str(e)}")
        return frame
