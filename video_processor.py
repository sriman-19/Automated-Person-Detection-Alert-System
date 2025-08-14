import cv2
import numpy as np
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from face_recognition_module import detect_faces_in_frame, compare_embeddings

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def extract_frames(video_path: str, interval: int = 2) -> List[Tuple[np.ndarray, float]]:
    """
    Extract frames from a video file at specified intervals

    Args:
        video_path: Path to the video file
        interval: Interval in seconds between frames

    Returns:
        List of tuples containing (frame, timestamp)
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        logger.info(f"Video info: {frame_count} frames, {fps} fps, {duration:.2f} seconds")

        # Calculate frame interval
        frame_interval = int(fps * interval)
        if frame_interval <= 0:
            frame_interval = 1

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame at regular intervals
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                frames.append((frame, timestamp))

            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames

    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return []


def process_video_file(video_path: str, missing_persons: List[Dict[str, Any]],
                       similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Process a video file to detect missing persons

    Args:
        video_path: Path to the video file
        missing_persons: List of missing person data
        similarity_threshold: Threshold for face match confidence

    Returns:
        List of detection results
    """
    try:
        # Extract frames from the video
        frames = extract_frames(video_path)
        if not frames:
            logger.warning("No frames extracted from video")
            return []

        matches = []

        for frame, timestamp in frames:
            # Detect faces in the frame
            faces = detect_faces_in_frame(frame)

            for face in faces:
                # Skip if no embeddings were extracted
                if face['embeddings'] is None:
                    continue

                # Compare with missing persons
                for person in missing_persons:
                    # Skip if no embedding for this person
                    if 'face_embedding' not in person:
                        continue

                    # Compare embeddings
                    similarity = compare_embeddings(face['embeddings'], person['face_embedding'])

                    # Check if it's a match
                    if similarity >= similarity_threshold:
                        # Add to matches
                        match = {
                            'person_id': person['id'],
                            'frame_time': timestamp,
                            'confidence': similarity,
                            'bbox': face['bbox']
                        }
                        matches.append(match)

                        logger.info(
                            f"Match found: {person['name']} at {timestamp:.2f}s with confidence {similarity:.2f}")

        # Sort matches by confidence (descending)
        matches.sort(key=lambda x: x['confidence'], reverse=True)

        return matches

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return []


def process_live_video(camera_id: int, missing_persons: List[Dict[str, Any]],
                       similarity_threshold: float = 0.7,
                       duration: int = 30) -> List[Dict[str, Any]]:
    """
    Process live video from a camera to detect missing persons

    Args:
        camera_id: Camera ID (usually 0 for webcam)
        missing_persons: List of missing person data
        similarity_threshold: Threshold for face match confidence
        duration: Duration in seconds to process the live video

    Returns:
        List of detection results
    """
    try:
        # Open the camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {camera_id}")
            return []

        matches = []
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 5th frame to reduce CPU usage
            if frame_count % 5 == 0:
                # Detect faces in the frame
                faces = detect_faces_in_frame(frame)

                for face in faces:
                    # Skip if no embeddings were extracted
                    if face['embeddings'] is None:
                        continue

                    # Compare with missing persons
                    for person in missing_persons:
                        # Skip if no embedding for this person
                        if 'face_embedding' not in person:
                            continue

                        # Compare embeddings
                        similarity = compare_embeddings(face['embeddings'], person['face_embedding'])

                        # Check if it's a match
                        if similarity >= similarity_threshold:
                            # Get current time
                            curr_time = time.time() - start_time

                            # Add to matches
                            match = {
                                'person_id': person['id'],
                                'frame_time': curr_time,
                                'confidence': similarity,
                                'bbox': face['bbox']
                            }
                            matches.append(match)

                            logger.info(
                                f"Live match found: {person['name']} at {curr_time:.2f}s with confidence {similarity:.2f}")

            frame_count += 1

        cap.release()

        # Sort matches by confidence (descending)
        matches.sort(key=lambda x: x['confidence'], reverse=True)

        return matches

    except Exception as e:
        logger.error(f"Error processing live video: {str(e)}")
        return []
