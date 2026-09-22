import cv2
import numpy as np
import os
import pickle
import gzip
from datetime import datetime
from pathlib import Path
import decord
import argparse
import json
import time
from typing import Dict, Optional, Tuple, List, Union, Any


class FaceExtractor:
    """
    A class for extracting face regions from videos based on pose and face landmarks.
    Creates face frames with only eyes and mouth visible on grey background.
    """
    
    def __init__(self, output_size: Tuple[int, int] = (224, 224), 
                 scale_factor: float = 1.2, grey_background_color: int = 128):
        """
        Initialize the FaceExtractor.
        
        Args:
            output_size: Size of the output face frames (width, height)
            scale_factor: Scale factor for bounding box expansion
            grey_background_color: Color value for grey background (0-255)
        """
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.grey_background_color = grey_background_color
        
        # Face landmark indices for eyes and mouth
        self.left_eye_indices = [69, 168, 156, 118, 54]
        self.right_eye_indices = [168, 299, 347, 336, 301]
        self.mouth_indices = [164, 212, 432, 18]

    def resize_frame(self, frame: np.ndarray, frame_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Resize frame to specified size."""
        if frame is not None and frame.size > 0:
            return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        else:
            return None

    def calculate_bounding_box(self, landmarks: List[List[float]], indices: List[int], 
                             image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Calculate bounding box for specific landmark indices."""
        x_coordinates = [landmarks[i][0] for i in indices]
        y_coordinates = [landmarks[i][1] for i in indices]

        left = min(x_coordinates)
        right = max(x_coordinates)
        top = min(y_coordinates)
        bottom = max(y_coordinates)

        return (int(left * image_shape[1]), int(top * image_shape[0]), 
                int(right * image_shape[1]), int(bottom * image_shape[0]))

    def crop_and_paste(self, src: np.ndarray, dst: np.ndarray, 
                      src_box: Tuple[int, int, int, int], dst_origin: Tuple[int, int]):
        """Crop region from source and paste to destination."""
        x1, y1, x2, y2 = src_box
        dx, dy = dst_origin
        crop = src[y1:y2, x1:x2]
        crop_height, crop_width = crop.shape[:2]
        dst[dy:dy+crop_height, dx:dx+crop_width] = crop

    def cues_on_grey_background(self, image: np.ndarray, facial_landmarks: List[List[float]]) -> np.ndarray:
        """
        Create face frame with only eyes and mouth visible on grey background.
        
        Args:
            image: Input image as numpy array
            facial_landmarks: Face landmarks from MediaPipe
            
        Returns:
            Face frame with eyes and mouth on grey background
        """
        image_shape = image.shape
        
        # Calculate bounding boxes for facial features
        left_eye_box = self.calculate_bounding_box(facial_landmarks, self.left_eye_indices, image_shape)
        right_eye_box = self.calculate_bounding_box(facial_landmarks, self.right_eye_indices, image_shape)
        mouth_box = self.calculate_bounding_box(facial_landmarks, self.mouth_indices, image_shape)

        # Calculate the overall bounding box
        min_x = min(left_eye_box[0], right_eye_box[0], mouth_box[0])
        min_y = min(left_eye_box[1], right_eye_box[1], mouth_box[1])
        max_x = max(left_eye_box[2], right_eye_box[2], mouth_box[2])
        max_y = max(left_eye_box[3], right_eye_box[3], mouth_box[3])

        # Add padding
        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.shape[1], max_x + padding)
        max_y = min(image.shape[0], max_y + padding)

        # Make the crop a square by adjusting either width or height
        width = max_x - min_x
        height = max_y - min_y
        side_length = max(width, height)

        # Adjust to ensure square
        if width < side_length:
            extra = side_length - width
            min_x = max(0, min_x - extra // 2)
            max_x = min(image.shape[1], max_x + extra // 2)

        if height < side_length:
            extra = side_length - height
            min_y = max(0, min_y - extra // 2)
            max_y = min(image.shape[0], max_y + extra // 2)

        # Create grey background image
        grey_background = np.ones((side_length, side_length, 3), dtype=np.uint8) * self.grey_background_color

        # Crop and paste facial features onto grey background
        self.crop_and_paste(image, grey_background, left_eye_box, (left_eye_box[0]-min_x, left_eye_box[1]-min_y))
        self.crop_and_paste(image, grey_background, right_eye_box, (right_eye_box[0]-min_x, right_eye_box[1]-min_y))
        self.crop_and_paste(image, grey_background, mouth_box, (mouth_box[0]-min_x, mouth_box[1]-min_y))

        return grey_background

    def select_face(self, pose_landmarks: List[List[float]], face_landmarks: List[List[List[float]]]) -> List[List[float]]:
        """
        Select the face that is closest to the pose nose landmark.
        
        Args:
            pose_landmarks: Pose landmarks from MediaPipe
            face_landmarks: List of face landmarks from MediaPipe
            
        Returns:
            Selected face landmarks
        """
        nose_landmark_from_pose = pose_landmarks[0]  # Nose from pose
        nose_landmarks_from_face = [face_landmarks[i][0] for i in range(len(face_landmarks))]
        
        # Find closest face based on nose landmark
        distances = [np.linalg.norm(np.array(nose_landmark_from_pose) - np.array(nose_landmark)) 
                    for nose_landmark in nose_landmarks_from_face]
        closest_nose_index = np.argmin(distances)
        
        return face_landmarks[closest_nose_index]

    def extract_face_crop_from_frame(
        self,
        frame_rgb: np.ndarray,
        frame_landmarks: Optional[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """
        Extract a single face cue crop from an RGB frame.

        This helper does not keep temporal state so callers can decide whether
        to reuse the previous face crop or emit a blank frame.
        """
        if frame_landmarks is None:
            return None
        if frame_landmarks.get("pose_landmarks") is None:
            return None
        if frame_landmarks.get("face_landmarks") is None:
            return None

        selected_face = self.select_face(
            frame_landmarks["pose_landmarks"][0],
            frame_landmarks["face_landmarks"],
        )
        face_frame = self.cues_on_grey_background(frame_rgb, selected_face)
        return self.resize_frame(face_frame, self.output_size)

    def extract_face_frames(self, video_input, landmarks_data: Dict[int, Any]) -> List[np.ndarray]:
        """
        Extract face frames from video based on landmarks.
        
        Args:
            video_input: Either a path to video file (str) or a decord.VideoReader object
            landmarks_data: Dictionary containing pose and face landmarks for each frame
            
        Returns:
            List of face frames as numpy arrays
        """
        # Handle different input types
        if isinstance(video_input, str):
            video_path = Path(video_input)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_input}")
            video = decord.VideoReader(str(video_path))
        # elif hasattr(video_input, '__len__') and hasattr(video_input, '__getitem__'):
        else:
            video = video_input
        # else:
        #     raise TypeError("video_input must be either a file path (str) or a VideoReader object")
        
        face_frames = []
        prev_face_frame = None
        prev_landmarks = None
        
        for i in range(len(video)):
            # frame = video[i].asnumpy()
            frame = video[i]
            if hasattr(video, 'seek'):
                video.seek(0)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get landmarks for this frame
            frame_landmarks = landmarks_data.get(i, None)
            
            # Handle missing landmarks
            if frame_landmarks is None:
                if prev_landmarks is not None:
                    frame_landmarks = prev_landmarks
                else:
                    # Use blank frame if no landmarks available
                    face_frames.append(np.zeros((*self.output_size, 3), dtype=np.uint8))
                    continue
            else:
                prev_landmarks = frame_landmarks
            
            # Check if pose landmarks exist
            if frame_landmarks.get('pose_landmarks') is None:
                if prev_face_frame is not None:
                    face_frames.append(prev_face_frame)
                else:
                    face_frames.append(np.zeros((*self.output_size, 3), dtype=np.uint8))
                continue
            
            # Process face if face landmarks exist
            if frame_landmarks.get('face_landmarks') is not None:
                # Select the face closest to the pose
                selected_face = self.select_face(
                    frame_landmarks['pose_landmarks'][0],
                    frame_landmarks['face_landmarks']
                )
                
                # Create face frame with cues on grey background
                face_frame = self.cues_on_grey_background(frame_rgb, selected_face)
                face_frame = self.resize_frame(face_frame, self.output_size)
                face_frames.append(face_frame)
                prev_face_frame = face_frame
                
            elif prev_face_frame is not None:
                face_frames.append(prev_face_frame)
            else:
                # Use blank frame if no face landmarks
                face_frames.append(np.zeros((*self.output_size, 3), dtype=np.uint8))
        
        return face_frames

    def extract_and_save_face_video(self, video_input, landmarks_data: Dict[int, Any], 
                                   output_dir: str, video_name: Optional[str] = None) -> str:
        """
        Extract face frames and save as video file.
        
        Args:
            video_input: Either a path to video file (str) or a decord.VideoReader object
            landmarks_data: Dictionary containing pose and face landmarks for each frame
            output_dir: Directory to save the face video
            video_name: Name for output video (auto-generated if not provided)
            
        Returns:
            Path to the saved face video
        """
        # Handle video input and get FPS
        if isinstance(video_input, str):
            video_path = Path(video_input)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_input}")
            video = decord.VideoReader(str(video_path))
            if video_name is None:
                video_name = video_path.stem
        # elif hasattr(video_input, '__len__') and hasattr(video_input, '__getitem__'):
        else:
            video = video_input
            if video_name is None:
                video_name = "video"
        # else:
        #     raise TypeError("video_input must be either a file path (str) or a VideoReader object")
        
        fps = video.get_avg_fps() if hasattr(video, 'get_avg_fps') else 30.0
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define output path
        face_video_path = output_path / f"{video_name}_face.mp4"
        
        # Remove existing file
        if face_video_path.exists():
            face_video_path.unlink()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(face_video_path), fourcc, fps, self.output_size)
        
        try:
            # Extract face frames
            face_frames = self.extract_face_frames(video, landmarks_data)
            
            # Write frames to video file
            for frame in face_frames:
                writer.write(frame)
                
        finally:
            # Clean up
            writer.release()
            del writer
        
        return str(face_video_path)


# Convenience function for backward compatibility
def extract_face_frames(video_input, landmarks_data: Dict[int, Any], 
                       output_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """
    Convenience function to extract face frames from video.
    
    Args:
        video_input: Either a path to video file (str) or a decord.VideoReader object
        landmarks_data: Dictionary containing pose and face landmarks for each frame
        output_size: Size of the output face frames (width, height)
        
    Returns:
        List of face frames as numpy arrays
    """
    extractor = FaceExtractor(output_size=output_size)
    return extractor.extract_face_frames(video_input, landmarks_data)


def video_holistic(video_file: str, face_path: str, problem_file_path: str, pose_path: str):
    """
    Original function for backward compatibility with command-line usage.
    """
    try:
        video = decord.VideoReader(video_file)
        fps = video.get_avg_fps()

        video_name = Path(video_file).stem
        clip_face_path = Path(face_path) / f"{video_name}_face.mp4"
        landmark_json_path = Path(pose_path) / f"{video_name}_pose.json"

        # Load landmarks
        with open(landmark_json_path, 'r') as rd:
            landmarks_data = json.load(rd)

        # Convert string keys to integers
        landmarks_data = {int(k): v for k, v in landmarks_data.items()}

        # Extract face video
        extractor = FaceExtractor()
        extractor.extract_and_save_face_video(video, landmarks_data, face_path, video_name)

    except Exception as e:
        print(f"Error processing {video_file}: {e}")
        with open(problem_file_path, "a") as p:
            p.write(video_file + "\n")


# Utility functions for batch processing
def load_file(filename: str):
    """Load a pickled and gzipped file."""
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def is_string_in_file(file_path: str, target_string: str) -> bool:
    """Check if a string exists in a file."""
    try:
        with Path(file_path).open("r") as f:
            for line in f:
                if target_string in line:
                    return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit')
    parser.add_argument('--files_list', type=str, required=True,
                        help='files list')
    parser.add_argument('--problem_file_path', type=str, required=True,
                        help='problem file path')
    parser.add_argument('--pose_path', type=str, required=True,
                        help='pose path')
    parser.add_argument('--face_path', type=str, required=True,
                        help='face path')

    args = parser.parse_args()
    start_time = time.time()

    # Load files list
    fixed_list = load_file(args.files_list)
    
    # Create problem file if it doesn't exist
    if not os.path.exists(args.problem_file_path):
        with open(args.problem_file_path, "w") as f:
            f.write("")
    
    # Process videos in batches
    video_batches = [fixed_list[i:i + args.batch_size] for i in range(0, len(fixed_list), args.batch_size)]
    
    for video_file in video_batches[args.index]:
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            print("Time limit reached. Stopping execution.")
            break

        video_name = Path(video_file).stem
        clip_face_path = Path(args.face_path) / f"{video_name}_face.mp4"
        
        if clip_face_path.exists():
            print(f"Skipping {video_file} - output already exists")
            continue
        elif is_string_in_file(args.problem_file_path, video_file):
            print(f"Skipping {video_file} - found in problem file")
            continue
        else:
            try:
                print(f"Processing {video_file}")
                video_holistic(video_file, args.face_path, args.problem_file_path, args.pose_path)
                print(f"Successfully processed {video_file}")
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                with open(args.problem_file_path, "a") as p:
                    p.write(video_file + "\n")


if __name__ == "__main__":
    main()
