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
import glob
import time
from typing import Dict, List, Optional, Tuple, Union, Any


class PoseProcessor:
    """
    A class for processing pose landmarks and converting them to normalized numpy arrays.
    """
    
    def __init__(self, pose_indices: Optional[List[int]] = None, 
                 normalize_keypoints: bool = True, fill_missing_value: float = -9999.0):
        """
        Initialize the PoseProcessor.
        
        Args:
            pose_indices: List of pose landmark indices to extract. 
                         Default is [0,11,12,13,14,15,16] (nose, shoulders, elbows, wrists)
            normalize_keypoints: Whether to normalize keypoints to signing space
            fill_missing_value: Value to use for missing keypoints
        """
        self.pose_indices = pose_indices if pose_indices else [0, 11, 12, 13, 14, 15, 16]
        self.normalize_keypoints = normalize_keypoints
        self.fill_missing_value = fill_missing_value
        
        # Number of coordinates per keypoint (x, y)
        self.coords_per_keypoint = 2
        self.output_shape = (len(self.pose_indices), self.coords_per_keypoint)
    
    def normalize_pose_keypoints(self, pose_landmarks: List[List[float]]) -> List[List[float]]:
        """
        Normalize pose keypoints to signing space.
        
        Args:
            pose_landmarks: List of pose landmarks from MediaPipe
            
        Returns:
            List of normalized pose keypoints
        """
        # Extract relevant landmarks for normalization
        left_shoulder = np.array(pose_landmarks[11][:2])
        right_shoulder = np.array(pose_landmarks[12][:2])
        left_eye = np.array(pose_landmarks[2][:2])
        nose = np.array(pose_landmarks[0][:2])

        # Calculate head unit in normalized space
        head_unit = np.linalg.norm(right_shoulder - left_shoulder) / 2

        # Define signing space dimensions in normalized space
        signing_space_width = 6 * head_unit
        signing_space_height = 7 * head_unit

        # Calculate signing space bounding box in normalized space
        signing_space_top = left_eye[1] - 0.5 * head_unit
        signing_space_bottom = signing_space_top + signing_space_height
        signing_space_left = nose[0] - signing_space_width / 2
        signing_space_right = signing_space_left + signing_space_width

        # Create transformation matrix
        translation_matrix = np.array([[1, 0, -signing_space_left],
                                       [0, 1, -signing_space_top],
                                       [0, 0, 1]])
        scale_matrix = np.array([[1 / signing_space_width, 0, 0],
                                 [0, 1 / signing_space_height, 0],
                                 [0, 0, 1]])
        shift_matrix = np.array([[1, 0, -0.5],
                                 [0, 1, -0.5],
                                 [0, 0, 1]])
        transformation_matrix = shift_matrix @ scale_matrix @ translation_matrix

        # Apply transformation to pose keypoints
        normalized_keypoints = []
        for landmark in pose_landmarks:
            keypoint = np.array([landmark[0], landmark[1], 1])
            normalized_keypoint = transformation_matrix @ keypoint
            normalized_keypoints.append(normalized_keypoint[:2].tolist())

        return normalized_keypoints
    
    def process_frame_landmarks(self, frame_landmarks: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Process landmarks for a single frame.
        
        Args:
            frame_landmarks: Dictionary containing pose landmarks for one frame
            
        Returns:
            Numpy array of processed pose keypoints
        """
        if frame_landmarks is None or frame_landmarks.get('pose_landmarks') is None:
            # Return missing value array
            return np.full(self.output_shape, self.fill_missing_value).flatten()
        
        # Get pose landmarks
        pose_landmarks = frame_landmarks['pose_landmarks'][0]
        
        # Normalize keypoints if required
        if self.normalize_keypoints:
            # Take first 25 landmarks for normalization (MediaPipe pose has 33 total)
            normalized_landmarks = self.normalize_pose_keypoints(pose_landmarks[:25])
        else:
            normalized_landmarks = pose_landmarks
        
        # Extract only the specified indices
        selected_landmarks = [normalized_landmarks[i] for i in self.pose_indices]
        
        # Convert to numpy array and flatten
        frame_keypoints = np.array(selected_landmarks).flatten()
        
        return frame_keypoints
    
    def process_landmarks_sequence(self, landmarks_data: Dict[int, Any]) -> np.ndarray:
        """
        Process landmarks for an entire sequence (video).
        
        Args:
            landmarks_data: Dictionary containing landmarks for each frame
            
        Returns:
            Numpy array of shape (num_frames, num_keypoints * 2)
        """
        # Get number of frames
        if not landmarks_data:
            return np.array([])
        
        max_frame = max(landmarks_data.keys())
        num_frames = max_frame + 1
        
        video_pose_landmarks = []
        prev_pose = None
        
        for i in range(num_frames):
            frame_landmarks = landmarks_data.get(i, None)
            
            if frame_landmarks is None:
                # Use previous pose if available, otherwise use missing values
                if prev_pose is not None:
                    frame_keypoints = prev_pose
                else:
                    frame_keypoints = np.full(self.output_shape, self.fill_missing_value).flatten()
            else:
                # Process current frame
                frame_keypoints = self.process_frame_landmarks(frame_landmarks)
                if not np.all(frame_keypoints == self.fill_missing_value):
                    prev_pose = frame_keypoints
            
            video_pose_landmarks.append(frame_keypoints)
        
        # Convert to numpy array
        video_pose_landmarks = np.array(video_pose_landmarks)
        
        # Apply any post-processing (like the original code's wrist masking)
        # video_pose_landmarks = self._apply_post_processing(video_pose_landmarks)
        
        return video_pose_landmarks
    
    def _apply_post_processing(self, pose_array: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to the pose array.
        
        Args:
            pose_array: Input pose array
            
        Returns:
            Post-processed pose array
        """
        # The original code fills left and right wrist with -9999
        # This corresponds to indices 15 and 16 in the original pose landmarks
        # In our selected indices [0,11,12,13,14,15,16], wrists are at positions 5 and 6
        # Each keypoint has 2 coordinates, so wrists are at positions 10-11 and 12-13
        
        # if len(self.pose_indices) >= 7 and 15 in self.pose_indices and 16 in self.pose_indices:
        #     # Find positions of wrists in our selected indices
        #     left_wrist_idx = self.pose_indices.index(15) * 2  # *2 because each keypoint has x,y
        #     right_wrist_idx = self.pose_indices.index(16) * 2
            
        #     # Fill wrist coordinates with missing value
        #     pose_array[:, left_wrist_idx:left_wrist_idx+2] = self.fill_missing_value
        #     pose_array[:, right_wrist_idx:right_wrist_idx+2] = self.fill_missing_value
        
        return pose_array
    
    def process_landmarks_from_file(self, pose_file_path: str) -> np.ndarray:
        """
        Process landmarks from a JSON file.
        
        Args:
            pose_file_path: Path to the pose landmarks JSON file
            
        Returns:
            Numpy array of processed pose keypoints
        """
        try:
            with open(pose_file_path, 'r') as f:
                landmarks_data = json.load(f)
            
            # Convert string keys to integers
            landmarks_data = {int(k): v for k, v in landmarks_data.items()}
            
            return self.process_landmarks_sequence(landmarks_data)
            
        except Exception as e:
            print(f"Error processing {pose_file_path}: {e}")
            return np.array([])
    
    def process_and_save_landmarks(self, landmarks_data: Dict[int, Any], 
                                  output_path: str, filename: str) -> str:
        """
        Process landmarks and save to file.
        
        Args:
            landmarks_data: Dictionary containing landmarks for each frame
            output_path: Directory to save the processed landmarks
            filename: Name for the output file (without extension)
            
        Returns:
            Path to the saved file
        """
        # Process landmarks
        processed_landmarks = self.process_landmarks_sequence(landmarks_data)
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        save_path = output_dir / f"{filename}.npy"
        np.save(save_path, processed_landmarks)
        
        return str(save_path)


# Convenience functions for backward compatibility
def process_pose_landmarks(landmarks_data: Dict[int, Any], 
                          normalize: bool = True, 
                          pose_indices: Optional[List[int]] = None) -> np.ndarray:
    """
    Convenience function to process pose landmarks.
    
    Args:
        landmarks_data: Dictionary containing landmarks for each frame
        normalize: Whether to normalize keypoints to signing space
        pose_indices: List of pose landmark indices to extract
        
    Returns:
        Numpy array of processed pose keypoints
    """
    processor = PoseProcessor(pose_indices=pose_indices, normalize_keypoints=normalize)
    return processor.process_landmarks_sequence(landmarks_data)


def process_frame_pose_landmarks(
    frame_landmarks: Optional[Dict[str, Any]],
    normalize: bool = True,
    pose_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Convenience helper for realtime processing of a single frame.
    """
    processor = PoseProcessor(pose_indices=pose_indices, normalize_keypoints=normalize)
    return processor.process_frame_landmarks(frame_landmarks)


def keypoints_to_numpy(pose_file: str, pose_emb_path: str):
    """
    Original function for backward compatibility with command-line usage.
    """
    try:
        processor = PoseProcessor()
        processed_landmarks = processor.process_landmarks_from_file(pose_file)
        
        if processed_landmarks.size > 0:
            # Save the processed landmarks
            video_name = Path(pose_file).stem
            save_path = Path(pose_emb_path) / f"{video_name}.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, processed_landmarks)
            
    except Exception as e:
        print(f"Error processing {pose_file}: {e}")


# Utility functions for batch processing
def get_mp4_files(directory: str) -> List[str]:
    """Get all MP4 files in a directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')
    
    mp4_files = glob.glob(os.path.join(directory, '*.mp4'))
    return [os.path.abspath(file) for file in mp4_files]


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
    parser.add_argument('--files_list', type=str, required=True,
                        help='path to the pose file')
    parser.add_argument('--pose_features_path', type=str, required=True,
                        help='path to the pose features file')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit')
    
    args = parser.parse_args()
    start_time = time.time()
    
    # Load files list
    fixed_list = load_file(args.files_list)
    
    # Initialize processor
    processor = PoseProcessor()
    
    # Process files in batches
    video_batches = [fixed_list[i:i + args.batch_size] for i in range(0, len(fixed_list), args.batch_size)]
    
    for pose_file in video_batches[args.index]:
        pose_file_path = Path(pose_file)
        output_path = Path(args.pose_features_path) / f"{pose_file_path.stem}.npy"
        
        if output_path.exists():
            print(f"Skipping {pose_file} - output already exists")
            continue
        
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            print("Time limit reached. Stopping execution.")
            break
        
        try:
            print(f"Processing {pose_file}")
            keypoints_to_numpy(pose_file, args.pose_features_path)
            print(f"Successfully processed {pose_file}")
        except Exception as e:
            print(f"Error processing {pose_file}: {e}")


if __name__ == "__main__":
    main()
