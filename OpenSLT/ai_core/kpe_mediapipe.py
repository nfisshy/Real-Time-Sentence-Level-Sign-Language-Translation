import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
import decord
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple, Any

_DETECTOR_CACHE = {}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class HolisticDetector:
    """
    A class for detecting face, hand, and pose landmarks in videos using MediaPipe.
    """
    
    def __init__(self, face_model_path: str, hand_model_path: str, 
                 min_detection_confidence: float = 0.1,
                 min_hand_detection_confidence: float = 0.05,
                 max_faces: int = 1, max_hands: int = 2):
        """
        Initialize the HolisticDetector with model paths and configuration.
        
        Args:
            face_model_path: Path to the face detection model
            hand_model_path: Path to the hand detection model
            min_detection_confidence: Minimum confidence for pose detection
            min_hand_detection_confidence: Minimum confidence for hand detection
            max_faces: Maximum number of faces to detect
            max_hands: Maximum number of hands to detect
        """
        self.face_model_path = face_model_path
        self.hand_model_path = hand_model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.max_faces = max_faces
        self.max_hands = max_hands
        self.parallel_detection_enabled = _env_flag(
            "SIGN_TRANSLATION_MEDIAPIPE_PARALLEL_DETECTORS",
            True,
        )
        self.parallel_workers = max(
            1,
            int(os.environ.get("SIGN_TRANSLATION_MEDIAPIPE_PARALLEL_WORKERS", "3")),
        )
        self.executor = None
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize the MediaPipe detectors."""
        # Initialize face detector
        base_options_face = python.BaseOptions(model_asset_path=self.face_model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=self.max_faces
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(options_face)

        # Initialize hand detector
        base_options_hand = python.BaseOptions(model_asset_path=self.hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options_hand)

        # Initialize holistic model for pose
        self.mp_holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence
        )
        if self.parallel_detection_enabled:
            self.executor = ThreadPoolExecutor(max_workers=self.parallel_workers)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        self.close()

    def _detect_pose(self, image: np.ndarray):
        return self.mp_holistic.process(image)

    def _detect_face(self, image: np.ndarray, timestamp_ms: Optional[int] = None):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        return self.face_detector.detect(mp_image)

    def _detect_hand(self, image: np.ndarray, timestamp_ms: Optional[int] = None):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        return self.hand_detector.detect(mp_image)

    def _serialize_landmarks(
        self,
        results,
        face_prediction,
        hand_prediction,
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        bounding_boxes = {}
        landmarks_data = {}

        # Process face landmarks
        if face_prediction.face_landmarks:
            bounding_boxes['#face'] = len(face_prediction.face_landmarks)
            landmarks_data['face_landmarks'] = []
            for face in face_prediction.face_landmarks:
                landmarks_face = [[landmark.x, landmark.y, landmark.z] for landmark in face]
                landmarks_data['face_landmarks'].append(landmarks_face)
        else:
            bounding_boxes['#face'] = 0
            landmarks_data['face_landmarks'] = None

        # Process hand landmarks
        if hand_prediction.hand_landmarks:
            bounding_boxes['#hands'] = len(hand_prediction.hand_landmarks)
            landmarks_data['hand_landmarks'] = []
            for hand in hand_prediction.hand_landmarks:
                landmarks_hand = [[landmark.x, landmark.y, landmark.z] for landmark in hand]
                landmarks_data['hand_landmarks'].append(landmarks_hand)
        else:
            bounding_boxes['#hands'] = 0
            landmarks_data['hand_landmarks'] = None

        # Process pose landmarks
        if results.pose_landmarks:
            bounding_boxes['#pose'] = 1
            landmarks_data['pose_landmarks'] = []
            pose_landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]
            landmarks_data['pose_landmarks'].append(pose_landmarks)
        else:
            bounding_boxes['#pose'] = 0
            landmarks_data['pose_landmarks'] = None

        return bounding_boxes, landmarks_data

    def _detect_frame_landmarks_with_timing(
        self,
        image: np.ndarray,
        frame_timestamp_ms: Optional[int] = None,
    ) -> Tuple[Dict[str, int], Dict[str, Any], Dict[str, float]]:
        timing = {
            "pose_elapsed_seconds": 0.0,
            "face_elapsed_seconds": 0.0,
            "hand_elapsed_seconds": 0.0,
        }

        if self.executor is None:
            stage_start = time.perf_counter()
            results = self._detect_pose(image)
            timing["pose_elapsed_seconds"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            face_prediction = self._detect_face(image, frame_timestamp_ms)
            timing["face_elapsed_seconds"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            hand_prediction = self._detect_hand(image, frame_timestamp_ms)
            timing["hand_elapsed_seconds"] = time.perf_counter() - stage_start
            return (*self._serialize_landmarks(results, face_prediction, hand_prediction), timing)

        def run_pose():
            stage_start = time.perf_counter()
            results = self._detect_pose(image)
            return results, time.perf_counter() - stage_start

        def run_face():
            stage_start = time.perf_counter()
            results = self._detect_face(image, frame_timestamp_ms)
            return results, time.perf_counter() - stage_start

        def run_hand():
            stage_start = time.perf_counter()
            results = self._detect_hand(image, frame_timestamp_ms)
            return results, time.perf_counter() - stage_start

        future_pose = self.executor.submit(run_pose)
        future_face = self.executor.submit(run_face)
        future_hand = self.executor.submit(run_hand)

        try:
            results, timing["pose_elapsed_seconds"] = future_pose.result()
            face_prediction, timing["face_elapsed_seconds"] = future_face.result()
            hand_prediction, timing["hand_elapsed_seconds"] = future_hand.result()
        except Exception:
            # Fall back to the sequential path so transient threading issues do not
            # take the whole pipeline down.
            stage_start = time.perf_counter()
            results = self._detect_pose(image)
            timing["pose_elapsed_seconds"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            face_prediction = self._detect_face(image, frame_timestamp_ms)
            timing["face_elapsed_seconds"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            hand_prediction = self._detect_hand(image, frame_timestamp_ms)
            timing["hand_elapsed_seconds"] = time.perf_counter() - stage_start

        return (*self._serialize_landmarks(results, face_prediction, hand_prediction), timing)

    def _build_timing_summary(self, branch_timings: list[Dict[str, float]]) -> Dict[str, Any]:
        if not branch_timings:
            return {
                "pose_elapsed_seconds": 0.0,
                "face_elapsed_seconds": 0.0,
                "hand_elapsed_seconds": 0.0,
                "pose_avg_ms": 0.0,
                "face_avg_ms": 0.0,
                "hand_avg_ms": 0.0,
                "frame_count": 0,
                "parallel_detection_enabled": bool(self.executor is not None),
                "parallel_workers": self.parallel_workers if self.executor is not None else 1,
            }

        pose_total = sum(item["pose_elapsed_seconds"] for item in branch_timings)
        face_total = sum(item["face_elapsed_seconds"] for item in branch_timings)
        hand_total = sum(item["hand_elapsed_seconds"] for item in branch_timings)
        frame_count = len(branch_timings)

        return {
            "pose_elapsed_seconds": round(pose_total, 3),
            "face_elapsed_seconds": round(face_total, 3),
            "hand_elapsed_seconds": round(hand_total, 3),
            "pose_avg_ms": round((pose_total / frame_count) * 1000.0, 3),
            "face_avg_ms": round((face_total / frame_count) * 1000.0, 3),
            "hand_avg_ms": round((hand_total / frame_count) * 1000.0, 3),
            "frame_count": frame_count,
            "parallel_detection_enabled": bool(self.executor is not None),
            "parallel_workers": self.parallel_workers if self.executor is not None else 1,
        }

    def detect_frame_landmarks(self, image: np.ndarray, frame_timestamp_ms: Optional[int] = None) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Detect landmarks in a single frame.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (bounding_boxes_count, landmarks_data)
        """
        bounding_boxes, landmarks_data, _timing = self._detect_frame_landmarks_with_timing(image, frame_timestamp_ms=frame_timestamp_ms)
        return bounding_boxes, landmarks_data

    def process_video(self, video_input, save_results: bool = False, 
                     output_dir: Optional[str] = None, video_name: Optional[str] = None,
                     return_timing: bool = False):
        """
        Process a video and extract landmarks from all frames.
        
        Args:
            video_input: Either a path to video file (str) or a decord.VideoReader object
            save_results: Whether to save results to files
            output_dir: Directory to save results (required if save_results=True)
            video_name: Name for output files (required if save_results=True and video_input is VideoReader)
            
        Returns:
            Dictionary containing landmarks for each frame
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If save_results=True but output_dir is None, or if video_name is None when needed
            TypeError: If video_input is neither string nor VideoReader
        """
        if save_results and output_dir is None:
            raise ValueError("output_dir must be provided when save_results=True")
        
        # Handle different input types
        if isinstance(video_input, str):
            # Input is a file path
            video_path = Path(video_input)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_input}")
            
            try:
                video = decord.VideoReader(str(video_path))
            except Exception as e:
                raise RuntimeError(f"Error loading video {video_input}: {e}")
                
            file_name = video_path.stem
            
        # elif hasattr(video_input, '__len__') and hasattr(video_input, '__getitem__'):
        else:
            # Input is a VideoReader object or similar
            video = video_input
            if save_results and video_name is None:
                raise ValueError("video_name must be provided when save_results=True and video_input is a VideoReader object")
            file_name = video_name or "video"
            
        # else:
        #     raise TypeError("video_input must be either a file path (str) or a VideoReader object")
        
        result_dict = {}
        stats = {}
        branch_timings = []
        
        # Process each frame
        for i in range(len(video)):
            try:
                # frame_rgb = video[i].asnumpy()
                frame_rgb = video[i]
                if hasattr(video, 'seek'):
                    video.seek(0)
                bounding_boxes, landmarks, timing = self._detect_frame_landmarks_with_timing(
                    frame_rgb,
                    frame_timestamp_ms=int(round((i / 30.0) * 1000.0)),
                )
                result_dict[i] = landmarks
                stats[i] = bounding_boxes
                branch_timings.append(timing)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                result_dict[i] = None
                stats[i] = {'#face': 0, '#hands': 0, '#pose': 0}
        
        # Save results if requested
        if save_results:
            self._save_results(file_name, result_dict, stats, output_dir)
        
        if return_timing:
            return result_dict, self._build_timing_summary(branch_timings)
        return result_dict
    
    def process_video_frames(self, frames: list, save_results: bool = False,
                           output_dir: Optional[str] = None, video_name: str = "video",
                           return_timing: bool = False) -> Dict[int, Any]:
        """
        Process a list of frames and extract landmarks.
        
        Args:
            frames: List of frame images as numpy arrays
            save_results: Whether to save results to files
            output_dir: Directory to save results (required if save_results=True)
            video_name: Name for output files
            
        Returns:
            Dictionary containing landmarks for each frame
        """
        if save_results and output_dir is None:
            raise ValueError("output_dir must be provided when save_results=True")
        
        result_dict = {}
        stats = {}
        branch_timings = []
        
        # Process each frame
        for i, frame in enumerate(frames):
            try:
                bounding_boxes, landmarks, timing = self._detect_frame_landmarks_with_timing(
                    frame,
                    frame_timestamp_ms=int(round((i / 30.0) * 1000.0)),
                )
                result_dict[i] = landmarks
                stats[i] = bounding_boxes
                branch_timings.append(timing)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                result_dict[i] = None
                stats[i] = {'#face': 0, '#hands': 0, '#pose': 0}
        
        # Save results if requested
        if save_results:
            self._save_results(video_name, result_dict, stats, output_dir)
        
        if return_timing:
            return result_dict, self._build_timing_summary(branch_timings)
        return result_dict

    def _save_results(self, video_name: str, landmarks_data: Dict, stats_data: Dict, output_dir: str):
        """Save landmarks and stats to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save landmarks
        landmarks_file = output_path / f"{video_name}_pose.json"
        with open(landmarks_file, 'w') as f:
            json.dump(landmarks_data, f)
        
        # Save stats
        stats_file = output_path / f"{video_name}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f)

    def compute_video_stats(self, landmarks_data: Dict) -> Dict[str, Any]:
        """
        Compute statistics from landmarks data.
        
        Args:
            landmarks_data: Dictionary containing landmarks for each frame
            
        Returns:
            Dictionary containing frame-by-frame stats and maximums
        """
        stats = {}
        max_counts = {'#face': 0, '#hands': 0, '#pose': 0}
        
        for frame, landmarks in landmarks_data.items():
            if landmarks is None:
                presence = {'#face': 0, '#hands': 0, '#pose': 0}
            else:
                presence = {
                    '#face': len(landmarks.get('face_landmarks', [])) if landmarks.get('face_landmarks') else 0,
                    '#hands': len(landmarks.get('hand_landmarks', [])) if landmarks.get('hand_landmarks') else 0,
                    '#pose': len(landmarks.get('pose_landmarks', [])) if landmarks.get('pose_landmarks') else 0
                }
            stats[frame] = presence
            
            # Update max counts
            for key in max_counts:
                max_counts[key] = max(max_counts[key], presence[key])
        
        stats['max'] = max_counts
        return stats


# Convenience function for backward compatibility and simple usage
def video_holistic(video_input, face_model_path: str, hand_model_path: str,
                  save_results: bool = False, output_dir: Optional[str] = None,
                  video_name: Optional[str] = None,
                  return_timing: bool = False) -> Dict[int, Any]:
    """
    Convenience function to process a video and extract holistic landmarks.
    
    Args:
        video_input: Either a path to video file (str) or a decord.VideoReader object
        face_model_path: Path to the face detection model
        hand_model_path: Path to the hand detection model
        save_results: Whether to save results to files
        output_dir: Directory to save results
        video_name: Name for output files (required if save_results=True and video_input is VideoReader)
        
    Returns:
        Dictionary containing landmarks for each frame
    """
    cache_key = (str(Path(face_model_path).resolve()), str(Path(hand_model_path).resolve()))
    detector = _DETECTOR_CACHE.get(cache_key)
    if detector is None:
        detector = HolisticDetector(face_model_path, hand_model_path)
        _DETECTOR_CACHE[cache_key] = detector
    return detector.process_video(video_input, save_results, output_dir, video_name, return_timing=return_timing)


# Utility functions for batch processing
def load_file(filename: str):
    """Load a pickled and gzipped file."""
    import pickle
    import gzip
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
    import argparse
    import time
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--pose_path', type=str, required=True,
                        help='path to where the pose data will be saved')
    parser.add_argument('--stats_path', type=str, required=True,
                        help='path to where the stats data will be saved')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit')
    parser.add_argument('--files_list', type=str, required=True,
                        help='files list')
    parser.add_argument('--problem_file_path', type=str, required=True,
                        help='problem file path')
    parser.add_argument('--face_model_path', type=str, required=True,
                        help='face model path')
    parser.add_argument('--hand_model_path', type=str, required=True,
                        help='hand model path')

    args = parser.parse_args()
    
    start_time = time.time()

    # Initialize detector
    detector = HolisticDetector(args.face_model_path, args.hand_model_path)

    # Load the files list
    fixed_list = load_file(args.files_list)

    # Create folders if they do not exist
    Path(args.pose_path).mkdir(parents=True, exist_ok=True)
    Path(args.stats_path).mkdir(parents=True, exist_ok=True)

    # Create problem file if it doesn't exist
    if not os.path.exists(args.problem_file_path):
        with open(args.problem_file_path, 'w') as f:
            pass

    # Process videos in batches
    video_batches = [fixed_list[i:i + args.batch_size] for i in range(0, len(fixed_list), args.batch_size)]
    
    for video_file in video_batches[args.index]:
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            print("Time limit reached. Stopping execution.")
            break

        # Check if output files already exist
        video_name = Path(video_file).stem
        landmark_json_path = Path(args.pose_path) / f"{video_name}_pose.json"
        stats_json_path = Path(args.stats_path) / f"{video_name}_stats.json"

        if landmark_json_path.exists() and stats_json_path.exists():
            print(f"Skipping {video_file} - output files already exist")
            continue
        elif is_string_in_file(args.problem_file_path, video_file):
            print(f"Skipping {video_file} - found in problem file")
            continue
        else:
            try:
                print(f"Processing {video_file}")
                result_dict = detector.process_video(
                    video_file_path=video_file,
                    save_results=True,
                    output_dir=args.pose_path
                )
                
                # Also save stats separately for compatibility
                stats = detector.compute_video_stats(result_dict)
                with open(stats_json_path, 'w') as f:
                    json.dump(stats, f)
                    
                print(f"Successfully processed {video_file}")
                
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                # Add to problem file
                with open(args.problem_file_path, "a") as p:
                    p.write(video_file + "\n")


if __name__ == "__main__":
    main()
