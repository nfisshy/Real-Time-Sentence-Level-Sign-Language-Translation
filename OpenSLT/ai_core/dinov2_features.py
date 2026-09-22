import torch
import torch.nn as nn
import logging
from torchvision import transforms
from PIL import Image
import decord
from decord import VideoReader
from decord import cpu, gpu
import numpy as np
import os
import pickle
import gzip
from pathlib import Path
import argparse
import json
import csv
import glob
import time 
from typing import List, Union, Optional, Tuple

_EMBEDDER_CACHE = {}
logger = logging.getLogger(__name__)


def _cuda_state() -> str:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    try:
        return (
            f"device_count={torch.cuda.device_count()} | "
            f"current_device={torch.cuda.current_device()} | "
            f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())} | "
            f"mem_alloc_mb={torch.cuda.memory_allocated() / (1024 * 1024):.1f} | "
            f"mem_reserved_mb={torch.cuda.memory_reserved() / (1024 * 1024):.1f}"
        )
    except Exception as exc:  # pragma: no cover - debug logging only
        return f"cuda_state_error={exc}"


class DINOEmbedder:
    """
    A class for extracting DINOv2 embeddings from video frames or images.
    """
    
    def __init__(self, dino_model_path: str, batch_size: int = 128, device: Optional[str] = None):
        """
        Initialize the DINOEmbedder.
        
        Args:
            dino_model_path: Path to the fine-tuned DINOv2 model
            batch_size: Batch size for processing frames
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None
        """
        self.dino_model_path = dino_model_path
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = self._load_dino_model()
        self.model.eval()
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"DINOEmbedder initialized on device: {self.device}")
    
    def _load_dino_model(self) -> nn.Module:
        """Load the fine-tuned DINOv2 model."""
        checkpoint_path = Path(self.dino_model_path)
        checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

        # Load the original DINOv2 model with the correct architecture
        logger.info(
            "DINO load start | base_repo=facebookresearch/dinov2 | checkpoint=%s | device=%s | checkpoint_size_mb=%.1f",
            self.dino_model_path,
            self.device,
            checkpoint_size_mb,
        )
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)
        logger.info("DINO hub model loaded | checkpoint=%s", self.dino_model_path)
        
        # Load fine-tuned weights on CPU first; direct CUDA deserialization on
        # Colab is noticeably less stable and makes stalls harder to diagnose.
        logger.info("DINO checkpoint torch.load start | checkpoint=%s | map_location=cpu", self.dino_model_path)
        load_start = time.perf_counter()
        pretrained = torch.load(self.dino_model_path, map_location="cpu")
        logger.info(
            "DINO checkpoint torch.load done | checkpoint=%s | elapsed=%.3fs",
            self.dino_model_path,
            time.perf_counter() - load_start,
        )
        
        # Make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained['teacher'].items():
            if 'dino_head' in key:
                continue  # Skip dino_head layers
            else:
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value
        
        # Change shape of pos_embed
        pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
        model.pos_embed = pos_embed
        
        # Load state dict
        logger.info("DINO load_state_dict start | checkpoint=%s | tensors=%d", self.dino_model_path, len(new_state_dict))
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("DINO load_state_dict done | checkpoint=%s", self.dino_model_path)
        
        # Move model to device
        logger.info(
            "DINO model.to start | checkpoint=%s | device=%s | cuda_state_before=%s",
            self.dino_model_path,
            self.device,
            _cuda_state(),
        )
        model.to(self.device)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        logger.info(
            "DINO model.to done | checkpoint=%s | device=%s | cuda_state_after=%s",
            self.dino_model_path,
            self.device,
            _cuda_state(),
        )
        return model
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame."""
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame
        
        tensor = self.transform(image)
        # Ensure only RGB channels are considered
        return tensor[:3]
    
    def _preprocess_frames_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of frames."""
        batch_tensors = torch.stack([self._preprocess_frame(frame) for frame in frames])
        return batch_tensors.to(self.device)
    
    def extract_embeddings_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract DINOv2 embeddings from a list of frames.
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            Numpy array of embeddings with shape (num_frames, embedding_dim)
        """
        all_embeddings = []
        
        # Process frames in batches
        for idx in range(0, len(frames), self.batch_size):
            batch_frames = frames[idx:idx + self.batch_size]
            
            # Preprocess batch
            batch_tensors = self._preprocess_frames_batch(batch_frames)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensors).cpu().numpy()
            
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings
    
    def extract_embeddings_from_video(self, video_input: Union[str, VideoReader], 
                                     target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Extract DINOv2 embeddings from a video.
        
        Args:
            video_input: Either a path to video file (str) or a VideoReader object
            target_size: Target size for video frames (width, height)
            
        Returns:
            Numpy array of embeddings with shape (num_frames, embedding_dim)
        """
        # Handle different input types
        if isinstance(video_input, str):
            video_path = Path(video_input)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_input}")
            try:
                vr = VideoReader(str(video_path), width=target_size[0], height=target_size[1])
            except Exception as e:
                raise RuntimeError(f"Error loading video {video_input}: {e}")
        # elif hasattr(video_input, 'get_batch'):
        else:
            vr = video_input
        # else:
        #     raise TypeError("video_input must be either a file path (str) or a VideoReader object")
        
        total_frames = len(vr)
        all_embeddings = []
        
        # Process video in batches
        for idx in range(0, total_frames, self.batch_size):
            batch_indices = range(idx, min(idx + self.batch_size, total_frames))
            # batch_frames = vr.get_batch(batch_indices).asnumpy()
            batch_frames = vr[batch_indices]
            
            # Preprocess batch
            batch_tensors = self._preprocess_frames_batch(batch_frames)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensors).cpu().numpy()
            
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings
    
    def extract_embeddings_from_video_and_save(self, video_path: str, output_folder: str) -> str:
        """
        Extract embeddings from video and save to file.
        
        Args:
            video_path: Path to the video file
            output_folder: Folder to save the embeddings
            
        Returns:
            Path to the saved embeddings file
        """
        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Extract embeddings
        embeddings = self.extract_embeddings_from_video(video_path)
        
        # Save embeddings
        video_name = Path(video_path).stem
        np_path = Path(output_folder) / f"{video_name}.npy"
        np.save(np_path, embeddings)
        
        return str(np_path)
    
    def extract_embedding_from_single_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract DINOv2 embedding from a single image.
        
        Args:
            image: Image as numpy array or PIL Image
            
        Returns:
            Numpy array of embedding with shape (1, embedding_dim)
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy()
        
        return embedding


# Convenience functions for backward compatibility
def extract_embeddings_from_frames(frames: List[np.ndarray], dino_model_path: str, 
                                  batch_size: int = 128,
                                  embedder: Optional["DINOEmbedder"] = None) -> np.ndarray:
    """
    Convenience function to extract embeddings from frames.
    
    Args:
        frames: List of frames as numpy arrays
        dino_model_path: Path to the fine-tuned DINOv2 model
        batch_size: Batch size for processing
        
    Returns:
        Numpy array of embeddings
    """
    embedder = embedder if embedder is not None else get_dino_embedder(dino_model_path, batch_size=batch_size)
    return embedder.extract_embeddings_from_frames(frames)


def extract_embeddings_from_video(video_path: str, dino_model_path: str, 
                                 batch_size: int = 128,
                                 embedder: Optional["DINOEmbedder"] = None) -> np.ndarray:
    """
    Convenience function to extract embeddings from video.
    
    Args:
        video_path: Path to the video file
        dino_model_path: Path to the fine-tuned DINOv2 model
        batch_size: Batch size for processing
        
    Returns:
        Numpy array of embeddings
    """
    embedder = embedder if embedder is not None else get_dino_embedder(dino_model_path, batch_size=batch_size)
    return embedder.extract_embeddings_from_video(video_path)


def get_dino_embedder(
    dino_model_path: str,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> DINOEmbedder:
    """
    Reuse initialized DINOv2 models across requests.

    The model weights are large and loading them repeatedly dominates latency.
    Caching by model path, batch size, and device keeps inference behavior
    unchanged while avoiding redundant work.
    """
    resolved_device = str(device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cache_key = (os.path.abspath(dino_model_path), batch_size, resolved_device)
    if cache_key not in _EMBEDDER_CACHE:
        logger.info("DINO embedder cache miss | checkpoint=%s | batch_size=%s | device=%s", dino_model_path, batch_size, resolved_device)
        _EMBEDDER_CACHE[cache_key] = DINOEmbedder(dino_model_path, batch_size=batch_size, device=resolved_device)
    else:
        logger.info("DINO embedder cache hit | checkpoint=%s | batch_size=%s | device=%s", dino_model_path, batch_size, resolved_device)
    return _EMBEDDER_CACHE[cache_key]


def video_to_embeddings(video_path: str, output_folder: str, dino_path: str, batch_size: int = 128):
    """
    Original function for backward compatibility with command-line usage.
    """
    try:
        embedder = DINOEmbedder(dino_path, batch_size)
        embedder.extract_embeddings_from_video_and_save(video_path, output_folder)
    except Exception as e:
        print(f'Error processing {video_path}: {e}')


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
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit in seconds')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='number of videos to process in this batch')
    parser.add_argument('--files_list', type=str, required=True,
                        help='path to the files list file')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='path to the output folder')
    parser.add_argument('--dino_path', type=str, required=True,
                        help='path to the dino model')
    
    args = parser.parse_args()
    start_time = time.time()
    
    # Load files list
    fixed_list = load_file(args.files_list)
    
    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Initialize embedder
    embedder = DINOEmbedder(args.dino_path, batch_size=512)
    
    # Process videos in batches
    video_batches = [fixed_list[i:i + args.batch_size] for i in range(0, len(fixed_list), args.batch_size)]
    print(f"Total number of video batches: {len(video_batches)}")
    
    for video_path in video_batches[args.index]:
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            print("Time limit reached. Stopping execution.")
            break
        
        video_name = Path(video_path).stem
        np_path = Path(args.output_folder) / f"{video_name}.npy"
        
        if np_path.exists():
            print(f"Skipping {video_path} - output already exists")
            continue
        else:
            try:
                print(f"Processing {video_path}")
                embedder.extract_embeddings_from_video_and_save(video_path, args.output_folder)
                print(f"Successfully processed {video_path}")
            except Exception as e:
                print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    main()
