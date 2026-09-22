import logging
import os
import time
import torch
import numpy as np
import decord
import torch.nn as nn
import json
import cv2
from .kpe_mediapipe import video_holistic
from .crop_hands import HandExtractor
from .crop_face import FaceExtractor
from .dinov2_features import extract_embeddings_from_frames, get_dino_embedder
from .body_features import process_pose_landmarks
# from shubert import SignHubertModel, SignHubertConfig
from .inference import test
import subprocess

logger = logging.getLogger(__name__)


class SHuBERTProcessor:

    def __init__(self, config):
        self.config = config
        self.hand_extractor = HandExtractor()
        self.face_extractor = FaceExtractor()
        self.hand_embedder = None
        self.face_embedder = None

    def _ensure_embedder_cache(self):
        if self.hand_embedder is None:
            self.hand_embedder = get_dino_embedder(self.config['dino_hands_model_path'])
        if self.face_embedder is None:
            self.face_embedder = get_dino_embedder(self.config['dino_face_model_path'])
    
    def process_video(self, video_path):
        overall_start = time.perf_counter()
        
        # output_file = f"{output_path}/{os.path.basename(video_file)}"
    
        
        # # Target FPS is 12.5
        # cmd = [
        #     'ffmpeg',
        #     '-i', video_path,
        #     '-filter:v', 'fps=15',
        #     '-c:v', 'libx264',
        #     '-preset', 'medium',  # Balance between speed and quality
        #     '-crf', '23',  # Quality level (lower is better)
        #     '-y',  # Overwrite output file if it exists
        #     video_path
        # ]
        

        # try:
        #     subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #     print(f"Saved to {video_path} at 15 fps")
        # except subprocess.CalledProcessError as e:
        #     print(f"Error reading video {video_path}: {e}")
        
        
        
        # Step 1: Change the fps to 15 by skipping frames
        stage_start = time.perf_counter()
        signer_video = decord.VideoReader(video_path)
        
        signer_video_fps = signer_video.get_avg_fps()
        # target_fps = 15
        stride = max(1, int(round(signer_video_fps / 15.0))) # Skip frames to reach 15 FPS
        index_list = list(range(0, len(signer_video), stride))
        signer_video = signer_video.get_batch(index_list)
        signer_video = signer_video.asnumpy()
        logger.info(
            "Timing | video_load_and_resample=%.3fs | original_fps=%.2f | sampled_frames=%d | stride=%d",
            time.perf_counter() - stage_start,
            signer_video_fps,
            len(signer_video),
            stride,
        )
        
        # Step 2: Extract pose using kpe_mediapipe
        stage_start = time.perf_counter()
        landmarks = video_holistic(
            video_input=signer_video,
            face_model_path=self.config['mediapipe_face_model_path'],
            hand_model_path=self.config['mediapipe_hands_model_path'],
        )
        logger.info("Timing | mediapipe_landmarks=%.3fs", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        self._ensure_embedder_cache()
        logger.info("Timing | ensure_model_cache=%.3fs", time.perf_counter() - stage_start)
             
        # Step 3: Extract stream features
        stage_start = time.perf_counter()
        left_hand_frames, right_hand_frames = self.hand_extractor.extract_hand_frames(signer_video, landmarks)
        logger.info(
            "Timing | crop_hands=%.3fs | left_frames=%d | right_frames=%d",
            time.perf_counter() - stage_start,
            len(left_hand_frames),
            len(right_hand_frames),
        )

        stage_start = time.perf_counter()
        hand_frames = left_hand_frames + right_hand_frames
        hand_embeddings = extract_embeddings_from_frames(
            hand_frames,
            self.config['dino_hands_model_path'],
            embedder=self.hand_embedder,
        )
        left_count = len(left_hand_frames)
        left_hand_embeddings = hand_embeddings[:left_count]
        right_hand_embeddings = hand_embeddings[left_count:]
        logger.info(
            "Timing | dino_hands=%.3fs | total_hand_frames=%d",
            time.perf_counter() - stage_start,
            len(hand_frames),
        )
        del left_hand_frames, right_hand_frames

        stage_start = time.perf_counter()
        face_frames = self.face_extractor.extract_face_frames(signer_video, landmarks)
        logger.info("Timing | crop_face=%.3fs | face_frames=%d", time.perf_counter() - stage_start, len(face_frames))

        stage_start = time.perf_counter()
        face_embeddings = extract_embeddings_from_frames(
            face_frames,
            self.config['dino_face_model_path'],
            embedder=self.face_embedder,
        )
        logger.info("Timing | dino_face=%.3fs", time.perf_counter() - stage_start)
        del face_frames, signer_video
          
        stage_start = time.perf_counter()
        pose_embeddings = process_pose_landmarks(landmarks)
        logger.info("Timing | pose_features=%.3fs", time.perf_counter() - stage_start)
        del landmarks

        stage_start = time.perf_counter()
        output_text = test(face_embeddings, 
             left_hand_embeddings, 
             right_hand_embeddings, 
             pose_embeddings, 
             self.config['slt_model_config'], 
             self.config['slt_model_checkpoint'], 
             self.config['slt_tokenizer_checkpoint'], 
             self.config['temp_dir'])
        logger.info("Timing | translation_generate=%.3fs", time.perf_counter() - stage_start)
        logger.info("Timing | end_to_end_pipeline=%.3fs", time.perf_counter() - overall_start)

        return output_text
        
if __name__ == "__main__":
    config = {
        'yolov8_model_path': '/share/data/pals/shester/inference/models/yolov8n.pt',
        'dino_face_model_path': '/share/data/pals/shester/inference/models/dinov2face.pth',
        'dino_hands_model_path': '/share/data/pals/shester/inference/models/dinov2hand.pth',
        'mediapipe_face_model_path': '/share/data/pals/shester/inference/models/face_landmarker_v2_with_blendshapes.task',
        'mediapipe_hands_model_path': '/share/data/pals/shester/inference/models/hand_landmarker.task',
        'shubert_model_path': '/share/data/pals/shester/inference/models/checkpoint_836_400000.pt',
        'temp_dir': '/share/data/pals/shester/inference',
        'slt_model_config': '/share/data/pals/shester/inference/models/byt5_base/config.json',
        'slt_model_checkpoint': '/share/data/pals/shester/inference/models/checkpoint-11625',
        'slt_tokenizer_checkpoint': '/share/data/pals/shester/inference/models/byt5_base',
    }
    
    # input_clip = "/share/data/pals/shester/datasets/openasl/clips_bbox/J-0KHhPS_m4.029676-029733.mp4"  
    # input_clip = "/share/data/pals/shester/inference/recordings/sabrin30fps.mp4"
    input_clip = "/share/data/pals/shester/inference/recordings/sample_sabrina.mp4"
    processor = SHuBERTProcessor(config) 
    output_text = processor.process_video(input_clip)
    print(f"The English translation is: {output_text}")
    
# /home-nfs/shesterg/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py
# /home-nfs/shesterg/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py
