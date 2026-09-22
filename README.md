# SHuBERT-ASL: Real-Time Sentence-Level ASL-to-English Translation
---

## 1. Objective

Sign language processing is conventionally approached at the **word level** — isolated sign language recognition (ISLR), where each sample is a short, cleanly segmented clip of a single sign. This is the natural starting point for a project: word-level annotation is straightforward to collect and label, and the model can focus purely on visual features — hand shape, hand position, motion direction, facial expression, body pose — without contending with segmentation ambiguity. This project's initial scope began from that framing.

Word-level recognition, however, has a fundamental semantic ceiling: a single isolated sign is rarely enough to convey a full communicative intent on its own. In natural ASL conversation, signers do not produce discrete signs with fixed boundaries — they produce continuous sequences combined with facial expression, mouth morphemes, eye gaze, movement rhythm, and upper-body posture, all contributing meaning simultaneously. Recognizing this limitation over the course of development motivated a shift toward **sentence-level, continuous translation** — video in, a full English sentence out — as the formulation that better preserves linguistic context. This sentence-level approach is what the final system, and this repository, implement.

The reference SHuBERT paper supports both formulations from a single pretrained encoder — a linear classification head for ISLR, or a ByT5 decoder for sentence-level translation — but only covers offline, batch fine-tuning for either; it does not address real-time inference. This repository fine-tunes SHuBERT with a ByT5 decoder for sentence-level ASL-to-English translation and adds the real-time inference layer — bounded queues, parallel perception workers, endpoint detection — required to run it live on an edge device.

---

## 2. System Architecture

```
Webcam → frame sampling (15 fps) → MediaPipe (face / hand / pose landmarks)
    → region cropping (privacy-preserving face processing)
    → DINOv2 ViT-S/14 (per-region visual features)
    → SHuBERT (temporal sign encoder, frozen)
    → linear projection → ByT5-Base decoder → English sentence
```

### 2.1. Landmark Extraction (MediaPipe)

Each frame is processed by three parallel MediaPipe branches — Face Landmarker, Hand Landmarker, Pose — rather than a single perception model, because each carries distinct linguistic information in ASL: hand shape and position carry the primary lexical signal, facial landmarks carry non-manual markers (question intonation, negation, affect), and upper-body pose stabilizes interpretation across frames. Frames with occluded or out-of-frame hands are interpolated from neighboring frames rather than dropped.

Face regions receive privacy-preserving processing before being passed downstream: the face bounding box is grayscaled and Gaussian-blurred **except for the eye and mouth regions**, which are preserved because they carry non-manual grammatical markers that a fully blurred face would destroy. Hand bounding boxes are dilated by ~20% before cropping to avoid clipping fingers and wrists at the frame boundary. All cropped regions are resized to 224×224.

### 2.2. Visual Feature Extraction (DINOv2 ViT-S/14)

Cropped face and hand regions are encoded independently through a DINOv2 ViT-S/14 backbone, producing a 384-dim vector per region per frame. Raw frames are never passed to the translation model directly — DINOv2 features remove background noise (lighting, clothing, wall clutter) and provide a more stable input than raw pixels, so the downstream translation model does not need to relearn low-level visual invariances from a comparatively small fine-tuning corpus.

Upper-body pose is represented separately and far more compactly: 7 landmarks (nose, shoulders, elbows, wrists) × 2 coordinates = a 14-dim vector, normalized relative to signing space. Pose only needs to supply coarse geometric context, not fine visual detail, so it is deliberately kept low-dimensional relative to the 384-dim hand/face streams.

### 2.3. Temporal Sign Encoding (SHuBERT)

| Parameter | Value |
|---|---|
| Transformer blocks | 12 |
| Embedding dim | 768 |
| Feed-forward dim | 3,072 |
| Attention heads | 12 |
| Total parameters | 86M |
| Encoder dropout | 0.05 |

The four per-frame streams (face, left hand, right hand, pose) are each projected to a shared 256-dim space, concatenated to 1024, then compressed to 768 — so each stream contributes comparably to the fused representation rather than the high-dimensional hand/face streams dominating the low-dimensional pose stream by sheer magnitude.

SHuBERT is a self-supervised encoder pretrained on ~1,000 hours of ASL video via masked cluster prediction, analogous to BERT/HuBERT's mask-and-predict objective applied to sign video. Its role in the pipeline: where DINOv2 answers "what does this hand look like in this frame," SHuBERT answers "what does this motion sequence mean in the context of the full sentence" — full self-attention across the sequence lets it disambiguate signs that share hand shape but differ in trajectory or timing.

### 2.4. Text Generation (ByT5-Base)

SHuBERT's contextual output is linearly projected into ByT5's hidden space, and the decoder autoregressively generates the English sentence. ByT5 operates at the byte level rather than a fixed subword vocabulary, which suits short, structurally variable output sentences without vocabulary-coverage constraints.

---

## 3. Fine-Tuning Setup

| Component | Setting |
|---|---|
| Infrastructure | 2× NVIDIA T4 (Modal serverless GPU) |
| Quantization | QLoRA, 4-bit NF4, FP16 compute dtype |
| LoRA rank | r = 4 |
| **Trainable** | ByT5-Base decoder q, v projections + cross-modal projection layer |
| **Frozen** | Full SHuBERT encoder (86M params) |
| Effective batch size | 16 × 16 (grad. accumulation) × 2 GPUs = 512 |
| Loss | Cross-entropy with label smoothing |
| Decoding | Beam width 2 |

**SHuBERT is kept fully frozen.** It was pretrained on ~1,000 hours of ASL video; the fine-tuning corpus here is ~9.8K sentences. Unfreezing 86M parameters against a dataset three orders of magnitude smaller than pretraining risks catastrophically overwriting the pretrained sign representation. Only the task-specific decoder head and the small cross-modal adapter are updated.
---

## 4. Data

| Attribute | Value |
|---|---|
| Total sentences | 9,872 |
| Unique labels | 9,746 (dataset is close to one-sample-per-sentence — avg. 1.013 samples/sentence) |
| Signers | 9 |
| Train / Val / Test split | 7,000 / 1,121 / 1,751 |
| Annotation level | Sentence-level video → English sentence pairs (no intermediate gloss) |

Each sample is a video clip paired with a full English sentence, organized directly for sign language translation rather than as an intermediate gloss-recognition task — this matches the sentence-level framing in Section 1 and avoids the error-propagation failure mode of the earlier ISLR pipeline.

---

## 5. Real-Time Inference Layer

This layer is not part of SHuBERT's reference implementation; it is the system-engineering work required to run sentence-level translation live rather than on pre-segmented offline clips.

### 5.1. Pipeline Scheduling

| Optimization | What it does |
|---|---|
| Early ingest | Frames are decoded and pushed into the pipeline as soon as a sub-batch is available, without waiting for the full HTTP request to complete |
| Bounded queues | Queue depth is capped to prevent unbounded backlog when input rate exceeds processing rate |
| Chunk merging under load | When the queue is full, incoming data is merged into the nearest pending chunk instead of queuing indefinitely |
| Dual perception workers | Two workers process landmark extraction in parallel |
| Parallel MediaPipe branches | Face / hand / pose branches run concurrently within each worker rather than sequentially |
| Reorder buffer | Chunks completed out of order (due to parallel workers) are held until temporal order can be restored before being passed downstream |
| Safe shutdown | On stop, in-flight chunks are allowed to finish processing before the session closes, preventing the last sentence from being silently dropped |

**Parallelizing the three MediaPipe branches** cuts per-frame landmark extraction from 107–111 ms to 45–48 ms (~2.3×). Running the branches sequentially, wall-clock time is the *sum* of the three; running them concurrently, it is bounded by the *slowest* branch — this is a scheduling change, not a reduction in input resolution, frame rate, or perception streams.

### 5.2. Endpoint Detection (Sentence Boundary State Machine)

Video is continuous; the system must decide when a signed sentence has ended without requiring the user to press a button.

| State | Transition condition |
|---|---|
| **Idle** | No stable signer detected |
| **Collecting** | Enter on 3/5 stable frames with active hands; false starts (< 600 ms) return to Idle |
| **Hold** | Hands have stopped moving; brief window to confirm true end-of-sentence before finalizing |
| **Finalized** | Sentence generated and output; 2 s cleanup timeout before accepting a new sentence |

Finalize triggers (any one):

- Hands disappear for ≥ 400 ms
- Hands lowered for ≥ 500 ms
- Hands idle (near-zero motion) for ≥ 900 ms

Motion thresholds: **0.004** for hand motion, **0.018** for body motion — tuned to distinguish linguistically meaningful movement from natural micro-jitter.

---

## 6. Evaluation Metric

Output is a generated sentence, not a fixed label set, so evaluation uses **BLEU** rather than classification accuracy — it captures both lexical correctness and structural closeness to the reference sentence, which a per-label accuracy metric (appropriate for ISLR) does not.

$$
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

where $p_n$ is the n-gram precision at order $n$, $w_n = \frac{1}{N}$ is the corresponding weight, and $BP$ is the brevity penalty:

$$
BP =
\begin{cases}
1 & \text{if } c > r \\
e^{(1 - r/c)} & \text{if } c \le r
\end{cases}
$$

with $c$ the length of the candidate (generated) sentence and $r$ the length of the reference sentence — the penalty discourages the model from generating unnaturally short output to inflate n-gram precision.

---

## 7. Results

| Split | BLEU |
|---|---|
| Validation | 16.7 |
| Test | 15.9 |

Training loss decreased from 2.91 to 1.47 over 8 epochs, converging in the final two epochs.

### End-to-end latency (system-level)

Measured from sentence finalization (endpoint detection) to generated text output, over 50 held-out test sentences:

| Metric | Baseline | Optimized | Improvement |
|---|---|---|---|
| Mean latency | 1.873 s | 1.354 s | −27.7% |
| P95 latency | 2.919 s | 2.130 s | −27.0% |

Decoded output was identical between baseline and optimized runs — the latency reduction comes entirely from pipeline scheduling (parallel MediaPipe, dual workers, reorder buffering), not from any reduction in input frame rate, resolution, or perception fidelity.

---

## 8. Citation

```bibtex
@inproceedings{gueuwou2025shubert,
    title={SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction},
    author={Gueuwou, Shester and M{\"u}ller, Xiaodan and Siegel, Matt and Shakhnarovich, Greg and Livescu, Karen},
    booktitle={ACL},
    year={2025}
}
```

Built on: MediaPipe, DINOv2, ByT5, HuBERT.
