import torch
import os

# --- PATHS ---
DATA_ROOT = "." # Root of the project or data
PROCESSED_FACES_DIR = "./processed_faces" # Directory containing 'Real' and 'Fake_...' folders
VIDEO_FEATURES_DIR = "./video_features"

# Models
SPATIAL_MODEL_PATH = "xception_spatial_stage1.pth"
SRM_MODEL_PATH = "xception_srm_stage2.pth"
TEMPORAL_MODEL_PATH = "final_bilstm_temporal.pth"

# --- HYPERPARAMETERS ---
IMG_SIZE = 299
SEQ_LENGTH = 10
BATCH_SIZE = 16  # Adjust based on GPU VRAM

# Stage 1: Spatial
SPATIAL_LR = 0.0001
SPATIAL_EPOCHS = 10

# Stage 2: SRM
SRM_LR = 0.0001
SRM_EPOCHS = 5

# Stage 3: LSTM
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_LR = 0.001
LSTM_EPOCHS = 20

# Generic / Fallbacks
LEARNING_RATE = 0.0001
EPOCHS = 10

# --- DEVICE ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- DATASETS ---
# Common dataset mapping
DATASETS = {
    # === FaceForensics++ ===
    # Original (Real) sequences
    'original_sequences/youtube/c23/videos': 0, 
    'original_sequences/actors/c23/videos': 0,
    # Manipulated (Fake) sequences
    'manipulated_sequences/Deepfakes/c23/videos': 1,
    'manipulated_sequences/Face2Face/c23/videos': 1,
    'manipulated_sequences/FaceSwap/c23/videos': 1,
    'manipulated_sequences/NeuralTextures/c23/videos': 1,
    'manipulated_sequences/DeepFakeDetection/c23/videos': 1,
    # === CelebDF-v2 (real only — fake data already plentiful from FF++) ===
    'Celeb-DF-v2/Celeb-real': 0,
    'Celeb-DF-v2/YouTube-real': 0,
}
