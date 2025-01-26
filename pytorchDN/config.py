# Hyperparameters
LEARNING_RATE = 0.0003
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 5  # For early stopping

# Model Architecture
IN_CHANNELS = 1
HIDDEN_CHANNELS = [32, 64]

# Data
SAMPLE_RATE = 16000   # Must match your audio files
AUDIO_LEN = 4         # Seconds
N_FFT = 512           # Spectrogram FFT size
HOP_LENGTH = 256      # Spectrogram hop length
NOISE_DIR = "data/noise"
CLEAN_DIR = "data/clean"

# Paths
MODEL_SAVE_PATH = "models/best_denoiser.pth"