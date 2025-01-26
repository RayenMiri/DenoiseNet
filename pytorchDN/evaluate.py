# evaluate.py

import torch
from model import DenoiseNet
from data_loader import get_dataloader
from config import NOISY_AUDIO_DIR, CLEAN_AUDIO_DIR, MODEL_SAVE_PATH

def evaluate():
    # Load the trained model
    model = DenoiseNet()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # Load data
    dataloader = get_dataloader(NOISY_AUDIO_DIR, CLEAN_AUDIO_DIR, batch_size=1)
    
    # Evaluate
    with torch.no_grad():
        for noisy, clean in dataloader:
            output = model(noisy)
            # Compare output with clean spectrogram
            # (You can also convert spectrograms back to audio and listen)

if __name__ == "__main__":
    evaluate()