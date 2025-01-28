import os
import random
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class NoiseDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, sr=16000, duration=1.0):
        self.sr = sr
        self.duration = duration
        self.clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir)]
        self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir)]
        
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Load clean audio as numpy array
        clean = self.load_audio(self.clean_files[idx])
        
        # Load noise and mix
        noise = self.load_audio(random.choice(self.noise_files))
        noisy = clean + 0.5 * noise  # Adjust SNR as needed
        
        # Convert to tensors
        return {
            'clean': torch.from_numpy(clean).float(),
            'noisy': torch.from_numpy(noisy).float()
        }
    
    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, duration=self.duration)
        return audio.astype(np.float32)