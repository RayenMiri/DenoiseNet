import os
import random
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class NoiseDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, sr=16000, segment_length=16000):
        self.sr = sr
        self.segment_length = segment_length
        self.clean_files = self._get_files(clean_dir)
        self.noise_files = self._get_files(noise_dir)
        
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Load clean audio
        clean = self._load_audio(self.clean_files[idx])
        
        # Load and process noise
        noise = self._load_audio(random.choice(self.noise_files))
        
        # Mix audio with length matching
        noisy = self._mix_audio(clean, noise)
        
        # Extract aligned segments
        clean_seg, noisy_seg = self._align_segments(clean, noisy)
        
        return {
            'clean': torch.from_numpy(clean_seg).float(),
            'noisy': torch.from_numpy(noisy_seg).float()
        }
    
    def _get_files(self, path):
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
    
    def _load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        return audio.astype(np.float32)
    
    def _mix_audio(self, clean, noise, snr_db=5):
        # Ensure noise matches clean length
        if len(noise) < len(clean):
            # Repeat noise if shorter
            noise = np.tile(noise, len(clean) // len(noise) + 1)[:len(clean)]
        else:
            # Trim noise if longer
            noise = noise[:len(clean)]
            
        # Calculate scaling factor based on original noise segment
        clean_power = np.mean(clean**2)
        noise_power = np.mean(noise**2)
        
        snr = 10**(snr_db / 10)
        scale = np.sqrt(clean_power / (snr * noise_power))
        
        return clean + scale * noise
    
    def _align_segments(self, clean, noisy):
        # Ensure both arrays are same length
        min_length = min(len(clean), len(noisy))
        clean = clean[:min_length]
        noisy = noisy[:min_length]
        
        # Random crop or pad
        if min_length < self.segment_length:
            # Pad with zeros
            pad_length = self.segment_length - min_length
            return (
                np.pad(clean, (0, pad_length), mode='constant'),
                np.pad(noisy, (0, pad_length), mode='constant')
            )
        else:
            # Random crop
            start = np.random.randint(0, min_length - self.segment_length)
            return (
                clean[start:start+self.segment_length],
                noisy[start:start+self.segment_length]
            )