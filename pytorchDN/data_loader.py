import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from config import *

class AudioDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_files = self._get_audio_files(noisy_dir,extensions=('.wav','.flac','.mp3'))
        self.clean_files = self._get_audio_files(clean_dir,extensions=('.wav','.flac','.mp3'))[:len(self.noisy_files)]
        print( len(self.noisy_files) , len(self.clean_files))
        assert len(self.noisy_files) == len(self.clean_files), "Mismatched files"
        print(f"Using {len(self.noisy_files)} paired files")

        # Spectrogram parameters
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.target_samples = SAMPLE_RATE * AUDIO_LEN  # Fixed audio length
        self.transform = T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=None)

    def _get_audio_files(self, root_dir, extensions):
        """Retrieve audio files from a directory and its subdirectories."""
        audio_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.lower().endswith(extensions):
                    audio_files.append(os.path.join(dirpath, file))
        return sorted(audio_files)

    def _process_audio(self, waveform, sample_rate):
        # Resample to target sample rate
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        
        # Convert to mono
        waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
        
        # Trim/pad to fixed length
        if waveform.shape[-1] > self.target_samples:
            waveform = waveform[:self.target_samples]
        else:
            pad = self.target_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        
        # Compute spectrogram (freq=257, time)
        spec = torch.abs(self.transform(waveform))
        return torch.log1p(spec)  # Log-scale normalization

    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        # Process noisy audio
        noisy, sr_noisy = torchaudio.load(self.noisy_files[idx])
        spec_noisy = self._process_audio(noisy, sr_noisy).unsqueeze(0)  # (1, 257, time)
        
        # Process clean audio
        clean, sr_clean = torchaudio.load(self.clean_files[idx])
        spec_clean = self._process_audio(clean, sr_clean).unsqueeze(0)
        
        return spec_noisy, spec_clean

def get_dataloaders():
    dataset = AudioDataset(NOISE_DIR, CLEAN_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    return train_loader, val_loader