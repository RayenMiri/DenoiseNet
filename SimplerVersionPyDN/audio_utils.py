import torch
import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, n_fft=512, hop_length=128, sr=16000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
    
    def stft(self, audio):
        # Convert tensor to numpy array if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # Ensure audio is 1D numpy array
        audio = audio.squeeze()
        spec = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(spec)
        phase = np.angle(spec)
        return mag, phase
    
    def istft(self, mag, phase):
        spec = mag * np.exp(1j * phase)
        return librosa.istft(spec, hop_length=self.hop_length)
    
    def preprocess(self, audio):
        # Handle both numpy and tensor inputs
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        mag, phase = self.stft(audio)
        mag = torch.from_numpy(mag).float()
        phase = torch.from_numpy(phase).float()
        return torch.stack([mag, phase], dim=0)
    
    def postprocess(self, mag_phase_tensor):
        mag = mag_phase_tensor[0].detach().cpu().numpy()
        phase = mag_phase_tensor[1].detach().cpu().numpy()
        return self.istft(mag, phase)