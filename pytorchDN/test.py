import torch
import torchaudio
from model import DenoiseUNet
from config import *
from utils import spec_to_audio

def denoise_audio(input_path, output_path):
    # Load model
    model = DenoiseUNet()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # Load and preprocess
    waveform, sr = torchaudio.load(input_path)
    waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
    transform = T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=None)
    spec = torch.log1p(torch.abs(transform(waveform))).unsqueeze(0)
    
    # Denoise
    with torch.no_grad():
        denoised_spec = torch.expm1(model(spec)).squeeze(0)
    
    # Save
    audio = spec_to_audio(denoised_spec)
    torchaudio.save(output_path, audio, sr)
    print(f"Denoised audio saved to {output_path}")

if __name__ == "__main__":
    denoise_audio("input_noisy.wav", "output_clean.wav")