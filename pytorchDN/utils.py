import torch
import torchaudio
import torchaudio.transforms as T

def hybrid_loss(clean, denoised):
    # Spectral MAE
    loss_spectral = torch.mean(torch.abs(clean - denoised))
    
    # SI-SNR (Waveform-level loss)
    inverse_transform = T.InverseSpectrogram(n_fft=512, hop_length=256)
    clean_wav = inverse_transform(clean.squeeze(1))
    denoised_wav = inverse_transform(denoised.squeeze(1))
    
    # SI-SNR calculation
    target = clean_wav - denoised_wav.mean(dim=-1, keepdim=True)
    projection = (denoised_wav * target).sum(dim=-1) / (target ** 2).sum(dim=-1) * target
    loss_sisnr = -20 * torch.log10(torch.norm(projection, dim=-1) / torch.norm(denoised_wav - projection, dim=-1)).mean()
    
    return loss_spectral + 0.5 * loss_sisnr

def spec_to_audio(spec):
    inverse_transform = T.InverseSpectrogram(n_fft=512, hop_length=256)
    return inverse_transform(spec.squeeze(0))