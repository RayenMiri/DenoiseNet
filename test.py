import torch
import librosa
import numpy as np
from denoisnet_model import DenoisNet  # Import your model class
import soundfile as sf  # For saving the output audio

# Load the trained model
def load_model(model_path, input_channels=1):
    model = DenoisNet(input_channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the audio file
def preprocess_audio(audio_path, target_length):
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Pad or truncate the audio to the target length
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    else:
        audio = audio[:target_length]
    
    # Convert to tensor and add batch and channel dimensions
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return audio_tensor, sr

# Denoise the audio
def denoise_audio(model, noisy_audio_tensor):
    with torch.no_grad():
        denoised_audio_tensor = model(noisy_audio_tensor)
    return denoised_audio_tensor.squeeze(0).squeeze(0).numpy()

# Save the denoised audio
def save_audio(audio, sr, output_path):
    sf.write(output_path, audio, sr)

# Main function to test the model
def test_model(model_path, noisy_audio_path, output_path, target_length=55128):
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the noisy audio
    noisy_audio_tensor, sr = preprocess_audio(noisy_audio_path, target_length)
    
    # Denoise the audio
    denoised_audio = denoise_audio(model, noisy_audio_tensor)
    
    # Save the denoised audio
    save_audio(denoised_audio, sr, output_path)
    print(f"Denoised audio saved to {output_path}")

# Example usage
if __name__ == "__main__":
    model_path = "denoise_model_final.pth"  # Path to your trained model
    noisy_audio_path = "data/test/test1.mp3"  # Path to the noisy audio file
    output_path = "denoised_output.wav"  # Path to save the denoised audio
    
    test_model(model_path, noisy_audio_path, output_path)