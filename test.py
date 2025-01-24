import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import soundfile as sf
import h5py

# Constants
SAMPLE_RATE = 16000  # Sample rate for audio processing
N_FFT = 512          # FFT window size
HOP_LENGTH = 128     # Hop length for STFT

# Load a pre-trained model (replace with your actual model path)
MODEL_PATH = "denoisnet.h5"

# Placeholder function for model inference
def suppress_noise(model, noisy_audio):
    # Convert audio to spectrogram
    stft = librosa.stft(noisy_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Normalize the magnitude spectrogram
    magnitude_normalized = magnitude / np.max(magnitude)

    # Transpose the spectrogram to match the model's input shape
    magnitude_normalized = magnitude_normalized.T  # Shape: (time_steps, n_features)

    # Pad the spectrogram to a compatible length
    original_length = magnitude_normalized.shape[0]
    target_length = int(np.ceil(original_length / 8) * 8)  # Ensure divisible by 8
    padding_length = target_length - original_length
    magnitude_normalized = np.pad(magnitude_normalized, ((0, padding_length), (0, 0)), mode='constant')

    # Reshape for model input (add batch dimension)
    input_data = np.expand_dims(magnitude_normalized, axis=0)  # Shape: (1, time_steps, n_features)

    # Predict the clean spectrogram
    clean_magnitude = model.predict(input_data)

    # Remove batch dimension and padding
    clean_magnitude = np.squeeze(clean_magnitude, axis=0)[:original_length, :]  # Shape: (time_steps, n_features)

    # Transpose back to (n_features, time_steps)
    clean_magnitude = clean_magnitude.T

    # Denormalize
    clean_magnitude = clean_magnitude * np.max(magnitude)

    # Reconstruct the audio using the original phase
    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft, hop_length=HOP_LENGTH)

    return clean_audio

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio

def save_audio(file_path, audio, sample_rate):
    sf.write(file_path, audio, sample_rate)

def main(input_file, output_file):
    # Load the audio file
    print(f"Loading audio file: {input_file}")
    noisy_audio = load_audio(input_file)

    # Load the pre-trained model
    print("Loading noise suppression model...")
    model = load_model(MODEL_PATH, compile=False)
    print("Model input shape:", model.input_shape)  # Should print (None, None, 257)

    # Apply noise suppression
    print("Applying noise suppression...")
    clean_audio = suppress_noise(model, noisy_audio)

    # Save the cleaned audio
    print(f"Saving cleaned audio to: {output_file}")
    save_audio(output_file, clean_audio, SAMPLE_RATE)

    print("Noise suppression complete!")

if __name__ == "__main__":
    input_file = "./data/test/test1.mp3" 
    output_file = "cleaned_audio.wav"
    
    with h5py.File("denoisnet.h5", "r") as f:
        print(f.attrs["training_config"])

    main(input_file, output_file)