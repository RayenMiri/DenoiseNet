import os
import numpy as np
import tensorflow as tf
from denoisnet_model import denoisnet_model
import librosa
from sklearn.model_selection import train_test_split

# Parameters
SAMPLE_RATE = 16000  # Audio sample rate
N_FFT = 512          # STFT window size
HOP_LENGTH = 128     # STFT hop length
EPOCHS = 50          # Number of training epochs
BATCH_SIZE = 32      # Batch size
INPUT_SHAPE = (None, N_FFT // 2 + 1)  # Input shape for U-Net (time_steps, n_features)

def load_audio(file_path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def audio_to_spectrogram(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    return spectrogram

def spectrogram_to_audio(spectrogram, hop_length=HOP_LENGTH):
    phase = np.exp(1j * np.angle(librosa.stft(audio, n_fft=N_FFT, hop_length=hop_length)))
    audio = librosa.istft(spectrogram * phase, hop_length=hop_length)
    return audio

def prepare_dataset(clean_files, noise_files, snr=5, fixed_length=128):
    X, y = [], []
    print(len(noise_files))
    for clean_file in clean_files:
        clean_audio = load_audio(clean_file)
        noise_audio = load_audio(np.random.choice(noise_files))
        
        # Ensure both audios are the same length
        min_len = min(len(clean_audio), len(noise_audio))
        clean_audio = clean_audio[:min_len]
        noise_audio = noise_audio[:min_len]
        
        # Mix clean and noisy audio
        noisy_audio = clean_audio + noise_audio * np.sqrt(np.sum(clean_audio**2) / (np.sum(noise_audio**2) * 10**(snr/10)))
        
        # Convert to spectrograms
        clean_spec = audio_to_spectrogram(clean_audio)
        noisy_spec = audio_to_spectrogram(noisy_audio)
        
        # Truncate or pad spectrograms to fixed length
        if clean_spec.shape[1] > fixed_length:
            clean_spec = clean_spec[:, :fixed_length]
            noisy_spec = noisy_spec[:, :fixed_length]
        elif clean_spec.shape[1] < fixed_length:
            pad_width = fixed_length - clean_spec.shape[1]
            clean_spec = np.pad(clean_spec, ((0, 0), (0, pad_width)), mode='constant')
            noisy_spec = np.pad(noisy_spec, ((0, 0), (0, pad_width)), mode='constant')
        
        X.append(noisy_spec.T)  # Transpose to (time_steps, n_features)
        y.append(clean_spec.T)
    
    return np.array(X), np.array(y)


def main():
    # Load dataset
    clean_files = []
    for root, dirs, files in os.walk("data/clean"):
        for file in files:
            if file.endswith((".wav",".mp3",".flac")):
                clean_files.append(os.path.join(root, file))

    
    noise_files = [os.path.join("data/noise", f) for f in os.listdir("data/noise") if f.endswith((".wav","mp3","flac"))]
    """
    //ill be adding that if the noise dir has sub dirs that contain the wanted files
    noise_files = []
    for root, dirs, files in os.walk("data/noise"):
        for file in files:
            clean_files.append(os.path.join(root, file))
    """
    # Prepare dataset
    X, y = prepare_dataset(clean_files, noise_files)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model or Load a saved one
    try:
        model = tf.keras.models.load_model("denoisnet.h5", compile=True)
        print("Loading a saved model...")
    except:
        print("No saved model was found . Building a new one...")
        model = denoisnet_model(input_shape=INPUT_SHAPE)

    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Save model
    model.save("denoisnet.h5", save_format="h5")

if __name__ == "__main__":
    main()