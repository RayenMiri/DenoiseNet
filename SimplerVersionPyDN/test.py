import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from audio_utils import AudioProcessor
from model import DenoiseNet
import argparse
import time
import librosa

class NoiseSuppressionTester:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = DenoiseNet(n_fft=512, hop_length=128).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        self.model.eval()
        
        self.ap = AudioProcessor(n_fft=512, hop_length=128, sr=16000)
        self.sample_rate = 16000
        self.buffer_size = 16000  # 1-second buffer

    def process_audio_file(self, input_path, output_path):
        """Process a WAV file and save cleaned version"""
        # Load audio with forced mono
        audio, sr = sf.read(input_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        audio = audio.squeeze()
        
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Initialize output with proper length
        cleaned = np.zeros_like(audio)
        total_samples = len(audio)
        
        # Process in chunks with overlap-add
        for i in range(0, total_samples, self.buffer_size):
            chunk = audio[i:i+self.buffer_size]
            chunk_length = len(chunk)
            
            # Pad if needed
            if chunk_length < self.buffer_size:
                chunk = np.pad(chunk, (0, self.buffer_size - chunk_length))
                
            # Process chunk
            cleaned_chunk = self._process_chunk(chunk)
            
            # Remove padding and insert
            cleaned[i:i+chunk_length] = cleaned_chunk[:chunk_length]
        
        # Save result
        sf.write(output_path, cleaned, self.sample_rate)
        return audio, cleaned

    def real_time_test(self):
        """Real-time audio processing test"""
        print("Starting real-time processing...")
        with sd.Stream(callback=self._audio_callback,
                      samplerate=self.sample_rate,
                      channels=1,
                      blocksize=self.buffer_size):
            while True:
                time.sleep(0.01)

    def _process_chunk(self, audio):
        """Process a single audio chunk"""
        with torch.no_grad():
            spec = self.ap.preprocess(audio).unsqueeze(0).to(self.device)
            enhanced_spec = self.model(spec)
            cleaned = self.ap.postprocess(enhanced_spec.squeeze().cpu())
        return cleaned

    def _audio_callback(self, indata, outdata, frames, time, status):
        """Sounddevice callback for real-time processing"""
        if status:
            print(status)
        cleaned = self._process_chunk(indata[:, 0])
        outdata[:] = cleaned.reshape(-1, 1)

    def visualize_results(self, original, cleaned):
        """Plot waveforms and spectrograms"""
        plt.figure(figsize=(15, 10))
        
        # Waveforms
        plt.subplot(2, 2, 1)
        plt.plot(original)
        plt.title("Original Waveform")
        
        plt.subplot(2, 2, 2)
        plt.plot(cleaned)
        plt.title("Cleaned Waveform")
        
        # Spectrograms
        plt.subplot(2, 2, 3)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
        librosa.display.specshow(S, sr=self.sample_rate, x_axis='time', y_axis='log')
        plt.title("Original Spectrogram")
        
        plt.subplot(2, 2, 4)
        S_clean = librosa.amplitude_to_db(np.abs(librosa.stft(cleaned)), ref=np.max)
        librosa.display.specshow(S_clean, sr=self.sample_rate, x_axis='time', y_axis='log')
        plt.title("Cleaned Spectrogram")
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test noise suppression model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, help='Input audio file to process')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--realtime', action='store_true', help='Run real-time test')
    args = parser.parse_args()

    tester = NoiseSuppressionTester(args.model, device='cuda' if torch.cuda.is_available() else 'cpu')

    if args.realtime:
        tester.real_time_test()
    elif args.input and args.output:
        original, cleaned = tester.process_audio_file(args.input, args.output)
        print(f"Processed audio saved to {args.output}")
        tester.visualize_results(original, cleaned)
    else:
        print("Please specify either --realtime or --input/--output arguments")