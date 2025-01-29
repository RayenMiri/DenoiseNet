import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from model import DenoiseNet
from audio_utils import AudioProcessor

class RealTimeProcessor:
    def __init__(self, model_path, buffer_size=16000, save_path="denoised_audio.wav", device='cpu'):
        self.device = device
        self.buffer_size = buffer_size
        self.save_path = save_path
        self.recorded_audio = []  # Store processed audio chunks

        # Load model
        self.model = DenoiseNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.ap = AudioProcessor()
        self.buffer = np.zeros((buffer_size,), dtype=np.float32)

    def process_frame(self, frame):
        """Process a single frame through the model."""
        with torch.no_grad():
            spec = self.ap.preprocess(frame).unsqueeze(0).to(self.device)
            enhanced_spec = self.model(spec)
            enhanced_audio = self.ap.postprocess(enhanced_spec.squeeze().cpu())
        return enhanced_audio

    def callback(self, indata, outdata, frames, time, status):
        """Audio processing callback function."""
        if status:
            print(f"Stream status: {status}")

        if np.all(indata == 0):
            print("Warning: No audio detected! Check your microphone.")

        self.buffer = np.roll(self.buffer, -frames)
        self.buffer[-frames:] = indata[:, 0]

        enhanced = self.process_frame(self.buffer)
        denoised_chunk = enhanced[-frames:].reshape(-1, 1)
        outdata[:] = denoised_chunk

        self.recorded_audio.append(denoised_chunk.copy())  # Store processed frames

    def save_audio(self):
        """Save the denoised audio if any was recorded."""
        if not self.recorded_audio:
            print("No audio recorded. Skipping save.")
            return
        
        denoised_audio = np.concatenate(self.recorded_audio, axis=0)  # Merge stored frames
        sf.write(self.save_path, denoised_audio, self.ap.sr)  # Save as WAV
        print(f"Saved denoised audio to {self.save_path}")

    def start(self):
        """Start real-time audio processing."""
        try:
            with sd.Stream(callback=self.callback, blocksize=self.buffer_size,
                           samplerate=self.ap.sr, channels=1):
                print("Processing... Press Ctrl+C to stop")
                while True:
                    pass
        except KeyboardInterrupt:
            print("\nStopping and saving audio...")
            self.save_audio()  # Save audio on exit

if __name__ == '__main__':
    processor = RealTimeProcessor('final_model.pth', device='cpu')
    processor.start()
