import torch
import numpy as np
import sounddevice as sd
from model import DenoiseNet
from audio_utils import AudioProcessor

class RealTimeProcessor:
    def __init__(self, model_path, buffer_size=16000, device='cpu'):
        self.device = device
        self.buffer_size = buffer_size
        self.model = DenoiseNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.ap = AudioProcessor()
        self.buffer = np.zeros((buffer_size,), dtype=np.float32)
        
    def process_frame(self, frame):
        with torch.no_grad():
            spec = self.ap.preprocess(frame).unsqueeze(0).to(self.device)
            enhanced_spec = self.model(spec)
            enhanced_audio = self.ap.postprocess(enhanced_spec.squeeze().cpu())
        return enhanced_audio
    
    def callback(self, indata, outdata, frames, time, status):
        self.buffer = np.roll(self.buffer, -frames)
        self.buffer[-frames:] = indata[:, 0]
        enhanced = self.process_frame(self.buffer)
        outdata[:] = enhanced[-frames:].reshape(-1, 1)

    def start(self):
        with sd.Stream(callback=self.callback, blocksize=self.buffer_size,
                       samplerate=self.ap.sr, channels=1):
            print("Processing... Press Ctrl+C to stop")
            while True:
                pass

if __name__ == '__main__':
    processor = RealTimeProcessor('best_model.pth', device='cpu')
    processor.start()