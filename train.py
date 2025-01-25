import torch
import torch.nn as nn
import torch.optim as optim
import os
import librosa
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from denoisnet_model import DenoisNet

class AudioDataset(Dataset):
    def __init__(self, clean_files, noise_files, transform=None, target_length=None):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.transform = transform
        self.target_length = target_length  # Target length for all audio files

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_file = self.clean_files[idx]
        noise_file = self.noise_files[idx % len(self.noise_files)]  # Use modulo to loop through noise files

        # Load audio data (waveform)
        clean_audio, sr = librosa.load(clean_file, sr=None)
        noise_audio, _ = librosa.load(noise_file, sr=None)

        # Ensure both audios have the same length
        min_length = min(len(clean_audio), len(noise_audio))
        clean_audio = clean_audio[:min_length]
        noise_audio = noise_audio[:min_length]

        # Optionally truncate or pad to a target length
        if self.target_length is not None:
            if min_length < self.target_length:
                # Pad with zeros if shorter than target length
                clean_audio = np.pad(clean_audio, (0, self.target_length - min_length), mode='constant')
                noise_audio = np.pad(noise_audio, (0, self.target_length - min_length), mode='constant')
            else:
                # Truncate if longer than target length
                clean_audio = clean_audio[:self.target_length]
                noise_audio = noise_audio[:self.target_length]

        # Add noise to the clean audio
        noisy_audio = clean_audio + 0.5 * noise_audio

        # Convert to tensors
        clean_tensor = torch.tensor(clean_audio, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        noisy_tensor = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0)

        # Apply any transformations (e.g., scaling, normalization)
        if self.transform:
            clean_tensor = self.transform(clean_tensor)
            noisy_tensor = self.transform(noisy_tensor)

        return noisy_tensor, clean_tensor

def collate_fn(batch):
    noisy_data = [item[0] for item in batch]
    clean_data = [item[1] for item in batch]
    
    # Pad sequences to the same length
    noisy_data = torch.nn.utils.rnn.pad_sequence(noisy_data, batch_first=True)
    clean_data = torch.nn.utils.rnn.pad_sequence(clean_data, batch_first=True)
    
    return noisy_data, clean_data

# Define the model, loss, and optimizer
input_channels = 1  # Mono audio
model = DenoisNet(input_channels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Hyperparameters
batch_size = 8
learning_rate = 1e-3
epochs = 20

# Prepare the dataset and dataloaders
clean_files = []
for root, dirs, files in os.walk("data/clean"):
    for file in files:
        if file.endswith((".wav",".mp3",".flac")):
            clean_files.append(os.path.join(root, file))

noise_files = [os.path.join("data/noise", f) for f in os.listdir("data/noise") if f.endswith((".wav","mp3","flac"))]

target_length = 55128  # Divisible by 8 (2^3)
train_dataset = AudioDataset(clean_files, noise_files, target_length=target_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for noisy_data, clean_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(noisy_data)

        # Compute the loss
        loss = criterion(output, clean_data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print loss for this epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Optionally save the model checkpoint
    torch.save(model.state_dict(), f"denoise_model_epoch_{epoch+1}.pth")

# After training, you can save the model
torch.save(model.state_dict(), "denoise_model_final.pth")
print("Training complete. Model saved.")