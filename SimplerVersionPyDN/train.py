import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import DenoiseNet
from audio_utils import AudioProcessor
from dataset import NoiseDataset
import os
import numpy as np

def train_model():
    # Configuration
    config = {
        'sr': 16000,
        'n_fft': 512,
        'hop_length': 128,
        'batch_size': 32,
        'epochs': 50,
        'lr': 1e-4,
        'train_ratio': 0.8,
        'model_save_path': 'best_noise_suppressor.pth',
        'data_dir': 'data/'
    }

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model and utilities
    model = DenoiseNet(n_fft=config['n_fft'], hop_length=config['hop_length']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.L1Loss()  # Using L1 loss for magnitude preservation
    ap = AudioProcessor(n_fft=config['n_fft'], hop_length=config['hop_length'], sr=config['sr'])

    # Dataset and DataLoader
    full_dataset = NoiseDataset(
        clean_dir=os.path.join(config['data_dir'], 'noisy_testset_wav'),
        noise_dir=os.path.join(config['data_dir'], 'noisy_testset_wav'),
        sr=config['sr']
    )
    
    # Split dataset
    train_size = int(config['train_ratio'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    # Training variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        epoch_train_loss = 0.0
        
        # Training phase
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Convert batch to numpy arrays for processing
            clean_audio = batch['clean'].numpy()
            noisy_audio = batch['noisy'].numpy()
            
            # Process audio
            clean_spec = torch.stack([ap.preprocess(a) for a in clean_audio]).to(device)
            noisy_spec = torch.stack([ap.preprocess(a) for a in noisy_audio]).to(device)
            
            # Forward pass
            enhanced_spec = model(noisy_spec)
            
            # Calculate loss
            loss = criterion(enhanced_spec, clean_spec)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * clean_audio.shape[0]
        
        # Calculate epoch metrics
        train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                clean_audio = batch['clean'].numpy()
                noisy_audio = batch['noisy'].numpy()
                
                clean_spec = torch.stack([ap.preprocess(a) for a in clean_audio]).to(device)
                noisy_spec = torch.stack([ap.preprocess(a) for a in noisy_audio]).to(device)
                
                enhanced_spec = model(noisy_spec)
                loss = criterion(enhanced_spec, clean_spec)
                epoch_val_loss += loss.item() * clean_audio.shape[0]
        
        val_loss = epoch_val_loss / len(val_dataset)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'config': config
            }, config['model_save_path'])
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

        # Print progress
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Best Val Loss: {best_val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Training complete. Final model saved.")

    # Save training history
    np.savez('training_history.npz', 
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses))

if __name__ == '__main__':
    train_model()
