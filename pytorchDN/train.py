import torch
from model import DenoiseUNet
from data_loader import get_dataloaders
from utils import hybrid_loss
from config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train():
    # Initialize
    model = DenoiseUNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    train_loader, val_loader = get_dataloaders()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = hybrid_loss(clean, outputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                outputs = model(noisy)
                val_loss += hybrid_loss(clean, outputs).item()
        
        # Stats
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == PATIENCE:
                print("Early stopping!")
                break
        
        scheduler.step(val_loss)

if __name__ == "__main__":
    train()