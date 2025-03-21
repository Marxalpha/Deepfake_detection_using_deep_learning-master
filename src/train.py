import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.enhanced_cnn import Enhanced3DCNN
from models.siamese_network import SiameseNetwork
from datetime import datetime
from tqdm import tqdm

class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        for label in ['real', 'fake']:
            path = os.path.join(root_dir, label)
            for file in os.listdir(path):
                if file.endswith('.mp4'):
                    self.files.append((os.path.join(path, file), label))
        
        self.label_map = {'real': 0, 'fake': 1}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        cap = cv2.VideoCapture(path)
        frames = []
        
        while len(frames) < 32:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128))
            frames.append(frame.transpose(2, 0, 1))  # (C, H, W)
        
        while len(frames) < 32:
            frames.append(np.zeros_like(frames[0]))
        
        cap.release()
        
        frames = np.stack(frames)
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        
        return frames, torch.tensor(self.label_map[label], dtype=torch.float32)

# Modify your train function in train.py to include accuracy calculation
# Add this import at the top of your file
from tqdm import tqdm

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    num_epochs = 15
    
    dataset = VideoDataset(root_dir='../Data')

    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)  # 20% for validation
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    enhanced_cnn = Enhanced3DCNN(input_channels=3, feature_dim=128).to(device)
    model = SiameseNetwork(enhanced_cnn, feature_dim=128).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_model_path = None

    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Add progress bar with tqdm
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for frames, labels in progress_bar:
            face_input = frames.to(device)
            background_input = frames.to(device)  # Use same frames as background (placeholder)
            labels = labels.to(device)
            # Keep only the first 3 channels (if the input is RGBA)
            face_input = face_input[:, :3, :, :, :]
            background_input = background_input[:, :3, :, :, :]

            optimizer.zero_grad()
            outputs = model(face_input, background_input)

            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)  # Only squeeze dim 1, preserving batch dim
            else:
                outputs = outputs.view(-1)  # Ensure it's at least 1D
            loss = criterion(outputs, labels)

            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += 1e-5 * l2_reg

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            if outputs.dim() > 1:
                predicted = (outputs.squeeze(1) > 0.5).float()
            else:
                predicted = (outputs.view(-1) > 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar with current loss and accuracy
            current_loss = running_loss / (progress_bar.n + 1)
            current_acc = 100 * correct / total
            progress_bar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc:.2f}%"})
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc="Validation"):
                face_input = frames.to(device)
                background_input = frames.to(device)
                labels = labels.to(device)
                face_input = face_input[:, :3, :, :, :]
                background_input = background_input[:, :3, :, :, :]
                
                outputs = model(face_input, background_input)
                if outputs.dim() > 1:
                    outputs_squeezed = outputs.squeeze(1)  # Only squeeze dimension 1, preserve batch dimension
                else:
                    # Handle case where tensor might already be squeezed too much
                    outputs_squeezed = outputs.view(-1)  # Reshape to ensure it's at least 1D
                loss = criterion(outputs_squeezed, labels)
                
                val_loss += loss.item()
                
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update LR scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] => Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {100*correct/total:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            time = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = os.path.join('saved_models', f'best_model{time}.pth')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")


        # # After the epoch is complete
        # epoch_loss = running_loss / len(dataloader)
        # epoch_acc = 100 * correct / total
        # print(f"Epoch [{epoch+1}/{num_epochs}] completed => Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # Save the trained model
    time=datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join('saved_models', f'saved_model{time}.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # return best_model_path if best_model_path else model_save_path

if __name__ == "__main__":
    train()
