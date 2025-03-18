import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from models.enhanced_cnn import Enhanced3DCNN
from models.siamese_network import SiameseNetwork
from models.feature_extraction import LBPModule, GLCMModule

class TestVideoDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        
        for label in ['real', 'fake']:
            path = os.path.join(root_dir, label)
            if not os.path.exists(path):
                continue
                
            for file in os.listdir(path):
                if file.endswith('.mp4'):
                    self.files.append(os.path.join(path, file))
                    self.labels.append(1 if label == 'fake' else 0)
        
        print(f"Found {len(self.files)} videos for testing")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        frames = self.extract_frames(path)
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        
        return frames, torch.tensor(label, dtype=torch.float32)
    
    def extract_frames(self, video_path, num_frames=4):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= num_frames:
            indices = list(range(frame_count))
        else:
            indices = [int(i * frame_count / num_frames) for i in range(num_frames)]

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.resize(frame, (128, 128))
                if frame.shape[2] > 3:
                    frame = frame[:, :, :3]
                frame = frame.transpose(2, 0, 1)  # (C, H, W)
                frames.append(frame)

        cap.release()

        # Ensure we have exactly `num_frames`
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((3, 128, 128), dtype=np.uint8))

        return np.stack(frames)

def find_latest_model(models_dir='saved_models'):
    """Find the most recent model file in the models directory"""
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found.")
        return None
        
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return None
        
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    
    return os.path.join(models_dir, model_files[0])

def create_model(device):
    """Create a model with the same architecture as used in training"""
    enhanced_cnn = Enhanced3DCNN(input_channels=3, feature_dim=128).to(device)
    
    # Set model to evaluation mode
    enhanced_cnn.eval()
    
    # Initialize the Siamese network
    model = SiameseNetwork(enhanced_cnn, feature_dim=128).to(device)
    model.eval()
    
    return model

def main():
    # Set test data directory
    test_dir = "test_data"
    if not os.path.exists(test_dir):
        print(f"Test data directory '{test_dir}' not found.")
        return
    
    # Find the most recent model
    model_path = find_latest_model()
    if model_path is None:
        return
    
    print(f"Using model: {model_path}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model with the same architecture
    model = create_model(device)
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Load state dict with strict=False to handle any mismatches
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        print("Model loaded with non-strict option")
        print(f"Missing keys: {len(incompatible_keys.missing_keys)}")
        print(f"Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = TestVideoDataset(root_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize lists to store results
    all_preds = []
    all_labels = []
    
    # Evaluate model
    print("Evaluating model...")
    
    with torch.no_grad():
        for frames, labels in test_loader:
            try:
                # Ensure frames have the right shape
                # Input shape should be [batch_size, channels, frames, height, width]
                # But current shape is [batch_size, frames, channels, height, width]
                # So we need to permute the dimensions
                frames = frames.permute(0, 2, 1, 3, 4)
                
                # Ensure we have exactly 3 channels (RGB)
                if frames.shape[1] > 3:
                    frames = frames[:, :3, :, :, :]
                
                # Move to device
                face_input = frames.to(device)
                background_input = frames.to(device)  # Use same input for both branches
                
                # Forward pass
                outputs = model(face_input, background_input).squeeze()
                
                # Convert to predictions
                preds = (outputs > 0.5).float().cpu().numpy()
                
                # Handle scalar case
                if np.isscalar(preds):
                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy()[0])
                else:
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                
                # Print progress
                print(f"Processed {len(all_preds)}/{len(test_dataset)} videos", end="\r")
                
            except Exception as e:
                print(f"Error during processing: {e}")
                print(f"Input shape: {frames.shape}")
                # Continue to next batch
                continue
    
    # Check if we have predictions
    if len(all_preds) == 0:
        print("No valid predictions were made. Please check the model architecture.")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save confusion matrix
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(results_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved as {cm_path}")

if __name__ == "__main__":
    main()