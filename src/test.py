import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from models.enhanced_cnn import Enhanced3DCNN
from models.siamese_network import SiameseNetwork
from tqdm import tqdm

class TestVideoDataset(Dataset):
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
        
        while len(frames) < 32:  # Get at least 4 frames
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128))
            frames.append(frame.transpose(2, 0, 1))  # (C, H, W)
        
        cap.release()
        
        # Pad if we couldn't get 4 frames
        while len(frames) < 32:
            frames.append(np.zeros_like(frames[0]))
        
        frames = np.stack(frames)
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        
        return frames, torch.tensor(self.label_map[label], dtype=torch.float32), path

def test_model(model_path='saved_models/saved_model.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the model
    enhanced_cnn = Enhanced3DCNN(input_channels=3, feature_dim=128).to(device)
    model = SiameseNetwork(enhanced_cnn, feature_dim=128).to(device)
    
    # Load the trained model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Load test data
    test_dataset = TestVideoDataset(root_dir='test_data')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    results = []
    correct = 0
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for frames, labels, file_paths in tqdm(test_loader, desc="Testing"):
            face_input = frames.to(device)
            background_input = frames.to(device)
            labels = labels.to(device)
            
            face_input = face_input[:, :3, :, :, :]
            background_input = background_input[:, :3, :, :, :]
            
            # Forward pass
            outputs = model(face_input, background_input)
            
            # Apply threshold for binary prediction
            predicted = (outputs.squeeze() > 0.5).float()
            
            # Update accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store results with detailed info
            for i in range(len(file_paths)):
                raw_output = outputs[i].item() if outputs.dim() > 1 else outputs.item()
                pred_value = 1 if raw_output > 0.5 else 0
                is_correct = (pred_value == labels[i].item())
                
                results.append({
                    'file_path': file_paths[i],
                    'actual_label': 'real' if labels[i].item() == 0 else 'fake',
                    'predicted_label': 'real' if pred_value == 0 else 'fake',
                    'raw_model_output': raw_output,
                    'correct': is_correct
                })
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Create DataFrame with results
    df = pd.DataFrame(results)
    print("\nTesting Results DataFrame:")
    print(df)
    
    # Add distribution analysis
    print("\nRaw output distribution:")
    print(f"Min: {df['raw_model_output'].min():.4f}, Max: {df['raw_model_output'].max():.4f}")
    print(f"Mean: {df['raw_model_output'].mean():.4f}, Median: {df['raw_model_output'].median():.4f}")
    
    # Try different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("\nAccuracy with different thresholds:")
    for threshold in thresholds:
        df[f'pred_{threshold}'] = df['raw_model_output'].apply(lambda x: 'fake' if x > threshold else 'real')
        acc = (df[f'pred_{threshold}'] == df['actual_label']).mean() * 100
        print(f"Threshold {threshold}: {acc:.2f}%")
    
    # Save results to CSV
    csv_path = 'test_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Calculate metrics per class
    real_accuracy = df[df['actual_label'] == 'real']['correct'].mean() * 100
    fake_accuracy = df[df['actual_label'] == 'fake']['correct'].mean() * 100
    
    print(f"\nReal videos detection accuracy: {real_accuracy:.2f}%")
    print(f"Fake videos detection accuracy: {fake_accuracy:.2f}%")
    
    return df

if __name__ == "__main__":
    # Specify the path to your trained model
    model_path = 'saved_models/saved_model20250319_020137.pth'  # Update this path as needed
    # If you have multiple models to test, you can list them here
    # model_paths = ['saved_models/saved_model20250318_120000.pth', 'saved_models/saved_model20250319_083000.pth']
    # for path in model_paths:
    #     print(f"\nTesting model: {path}")
    #     test_model(path)
    
    # Test the default model
    test_model(model_path)