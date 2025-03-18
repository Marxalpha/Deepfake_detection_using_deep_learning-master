import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.enhanced_cnn import Enhanced3DCNN
from models.siamese_network import SiameseNetwork

class TestVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = []
        self.transform = transform
        self.labels = []
        
        for label in ['real', 'fake']:
            path = os.path.join(root_dir, label)
            if not os.path.exists(path):
                continue
                
            for file in os.listdir(path):
                if file.endswith('.mp4'):
                    self.files.append(os.path.join(path, file))
                    self.labels.append(1 if label == 'fake' else 0)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        frames = self.extract_frames(path)
        
        # Convert to tensor
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        
        return frames, torch.tensor(label, dtype=torch.float32), path
    
    def extract_frames(self, video_path, num_frames=4):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract (evenly distributed)
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
                frames.append(frame.transpose(2, 0, 1))  # Convert to (C, H, W)
                
                if len(frames) == num_frames:
                    break
        
        cap.release()
        
        # If we couldn't extract enough frames, duplicate the last one
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((3, 128, 128), dtype=np.uint8))
            
        return np.stack(frames)

def predict_single_video(model, video_path, device):
    """
    Function to predict on a single video file
    """
    dataset = TestVideoDataset(root_dir=None, transform=None)
    frames = dataset.extract_frames(video_path)
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0) / 255.0
    
    model.eval()
    with torch.no_grad():
        face_input = frames.to(device)
        background_input = frames.to(device)
        outputs = model(face_input, background_input)
        
    probability = outputs.item()
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def evaluate_model(model_path, test_dir, batch_size=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    enhanced_cnn = Enhanced3DCNN(input_channels=3, feature_dim=128).to(device)
    model = SiameseNetwork(enhanced_cnn, feature_dim=128).to(device)
    
    # Load trained model weights
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create test dataset and dataloader
    test_dataset = TestVideoDataset(root_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Testing on {len(test_dataset)} videos from {test_dir}")
    
    # Lists to store results
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    
    # Process each batch
    with torch.no_grad():
        for frames, labels, filenames in test_loader:
            face_input = frames.to(device)
            background_input = frames.to(device)
            outputs = model(face_input, background_input).squeeze()
            
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_filenames.extend(filenames)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    # Print results
    print(f"Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print(f"Confusion matrix saved as confusion_matrix.png")
    
    # Display incorrect predictions
    incorrect_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_preds)) if true != pred]
    
    if incorrect_indices:
        print("\nIncorrect predictions:")
        for i in incorrect_indices:
            label = "Real" if all_labels[i] == 0 else "Fake"
            pred = "Real" if all_preds[i] == 0 else "Fake"
            print(f"File: {all_filenames[i]}, True: {label}, Predicted: {pred}, Probability: {all_probs[i]:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': list(zip(all_filenames, all_labels, all_preds, all_probs))
    }

def test_on_directory(model_path, video_dir):
    """
    Test the model on all videos in a directory
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    enhanced_cnn = Enhanced3DCNN(input_channels=3, feature_dim=128).to(device)
    model = SiameseNetwork(enhanced_cnn, feature_dim=128).to(device)
    
    # Load trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    results = []
    
    for file in os.listdir(video_dir):
        if file.endswith('.mp4') or file.endswith('.avi'):
            video_path = os.path.join(video_dir, file)
            prediction, probability = predict_single_video(model, video_path, device)
            
            result = "FAKE" if prediction == 1 else "REAL"
            confidence = probability if prediction == 1 else 1 - probability
            
            print(f"Video: {file} | Prediction: {result} | Confidence: {confidence:.2f}")
            results.append({
                'file': file,
                'prediction': result,
                'confidence': confidence
            })
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test deepfake detection model")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'predict'], 
                        help='Mode: evaluate (with labeled data) or predict (unlabeled)')
    parser.add_argument('--data', type=str, required=True, help='Path to the test data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        evaluate_model(args.model, args.data, args.batch_size)
    else:
        test_on_directory(args.model, args.data)