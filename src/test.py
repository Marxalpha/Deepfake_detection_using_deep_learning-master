import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.enhanced_cnn import Enhanced3DCNN
from models.siamese_network import SiameseNetwork

class TestVideoDataset(Dataset):
    def __init__(self, root_dir=None):
        self.files = []
        self.labels = []
        
        if root_dir is not None:
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

        # Convert to tensor and normalize
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0
        
        return frames, torch.tensor(label, dtype=torch.float32), path
    
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
                # Ensure frame has 3 channels (RGB)
                if frame.shape[2] > 3:
                    frame = frame[:, :, :3]
                frame = frame.transpose(2, 0, 1)  # (C, H, W)
                frames.append(frame)

        cap.release()

        # Ensure we have exactly `num_frames`
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((3, 128, 128), dtype=np.uint8))

        return np.stack(frames)

def debug_model_input_shapes(model, device, batch_size=1):
    """Helper function to print shapes at various stages of the model"""
    # Generate random input of same shape as our data
    dummy_input = torch.rand(batch_size, 3, 4, 128, 128).to(device)
    print(f"Generated dummy input with shape: {dummy_input.shape}")
    
    # Hook to capture intermediate tensor shapes
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hooks for the model's modules
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv3d):
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Forward pass
    try:
        with torch.no_grad():
            model(dummy_input, dummy_input)
            
        # Print shapes
        print("\nIntermediate tensor shapes:")
        for name, act in activation.items():
            if isinstance(act, tuple):
                print(f"{name}: {[t.shape for t in act]}")
            else:
                print(f"{name}: {act.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
    finally:
        for hook in hooks:
            hook.remove()

def predict_single_video(model, video_path, device):
    """
    Function to predict on a single video file
    """
    dataset = TestVideoDataset(root_dir=None)
    frames = dataset.extract_frames(video_path)
    print(f"Extracted frames shape: {frames.shape}")

    # Normalize and create batch dimension
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0) / 255.0
    print(f"Input tensor shape after unsqueeze: {frames.shape}")

    # Ensure 3 channels and rearrange to match model's expected format
    if frames.shape[2] > 3:
        frames = frames[:, :, :3, :, :]
    
    # Rearrange dimensions to match model input shape
    frames = frames.permute(0, 2, 1, 3, 4)  # [1, 3, 4, 128, 128]
    print(f"Input tensor shape after permute: {frames.shape}")

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
    
    # Debug model with dummy input
    print("\nDebugging model with dummy input...")
    debug_model_input_shapes(model, device)

    # Create test dataset and dataloader
    test_dataset = TestVideoDataset(root_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTesting on {len(test_dataset)} videos from {test_dir}")

    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []

    # Try with smaller batch size if regular fails
    current_batch_size = batch_size
    
    try:
        with torch.no_grad():
            # Process first batch
            batch_data = next(iter(test_loader))
            frames, labels, filenames = batch_data
            
            print(f"First batch frame tensor shape: {frames.shape}")
            
            # Ensure consistent channel count
            frames = frames[:, :, :3, :, :]  # Keep only 3 channels
            print(f"After channel adjustment: {frames.shape}")

            # Rearrange dimensions before feeding into the model
            frames = frames.permute(0, 2, 1, 3, 4)  # [batch_size, 3, num_frames, 128, 128]
            print(f"After permute: {frames.shape}")

            face_input = frames.to(device)
            background_input = frames.to(device)

            # This might raise the matrix multiplication error
            outputs = model(face_input, background_input).squeeze()
            
            # If we get here, we can proceed with the full evaluation
            print("First batch processed successfully, continuing with evaluation...")
            
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_filenames.extend(filenames)
            
            # Continue with rest of batches
            for frames, labels, filenames in itertools.islice(test_loader, 1, None):
                frames = frames[:, :, :3, :, :]
                frames = frames.permute(0, 2, 1, 3, 4)
                
                face_input = frames.to(device)
                background_input = frames.to(device)
                outputs = model(face_input, background_input).squeeze()
                
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
                all_filenames.extend(filenames)
    
    except RuntimeError as e:
        print(f"Error with batch size {current_batch_size}: {e}")
        print("\nAttempting with batch size 1...")
        
        # Fallback to batch size 1
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_filenames = []
        
        try:
            with torch.no_grad():
                for frames, labels, filenames in test_loader:
                    frames = frames[:, :, :3, :, :]
                    frames = frames.permute(0, 2, 1, 3, 4)
                    
                    face_input = frames.to(device)
                    background_input = frames.to(device)
                    outputs = model(face_input, background_input).squeeze()
                    
                    preds = (outputs > 0.5).float().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(outputs.cpu().numpy())
                    all_filenames.extend(filenames)
        except Exception as e:
            print(f"Error with batch size 1: {e}")
            print("Please check model architecture and training parameters.")
            return None

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
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

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
    plt.savefig('confusion_matrix.png')
    print(f"Confusion matrix saved as confusion_matrix.png")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': list(zip(all_filenames, all_labels, all_preds, all_probs))
    }

if __name__ == "__main__":
    import argparse
    import itertools

    parser = argparse.ArgumentParser(description="Test deepfake detection model")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to the test data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')

    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.batch_size)