import torch
import torch.nn as nn
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

class GLCMModule(nn.Module):
    """
    Gray Level Co-occurrence Matrix (GLCM) feature extraction module.
    Calculates GLCM features from the input image.
    """
    def __init__(self):
        super(GLCMModule, self).__init__()
    
    def forward(self, x):
        batch_size = x.size(0)
        features = []
        
        for i in range(batch_size):
            sample_features = []
            for f in range(x.size(2)):  # Iterate through frames
                # Convert to grayscale
                gray_frame = x[i, :, f].mean(dim=0).to('cpu').numpy()
                gray_frame = (gray_frame * 255).astype(np.uint8)
                
                # Calculate standard deviation
                std_dev = np.std(gray_frame)
                
                # Calculate GLCM and its properties
                try:
                    distances = [1]
                    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                    # Ensure we have proper pixel values for GLCM
                    levels = min(256, gray_frame.max() + 1)
                    if levels < 2:  # Need at least 2 levels for GLCM
                        levels = 2
                    
                    glcm = graycomatrix(gray_frame, distances, angles, levels=levels, symmetric=True, normed=True)
                    
                    contrast = np.mean(graycoprops(glcm, 'contrast'))
                    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
                    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
                    asm = np.mean(graycoprops(glcm, 'ASM'))
                    energy = np.mean(np.sqrt(asm))
                    
                    # Combine features
                    frame_features = [std_dev, contrast, dissimilarity, homogeneity, asm, energy]
                except Exception as e:
                    print(f"GLCM calculation failed: {e}")
                    # Use default values if GLCM fails
                    frame_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                
                sample_features.extend(frame_features)
            
            # Ensure we have a valid feature vector
            if not sample_features:
                sample_features = [0.0] * 24  # Default size: 6 features * 4 frames
            
            features.append(sample_features)
        
        # Convert to tensor with proper handling of empty lists
        if not features:
            return torch.zeros((batch_size, 24), dtype=torch.float32, device=x.device)
        
        # Ensure all feature vectors have the same length
        max_len = max(len(f) for f in features)
        for i in range(len(features)):
            if len(features[i]) < max_len:
                features[i].extend([0.0] * (max_len - len(features[i])))
        
        return torch.tensor(features, dtype=torch.float32, device=x.device)

class LBPModule(nn.Module):
    """
    Local Binary Pattern (LBP) feature extraction module.
    """
    def __init__(self, enhanced_3d_cnn):
        super(LBPModule, self).__init__()
        self.enhanced_3d_cnn = enhanced_3d_cnn
    
    def extract_lbp(self, x):
        batch_size, channels, frames, height, width = x.size()
        lbp_tensor = torch.zeros((batch_size, 3, frames, height, width), device=x.device)
        
        for i in range(batch_size):
            for f in range(frames):
                try:
                    # Convert to grayscale
                    gray_frame = x[i, :, f].mean(dim=0).to('cpu').numpy()
                    gray_frame = (gray_frame * 255).astype(np.uint8)
                    
                    # Calculate LBP
                    lbp = local_binary_pattern(gray_frame, P=8, R=1, method='uniform')
                    # Normalize and ensure no NaN values
                    max_val = lbp.max()
                    if max_val > 0:
                        lbp = lbp / max_val
                    
                    lbp_value = torch.tensor(lbp, dtype=torch.float32, device=x.device)
                    for c in range(3):  # Copy the same LBP values to all 3 channels
                        lbp_tensor[i, c, f] = lbp_value
                except Exception as e:
                    print(f"LBP calculation failed: {e}")
                    # Leave as zeros if LBP fails
        
        return lbp_tensor
    
    def forward(self, x):
        try:
            lbp_tensor = self.extract_lbp(x)
            # Ensure model is in eval mode for single-sample batches
            self.enhanced_3d_cnn.eval()
            with torch.no_grad():
                lbp_features = self.enhanced_3d_cnn(lbp_tensor)
            return lbp_features
        except Exception as e:
            print(f"LBP feature extraction failed: {e}")
            batch_size = x.size(0)
            return torch.zeros((batch_size, 128), dtype=torch.float32, device=x.device)