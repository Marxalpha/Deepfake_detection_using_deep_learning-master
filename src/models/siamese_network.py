import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_extraction import GLCMModule, LBPModule

class SiameseNetwork(nn.Module):
    def __init__(self, enhanced_3d_cnn, feature_dim):
        super(SiameseNetwork, self).__init__()
        self.branch = enhanced_3d_cnn
        self.glcm = GLCMModule()
        self.lbp = LBPModule(enhanced_3d_cnn)
        
        # Combined feature vector size
        self.fc_combined = None

    def forward(self, face_input, background_input):
        # Set to evaluation mode for inference
        if not self.training:
            self.branch.eval()
            self.lbp.enhanced_3d_cnn.eval()
        
        try:
            # Deep learning features
            face_features = self.branch(face_input)
            background_features = self.branch(background_input)
            
            # Texture features
            glcm_feat = self.glcm(face_input)
            lbp_feat = self.lbp(face_input)
            
            # Handle potential shape issues
            batch_size = face_input.size(0)
            
            # Ensure all features have the correct shape
            if face_features.dim() == 1:
                face_features = face_features.unsqueeze(0)
            if background_features.dim() == 1:
                background_features = background_features.unsqueeze(0)
            if glcm_feat.dim() == 1:
                glcm_feat = glcm_feat.unsqueeze(0)
            if lbp_feat.dim() == 1:
                lbp_feat = lbp_feat.unsqueeze(0)
            
            # Combine all features
            combined = torch.cat([face_features, background_features, glcm_feat, lbp_feat], dim=1)
            
            # Initialize combined FC layer if needed
            if self.fc_combined is None:
                feature_size = combined.size(1)
                self.fc_combined = nn.Linear(feature_size, 1).to(combined.device)
            
            # Get final output
            output = torch.sigmoid(self.fc_combined(combined))
            
            return output
            
        except Exception as e:
            print(f"Error in SiameseNetwork forward pass: {e}")
            import traceback
            traceback.print_exc()
            # Return default prediction as fallback
            return torch.tensor([0.5], device=face_input.device)