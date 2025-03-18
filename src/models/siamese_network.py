import torch
import torch.nn as nn
import torch.nn.functional as F
# from feature_extraction import GLCMModule, LBPModule
from .feature_extraction import GLCMModule, LBPModule

class SiameseNetwork(nn.Module):
    def __init__(self, enhanced_3d_cnn, feature_dim):
        super(SiameseNetwork, self).__init__()
        self.branch = enhanced_3d_cnn
        self.glcm = GLCMModule()
        self.lbp = LBPModule(enhanced_3d_cnn)
        
        # Combined feature vector size
        self.fc_combined = nn.Linear(402, 1)

    def forward(self, face_input, background_input):
        # Deep learning features
        face_features = self.branch(face_input)
        background_features = self.branch(background_input)
        
        # Texture features
        glcm_feat = self.glcm(face_input)
        lbp_feat = self.lbp(face_input)
        
        # Combine all features
        combined = torch.cat([face_features, background_features, glcm_feat, lbp_feat], dim=1)
        output = torch.sigmoid(self.fc_combined(combined))
        return output
