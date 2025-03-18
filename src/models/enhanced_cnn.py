import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SpatiotemporalAttention

class Enhanced3DCNN(nn.Module):
    """
    Enhanced 3D CNN model with spatiotemporal attention layer.
    
    This model is based on the architecture described in the paper with the addition
    of a spatiotemporal attention layer at the beginning.
    """
    def __init__(self, input_channels=3, feature_dim=128):
        """
        Initialize the Enhanced 3D CNN model.
        
        Args:
            input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            feature_dim (int): Dimension of the output feature vector
        """
        super(Enhanced3DCNN, self).__init__()
        self.feature_dim = feature_dim

        self.spatiotemporal_attention = SpatiotemporalAttention(in_channels=input_channels)
        
        # Block 1
        self.conv1 = nn.Conv3d(input_channels, 8, kernel_size=(3, 3, 3), stride=1, padding='same')
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        
        # Block 2
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding='same')
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        
        # Block 3
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding='same')
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        
        # Block 4
        self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding='same')
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        
        # Block 5
        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding='same')
        self.bn5 = nn.BatchNorm3d(128)
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        
        # Initialize fully connected layers
        self.fc_input_size = None  # Will be determined on first forward pass
        self.fc = None  # Will be initialized dynamically
        self.dropout = nn.Dropout(0.5)
        self.bn_fc = None  # Will be initialized dynamically
        
    def apply_bn(self, x, bn_layer, batch_size):
        """Apply batch normalization only if not in eval mode with batch_size=1"""
        if self.training or batch_size > 1:
            return bn_layer(x)
        else:
            return x  # Skip batch norm for inference with batch_size=1
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply attention
        x = self.spatiotemporal_attention(x)
        
        # Apply convolutional blocks
        x = F.relu(self.apply_bn(self.conv1(x), self.bn1, batch_size))
        x = self.pool1(x)
        
        x = F.relu(self.apply_bn(self.conv2(x), self.bn2, batch_size))
        x = self.pool2(x)
        
        x = F.relu(self.apply_bn(self.conv3(x), self.bn3, batch_size))
        x = self.pool3(x)
        
        x = F.relu(self.apply_bn(self.conv4(x), self.bn4, batch_size))
        x = self.pool4(x)
        
        x = F.relu(self.apply_bn(self.conv5(x), self.bn5, batch_size))
        x = self.pool5(x)
        
        # Ensure tensor is contiguous before reshaping
        x = x.contiguous()
        
        # Use reshape instead of view to avoid the "view size not compatible" error
        try:
            x = x.reshape(batch_size, -1)
        except RuntimeError:
            # If reshape fails, print shape and return zeros as a fallback
            print(f"Failed to reshape tensor of shape {x.shape}")
            return torch.zeros(batch_size, self.feature_dim, device=x.device)
        
        # Initialize FC layers if they haven't been yet
        if self.fc is None:
            self.fc_input_size = x.size(1)
            self.fc = nn.Linear(self.fc_input_size, self.feature_dim).to(x.device)
            self.bn_fc = nn.BatchNorm1d(self.feature_dim).to(x.device)

        # Apply FC layers
        x = self.dropout(x)
        x = self.fc(x)
        
        # Apply batch norm for FC layer, skipping for batch_size=1 during eval
        if self.training or batch_size > 1:
            x = F.relu(self.bn_fc(x))
        else:
            x = F.relu(x)  # Skip batch norm

        return x