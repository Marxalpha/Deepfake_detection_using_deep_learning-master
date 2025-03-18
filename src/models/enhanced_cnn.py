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
        # self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
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
        
        # Calculate size after convolutions and pooling
        # Input: 128x128x4x3
        # After pool1: 128x64x4x8
        # After pool2: 64x32x2x16
        # After pool3: 32x16x1x32
        # After pool4: 16x8x0x64 (actually 16x8x1x64 with padding)
        # After pool5: 8x4x0x128 (actually 8x4x1x128 with padding)
        
        # Assuming we have padding that keeps the time dimension as 1 after pooling
        self.fc_input_size = None
        self.fc=None
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(self.fc_input_size, feature_dim)
        self.bn_fc = None
        
    def forward(self, x):
        # print("Input shape:", x.shape)
        x = self.spatiotemporal_attention(x)
        # print("After attention:", x.shape)

        x = F.relu(self.bn1(self.conv1(x)))
        # print("After conv1:", x.shape)
        x = self.pool1(x)
        # print("After pool1:", x.shape)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # print("After pool2:", x.shape)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # print("After pool3:", x.shape)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        # print("After pool4:", x.shape)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        # print("After pool5:", x.shape)

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        if self.fc is None:
            self.fc_input_size = x.size(1)
            self.fc = nn.Linear(self.fc_input_size, self.feature_dim).to(x.device)
            self.bn_fc = nn.BatchNorm1d(self.feature_dim).to(x.device)

        x = self.dropout(x)
        x = self.fc(x)
        x = F.relu(self.bn_fc(x))

        return x
