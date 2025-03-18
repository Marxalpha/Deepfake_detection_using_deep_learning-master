import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatiotemporalAttention(nn.Module):
    """
    Spatiotemporal Attention Layer that focuses on important regions in both spatial and temporal dimensions.
    
    As described in the paper, this layer applies attention to both spatial and temporal dimensions
    to filter out irrelevant parts of the input.
    """
    def __init__(self, in_channels=3):
        super(SpatiotemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(None, None, None))
        
        # 3D convolution for spatiotemporal attention
        self.attention_conv = nn.Conv3d(
            in_channels=1,  # After converting to grayscale
            out_channels=1,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Input shape: (batch_size, channels, frames, height, width)
        batch_size, channels, frames, height, width = x.size()
        
        # Convert to grayscale by taking average across channels
        # This is equivalent to adaptive average pooling along the channel dimension
        grayscale = self.avg_pool(x).mean(dim=1, keepdim=True)
        
        # Apply 3D convolution to generate attention map
        attention_map = self.attention_conv(grayscale)
        
        # Apply sigmoid to get attention weights
        attention_weights = self.sigmoid(attention_map)
        
        # Apply attention to the input
        output = x * attention_weights
        
        return output