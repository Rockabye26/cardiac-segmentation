import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DoubleConv(nn.Module):
    """Double Convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: Upsample -> Concat -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, dropout: float = 0.1):
        super().__init__()
        
        if bilinear:
            # Use bilinear upsampling + conv to reduce channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
        else:
            # Use transpose convolution for upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder path
            x2: Feature map from encoder path (skip connection)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch between x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution: 1x1 Conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net Architecture for Cardiac MRI Segmentation
    
    Architecture:
    - Input: (batch, 1, 256, 256) - grayscale cardiac MRI
    - Output: (batch, 4, 256, 256) - 4-class segmentation logits
    - 4 encoding levels, 4 decoding levels with skip connections
    """
    
    def __init__(
        self, 
        n_channels: int = 1, 
        n_classes: int = 4, 
        bilinear: bool = True,
        base_channels: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (Contracting path)
        self.inc = DoubleConv(n_channels, base_channels, dropout)
        self.down1 = Down(base_channels, base_channels * 2, dropout)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dropout)
        
        # Decoder (Expansive path)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, dropout)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, dropout)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, dropout)
        self.up4 = Up(base_channels * 2, base_channels, bilinear, dropout)
        
        # Output layer
        self.outc = OutConv(base_channels, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using kaiming normal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass with skip connections"""
        # Encoder
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 1024/512 channels (bottleneck)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # 512/256 channels
        x = self.up2(x, x3)   # 256/128 channels
        x = self.up3(x, x2)   # 128/64 channels
        x = self.up4(x, x1)   # 64 channels
        
        # Output (no activation - use with CrossEntropyLoss)
        logits = self.outc(x)  # n_classes channels
        
        return logits
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'input_shape': f"({self.n_channels}, H, W)",
            'output_shape': f"({self.n_classes}, H, W)",
            'architecture': 'U-Net'
        }


class UNetSmall(UNet):
    """Smaller U-Net for limited GPU memory"""
    
    def __init__(self, n_channels: int = 1, n_classes: int = 4, dropout: float = 0.1):
        super().__init__(
            n_channels=n_channels, 
            n_classes=n_classes, 
            base_channels=32,  # Reduced from 64
            dropout=dropout
        )


class UNetLarge(UNet):
    """Larger U-Net for better performance with more GPU memory"""
    
    def __init__(self, n_channels: int = 1, n_classes: int = 4, dropout: float = 0.1):
        super().__init__(
            n_channels=n_channels, 
            n_classes=n_classes, 
            base_channels=96,  # Increased from 64
            dropout=dropout
        )


def create_unet_model(
    model_size: str = 'standard',
    n_channels: int = 1,
    n_classes: int = 4,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
) -> UNet:
    """
    Factory function to create U-Net models
    
    Args:
        model_size: 'small', 'standard', or 'large'
        n_channels: Number of input channels
        n_classes: Number of output classes
        dropout: Dropout rate
        device: Device to move model to
        
    Returns:
        U-Net model instance
    """
    if model_size == 'small':
        model = UNetSmall(n_channels, n_classes, dropout)
    elif model_size == 'large':
        model = UNetLarge(n_channels, n_classes, dropout)
    else:
        model = UNet(n_channels, n_classes, dropout=dropout)
    
    if device is not None:
        model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_unet_model('standard', device=device)
    
    # Print model info
    info = model.get_model_info()
    print("U-Net Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 2
    height, width = 256, 256
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 1, height, width).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nTest Results:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test different model sizes
    for size in ['small', 'standard', 'large']:
        test_model = create_unet_model(size)
        info = test_model.get_model_info()
        print(f"\n{size.title()} U-Net:")
        print(f"  Parameters: {info['total_parameters']:,}")
        print(f"  Model size: {info['model_size_mb']:.1f} MB")