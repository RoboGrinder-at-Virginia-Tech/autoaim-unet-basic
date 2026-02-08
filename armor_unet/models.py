import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)


class DoubleConv(nn.Module):
    """Double convolution block with optional dropout"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net architecture for image segmentation"""
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        # Encoder (4 Layers)
        self.enc1 = DoubleConv(in_channels, base_channels, dropout=0.1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels*2, dropout=0.1)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels*2, base_channels*4, dropout=0.2)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_channels*4, base_channels*8, dropout=0.2)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels*8, base_channels*16, dropout=0.3)

        # Decoder (4 Layers)
        self.upconv4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels*16, base_channels*8, dropout=0.2)
        self.upconv3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels*8, base_channels*4, dropout=0.2)
        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels*4, base_channels*2, dropout=0.1)
        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels*2, base_channels, dropout=0.1)

        # Output
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)

#  MobileNetV2 U-Net model for comparison
class MobileNetUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetUNet, self).__init__()
        
        # Load the MobileNetV2 backbone
        backbone = mobilenet_v2(pretrained=True).features
        self.encoder = backbone
        
        # Decoder layers for upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)  # Final layer for segmentation mask
        )

    def forward(self, x):
        # Pass input through the encoder
        x = self.encoder(x)
        
        # Pass the encoder output through the decoder
        x = self.decoder(x)
        return x
    

class TorchVisionAdapter(nn.Module):
    """Adapter for TorchVision segmentation models to match the project API"""
    def __init__(self, model_name, in_channels=3, out_channels=1):
        super().__init__()
        
        # Handle input channels if not RGB (3 channels)
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_adapter = nn.Identity()

        if model_name == 'deeplabv3':
            # DeepLabV3 with ResNet50 backbone
            self.model = deeplabv3_resnet50(pretrained=True)
            # Replace the classifier head (DeepLabHead)
            # The last layer in DeepLabHead is classifier[4]
            self.model.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1)
            self.model.aux_classifier = None
            
        elif model_name == 'fcn':
            # FCN with ResNet50 backbone
            self.model = fcn_resnet50(pretrained=True)
            # Replace the classifier head (FCNHead)
            # The last layer in FCNHead is classifier[4]
            self.model.classifier[4] = nn.Conv2d(512, out_channels, kernel_size=1)
            self.model.aux_classifier = None
            
        elif model_name == 'lraspp':
            # LRASPP with MobileNetV3-Large backbone (High efficiency)
            self.model = lraspp_mobilenet_v3_large(pretrained=True)
            # LRASPPHead has two classifiers (low and high) that need to be replaced
            self.model.classifier.low_classifier = nn.Conv2d(40, out_channels, kernel_size=1)
            self.model.classifier.high_classifier = nn.Conv2d(128, out_channels, kernel_size=1)
            self.model.aux_classifier = None
            
        else:
            raise ValueError(f"Unknown torchvision model: {model_name}")

    def forward(self, x):
        x = self.input_adapter(x)
        # Torchvision models return an OrderedDict {'out': tensor, 'aux': tensor}
        return self.model(x)['out']


# function to select model type from small-medium-large
def get_model(model_name="small", in_channels=3, out_channels=1, base_channels=None):
    """
    Select and return a U-Net model based on the specified size.

    Parameters:
    - model_name (str): "small", "medium", "large", "mobilenet", "deeplabv3", "fcn", "lraspp".
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - base_channels (int or None): Base number of channels for the U-Net. If None, defaults are used based on model size.

    Returns:
    - UNet: An instance of the UNet model configured according to the specified size.
    """
    model_name = model_name.lower()
    if model_name == "small":
        bc = base_channels if base_channels is not None else 32
        return UNet(in_channels, out_channels, base_channels=bc)
    elif model_name == "medium":
        bc = base_channels if base_channels is not None else 64
        return UNet(in_channels, out_channels, base_channels=bc)
    elif model_name == "large":
        bc = base_channels if base_channels is not None else 128
        return UNet(in_channels, out_channels, base_channels=bc)
    elif model_name == "mobilenet":
        return MobileNetUNet(num_classes=out_channels)
    elif model_name in ['deeplabv3', 'fcn', 'lraspp']:
        return TorchVisionAdapter(model_name, in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model_name '{model_name}'.")