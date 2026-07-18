import torch
import torch.nn as nn

class ResNetBlock1D(nn.Module):
    """A standard 1D ResNet Residual Block with projection shortcut."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class ECGEncoder1D(nn.Module):
    """High-Performance 1D ResNet Feature Extractor for 12-lead ECG time-series signals."""
    
    def __init__(self, in_channels=12, base_filters=64, latent_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.latent_dim = latent_dim
        
        # Initial convolution with large kernel to capture QRS and wider structures
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        
        # Standard residual stages
        self.in_planes = base_filters
        self.stage1 = self._make_layer(base_filters, blocks=2, stride=2)
        self.stage2 = self._make_layer(base_filters * 2, blocks=2, stride=2)
        self.stage3 = self._make_layer(base_filters * 4, blocks=2, stride=2)
        
        # Global Average Pooling and latent projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 4, latent_dim)
        
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )
            
        layers = []
        layers.append(ResNetBlock1D(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(ResNetBlock1D(self.in_planes, planes))
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Input shape: [B, Leads, Length] (e.g. [256, 12, 1000])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.pool(x).squeeze(-1) # [B, planes*4]
        z = self.fc(x) # Output representation: [B, latent_dim]
        return z

class ECGClassifier1D(nn.Module):
    """Linear or non-linear classifier probe for supervised training or downstream evaluation."""
    
    def __init__(self, encoder: ECGEncoder1D, num_classes=5, hidden_dim=None):
        super().__init__()
        self.encoder = encoder
        
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(encoder.latent_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(encoder.latent_dim, num_classes)
            
    def forward(self, x):
        # x shape: [B, Leads, Length]
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits
