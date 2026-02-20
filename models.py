import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# --- SRM FILTER LAYER ---
class SRMConv2d(nn.Module):
    def __init__(self):
        super(SRMConv2d, self).__init__()
        self.channels = 3
        q1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]
        q2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]
        q3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]
        
        q = torch.FloatTensor([q1, q2, q3])
        self.weight = nn.Parameter(q.unsqueeze(1), requires_grad=False)

    def forward(self, x):
        r = F.conv2d(x[:,0:1,:,:], self.weight, padding=2)
        g = F.conv2d(x[:,1:2,:,:], self.weight, padding=2)
        b = F.conv2d(x[:,2:3,:,:], self.weight, padding=2)
        noise_map = torch.cat([r, g, b], dim=1)
        return noise_map

# --- SPATIAL MODEL ---
# Using Xception directly via timm is preferred, but wrapping for consistency
class SpatialXception(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    # Helper to get features (removes fc)
    def features(self, x):
        return self.model.forward_features(x)

# --- SRM MODEL ---
class SRMXception(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.srm = SRMConv2d()
        self.compress = nn.Conv2d(9, 3, kernel_size=1) 
        self.backbone = timm.create_model('xception', pretrained=pretrained)
        if num_classes > 0:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        else:
            self.backbone.fc = nn.Identity()
        
    def forward(self, x):
        noise = self.srm(x)
        noise = self.compress(noise)
        out = self.backbone(noise)
        return out

# --- TEMPORAL MODEL (BiLSTM) ---
class DeepfakeLSTM(nn.Module):
    def __init__(self, input_size=4096, hidden_size=128, num_layers=2):
        super(DeepfakeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # BiLSTM output dim is hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :] 
        out = self.fc(last_time_step)
        return out
