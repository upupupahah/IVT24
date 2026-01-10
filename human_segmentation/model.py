import torch
import torch.nn as nn

class HumanSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU()
        )
        self.final = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u2 = self.upconv2(b)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)
        u1 = self.upconv1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        out = self.sigmoid(self.final(d1))
        return out