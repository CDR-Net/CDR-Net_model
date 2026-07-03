# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        r = self.shortcut(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.bn2(self.conv2(x))
        return F.leaky_relu(x + r, 0.01)

class UNetNoPoolingDualDecoder(nn.Module):
    """
    Shared residual encoder + dual decoders
    output:
      - out_ed: (B,2,H,W)  # [INC, DEC]
      - out_at: (B,2,H,W)  # [INC, DEC]
    """
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_channels, stride=1)
        self.enc2 = ResidualBlock(base_channels, base_channels*2, stride=2)
        self.enc3 = ResidualBlock(base_channels*2, base_channels*4, stride=2)
        self.enc4 = ResidualBlock(base_channels*4, base_channels*8, stride=2)
        self.bottleneck = ResidualBlock(base_channels*8, base_channels*16, stride=2)

        # ED decoder (transpose conv)
        self.up_a1  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, stride=2)
        self.dec_a1 = ResidualBlock(base_channels*16, base_channels*8)
        self.up_a2  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec_a2 = ResidualBlock(base_channels*8, base_channels*4)
        self.up_a3  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec_a3 = ResidualBlock(base_channels*4, base_channels*2)
        self.up_a4  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec_a4 = ResidualBlock(base_channels*2, base_channels)
        self.final_a = nn.Conv2d(base_channels, 2, 1)  # ED: INC/DEC

        # AT decoder (bilinear + 1x1)
        self.red_b1 = nn.Conv2d(base_channels*16, base_channels*8, 1, bias=False)
        self.dec_b1 = ResidualBlock(base_channels*16, base_channels*8)
        self.red_b2 = nn.Conv2d(base_channels*8, base_channels*4, 1, bias=False)
        self.dec_b2 = ResidualBlock(base_channels*8, base_channels*4)
        self.red_b3 = nn.Conv2d(base_channels*4, base_channels*2, 1, bias=False)
        self.dec_b3 = ResidualBlock(base_channels*4, base_channels*2)
        self.red_b4 = nn.Conv2d(base_channels*2, base_channels, 1, bias=False)
        self.dec_b4 = ResidualBlock(base_channels*2, base_channels)
        self.final_b = nn.Conv2d(base_channels, 2, 1)  # AT: INC/DEC

        # 희소 타깃 안정화용 bias (두 채널 모두 음수로)
        nn.init.constant_(self.final_a.bias, -2.0)  # ED inc/dec
        nn.init.constant_(self.final_b.bias, -4.0)  # AT inc/dec (더 희소)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)

        # ED
        a = self.up_a1(b)
        a = self.dec_a1(torch.cat([a, e4], dim=1))
        a = self.up_a2(a)
        a = self.dec_a2(torch.cat([a, e3], dim=1))
        a = self.up_a3(a)
        a = self.dec_a3(torch.cat([a, e2], dim=1))
        a = self.up_a4(a)
        a = self.dec_a4(torch.cat([a, e1], dim=1))
        out_ed = self.final_a(a)

        # AT
        u = F.interpolate(self.red_b1(b), scale_factor=2, mode="bilinear", align_corners=False)
        u = self.dec_b1(torch.cat([u, e4], dim=1))
        u = F.interpolate(self.red_b2(u), scale_factor=2, mode="bilinear", align_corners=False)
        u = self.dec_b2(torch.cat([u, e3], dim=1))
        u = F.interpolate(self.red_b3(u), scale_factor=2, mode="bilinear", align_corners=False)
        u = self.dec_b3(torch.cat([u, e2], dim=1))
        u = F.interpolate(self.red_b4(u), scale_factor=2, mode="bilinear", align_corners=False)
        u = self.dec_b4(torch.cat([u, e1], dim=1))
        out_at = self.final_b(u)

        return out_ed, out_at
