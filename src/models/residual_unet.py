import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """DoubleCNN con shortcut residual. Conv1x1 si in_ch != out_ch."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class ResidualEncoder(nn.Module):
    def __init__(self, in_channels=1, base_features=64, depth=4):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList()
        for i in range(depth):
            out_ch = base_features * (2 ** i)
            self.encoders.append(ResidualBlock(in_channels, out_ch))
            in_channels = out_ch

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips


class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(in_channels, in_channels * 2),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualDecoder(nn.Module):
    def __init__(self, base_features=64, depth=4):
        super().__init__()
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = base_features * (2 ** (i + 1))
            out_ch = base_features * (2 ** i)
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.decs.append(ResidualBlock(in_ch, out_ch))

    def forward(self, x, skips):
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return x


class ResidualUNet(nn.Module):
    def __init__(self, base_features=64, depth=4, dropout=0.0):
        super().__init__()
        self.encoder = ResidualEncoder(1, base_features, depth)
        self.bottleneck = ResidualBottleneck(base_features * (2 ** (depth - 1)), dropout=dropout)
        self.decoder = ResidualDecoder(base_features, depth)
        self.output = nn.Conv2d(base_features, 1, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.output(x)
