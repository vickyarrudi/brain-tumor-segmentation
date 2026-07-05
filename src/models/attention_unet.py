import torch.nn as nn
from src.models.unet import Encoder, Bottleneck, DoubleCNN

class AttentionGate(nn.Module):
    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # x filtrado por el mapa de atención

class AttentionDecoder(nn.Module):
    def __init__(self, base_features=64, depth=4):
        super().__init__()
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.decs = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            in_ch = base_features * (2 ** (i + 1))
            out_ch = base_features * (2 ** i)

            self.ups.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.attentions.append(
                AttentionGate(f_g=out_ch, f_x=out_ch, f_int=out_ch // 2)
            )
            self.decs.append(
                DoubleCNN(in_ch, out_ch)
            )

    def forward(self, x, skips):
        for i, (up, attn, dec) in enumerate(zip(self.ups, self.attentions, self.decs)):
            x = up(x)
            skip = attn(g=x, x=skips[-(i + 1)])  # skip filtrado
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return x

class AttentionUNet(nn.Module):
    def __init__(self, base_features=64, depth=4, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(1, base_features, depth)
        self.bottleneck = Bottleneck(base_features * (2 ** (depth - 1)), dropout=dropout)
        self.decoder = AttentionDecoder(base_features, depth)
        self.output = nn.Conv2d(base_features, 1, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.output(x)