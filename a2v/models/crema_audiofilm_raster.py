"""A compact audio-conditioned raster renderer for the sealed CREMA protocol."""

from __future__ import annotations

import torch
from torch import nn


class AudioFiLMRasterRenderer(nn.Module):
    """Predict a 64x64 RGB target from an identity raster and log-mel audio."""

    def __init__(self) -> None:
        super().__init__()
        self.identity_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.audio_encoder = nn.Sequential(nn.Flatten(), nn.Linear(40 * 48, 256), nn.ReLU(inplace=True), nn.Linear(256, 512))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, identity_rgb: torch.Tensor, audio_log_mel: torch.Tensor) -> torch.Tensor:
        bottleneck = self.identity_encoder(identity_rgb)
        scale, bias = self.audio_encoder(audio_log_mel).chunk(2, dim=1)
        conditioned = bottleneck * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        return self.decoder(conditioned)


class AudioConcatRasterRenderer(nn.Module):
    """Predict a 64x64 RGB target with concatenative audio/identity fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.identity_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.audio_encoder = nn.Sequential(nn.Flatten(), nn.Linear(40 * 48, 256), nn.ReLU(inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Conv2d(512, 256, 1), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, identity_rgb: torch.Tensor, audio_log_mel: torch.Tensor) -> torch.Tensor:
        visual = self.identity_encoder(identity_rgb)
        audio = self.audio_encoder(audio_log_mel)[:, :, None, None].expand(-1, -1, visual.shape[2], visual.shape[3])
        return self.decoder(self.fusion(torch.cat([visual, audio], dim=1)))
