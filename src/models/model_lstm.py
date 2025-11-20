# -*- coding: utf-8 -*-
"""
Acoustic scene classification model (CNN + BiLSTM).

Expected input shape:
    (batch, channels, freq_bins, time_frames)
For example:
    (batch, 8, 512, 25)

"""

import torch
from torch import nn


class AcousticSceneLSTMClassifier(nn.Module):
    """
    CNN + BiLSTM classifier for multi-microphone STFT inputs.

    Args
    ----
    num_chnls : int
        Number of input channels (2*microphones).
    num_classes : int
        Number of output classes.
    freq_bins : int, optional
        Number of frequency bins in the STFT (default: 512).
    time_frames : int, optional
        Number of time frames in each segment (default: 25).
    lstm_hidden_size : int, optional
        Hidden size of the LSTM (per direction).
    lstm_num_layers : int, optional
        Number of LSTM layers.
    dropout : float, optional
        Dropout used in LSTM and classifier head.
    """

    def __init__(
        self,
        num_chnls: int,
        num_classes: int,
        freq_bins: int = 512,
        time_frames: int = 25,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        self.num_chnls = num_chnls
        self.num_classes = num_classes
        self.freq_bins = freq_bins
        self.time_frames = time_frames

        # ---------------------------------------------------------------------
        # 1) CNN feature extractor over (channels, freq, time)
        # ---------------------------------------------------------------------

        self.conv = nn.Sequential(
            nn.Conv2d(num_chnls, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(8, 1)),  # (F/8, T)

            nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(8, 1)),  # (F/64, T)

            nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)),  # (F/256, T)
        )

        # ---------------------------------------------------------------------
        # 2) BiLSTM along time
        # ---------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True)

        # ---------------------------------------------------------------------
        # 3) Classifier head (on top of the last LSTM time step)
        # ---------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, channels, freq_bins, time_frames).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        # x: (B, C, F, T)
        x = self.conv(x)  # (B, C', F', T')
 
        # Prepare sequence for LSTM: collapse (C', F') into feature dimension
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T', C', F')
        b, t, c, f = x.shape
        x = x.view(b, t, c * f)  # (B, T', C'*F')

        # BiLSTM over time
        x, _ = self.lstm(x)  # (B, T', 2*hidden)

        # Average pooling over the sequence dimension
        x = x.mean(dim=1)
        
        # Classifier head
        logits = self.classifier(x)  # (B, num_classes)
        return logits


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    model = AcousticSceneLSTMClassifier(num_chnls=16, num_classes=4)
    dummy = torch.randn(4, 16, 512, 25)
    out = model(dummy)
    print("Output shape:", out.shape)
    print("Num parameters:", sum(p.numel() for p in model.parameters()))
