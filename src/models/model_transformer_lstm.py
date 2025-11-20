# -*- coding: utf-8 -*-
"""
Acoustic scene classification model (CNN + Transformer + BiLSTM).

Expected input shape:
    (batch, channels, freq_bins, time_frames)
For example:
    (batch, 16, 512, 25)
"""

import torch
from torch import nn


class AcousticSceneTransformerLSTM(nn.Module):
    """
    CNN + Transformer + BiLSTM classifier for multi-microphone STFT inputs.

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
        Dropout used in Transformer, LSTM and classifier head.
    transformer_nhead : int, optional
        Number of attention heads in the Transformer encoder.
    transformer_num_layers : int, optional
        Number of Transformer encoder layers.
    transformer_dim_feedforward : int, optional
        Dimension of the Transformer feed-forward layer.
    """

    def __init__(
        self,
        num_chnls: int,
        num_classes: int,
        freq_bins: int = 512,
        time_frames: int = 25,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        dropout: float = 0.0,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 1,
        transformer_dim_feedforward: int = 256,
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
            nn.Conv2d(num_chnls, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(8, 1)),   # (F/8,  T)

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(8, 1)),   # (F/64, T)

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)),   # (F/256, T)
        )

        # Compute feature dimension after CNN to set Transformer/LSTM sizes
        # Start from freq_bins, apply the same pooling factors: 8, 8, 4
        conv_freq = freq_bins
        for k in (8, 8, 4):
            conv_freq = conv_freq // k  # assumes divisibility

        cnn_out_channels = 64
        self.seq_feature_dim = cnn_out_channels * conv_freq  # this was 128 for 512 bins

        # ---------------------------------------------------------------------
        # 2) Transformer encoder along time
        # ---------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.seq_feature_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
        )

        # ---------------------------------------------------------------------
        # 3) BiLSTM along time (on top of Transformer features)
        # ---------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=self.seq_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # ---------------------------------------------------------------------
        # 4) Classifier head (on top of the pooled LSTM output)
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

        # Prepare sequence for Transformer/LSTM: collapse (C', F') into feature dim
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T', C', F')
        b, t, c, f = x.shape
        x = x.view(b, t, c * f)  # (B, T', C'*F') == (B, T', seq_feature_dim)

        # Transformer encoder over time
        x = self.transformer_encoder(x)  # (B, T', seq_feature_dim)

        # BiLSTM over time
        x, _ = self.lstm(x)  # (B, T', 2*hidden)

        # Average pooling over the sequence dimension
        x = x.mean(dim=1)  # (B, 2*hidden)

        # Classifier head
        logits = self.classifier(x)  # (B, num_classes)
        return logits


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    model = AcousticSceneLSTMClassifier(
        num_chnls=16,
        num_classes=4,
        freq_bins=512,
        time_frames=25,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        dropout=0.1,
        transformer_nhead=4,
        transformer_num_layers=1,
        transformer_dim_feedforward=256,
    )

    dummy = torch.randn(4, 16, 512, 25)
    out = model(dummy)
    print("Output shape:", out.shape)
    print("Num parameters:", sum(p.numel() for p in model.parameters()))
