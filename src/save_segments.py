# -*- coding: utf-8 -*-
"""
Utilities to convert simulated multi-microphone recordings into
fixed-length STFT segments saved as PyTorch tensors, together with labels.

"""


import math
from pathlib import Path
from typing import List, Tuple, Optional

import librosa
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------
def read_recordings_and_labels(data_dir: Path) -> Tuple[List[Path], pd.DataFrame]:
    """
    Scan a directory for recordings (*.wav) and a single labels file (*.txt or *.csv).

    The labels file is expected to have columns:
        File_Name, Location

    Parameters
    ----------
    data_dir : Path
        Directory that contains simulated recordings and a labels file.

    Returns
    -------
    rec_paths : list[Path]
        Sorted list of all *.wav files.
    labels_df : pd.DataFrame
        DataFrame with one row per recording (same order as rec_paths).
    """
    data_dir = Path(data_dir)

    rec_paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".wav")
    if not rec_paths:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")

    # Try to find a labels file: prefer .csv, then .txt
    label_candidates = sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.txt")))
    if not label_candidates:
        raise FileNotFoundError(f"No labels file (*.csv / *.txt) found in {data_dir}")

    labels_path = label_candidates[0]


    labels_df = pd.read_csv(labels_path)

    if len(labels_df) != len(rec_paths):
        print(
            f"[Warning] Number of recordings ({len(rec_paths)}) and rows in labels "
            f"({len(labels_df)}) differ. Matching will be done by order."
        )

    print(f"Found {len(rec_paths)} recordings in {data_dir}")
    print(f"Using labels file: {labels_path}")
    return rec_paths, labels_df


def load_recording(path: Path, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load a recording with librosa (mono or multi-channel).

    Parameters
    ----------
    path : Path
        Path to the .wav file.
    sr : int or None
        Target sample rate. If None, use file's native sampling rate.

    Returns
    -------
    audio : np.ndarray
        Shape (num_channels, num_samples). Mono files are converted to (1, N).
    fs : int
        Sampling rate in Hz.
    """
    path = Path(path)
    # mono=False will keep channels separate if present
    audio, fs = librosa.load(path.as_posix(), sr=sr, mono=False)

    # librosa returns shape (N,) for mono, (C, N) for multi-channel
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # (1, N)

    return audio, fs


def stft_multichannel_real_imag(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """
    Compute STFT for each channel and stack real + imaginary parts.

    Parameters
    ----------
    audio : np.ndarray
        Shape (num_channels, num_samples).
    n_fft : int
        FFT size.
    hop_length : int
        Hop length in samples.

    Returns
    -------
    stft_ri : np.ndarray
        Shape (2*num_channels, n_freq_bins, num_frames).
        First half along axis=0 is real, second half is imag.
    """

    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)  # (F, T)
    stft_ri = np.concatenate((S.real, S.imag), axis=0)[:,:-1,:]


    return stft_ri


def segment_stft(
    stft_ri: np.ndarray,
    segment_length: int,
) -> np.ndarray:
    """
    Split an STFT tensor into fixed-length time segments (with zero-padding).

    Parameters
    ----------
    stft_ri : np.ndarray
        Shape (2C, F, T).
    segment_length : int
        Segment length in frames (time axis).

    Returns
    -------
    segments : np.ndarray
        Shape (num_segments, 2C, F, segment_length).
    """
    C, F, T = stft_ri.shape
    
    num_segments = int(math.ceil(T / segment_length))
    padded_T = num_segments * segment_length

    # pad along time axis
    pad_width = ((0, 0), (0, 0), (0, padded_T - T))
    stft_padded = np.pad(stft_ri, pad_width, mode="constant")

    # reshape into (2C, F, num_segments, segment_length) then reorder
    stft_padded = stft_padded.reshape(C, F, num_segments, segment_length)
    segments = np.transpose(stft_padded, (2, 0, 1, 3))  # (num_segments, 2C, F, L)
    return segments


# ---------------------------------------------------------------------
# Main function: process a directory and save segments
# ---------------------------------------------------------------------
def save_segments_from_directory(
    data_dir: Path,
    out_dir: Path,
    segment_length: int = 25,
    n_fft: int = 1024,
    hop_length: int = 512,
    target_sr: Optional[int] = None,
) -> None:
    """
    Process all recordings in `data_dir` and save fixed-length STFT segments.

    For each input file, this will:
      1. Load the recording (mono or multi-channel)
      2. Compute per-channel STFT, stack real+imag parts
      3. Split along time into segments of `segment_length` frames
      4. Save each segment as a .pt tensor:
         out_dir / "<file_stem>_seg<k>.pt"
      5. Store labels per segment in a single CSV:
         out_dir / "segments_labels.csv"

    Parameters
    ----------
    data_dir : Path
        Directory that contains .wav recordings and a labels file.
    out_dir : Path
        Destination directory for segments and labels.
    segment_length : int, optional
        Length in STFT frames of each segment.
    n_fft : int, optional
        STFT FFT size.
    hop_length : int, optional
        STFT hop length in samples.
    target_sr : int or None, optional
        If not None, resample audio to this rate when loading.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_paths, labels_df = read_recordings_and_labels(data_dir)

    all_rows = []  # for the final segments_labels.csv

    for idx, rec_path in enumerate(rec_paths):
        print(f"[{idx+1}/{len(rec_paths)}] Processing {rec_path.name}")

        audio, fs = load_recording(rec_path, sr=target_sr)
        stft_ri = stft_multichannel_real_imag(audio, n_fft=n_fft, hop_length=hop_length)
        segments = segment_stft(stft_ri, segment_length=segment_length)

        # Get label row if available, else default
        if idx < len(labels_df):
            row = labels_df.iloc[idx]
            location = row["location"]
        else:
            location  = "unknown"

        for seg_idx, seg in enumerate(segments):
            seg_filename = f"{rec_path.stem}_seg{seg_idx:03d}.pt"
            seg_path = out_dir / seg_filename

            # Save as torch tensor: shape (2C, F, L)
            tensor = torch.from_numpy(seg.astype(np.float32))

            torch.save(tensor, seg_path)

            all_rows.append(
                {
                    "segment_file": seg_filename,
                    "source_file": rec_path.name,
                    "location": location,
                    "sample_rate": fs,
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "segment_length": segment_length,
                }
            )

    # Save all segment labels to a single CSV
    labels_out_path = out_dir / "segments_labels.csv"
    pd.DataFrame(all_rows).to_csv(labels_out_path, index=False)
    print(f"Segmentation complete. Saved segments to: {out_dir}")
    print(f"Segment-level labels CSV: {labels_out_path}")


# ---------------------------------------------------------------------
# CLI / script entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Adjust these paths for your machine before running as a script.
    base_dir = Path(__file__).resolve().parent

    data_dir = base_dir / "data/processed/simulated_room"
    out_dir = base_dir / "data/processed/simulated_room_stft"

    segment_length = 25
    n_fft = 1024
    hop_length = 512
    target_sr = 16000  # keep None to use the original SR

    save_segments_from_directory(
        data_dir=data_dir,
        out_dir=out_dir,
        segment_length=segment_length,
        n_fft=n_fft,
        hop_length=hop_length,
        target_sr=target_sr,
    )
