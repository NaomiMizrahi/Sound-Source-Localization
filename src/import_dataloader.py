# -*- coding: utf-8 -*-
"""
Data loading utilities for SSL / acoustic scene classification.

This module provides:
    - SpectrogramDataset: loads STFT segments (torch tensors or pickles)
    - ImportData: builds train/val DataLoaders from:
        * a single directory of segments (with internal train/val split), or
        * two explicit directories: one for train, one for val.

Expected file naming convention (default):
    <RecName>_loc-<LABEL>_segXXX.(pt|pkl)

Example:
    19-198-0001_loc-desk_seg000.pt  -> label "desk"
    19-198-0001_loc-sofa_seg001.pkl -> label "sofa"

"""

from __future__ import annotations

import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Configuration: how to extract labels from filenames
# ---------------------------------------------------------------------
# Default: match "..._loc-<label>..." -> group(1) is the label
LABEL_PATTERN = re.compile(r"loc-([A-Za-z0-9]+)(?=_)")


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class SpectrogramDataset(Dataset):
    """
    Dataset for precomputed STFT segments stored as .pt or .pkl files.

    Each item returns:
        - STFT tensor of shape (2C, F, T) or whatever you saved
        - integer label
    """

    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        use_pickle: bool = False,
    ):
        """
        Parameters
        ----------
        file_paths : list[Path]
            List of segment files (.pt or .pkl), one per sample.
        labels : list[int]
            Integer labels aligned with file_paths.
        use_pickle : bool
            If True, load features using pickle.load. Otherwise, use torch.load.
        """
        assert len(file_paths) == len(labels), "file_paths and labels must have same length"
        self.file_paths = file_paths
        self.labels = labels
        self.use_pickle = use_pickle

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        label = self.labels[idx]

        if self.use_pickle:
            with open(path, "rb") as f:
                feat = pickle.load(f)
        else:
            feat = torch.load(path)
            

        # Ensure tensor
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat, dtype=torch.float32)
        else:
            feat = feat.float()

        return feat, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------
# Helper: scan directory and infer labels from filenames
# ---------------------------------------------------------------------
def _find_segment_files(root: Path) -> Tuple[List[Path], bool]:
    """
    Find all segment files (.pt or .pkl) in the given directory.

    Returns
    -------
    files : list[Path]
        Sorted list of segment files.
    use_pickle : bool
        True if .pkl files are used, False if .pt files are used.
    """
    root = Path(root)

    pt_files = sorted(root.glob("*.pt"))
    pkl_files = sorted(root.glob("*.pkl"))

    if pt_files:
        return pt_files, False
    if pkl_files:
        return pkl_files, True

    raise FileNotFoundError(f"No .pt or .pkl segment files found in {root}")


def _extract_label_from_name(path: Path) -> str:
    """
    Extract label from filename using LABEL_PATTERN.

    Example:
        '19-198-0001_loc-desk_seg000.pt' -> 'desk'
    """
    m = LABEL_PATTERN.search(path.name)
    if not m:
        raise ValueError(
            f"Could not extract label from filename {path.name!r} "
            f"using pattern {LABEL_PATTERN.pattern!r}"
        )
    return m.group(1)


def _split_train_val_indices(
    labels: List[int],
    train_ratio: float = 0.8,
) -> Tuple[List[int], List[int]]:
    """
    Split indices into train/val, preserving label distribution roughly.

    Parameters
    ----------
    labels : list[int]
        Numerical labels per sample.
    train_ratio : float
        Fraction of samples per label to put in train split.

    Returns
    -------
    train_idx : list[int]
    val_idx : list[int]
    """
    from collections import defaultdict

    label_to_indices = defaultdict(list)
    for i, y in enumerate(labels):
        label_to_indices[y].append(i)

    train_idx, val_idx = [], []
    for y, idx_list in label_to_indices.items():
        n = len(idx_list)
        n_train = max(1, int(round(train_ratio * n)))
        idx_list_sorted = sorted(idx_list)
        train_idx.extend(idx_list_sorted[:n_train])
        val_idx.extend(idx_list_sorted[n_train:])

    # sort for reproducibility
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    return train_idx, val_idx


# ---------------------------------------------------------------------
# Main Function: ImportData
# ---------------------------------------------------------------------
def ImportData(
    DataPath: Optional[Path],
    ModelParam: Dict,
    TrainFiles: Optional[List[Path]] = None,
    ValFiles: Optional[List[Path]] = None,
) -> Tuple[List[Path], List[Path], DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Two modes:

    1) Single directory + internal split
        - Pass DataPath (directory containing .pt/.pkl)
        - Leave TrainFiles and ValFiles as None
        - Uses 'TrainRatio' from ModelParam to split per label.

    2) Explicit file lists
        - Pass TrainFiles and ValFiles (lists of Paths or path-like)
        - DataPath can be None
        - All files from TrainFiles go to train, all from ValFiles go to val.
          Label mapping is built jointly so indices are consistent.

    Parameters
    ----------
    DataPath : Path or None
        Directory containing segment files (.pt or .pkl) when using single-dir mode.
    ModelParam : dict
        Dictionary with configuration. Recognized keys:
            - 'BatchSize' (int): batch size for training loader (default 16)
            - 'TrainRatio' (float): train/val split ratio (default 0.8, only used in mode 1)
            - 'NumWorkers' (int): DataLoader workers (default 0)
            - 'PinMemory' (bool): pin_memory flag (default True)
    TrainFiles : list[Path] or None
        Explicit list of train segment files (.pt/.pkl) for explicit-split mode.
    ValFiles : list[Path] or None
        Explicit list of val segment files (.pt/.pkl) for explicit-split mode.

    Returns
    -------
    seg_train : list[Path]
    seg_val : list[Path]
    dataloader_train : DataLoader
    dataloader_val : DataLoader
    """
    batch_size = int(ModelParam.get("BatchSize", 16))
    train_ratio = float(ModelParam.get("TrainRatio", 0.8))
    num_workers = int(ModelParam.get("NumWorkers", 0))
    pin_memory = bool(ModelParam.get("PinMemory", True))

    # -----------------------------------------------------------------
    # Mode selection: explicit file lists vs single path + split
    # -----------------------------------------------------------------
    using_explicit_lists = (TrainFiles is not None) or (ValFiles is not None)

    if using_explicit_lists:
        if (TrainFiles is None) or (ValFiles is None):
            raise ValueError("If using explicit file lists, both TrainFiles and ValFiles must be provided.")

        # Convert to Path objects, just in case strings are passed
        seg_train = [Path(p) for p in TrainFiles]
        seg_val = [Path(p) for p in ValFiles]

        if len(seg_train) == 0:
            raise ValueError("TrainFiles list is empty.")
        if len(seg_val) == 0:
            raise ValueError("ValFiles list is empty.")

        # Infer file type (.pt vs .pkl) and ensure consistency
        def _infer_use_pickle(files: List[Path]) -> bool:
            exts = {p.suffix for p in files}
            allowed = {".pt", ".pkl"}
            exts = exts & allowed
            if not exts:
                raise ValueError("No .pt or .pkl files found in provided file list.")
            if len(exts) > 1:
                raise ValueError(f"Mixed file types not supported in one list: {exts}")
            return ".pkl" in exts

        use_pickle_train = _infer_use_pickle(seg_train)
        use_pickle_val = _infer_use_pickle(seg_val)

        if use_pickle_train != use_pickle_val:
            raise ValueError(
                f"Train and val file lists must use the same file type (.pt or .pkl). "
                f"Got train use_pickle={use_pickle_train}, val use_pickle={use_pickle_val}"
            )

        use_pickle = use_pickle_train

        # Extract labels and build joint mapping
        str_labels_train: List[str] = [_extract_label_from_name(p) for p in seg_train]
        str_labels_val: List[str] = [_extract_label_from_name(p) for p in seg_val]

        all_labels = str_labels_train + str_labels_val
        unique_labels = sorted(set(all_labels))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

        numerical_labels_train = [label_to_idx[lab] for lab in str_labels_train]
        numerical_labels_val = [label_to_idx[lab] for lab in str_labels_val]

        print(f"[Explicit file lists] Received {len(seg_train)} train segment files")
        print(f"[Explicit file lists] Received {len(seg_val)} val segment files")
        print(f"Detected labels: {unique_labels}")
        print(f"labels mapping: {label_to_idx}")

        labels_train = numerical_labels_train
        labels_val = numerical_labels_val

    else:
        if DataPath is None:
            raise ValueError(
                "Either DataPath (for single-dir mode) or TrainFiles+ValFiles (for explicit mode) must be provided."
            )

        root = Path(DataPath)

        # 1) Find segment files
        all_files, use_pickle = _find_segment_files(root)

        # 2) Extract string labels from filenames
        str_labels: List[str] = []
        for p in all_files:
            label = _extract_label_from_name(p)
            str_labels.append(label)

        # 3) Map labels to integers
        unique_labels = sorted(set(str_labels))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        numerical_labels = [label_to_idx[lab] for lab in str_labels]

        print(f"[Single dir] Found {len(all_files)} segment files in {root}")
        print(f"Detected labels: {unique_labels}")
        print(f"Using train_ratio={train_ratio}")
        print(f"labels mapping: {label_to_idx}")

        # 4) Train/val split (per label)
        train_idx, val_idx = _split_train_val_indices(numerical_labels, train_ratio=train_ratio)

        seg_train = [all_files[i] for i in train_idx]
        seg_val = [all_files[i] for i in val_idx]
        labels_train = [numerical_labels[i] for i in train_idx]
        labels_val = [numerical_labels[i] for i in val_idx]

    # -----------------------------------------------------------------
    # Build Datasets and DataLoaders (shared for both modes)
    # -----------------------------------------------------------------
    dataset_train = SpectrogramDataset(seg_train, labels_train, use_pickle=use_pickle)
    dataset_val = SpectrogramDataset(seg_val, labels_val, use_pickle=use_pickle)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return seg_train, seg_val, dataloader_train, dataloader_val


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage: single directory with internal split
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data/processed/simulated_room_stft"

    ModelParam = {
        "BatchSize": 16,
        "TrainRatio": 0.8,
        "NumWorkers": 0,
        "PinMemory": True,
    }

    # Mode 1: single dir
    #train_path, val_path, dataloader_train, dataloader_val = ImportData(data_dir, ModelParam)

    # Mode 2 (example): explicit train/val dirs
    train_dir = base_dir / "data/processed/simulated_room_stft"
    train_files = sorted(train_dir.glob("*.pt"))
    val_dir = base_dir / "data/processed/simulated_room_stft"
    val_files = sorted(val_dir.glob("*.pt"))
    train_path, val_path, dataloader_train, dataloader_val = ImportData(
         DataPath=None,
         ModelParam=ModelParam,
         TrainFiles=train_files,
         ValFiles=val_files,
     )

    # Dimension test
    batch = next(iter(dataloader_train))
    STFT, labels = batch
    print("Batch STFT shape:", STFT.shape)
    print("Batch labels shape:", labels.shape)
    
    dataset_train = dataloader_train.dataset  # SpectrogramDataset
