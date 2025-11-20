# -*- coding: utf-8 -*-
"""
Generic training utilities for SSL / acoustic scene models.

This module provides:
    - run_hyperparameter_search: simple grid search over hyperparameters
    - train_single_model: standard train/val loop for one model instance

"""

from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from pathlib import Path
from import_dataloader import ImportData 
from evaluation import evaluate_classification_model



# ---------------------------------------------------------------------
# Hyperparameter search (grid search)
# ---------------------------------------------------------------------
def run_hyperparameter_search(
    train_cfg: Dict[str, Any],
    train_path: list,
    hparam_space: Dict[str, List[Any]],
    model: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run k-fold or simple train/val cross-validation over a grid of hyperparameters.

    Parameters
    ----------
    train_cfg : dict
        Base training configuration (BatchSize, LearningRate, Epochs, etc.).
    hparam_space : dict
        Search space for hyperparameters, e.g.:
            {
                "BatchSize": [16, 32],
                "LearningRate": [1e-3, 5e-4],
                "Epochs": [30, 50],
            }
        Keys should match those in train_cfg so we can override them.
    build_loaders_fn : callable
        Function that builds (train_loader, val_loader) given a config dict.
    build_model_fn : callable
        Function that builds and returns a fresh model given a config dict.
    val_score_fn : callable
        Function that computes a validation score (e.g. accuracy) for a model.
    device : torch.device
        Device for model and tensors.

    Returns
    -------
    best_params : dict
        Dictionary with the best hyperparameters found.
    """
    # Default search ranges if not provided
    batch_sizes = hparam_space.get("BatchSize", [train_cfg.get("BatchSize", 64)])
    learning_rates = hparam_space.get("LearningRate", [train_cfg.get("LearningRate", 1e-3)])
    epochs_list = hparam_space.get("Epochs", [train_cfg.get("Epochs", 20)])
    
    # k-fold spliting
    kf = KFold(n_splits=train_cfg.get("NumFolds", 3))

    best_score = -float("inf")
    best_params: Dict[str, Any] = {}

    for bs in batch_sizes:
        for lr in learning_rates:
            for ep in epochs_list:
                # Create a local copy of train_cfg with these hyperparams
                cfg = dict(train_cfg)
                cfg["BatchSize"] = bs
                cfg["LearningRate"] = lr
                cfg["Epochs"] = ep

                print(f"\n[GridSearch] Trying BatchSize={bs}, LR={lr}, Epochs={ep}")
                
                acc = []
                for train_index, val_index in kf.split(train_path):
                    # Split audio paths into training and validation sets for the current fold
                    audio_paths_train_fold = [train_path[i] for i in train_index]
                    audio_paths_val_fold = [train_path[i] for i in val_index]
                    
                    # Create custom datasets and data loaders for training and validation
                    _, _, train_loader, val_loader = ImportData(DataPath=None,
                                                                  ModelParam=cfg,
                                                                  TrainFiles=audio_paths_train_fold,
                                                                  ValFiles=audio_paths_val_fold,
                                                                  )
 
                    # Simple training loop for this config
                    model = train_model(
                        model=model,
                        train_loader=train_loader,
                        num_epochs=ep,
                        lr=lr,
                        device=device,
                        )

                    # Evaluate
                    metrics = evaluate_classification_model(
                        model,
                        val_loader,
                        device=device,
                        save_dir="",
                        save_excel=False,
                        )
                    acc.append(metrics["accuracy"])
                    print(f"fold Accuracy: {metrics['accuracy']}")
                
                avg_score = sum(acc) / len(acc)   
                print(f"[GridSearch] Averag score: {avg_score:.4f}")

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {
                        "batch_size": bs,
                        "learning_rate": lr,
                        "epochs": ep,
                    }
                    print(f"[GridSearch] New best score: {best_score:.4f}")

    print(f"\n[GridSearch] Best params: {best_params}, score={best_score:.4f}")
    return best_params



# ---------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    """
    Training loop with cross-entropy loss, Adam optimizer, and schedular.

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        scheduler.step()
        print(f"  Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    return model
