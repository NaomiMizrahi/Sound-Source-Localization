# -*- coding: utf-8 -*-
"""
Refactored experiment runner for acoustic scene classification:
- simulation
- STFT segmentation
- training (with optional hyperparameter search)
- evaluation

Usage (example):
    python run_experiment.py -c configs/Param_CV.yaml
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List


import yaml
import torch


from ados_room_simulation import RoomSimulation
from save_segments import save_segments_from_directory
from import_dataloader import ImportData 
from models.model_lstm import AcousticSceneLSTMClassifier
from models.model_transformer_lstm import AcousticSceneTransformerLSTM
from training_utils import run_hyperparameter_search, train_model
from evaluation import evaluate_classification_model

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
def set_parser() -> argparse.Namespace:
    """CLI argument parser for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Acoustic scene classification experiment"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g. configs/Param_CV.yaml)",
    )
    return parser.parse_args()


def load_yaml(file_pointer: str) -> Dict[str, Any]:
    """Load YAML config."""
    cfg_path = Path(file_pointer)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------------
# Main pipeline class
# -------------------------------------------------------------------------
class AcousticScenePipeline:
    """
    Main end-to-end pipeline.

    """

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace) -> None:
        self.config = config
        self.args = args

        # --- Extract common config sections (rename to match your YAML) ---
        self.paths_cfg: Dict[str, Any] = config.get("Paths", config.get("paths", {}))
        self.mode_cfg: Dict[str, Any] = config.get("Mode", config.get("mode", {}))
        self.model_cfg: Dict[str, Any] = config.get("ModelParam", {})
        self.sim_cfg: Dict[str, Any] = config.get("SimulationParam", {})
        self.stft_cfg: Dict[str, Any] = config.get("STFTParam", {})
        self.train_cfg: Dict[str, Any] = config.get("TrainParam", {})

        # --- Root paths (project-relative) ---
        self.project_root = Path(__file__).resolve().parent
        self.clean_path = self.project_root / self.paths_cfg.get("CleanPath", "")
        self.noise_path = self.project_root / self.paths_cfg.get("NoisePath", "")
        self.simulated_path = self.project_root / self.paths_cfg.get(
            "SimulatedPath", "data/processed/simulated"
        )
        self.stft_path = self.project_root / self.paths_cfg.get(
            "STFTPath", "data/processed/stft"
        )
        self.results_root = self.project_root / self.paths_cfg.get(
            "ResultsPath", "outputs"
        )

        # --- Device / GPU selection ---
        gpu_num = self.model_cfg.get("GPUNum", 0)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Save folder for this experiment ---
        # You can make this more sophisticated (timestamp, config hash, etc.)
        self.experiment_name = self.model_cfg.get("ExperimentName", "acoustic_ssl")
        self.save_dir = self.results_root / self.experiment_name

        # Hyperparam search space (if you use it)
        self.hparam_space = self.train_cfg.get("HyperParamSpace", {})


    def run_all(self) -> None:
        """Run the full pipeline based on mode flags in the config."""
        start_time = time.time()
        self._prepare_output_dir()

        print(f"Running AcousticScenePipeline on device: {self.device}")
        print(f"Results will be saved to: {self.save_dir}")

        # 1) Simulation
        if self.mode_cfg.get("Simulation", False):
            print("Running room simulation")
            self.run_simulation()
        else:
            print("Skipping simulation (Simulation=False)")

        # 2) STFT segmentation
        if self.mode_cfg.get("STFT", False):
            print("Computing STFT segments")
            self.run_stft()
        else:
            print("Skipping STFT (STFT=False)")

        # 3) Training / Hyperparameter search
        if self.mode_cfg.get("Train", False):
            print("Training model")
            self.run_training()
        else:
            print("Skipping training (Train=False)")

        # 4) Evaluation / summary (optional hook)
        if self.mode_cfg.get("Evaluate", True):
            print("Final evaluation")
            self._evaluate_model()
        else:
            print("Skipping evaluation (Evaluate=False)")

        total_time = time.time() - start_time
        print(f"Pipeline completed in {total_time:.1f} seconds.")

    # ---------------------------------------------------------------------
    # Internal helper methods
    # ---------------------------------------------------------------------
    def _prepare_output_dir(self) -> None:
        """Create the experiment save directory if it doesn't exist."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Optionally copy config file into save_dir for reproducibility:
        cfg_dest = self.save_dir / "config_used.yaml"
        if self.args.config:
            cfg_src = Path(self.args.config).resolve()
            if cfg_src.is_file():
                cfg_dest.write_text(cfg_src.read_text())
        print(f"Experiment directory prepared: {self.save_dir}")

    # ---------------------------------------------------------------------
    # 1) Simulation
    # ---------------------------------------------------------------------
    def run_simulation(self) -> None:
        """
        Run room simulation to generate multi-mic recordings.

        """
        self.simulated_path.mkdir(parents=True, exist_ok=True)

        # Example: typical parameters from your sim_cfg; adjust names to match YAML.
        snr_range = self.sim_cfg.get("SNRRange", [5, 15])
        rt60 = self.sim_cfg.get("RT60", [0.5])

        print(
            f"Simulatig ADOS room with SNR range={snr_range},"
            f"and RT60={rt60}"
        )



        sim = RoomSimulation(
            clean_path=self.clean_path,
            noise_path=self.noise_path,
            save_path=self.simulated_path,
            snr_range=snr_range,
            rt60=rt60,
            sim_cfg=self.sim_cfg,
            )
        sim.run()

        print("Simulation step finished (implement details inside run_simulation).")

    # ---------------------------------------------------------------------
    # 2) STFT segmentation
    # ---------------------------------------------------------------------
    def run_stft(self) -> None:
        """
        Compute STFT on simulated data and save segments.

        """
        self.stft_path.mkdir(parents=True, exist_ok=True)

        seg_len = self.stft_cfg.get("SegLen", 25)
        n_fft = self.stft_cfg.get("n_fft", 1024)
        hop_length = self.stft_cfg.get("hop_length", 512)
        target_sr = self.stft_cfg.get("SampleRate", 16000)
        

        print(
            f"Running STFT segmentation: SegLen={seg_len}, n_fft={n_fft}, "
            f"hop_length={hop_length}"
        )
        print(f"Input dir:  {self.simulated_path}")
        print(f"Output dir: {self.stft_path}")


        save_segments_from_directory(
            data_dir=self.simulated_path,  # list of folders
            out_dir=self.stft_path,
            segment_length=seg_len,
            n_fft=n_fft,
            hop_length=hop_length,
            target_sr=target_sr,
            )

        print("STFT segmentation step finished.")

    # ---------------------------------------------------------------------
    # 3) Training (with optional hyperparameter search)
    # ---------------------------------------------------------------------
    def run_training(self) -> None:
        """
        Train models on the STFT dataset.

        If HyperTune=True in config, perform hyperparameter search
        before training the final model.
        """
        
        num_chnls = self.train_cfg.get("num_chnls", 16)
        num_classes = self.train_cfg.get("num_classes", 4)
        
        if self.train_cfg.get("model")=="lstm":
            self.model = AcousticSceneLSTMClassifier(num_chnls=num_chnls, num_classes=num_classes)
        else:
            self.model = AcousticSceneTransformerLSTM(num_chnls=num_chnls, num_classes=num_classes)
        
        train_path, val_path, self.train_loader, self.val_loader = ImportData(self.stft_path, self.train_cfg)
        hyper_tune = self.train_cfg.get("HyperTune", False)

        if hyper_tune:
            print("Hyperparameter search enabled.")
            best_params = run_hyperparameter_search(
                self.train_cfg,
                train_path,
                self.hparam_space,
                self.model,
                device = self.device 
                )
            print(f"Best hyperparameters found: {best_params}")
            self.trained_model = train_model(
                model=self.model,
                train_loader=self.train_loader,
                num_epochs=best_params.get("epochs"),
                lr=best_params.get("learning_rate"),
                device=self.device,
            )
        else:
            print("Hyperparameter search disabled. Using fixed params from config.")
            fixed_params = {
                "batch_size": self.train_cfg.get("BatchSize", 64),
                "epochs": self.train_cfg.get("Epochs", 20),
                "learning_rate": self.train_cfg.get("LearningRate", 1e-3),
            }
            self.trained_model = train_model(
                    model=self.model,
                    train_loader=self.train_loader,
                    num_epochs=fixed_params.get("epochs"),
                    lr=fixed_params.get("learning_rate"),
                    device=self.device,
                    )

    # ---------------------------------------------------------------------
    # 4) Evaluation / summary
    # ---------------------------------------------------------------------
    def _evaluate_model(self) -> None:
        """
        Model evaluation
        """
        metrics = evaluate_classification_model(
            self.trained_model,
            self.val_loader,
            device=self.device,
            save_dir=self.results_root,
            save_excel=True,
        )
        score = metrics["accuracy"]
        print(f"Evaluation finished, model Accuracy: {score}")



# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main() -> AcousticScenePipeline:
    args = set_parser()
    cfg = load_yaml(file_pointer=args.config)
    pipeline = AcousticScenePipeline(cfg, args)
    return pipeline


if __name__ == "__main__":
    pipeline = main()
    pipeline.run_all()
