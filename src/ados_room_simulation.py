# -*- coding: utf-8 -*-
import pyroomacoustics as pra
import librosa
import numpy as np
import random
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional

random.seed(100)


class RoomSimulation:
    def __init__(
        self,
        clean_path,
        noise_path,
        save_path,
        snr_range=(5, 15),
        rt60=None,
        sim_cfg=None,
        ):
        
        """

        Parameters
        ----------
        clean_path : str or Path
            Folder containing clean speech (LibriSpeech, etc.).
        noise_path : str or Path
            Folder containing noise wav files (MS-SNSD, etc.).
        save_path : str or Path
            Folder where simulated multi-mic wavs will be saved.
        snr_range : (float, float)
            Min and max SNR in dB. E.g. (5, 15).
        rt60_list : list[float] or None
            List of RT60 values to sample from. If None, uses a default.
        sim_cfg : dict or None
            Extra simulation parameters from YAML (can include num_mics, sr, etc.).
        """
        

        self.audio_folder = Path(clean_path)
        self.noise_folder = Path(noise_path)
        self.output_folder = Path(save_path)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.label_file = self.output_folder / "labels.csv"

        self.snr_range = snr_range  # (min_snr, max_snr)
        self.rt60 = rt60 if rt60 is not None else 0.5
        self.sim_cfg = sim_cfg or {}


       # Default room / sim settings
        self.room_dims = [4.8, 5.4, 3]
        self.fs = int(self.sim_cfg.get("SampleRate", 16000))
        self.num_mics = int(self.sim_cfg.get("NumMics", 8))
        self.max_order = int(self.sim_cfg.get("MaxOrder", 10))

        # Semantic locations
        self.location_choices = ["desk", "carpet", "sofa", "other"]

        # Location-specific ranges (x, y, z) 
        self.location_ranges = {
            "desk": {
                "x": [1.8, 2.4],
                "y": [2.7, 3.7],
                "z": [0.5, 1.6],
            },
            "carpet": {
                "x": [2.4, 3.0],
                "y": [2.1, 5.1],
                "z": [0.2, 1.6],
            },
            "sofa": {
                "x": [2.0, 4.8],
                "y": [0.0, 2.1],
                "z": [0.2, 1.6],
            },
            "other": {
                "x": [0.0, 2.4],
                "y": [1.35, 2.7],
                "z": [0.5, 1.6],
            },
        }

    # ------------------------------------------------------------------
    # Run the simulation
    # ------------------------------------------------------------------
    def run(self):
        """Wrapper so pipeline can call `sim.run()`."""
        return self.simulate()

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------
    def simulate(self) -> List[str]:
        """
        Main simulation loop.

        Returns
        -------
        labels : list[str]
            List of "filename,location" strings.
        """
        clean_files = self._collect_audio_files(self.audio_folder)
        noise_files = self._collect_audio_files(self.noise_folder)

        if len(clean_files) == 0:
            raise RuntimeError(f"No clean audio files found in {self.audio_folder}")
        if len(noise_files) == 0:
            raise RuntimeError(f"No noise audio files found in {self.noise_folder}")


        labels = []

        print(f"Found {len(clean_files)} clean files, {len(noise_files)} noise files.")
        print(f"Output folder: {self.output_folder}")
        print(f"Room dims: {self.room_dims}, fs: {self.fs}, num_mics: {self.num_mics}")

        for idx, audio_file in enumerate(clean_files, 1):
            #print(f"\n[{idx}/{len(clean_files)}] Simulating file: {audio_file.name}")

            # --------------------------------------------------------------
            # 1) Load clean speech
            # --------------------------------------------------------------
            clean, _ = librosa.load(audio_file, sr=self.fs)
            if clean.size == 0:
                print("  -> Skip (empty audio).")
                continue

            # --------------------------------------------------------------
            # 2) Build non-shoebox room from corners and extrude to 3D
            # --------------------------------------------------------------
            floor = np.array(
                [
                [0.0, 0.0],
                [0.0, 2.7],
                [1.8, 2.7],
                [1.8, 5.4],
                [4.8, 5.4],
                [4.8, 0.0],
                ]
                ).T  # shape (2, N_vertices)
            

            # RT=0.5
            if self.rt60==0.5:
                wall_mat = {
                     "description": "wall metrials",
                     "coeffs": [0.25, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15],
                     "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],}
            #RT=0.7    
            elif self.rt60==0.7:
                wall_mat = {
                     "description": "wall metrials",
                     "coeffs": [0.2, 0.2, 0.22, 0.22, 0.11, 0.11, 0.1],
                     "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],}
            
            #RT=0.9
            elif self.rt60==0.9:
                wall_mat = {
                    "description": "wall metrials",
                    "coeffs": [0.19, 0.19, 0.19, 0.19, 0.085, 0.085, 0.085],
                    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],}
            else: 
                print("RT60 is not valid")
                continue

            # One material per wall
            m = pra.make_materials(*[(wall_mat,) for _ in range(floor.shape[1])])
    
            # 2D room from polygonal floor
            room = pra.Room.from_corners(
                floor,
                fs=self.fs,
                materials=m,
                max_order=self.max_order,
                ray_tracing=True,
                air_absorption=True,
                humidity=self.sim_cfg.get("Humidity", 25),
            )
    
            # Extrude floor to full 3D room with given height
            room_height = self.room_dims[2]
            floor_and_ceiling_mat = pra.make_materials(
                ceiling=(wall_mat,),
                floor=(wall_mat,),
            )
            room.extrude(room_height, materials=floor_and_ceiling_mat)
    
            # --------------------------------------------------------------
            # 3) Microphone array: fixed positions + small perturbations
            # --------------------------------------------------------------
            # Room bounds for each axis
            MinMaxLoc = np.array(
                [
                    [0.0, self.room_dims[0]],        
                    [0.0, self.room_dims[1]],        
                    [0.0, self.room_dims[2]] 
                ]
            )
    
            # 8-mic layout 
            mic_locations = np.array(
                [
                    [1.8, 4.8, 4.8, 4.8, 1.95, 3.9, 1.8, 3.9],   # x
                    [2.85, 2.85, 3.15, 2.3, 3.05, 3.0, 1.05, 5.4],  # y
                    [1.7, 1.3, 2.75, 2.75, 1.05, 1.95, 1.2, 2.15],  # z
                ]
            )
    
            # small random perturbation 
            valid_locations = np.zeros_like(mic_locations)
            
            for i in range(mic_locations.shape[1]):
                base_loc = mic_locations[:, i]
            
                for _ in range(50):  
                    candidate = base_loc.copy()
                    for j in range(3):
                        low = max(-0.1, round(MinMaxLoc[j, 0] - base_loc[j], 2))
                        high = min(0.1, round(MinMaxLoc[j, 1] - base_loc[j], 2))
                        perturbation = np.random.uniform(low=low, high=high)
                        candidate[j] = base_loc[j] + perturbation
            
                    if room.is_inside(candidate):
                        valid_locations[:, i] = candidate
                        break
                else:
                    # if no valid perturbation, fall back to base (which we know is inside)
                    if not room.is_inside(base_loc):
                        raise ValueError(f"Base mic #{i} at {base_loc} is not inside the room!")
                    valid_locations[:, i] = base_loc
            
            mic_locations_perturbed_valid = valid_locations
            #print("Mic positions (perturbed & validated):\n", mic_locations_perturbed_valid)
    
            room.add_microphone_array(mic_locations_perturbed_valid)

            # --------------------------------------------------------------
            # 4) Choose semantic location for the speech source
            # --------------------------------------------------------------
            location_type = random.choice(self.location_choices)
            source_location = self._sample_location_for_type(location_type)

            #print(f"Location type: {location_type}, coord: {source_location}")

            room.add_source(source_location, signal=clean)

            # --------------------------------------------------------------
            # 5) Add noise as second source with desired SNR
            # --------------------------------------------------------------
            noise_file = random.choice(noise_files)

            noise, _ = librosa.load(noise_file, sr=self.fs, mono=True)

            extracted_noise = self._extract_random_noise_segment(noise, len(clean))
            scaled_noise = self._scale_noise(clean, extracted_noise)

            noise_location = [
                np.random.uniform(1.8, 4.8),  # x-coordinate
                np.random.uniform(0, 5.4),  # y-coordinate
                np.random.uniform(2.2, 2.4)  # z-coordinate (height between 2.2 and 2.4)
                ]

            room.add_source(noise_location, signal=scaled_noise)
            #print("Finished add noise")

            # --------------------------------------------------------------
            # 6) Run simulation
            # --------------------------------------------------------------

            room.simulate()

            # mic_signals: shape (num_mics, n_samples)
            mic_signals = room.mic_array.signals  # (n_samples, num_mics)

            # Normalize to avoid clipping
            #max_abs = np.max(np.abs(mic_signals)) + 1e-12
            #mic_signals = mic_signals / max_abs * 0.99
            peak = float(np.max(np.abs(mic_signals))) + 1e-12
            if peak > 0:
                mic_signals = 0.99 * mic_signals / peak
            #print("finished rec simulation")

            # --------------------------------------------------------------
            # 7) Save multichannel WAV
            # --------------------------------------------------------------
            out_name = (
                f"{audio_file.stem}_loc-{location_type}.wav"
            )
            out_path = self.output_folder / out_name

            sf.write(out_path, mic_signals.T.astype(np.float32), self.fs)
            #print(f"Saved: {out_path}")

            # --------------------------------------------------------------
            # 8) Save label entry
            # --------------------------------------------------------------
            labels.append(f"{out_name},{location_type}")

        # --------------------------------------------------------------
        # 9) Write labels file
        # --------------------------------------------------------------
        with self.label_file.open("w", encoding="utf-8") as f:
            f.write("filename,location\n")
            for line in labels:
                f.write(line + "\n")

        print(f"Simulations completed. Labels saved to: {self.label_file}")
        return labels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_audio_files(folder: Path) -> List[Path]:
        exts = (".wav", ".flac", ".ogg")
        files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
        return sorted(files)

    def is_inside_room(self, location: np.ndarray) -> bool:
        """Check if a 3D point is inside the room box."""
        x, y, z = location
        return (
            0.0 <= x <= self.room_dims[0]
            and 0.0 <= y <= self.room_dims[1]
            and 0.0 <= z <= 1.6
        )

    def _sample_location_for_type(self, loc_type: str) -> np.ndarray:
        """Sample a random (x, y, z) within the given semantic region."""

        ranges = self.location_ranges[loc_type]
        for _ in range(100):
            x = random.uniform(*ranges["x"])
            y = random.uniform(*ranges["y"])
            z = random.uniform(*ranges["z"])
            loc = np.array([x, y, z])
            if self.is_inside_room(loc):
                return loc
            

        # if we fail many times, just return zone center
        center_x = sum(ranges["x"]) / 2
        center_y = sum(ranges["y"]) / 2
        center_z = sum(ranges["z"]) / 2
        return np.array([center_x, center_y, center_z])

    @staticmethod
    def _extract_random_noise_segment(noise: np.ndarray, length: int) -> np.ndarray:
        """Cut a random contiguous segment of given length from noise."""
        if len(noise) < length:
            # tile noise if it's too short
            reps = int(np.ceil(length / len(noise)))
            noise = np.tile(noise, reps)

        max_start = len(noise) - length
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + length
        return noise[start_idx:end_idx]

    def _scale_noise(self, clean: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Scale noise to achieve a random SNR within self.snr_range.

        clean, noise are 1D arrays with the same length.
        """
        rms_clean = np.sqrt(np.mean(clean**2))
        rms_noise = np.sqrt(np.mean(noise**2))

        snr_min, snr_max = self.snr_range
        desired_snr = random.uniform(snr_min, snr_max)  # dB

        # clean_rms / noise_rms = 10^(SNR/20)
        scaling_factor = (rms_clean / rms_noise) / (10.0 ** (desired_snr / 20.0))
        scaled_noise = noise * scaling_factor

        #print(
        #    f"  Noise scaled to SNR={desired_snr:.2f} dB "
         #   f"(scaling_factor={scaling_factor:.4f})"
        #)
        return scaled_noise


# ----------------------------------------------------------------------
# Standalone usage example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    BASE_DIR = Path(__file__).resolve().parent   # folder where the script is

    sim = RoomSimulation(
        clean_path=BASE_DIR / Path("data/raw/clean_speech"),
        noise_path=BASE_DIR / Path("data/raw/noise"),
        save_path=BASE_DIR / Path("data/processed/simulated_room"),
        snr_range=(5, 15),
        rt60=0.5,
        sim_cfg={
            "RoomDims": [4.8, 5.4, 3],
            "SampleRate": 16000,
            "NumMics": 8,
            "MaxOrder": 1,
        },
    )
    sim.run()

