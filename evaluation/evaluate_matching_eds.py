"""
validate_matching.py — Production-ready event-based matching evaluator for REALM models.

Usage:
    python validate_matching.py --config configs/matching.yaml
    python validate_matching.py --config configs/matching.yaml --model minima --device cuda:1
    python validate_matching.py --config configs/matching.yaml --sequences 00_peanuts_dark 01_peanuts_light
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
import yaml
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from tabulate import tabulate
from tqdm import tqdm

from realm.model_factory import REALM_creator

# ---------------------------------------------------------------------------
# Deterministic / performance flags
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default EDS sequences
# ---------------------------------------------------------------------------
EDS_DEFAULT_SEQUENCES = [
    "00_peanuts_dark",
    "01_peanuts_light",
    "02_rocket_earth_light",
    "03_rocket_earth_dark",
    "06_ziggy_and_fuzz",
    "07_ziggy_and_fuzz_hdr",
    "08_peanuts_running",
    "11_all_characters",
]


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and minimally validate a YAML config file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def events_to_image(events: torch.Tensor, H: int, W: int) -> np.ndarray:
    """Render an event tensor (N, 4) into a log-normalised (H, W) uint8 image."""
    if len(events) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    xs = events[:, 0].long().clamp(0, W - 1)
    ys = events[:, 1].long().clamp(0, H - 1)
    img = torch.zeros((H, W), device=events.device, dtype=torch.float32)
    img.index_put_((ys, xs), torch.ones_like(xs, dtype=torch.float32), accumulate=True)
    img = torch.log1p(img)
    img = img / (img.max() + 1e-5) * 255.0
    return img.cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_auc(results: dict, thresholds: list[float]) -> list[float]:
    """Compute AUC under the pose-error recall curve for each threshold."""
    if not results["r_error"]:
        return [0.0] * len(thresholds)

    errors = [0.0] + list(np.sort(results["r_error"]))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(float(np.trapezoid(y, x) / thr * 100))
    return aucs


# ---------------------------------------------------------------------------
# Data loading helpers  (left unchanged per spec)
# ---------------------------------------------------------------------------

def get_calibration_from_eds_txt(calib_path: Path):
    data = np.genfromtxt(calib_path)
    K = np.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = data[0], data[1], data[2], data[3]
    D = data[4:]
    return K, D


def get_events_from_hdf5(
    event_proxy,
    timestamp_us: int,
    delta_t_us: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    t_end = timestamp_us
    t_start = t_end - delta_t_us
    ev_times = event_proxy.t
    idx_end = np.searchsorted(ev_times, t_end)
    idx_start = max(0, np.searchsorted(ev_times, t_start))

    if idx_start >= idx_end:
        return None

    x = event_proxy.x[idx_start:idx_end]
    y = event_proxy.y[idx_start:idx_end]
    p = event_proxy.p[idx_start:idx_end]
    t = ev_times[idx_start:idx_end]

    if len(t) == 0:
        return None

    t_sec = t.astype(np.float64) * 1e-6
    events_np = np.stack([x, y, t_sec, p], axis=1)
    return torch.from_numpy(events_np).to(device, dtype=torch.float64)


class EventProxyWrapper:
    def __init__(self, h5_grp) -> None:
        self.t = h5_grp["t"]
        self.x = h5_grp["x"]
        self.y = h5_grp["y"]
        self.p = h5_grp["p"]


def interpolate_poses(
    gt_times: np.ndarray,
    gt_poses: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    t_interp = interp1d(gt_times, gt_poses[:, :3], axis=0, fill_value="extrapolate")(query_times)
    q_interp = interp1d(gt_times, gt_poses[:, 3:], axis=0, fill_value="extrapolate")(query_times)
    q_interp /= np.linalg.norm(q_interp, axis=1, keepdims=True)
    return np.hstack([t_interp, q_interp])


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class MatchingValidator:
    """
    End-to-end pose-error evaluator for REALM event-based matching models.

    Responsibilities
    ----------------
    * Model initialisation via REALM_creator (mirrors SegmentationValidator)
    * Per-sequence evaluation over the EDS benchmark
    * AUC computation across configurable rotation thresholds
    * Results saved to a timestamped text file and .npz archive
    """

    # Fixed EDS sensor resolution
    H: int = 480
    W: int = 640

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)

        # Config ---------------------------------------------------------------
        self.cfg = load_config(args.config)
        eval_cfg: dict = self.cfg.get("evaluation", {})

        # Model ----------------------------------------------------------------
        logger.info("Building model...")
        self.model = REALM_creator(self.cfg).to(self.device)
        self.model.eval()
        logger.info("Model type: %s", self.model.model_type.value)

        # Evaluation parameters ------------------------------------------------
        self.delta_t_us: int = int(eval_cfg.get("model_delta_t", args.model_delta_t) * 1e6)
        self.max_delta_t_us: float = eval_cfg.get("max_eval_delta_t", args.max_eval_delta_t) * 1e6
        self.ransac_threshold: float = eval_cfg.get("ransac_threshold", args.ransac_threshold)
        self.auc_thresholds: list[float] = eval_cfg.get("auc_thresholds", args.auc_thresholds)

        # Output paths ---------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_tag = f"eds_{self.model.model_type.value}_{timestamp}"
        self.output_dir = Path("results") / "matching" / run_tag
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_txt_path = self.output_dir / "results.txt"
        logger.info("Run output directory: %s", self.output_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self, sequences: list[str], dataset_path: Path) -> dict:
        """
        Evaluate over a list of sequence names and return aggregated results.

        Args:
            sequences:    List of EDS sequence folder names.
            dataset_path: Root directory containing the sequence sub-folders.
        """
        all_results: dict = {"r_error": [], "inlier_ratio": [], "gt_rot": [], "dt": []}
        scene_results: dict = {}

        logger.info("Model type : %s", self.model.model_type.value)
        logger.info("Sequences  : %d", len(sequences))

        for seq_name in sequences:
            seq_path = dataset_path / seq_name
            if not seq_path.exists():
                logger.warning("Skipping missing sequence: %s", seq_path)
                continue

            logger.info("Processing sequence: %s", seq_name)
            try:
                res = self._evaluate_sequence(seq_path)
            except Exception:
                logger.exception("Failed to evaluate sequence: %s", seq_name)
                continue

            scene_auc = calculate_auc(res, self.auc_thresholds)
            scene_results[seq_name] = scene_auc
            logger.info(
                "  %s — AUC@%s: %s",
                seq_name,
                "/".join(str(int(t)) for t in self.auc_thresholds),
                "  ".join(f"{a:.2f}" for a in scene_auc),
            )

            for k, v in res.items():
                all_results[k].extend(v)

        self._save_results(all_results, scene_results)
        return all_results

    # ------------------------------------------------------------------
    # Per-sequence evaluation
    # ------------------------------------------------------------------

    def _evaluate_sequence(self, seq_path: Path) -> dict:
        """Run the matching + pose recovery loop for one sequence."""
        results: dict = {"r_error": [], "inlier_ratio": [], "gt_rot": [], "dt": []}

        K, D = get_calibration_from_eds_txt(seq_path / "calib.txt")

        gt_data = np.genfromtxt(seq_path / "stamped_groundtruth.txt")
        gt_data[:, 0] *= 1e6  # s → µs
        image_ts = np.genfromtxt(seq_path / "images_timestamps.txt")

        poses_interp = interpolate_poses(gt_data[:, 0], gt_data[:, 1:], image_ts)
        poses_gt = np.zeros((len(image_ts), 8))
        poses_gt[:, 0] = image_ts
        poses_gt[:, 1:] = poses_interp

        deg_intervals = np.arange(1, 46)  # 1° … 45°

        with h5py.File(seq_path / "events.h5", "r") as f_events:
            event_proxy = EventProxyWrapper(f_events)

            step = 0
            skip_until_step = 0
            pbar = tqdm(total=len(image_ts), desc=seq_path.name, unit="frame")

            while step < len(image_ts):
                pbar.update(1)

                if step < skip_until_step:
                    step += 1
                    continue

                # Scan ahead to find frames that span the desired rotation range
                gt_rot_list, abs_rot_list = [], []
                has_anchor = False

                for lookahead in range(step + 1, len(image_ts)):
                    dt = poses_gt[lookahead, 0] - poses_gt[step, 0]
                    if dt > self.max_delta_t_us:
                        break

                    r_step = R.from_quat(poses_gt[step, 4:8])
                    r_next = R.from_quat(poses_gt[lookahead, 4:8])
                    rel_rot = r_next.inv() * r_step
                    deg = float(np.linalg.norm(rel_rot.as_rotvec(degrees=True)))

                    gt_rot_list.append(rel_rot)
                    abs_rot_list.append(deg)

                    if deg > float(deg_intervals.max()):
                        has_anchor = True
                        break

                if not has_anchor:
                    step += 1
                    continue

                data_view1 = get_events_from_hdf5(
                    event_proxy, int(image_ts[step]), self.delta_t_us, self.device
                )
                if data_view1 is None:
                    step += 1
                    continue

                last_lookahead = step + len(gt_rot_list)
                skip_until_step = last_lookahead + 1
                interval_idx = 0

                for lookahead in range(step + 1, last_lookahead + 1):
                    list_idx = lookahead - step - 1
                    if abs_rot_list[list_idx] < deg_intervals[interval_idx]:
                        continue

                    interval_idx += 1
                    data_view2 = get_events_from_hdf5(
                        event_proxy, int(image_ts[lookahead]), self.delta_t_us, self.device
                    )
                    if data_view2 is None:
                        continue

                    error_angle, inlier_ratio = self._match_and_recover_pose(
                        data_view1, data_view2, K, D, gt_rot_list[list_idx]
                    )

                    results["r_error"].append(error_angle)
                    results["inlier_ratio"].append(inlier_ratio)
                    results["gt_rot"].append(abs_rot_list[list_idx])
                    results["dt"].append(poses_gt[lookahead, 0] - poses_gt[step, 0])

                step += 1

            pbar.close()

        return results

    # ------------------------------------------------------------------
    # Matching + pose recovery
    # ------------------------------------------------------------------

    def _match_and_recover_pose(
        self,
        data_view1: torch.Tensor,
        data_view2: torch.Tensor,
        K: np.ndarray,
        D: np.ndarray,
        gt_rot: R,
    ) -> tuple[float, float]:
        """
        Run the model matcher and recover pose via RANSAC + Essential matrix.

        Returns:
            (rotation_error_deg, inlier_ratio)
        """
        try:
            kpts1, kpts2, _ = self.model(data_view1, data_view2)
            num_inliers, _E, est_R, _t, _mask = cv2.recoverPose(
                kpts1.astype(np.float64),
                kpts2.astype(np.float64),
                K, D, K, D,
                method=cv2.RANSAC,
                threshold=self.ransac_threshold,
            )
            error_angle = float(
                np.linalg.norm(
                    (R.from_matrix(est_R) * gt_rot.inv()).as_rotvec(degrees=True)
                )
            )
            inlier_ratio = num_inliers / max(len(kpts1), 1)
        except Exception:
            logger.debug("Pose recovery failed for a pair — marking as failed match.")
            error_angle = 180.0
            inlier_ratio = 0.0

        return error_angle, inlier_ratio

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _save_results(self, all_results: dict, scene_results: dict) -> None:
        """Print, log, and persist evaluation results."""
        if not all_results["gt_rot"]:
            logger.warning("No pairs were evaluated — nothing to save.")
            return

        final_aucs = calculate_auc(all_results, self.auc_thresholds)
        thr_header = "/".join(str(int(t)) for t in self.auc_thresholds)

        global_rows = [[f"AUC@{thr_header}", "  ".join(f"{a:.2f}" for a in final_aucs)],
                       ["Total pairs", len(all_results["gt_rot"])]]
        per_scene_rows = [
            [name, "  ".join(f"{a:.2f}" for a in aucs)]
            for name, aucs in scene_results.items()
        ]

        sep = "=" * 50
        output = "\n".join([
            f"\n{sep}",
            "   FINAL MATCHING RESULTS",
            sep,
            tabulate(global_rows, headers=["Metric", "Value"], tablefmt="grid"),
            "",
            f"--- Per-sequence AUC@{thr_header} ---",
            tabulate(per_scene_rows, headers=["Sequence", "AUC"], tablefmt="simple"),
            sep,
        ])

        logger.info(output)

        try:
            with self.results_txt_path.open("a") as fh:
                fh.write(output + "\n")
            logger.info("Results saved to %s", self.results_txt_path)
        except OSError as exc:
            logger.warning("Could not write results file: %s", exc)

        npz_path = self.output_dir / "all_results.npz"
        np.savez(npz_path, **all_results)
        logger.info("Raw results saved to %s", npz_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REALM event-based matching evaluator (EDS benchmark)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/matching.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/media/stonehenge/datasets/event_datasets/eds_hdf5",
        help="Root directory of the EDS dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device string, e.g. 'cuda', 'cuda:1', 'cpu'.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=EDS_DEFAULT_SEQUENCES,
        help="EDS sequence folder names to evaluate.",
    )
    parser.add_argument(
        "--model_delta_t",
        type=float,
        default=0.03,
        help="Event window duration fed to the model (seconds).",
    )
    parser.add_argument(
        "--max_eval_delta_t",
        type=float,
        default=2.0,
        help="Maximum time gap between anchor and query frames (seconds).",
    )
    parser.add_argument(
        "--auc_thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0],
        help="Rotation thresholds (degrees) for AUC computation.",
    )
    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=1.0,
        help="RANSAC reprojection threshold (pixels).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    validator = MatchingValidator(args)
    validator.run(args.sequences, Path(args.dataset_path))
    