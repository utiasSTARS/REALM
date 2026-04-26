"""
validate_matches.py — Production-ready matches robustness evaluator for REALM models.

Evaluates relative pose estimation on the VECtor dataset across a range of rotation
magnitudes, reporting median error, accuracy, and AUC per rotation bin.

Usage:
    python validate_matches.py --config configs/matching.yaml --dataset_path datasets/VECtor/robot-fast
    python validate_matches.py --config configs/matching.yaml --dataset_path datasets/VECtor/robot-fast --mode ee --device cuda:1
    python validate_matches.py --config configs/matching.yaml --dataset_path datasets/VECtor/robot-fast --fp32
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import poselib
import torch
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from realm.utils.transform import Resize, rescale_matches
from realm.utils.vis import vis_matches, matches, voxel_to_rgb_image
from realm.model_factory import REALM_creator
from dataloaders.vector_dataset import build_vector_datasets

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
# Camera alignment helper
# ---------------------------------------------------------------------------

class AlignData:
    """
    Transforms an absolute mocap pose into the optical frame of the requested
    camera using pre-calibrated body→camera extrinsics.

    Args:
        camera_type: 'i' for the regular (inertial) camera,
                     'e' for the event camera.
    """

    def __init__(self) -> None:
        # T_body_cam0: body → left regular camera
        T_body_cam0 = np.array([
            -0.8571370239765715,  0.01322063096422758, -0.5149187674240417,  0.06315837246891948,
             0.03276713258773899, -0.9962462506036182, -0.08012317505073686, -0.02785306005411673,
            -0.514045170340666,  -0.08554895133864117,  0.853486344222504,   0.0704789883105976,
             0,                   0,                    0,                    1,
        ]).reshape(4, 4)

        # T_cam0_cam2: left regular camera → left event camera
        T_cam0_cam2 = np.array([
             0.9999407352369797,   0.009183655542749752,  0.005846920950435052,  0.0005085820608404798,
            -0.009131364645448854, 0.9999186289230431,   -0.008908070070089353, -0.04081979450823404,
            -0.005928253827254812, 0.008854151768176144,  0.9999432282899994,   -0.0140781304960408,
             0,                    0,                     0,                     1,
        ]).reshape(4, 4)

        self._R_body_cam0 = R.from_matrix(T_body_cam0[:3, :3])
        self._R_cam0_cam2 = R.from_matrix(T_cam0_cam2[:3, :3])
        self._R_body_cam2 = self._R_body_cam0 * self._R_cam0_cam2

    def __call__(self, R_mocap: R, camera_type: str) -> R:
        if camera_type == "i":
            return R_mocap * self._R_body_cam0
        if camera_type == "e":
            return R_mocap * self._R_body_cam2
        raise ValueError(f"Unknown camera_type {camera_type!r}. Expected 'i' or 'e'.")


# ---------------------------------------------------------------------------
# Calibration / pose helpers
# ---------------------------------------------------------------------------

def load_vector_calibration(calib_file: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load camera intrinsics and distortion from a VECtor YAML calibration file."""
    with calib_file.open() as fh:
        cam = yaml.load(fh, Loader=yaml.FullLoader)
    W = cam["image_width"]
    H = cam["image_height"]
    K = np.array(cam["camera_matrix"]["data"]).reshape(3, 3)
    D = np.array(cam["distortion_coefficients"]["data"])
    return K, D, H, W


def interpolate_pose(
    poses_gt: np.ndarray,
    target_time: float,
) -> tuple[np.ndarray, R]:
    """Interpolate position (linear) and rotation (SLERP) at *target_time*."""
    times = poses_gt[:, 0]
    idx = np.searchsorted(times, target_time)

    if idx == 0:
        return poses_gt[0, 1:4], R.from_quat(poses_gt[0, 4:8])
    if idx >= len(times):
        return poses_gt[-1, 1:4], R.from_quat(poses_gt[-1, 4:8])

    t0, t1 = times[idx - 1], times[idx]
    alpha = (target_time - t0) / (t1 - t0)
    pos = poses_gt[idx - 1, 1:4] + alpha * (poses_gt[idx, 1:4] - poses_gt[idx - 1, 1:4])

    rots = R.from_quat([poses_gt[idx - 1, 4:8], poses_gt[idx, 4:8]])
    rot = Slerp([t0, t1], rots)([target_time])[0]
    return pos, rot


# ---------------------------------------------------------------------------
# Visualisation / reporting helpers
# ---------------------------------------------------------------------------

def plot_robustness_analysis(
    results: dict,
    output_dir: Path,
    timestamp: str,
    mode: str,
) -> None:
    """Save a three-panel scatter + median trendline figure."""

    def _add_trendline(ax, x: np.ndarray, y: np.ndarray, num_bins: int = 15) -> None:
        if len(x) == 0:
            return
        bins = np.linspace(x.min(), x.max(), num_bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        medians, p25, p75 = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (x >= lo) & (x < hi)
            if mask.any():
                medians.append(np.median(y[mask]))
                p25.append(np.percentile(y[mask], 25))
                p75.append(np.percentile(y[mask], 75))
            else:
                medians.append(np.nan)
                p25.append(np.nan)
                p75.append(np.nan)
        ax.plot(centers, medians, color="black", linewidth=2, label="Median")
        ax.fill_between(centers, p25, p75, color="black", alpha=0.3, label="IQR (25%–75%)")
        ax.legend()

    abs_rots      = np.array(results["abs_rot"])
    norm_matches  = np.array(results["norm_matches"])
    inlier_ratios = np.array(results["inlier_ratio"])
    rot_errors    = np.array(results["rot_error"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(abs_rots, norm_matches,  alpha=0.3, color="blue",  s=10)
    axes[1].scatter(abs_rots, inlier_ratios, alpha=0.3, color="green", s=10)
    axes[2].scatter(abs_rots, rot_errors,    alpha=0.3, color="red",   s=10)

    _add_trendline(axes[0], abs_rots, norm_matches)
    _add_trendline(axes[1], abs_rots, inlier_ratios)
    _add_trendline(axes[2], abs_rots, rot_errors)

    axes[0].set_title(f"Normalised matches ({mode})")
    axes[1].set_title("Inlier percentage")
    axes[2].set_title("Rotation error (deg)")
    axes[2].set_ylim(0, 50)

    for ax in axes:
        ax.set_xlabel("Absolute rotation angle (deg)")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out = output_dir / f"matches_{mode}_{timestamp}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info("Robustness plot saved → %s", out)


def build_results_table(results: dict) -> str:
    """Return a formatted per-rotation-bin summary string."""
    errors   = np.array(results["rot_error"])
    abs_rots = np.array(results["abs_rot"])

    bins = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90), (90, 180)]
    header = f"\n{'Rot bin':<12} | {'Count':<6} | {'Med. err':<10} | {'Acc@5°':<8} | {'Acc@10°':<8} | {'AUC@20°':<8}"
    rows = [header, "-" * 75]

    for start, end in bins:
        mask = (abs_rots >= start) & (abs_rots < end)
        bin_errs = errors[mask]
        if len(bin_errs) == 0:
            continue
        label  = f"{start}-{end}°" if not (start == 0 and end == 180) else "TOTAL"
        med    = np.median(bin_errs)
        acc5   = np.mean(bin_errs < 5) * 100
        acc10  = np.mean(bin_errs < 10) * 100
        auc20  = float(np.mean([np.mean(bin_errs < t) for t in range(1, 21)]))
        rows.append(
            f"{label:<12} | {len(bin_errs):<6} | {med:>8.2f}° | {acc5:>7.1f}% | {acc10:>7.1f}% | {auc20:>8.3f}"
        )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class MatchesValidator:
    """
    Matches robustness evaluator for REALM models on the VECtor benchmark.

    Responsibilities
    ----------------
    * Model initialisation via REALM_creator
    * Anchor-vs-all relative pose estimation with poselib RANSAC
    * Per-rotation-bin accuracy / AUC reporting
    * Optional match visualisations every N frames + MP4 stitching
    * Results persisted as .txt summary and .npz archive
    """

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
        self.mode: str           = args.mode
        self.target_shape: Optional[int] = eval_cfg.get("target_shape", args.target_shape)
        self.ransac_threshold: float     = eval_cfg.get("ransac_threshold", args.ransac_threshold)
        self.vis_stride: int             = eval_cfg.get("vis_stride", args.vis_stride)

        # AMP ------------------------------------------------------------------
        self.use_amp: bool = not args.fp32
        logger.info(
            "Using %s inference.",
            "mixed-precision (AMP)" if self.use_amp else "full-precision (FP32)",
        )

        # Camera alignment -----------------------------------------------------
        self.aligner = AlignData()
        self.anchor_type = "e" if self.mode == "ee" else "i"
        self.target_type = "i" if self.mode == "ii" else "e"

        # Output paths ---------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        seq_name  = Path(args.dataset_path).name
        run_tag   = f"matches_{self.model.model_type.value}_{seq_name}_{self.mode}_{timestamp}"
        self.output_dir = Path("results") / "matches" / run_tag
        self.img_dir    = self.output_dir / "imgs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp  = timestamp
        logger.info("Run output directory: %s", self.output_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self, dataset, anchor_dataset=None) -> dict:
        """
        Run the full matches evaluation loop.

        Args:
            dataset:        Primary VECtorDataset (event-mode for 'ee'/'ie',
                            image-mode for 'ii').
            anchor_dataset: Image-mode VECtorDataset used only for the anchor
                            frame in 'ie' mode.  Pass ``None`` for 'ee'/'ii'.

        Returns:
            Dict with keys 'abs_rot', 'norm_matches', 'inlier_ratio',
            'rot_error', 'failed'.
        """
        # Calibration for the target (query) view
        K, D, H, W = load_vector_calibration(dataset.calib_file)
        cam_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D.tolist()]
        camera_dict = {"model": "OPENCV", "width": W, "height": H, "params": cam_params}

        # Calibration for the anchor view (differs only in 'ie' mode)
        if self.mode == "ie" and anchor_dataset is not None:
            Ka, Da, Ha, Wa = load_vector_calibration(anchor_dataset.calib_file)
            cam_params_anchor = [Ka[0, 0], Ka[1, 1], Ka[0, 2], Ka[1, 2], *Da.tolist()]
            camera_dict_anchor = {"model": "OPENCV", "width": Wa, "height": Ha, "params": cam_params_anchor}
        else:
            camera_dict_anchor = camera_dict

        poses_gt = self._load_poses(dataset.dataset_path)

        # Anchor ---------------------------------------------------------------
        # In 'ie' mode the anchor is an RGB frame; use the image dataset for it.
        anchor_src = anchor_dataset if (self.mode == "ie" and anchor_dataset is not None) else dataset
        anchor_idx = len(anchor_src) // 2
        data_anchor, t_anchor = self._prepare_input(anchor_src, anchor_idx, anchor=True)
        _, r_anchor_mocap = interpolate_pose(poses_gt, t_anchor)
        r_anchor_cam = self.aligner(r_anchor_mocap, self.anchor_type)

        results: dict = {
            "abs_rot": [], "norm_matches": [], "inlier_ratio": [], "rot_error": [], "failed": 0
        }

        autocast_ctx = (
            torch.amp.autocast("cuda")
            if self.use_amp
            else torch.amp.autocast("cuda", enabled=False)
        )

        max_kps = 1
        failed  = 0

        with autocast_ctx:
            for counter, (t, data_raw) in enumerate(
                tqdm(dataset, desc=f"Matches [{self.mode}]", unit="frame")
            ):
                if data_raw is None:
                    continue

                t_target = float(t)
                _, r_target_mocap = interpolate_pose(poses_gt, t_target)
                r_target_cam = self.aligner(r_target_mocap, self.target_type)
                rel_rot_gt   = r_target_cam.inv() * r_anchor_cam
                abs_theta    = float(np.linalg.norm(rel_rot_gt.as_rotvec(degrees=True)))

                data_target = self._prepare_raw(data_raw, anchor=False)

                kpts1_vis, kpts2_vis, kpts1, kpts2, inliers_mask, num_inliers, error_deg, ransac_crashed = \
                    self._match_and_solve(
                        data_anchor, data_target, camera_dict_anchor, camera_dict, rel_rot_gt
                    )

                # Mirror the original three-way failed counting:
                #   1) len(matches) < 5        → else branch (no RANSAC attempted)
                #   2) cv2.error during RANSAC → ransac_crashed flag
                #   3) num_inliers < 5         → checked after RANSAC
                # Cases 2 and 3 both fire for the same frame when RANSAC crashes
                # (num_inliers stays 0), matching the original script's behaviour.
                if len(kpts1) < 5:
                    failed += 1
                else:
                    if ransac_crashed:
                        failed += 1
                    if num_inliers < 5:
                        failed += 1

                # Periodic visualisation — every vis_stride frames (unconditional,
                # matching the original script which always wrote every 10th frame).
                # Use vis keypoints (target_shape space) not the RANSAC-rescaled ones,
                # matching the original which passes raw model output to vis_matches.
                if counter % self.vis_stride == 0:
                    self._save_match_vis(
                        data_anchor, data_target, kpts1_vis, kpts2_vis, inliers_mask, t_target
                    )

                if len(kpts1) > max_kps:
                    max_kps = len(kpts1)

                results["abs_rot"].append(abs_theta)
                results["norm_matches"].append(num_inliers)
                results["inlier_ratio"].append(num_inliers / max(len(kpts1), 1))
                results["rot_error"].append(error_deg)

        results["failed"]       = failed
        results["norm_matches"] = (np.array(results["norm_matches"]) / max_kps).tolist()

        logger.warning("Failed frames: %d / %d", failed, len(results["rot_error"]))

        self._save_results(results)

        if self.args.save_vis:
            self._create_video()

        return results

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_input(self, dataset, idx: int, anchor: bool) -> tuple[torch.Tensor, float]:
        """Load and pre-process a single frame from the dataset."""
        t, data_raw = dataset[idx]
        return self._prepare_raw(data_raw, anchor=anchor).unsqueeze(0).to(self.device), float(t)

    def _prepare_raw(self, data_raw: torch.Tensor, anchor: bool) -> torch.Tensor:
        """Resize (if needed) and move to the correct shape."""
        if self.target_shape is not None:
            data_raw = Resize(
                (self.target_shape, self.target_shape), keep_aspect_ratio=True
            )(data_raw)
        return data_raw

    # ------------------------------------------------------------------
    # Matching + pose recovery
    # ------------------------------------------------------------------

    def _match_and_solve(
        self,
        data_anchor: torch.Tensor,
        data_target: torch.Tensor,
        camera_dict_1: dict,
        camera_dict_2: dict,
        rel_rot_gt: R,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int, float, bool]:
        """
        Run the REALM siamese forward pass and recover relative pose.

        Returns:
            (kpts1, kpts2, inliers_mask, num_inliers, rotation_error_deg, ransac_crashed)

        ``ransac_crashed`` is True when poselib raises a cv2.error so the caller
        can replicate the original script's failed-frame counting exactly.

        Returns an 8-tuple:
            (kpts1_vis, kpts2_vis,          ← raw model output, target_shape space
             kpts1_ransac, kpts2_ransac,    ← rescaled to original camera resolution
             inliers_mask, num_inliers, rotation_error_deg, ransac_crashed)
        """
        view2 = data_target.unsqueeze(0).to(self.device)
        res = self.model({"view1": data_anchor, "view2": view2})

        res1, res2 = res
        desc1 = res1['desc'].squeeze(0).float()
        desc2 = res2['desc'].squeeze(0).float()

        with torch.amp.autocast("cuda", enabled=False):
            mkpts1, mkpts2 = matches(
                desc1,
                desc2,
                data_anchor.shape[-1],
                data_anchor.shape[-1],
                device=self.device
            )

        # Keep a copy in target_shape pixel space — used only for visualisation.
        # The original script passes raw model-output keypoints to vis_matches
        # while RANSAC uses the rescaled-to-original-resolution versions.
        mkpts1_vis = mkpts1.copy()
        mkpts2_vis = mkpts2.copy()

        if self.target_shape is not None:
            mkpts1 = rescale_matches(mkpts1, camera_dict_1["height"], camera_dict_1["width"],
                                     self.target_shape, self.target_shape)
            mkpts2 = rescale_matches(mkpts2, camera_dict_2["height"], camera_dict_2["width"],
                                     self.target_shape, self.target_shape)

        num_inliers = 0
        inliers_mask: Optional[np.ndarray] = None
        error_deg = 180.0
        ransac_crashed = False

        if len(mkpts1) >= 5:
            try:
                pose_result, info = poselib.estimate_relative_pose(
                    mkpts1.astype(np.float64),
                    mkpts2.astype(np.float64),
                    camera_dict_1,
                    camera_dict_2,
                    {"max_epipolar_error": self.ransac_threshold},
                )
                if pose_result is not None:
                    inliers_mask = info["inliers"]
                    num_inliers  = int(np.sum(inliers_mask))
                    # poselib quaternion: (w, x, y, z) → scipy: (x, y, z, w)
                    q = pose_result.q
                    est_rot = R.from_quat([q[1], q[2], q[3], q[0]])
                    error_deg = float(
                        np.linalg.norm((est_rot * rel_rot_gt.inv()).as_rotvec(degrees=True))
                    )
            except cv2.error:
                # Matches the original script's except cv2.error branch which
                # increments failed and leaves num_inliers at 0.
                ransac_crashed = True
                logger.debug("cv2.error during pose recovery.", exc_info=True)

        return mkpts1_vis, mkpts2_vis, mkpts1, mkpts2, inliers_mask, num_inliers, error_deg, ransac_crashed

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _save_match_vis(
        self,
        data_anchor: torch.Tensor,
        data_target: torch.Tensor,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        inliers_mask: Optional[np.ndarray],
        t_target: float,
    ) -> None:
        """Save a side-by-side match visualisation for this frame."""
        # Convert event voxel grids to an RGB image before rendering.
        # voxel_to_rgb_image returns a float tensor in [0, 1]; _tensor_to_bgr
        # handles the [0,1]→[0,255] scaling, matching the original script's
        #   img = image_generator(data) * 255  →  cv2.cvtColor(..., COLOR_RGB2BGR)
        if self.mode[0] == "e":  # Anchor is event-mode
            data_anchor = voxel_to_rgb_image(data_anchor).unsqueeze(0)
        if self.mode[1] == "e":  # Target is event-mode
            data_target = voxel_to_rgb_image(data_target)

        img1 = self._tensor_to_bgr(data_anchor.squeeze(0))
        img2 = self._tensor_to_bgr(data_target)

        # Align heights if the two modalities differ in resolution
        if img1.shape[:2] != img2.shape[:2]:
            scale = img2.shape[0] / img1.shape[0]
            img1 = cv2.resize(img1, (int(img1.shape[1] * scale), img2.shape[0]))
            kpts1 = kpts1 * scale

        res_img = vis_matches(img1, img2, kpts1, kpts2, inliers_mask=inliers_mask, n_viz=20)
        save_path = self.img_dir / f"matches_{t_target:.3f}.png"
        cv2.imwrite(str(save_path), res_img)

    @staticmethod
    def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
        """Convert a (C, H, W) float tensor in [0, 1] or [0, 255] to a uint8 BGR image."""
        img = t.cpu().float()
        if img.max() <= 1.0:
            img = img * 255.0
        # check if needs to permute (i.e. has 3 channels) before converting to numpy
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        return cv2.cvtColor(img.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _create_video(
        self,
        fps: int = 10,
    ) -> None:
        """Stitch saved match PNGs into an MP4."""
        frames = sorted(self.img_dir.glob("matches_*.png"),
                        key=lambda p: float(p.stem.split("_")[1]))
        if not frames:
            logger.warning("No match images found — skipping video export.")
            return

        first = cv2.imread(str(frames[0]))
        if first is None:
            logger.error("Cannot read first frame: %s", frames[0])
            return

        h, w = first.shape[:2]
        video_path = self.output_dir / f"matches_video_{self.mode}_{self.timestamp}.mp4"

        for fourcc_tag in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            if writer.isOpened():
                logger.info("Using codec: %s", fourcc_tag)
                break
            writer.release()
        else:
            logger.error("Could not open a VideoWriter — skipping video export.")
            return

        for fp in tqdm(frames, desc="Encoding video"):
            frame = cv2.imread(str(fp))
            if frame is not None:
                writer.write(frame)
        writer.release()
        logger.info("Video ready: %s", video_path)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @staticmethod
    def _load_poses(sequence_path: Path) -> np.ndarray:
        """Load ground truth poses from the sequence directory."""
        pose_files = list(sequence_path.glob("*.gt.txt"))
        if not pose_files:
            raise FileNotFoundError(f"No *.gt.txt pose file found in {sequence_path}")
        return np.genfromtxt(pose_files[0])

    def _save_results(self, results: dict) -> None:
        """Print, log, and persist evaluation results."""
        summary = build_results_table(results)
        logger.info(summary)

        txt_path = self.output_dir / f"matches_summary_{self.mode}_{self.timestamp}.txt"
        try:
            txt_path.write_text(summary)
            logger.info("Summary saved → %s", txt_path)
        except OSError as exc:
            logger.warning("Could not write summary file: %s", exc)

        npz_path = self.output_dir / f"matches_results_{self.mode}_{self.timestamp}.npz"
        np.savez(npz_path, **{k: np.array(v) for k, v in results.items() if k != "failed"})
        logger.info("Raw results saved → %s", npz_path)

        plot_robustness_analysis(results, self.output_dir, self.timestamp, self.mode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REALM matches robustness evaluator (VECtor benchmark)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./realm/realm/configs/mast3r.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ie",
        choices=["ii", "ee", "ie"],
        help="Matching modality: ii=image-image, ee=event-event, ie=image-event.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device string, e.g. 'cuda', 'cuda:1', 'cpu'.",
    )
    parser.add_argument(
        "--target_shape",
        type=int,
        default=448,
        help="Spatial resolution fed to the model (square). Set to 0 to disable resizing.",
    )
    parser.add_argument(
        "--model_delta_t",
        type=float,
        default=33000.0,
        help="Event window duration (µs).",
    )
    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=1.0,
        help="Maximum epipolar error for poselib RANSAC (pixels).",
    )
    parser.add_argument(
        "--vis_stride",
        type=int,
        default=10,
        help="Save a match visualisation every N frames.",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save per-frame match visualisations and stitch an MP4.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable AMP (mixed precision) — useful for debugging.",
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

    # target_shape=0 is a CLI convention for "no resizing"
    if args.target_shape == 0:
        args.target_shape = None

    cfg = load_config(args.config)

    # Build dataset(s) — representation is driven by cfg[data][representation]
    primary_dataset, anchor_dataset = build_vector_datasets(
        dataset_path=cfg['data']['dataset_path'],
        mode=args.mode,
        ev_us=args.model_delta_t,
        cfg=cfg,
    )

    validator = MatchesValidator(args)
    validator.run(primary_dataset, anchor_dataset=anchor_dataset)