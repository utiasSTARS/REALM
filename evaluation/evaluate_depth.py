"""
validate_depth.py — depth evaluator for REALM models.

Usage:
    python validate_depth.py --config configs/depth.yaml
    python validate_depth.py --config configs/depth.yaml --save_vis --device cuda:1
    python validate_depth.py --config configs/depth.yaml --save_vis --fp32
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset_builder import REALM_Dataset
from dataloaders.utils_dataloaders import img_undo_normalize, voxel_to_rgb_image
from metrics.metrics_depth import MetricsDepth
from realm.model import ModelType
from realm.model_factory import REALM_creator
from realm.utils.log import get_logger

# ---------------------------------------------------------------------------
# Deterministic / performance flags
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = get_logger("validate_depth", level=logging.INFO)


# ---------------------------------------------------------------------------
# Depth visualisation helper
# ---------------------------------------------------------------------------

def vis_depth(depth_map, cmap_name: str = "inferno") -> np.ndarray:
    """
    Visualise a depth map with logarithmic scaling and robust percentile normalisation.

    Args:
        depth_map: (H, W) or (1, H, W) numpy array or torch.Tensor.
        cmap_name: Matplotlib colormap name.

    Returns:
        (H, W, 3) uint8 RGB image.  Invalid pixels are rendered black.
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(0)

    valid = (depth_map > 1e-6) & np.isfinite(depth_map)
    if valid.sum() == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)

    log_depth = np.zeros_like(depth_map)
    log_depth[valid] = np.log(depth_map[valid])

    log_min = np.percentile(log_depth[valid], 2)
    log_max = np.percentile(log_depth[valid], 95)

    log_clipped = np.clip(log_depth, log_min, log_max)

    norm = np.zeros_like(depth_map)
    if log_max > log_min:
        norm[valid] = (log_clipped[valid] - log_min) / (log_max - log_min)
    else:
        norm[valid] = 0.5

    colored = plt.get_cmap(cmap_name)(norm)[:, :, :3]
    colored[~valid] = 0.0
    return (colored * 255).astype(np.uint8)


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
# Validator
# ---------------------------------------------------------------------------

class DepthValidator:
    """
    End-to-end depth evaluator for REALM models.

    Responsibilities
    ----------------
    * Per-sequence inference over a list of dataset folders
    * Zero-event holdover for event-camera modalities: when a voxel frame
      contains no events (all-zero), the last valid model prediction is
      reused rather than passing a blank frame through inference.
    * Abs Rel / RMSE / delta metrics accumulation
    * Optional per-frame PNG visualizations + MP4 stitching
    * Results appended to a timestamped text file
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)

        # Config ---------------------------------------------------------------
        self.cfg = load_config(args.config)
        data_cfg: dict = self.cfg.get("data", {})

        # Model ----------------------------------------------------------------
        logger.info("Building model...")
        self.model = REALM_creator(self.cfg).to(self.device)
        self.model.eval()

        # Metrics --------------------------------------------------------------
        self.metrics_evaluator = MetricsDepth(
            min_depth=2,
            max_depth=80,
            ranges=[10, 20, 30],
        )

        # AMP ------------------------------------------------------------------
        self.use_amp: bool = not args.fp32
        logger.info(
            "Using %s inference.",
            "mixed-precision (AMP)" if self.use_amp else "full-precision (FP32)",
        )

        # Datasets (one DataLoader built per sequence inside _run_sequence) ----
        self.datasets = REALM_Dataset(data_cfg.get("datasets_config", {}), mode="test")
        self.batch_size: int = data_cfg.get("batch_size", 64)
        self.num_workers: int = data_cfg.get("num_workers", 8)
        self.prefetch_factor: int = data_cfg.get("prefetch_factor", 2)

        # Output paths ---------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_tag = f"depth_{self.model.model_type.value}_{timestamp}"
        self.output_dir = Path("results") / "depth" / run_tag
        self.results_txt_path = self.output_dir / "results.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run output directory: %s", self.output_dir)

        # Visualisation base dir (per-sequence sub-dirs created in _run_sequence)
        self.vis_base: Optional[Path] = (
            self.output_dir / "imgs" if args.save_vis else None
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, sequences: list[str]) -> None:
        """Evaluate over a list of sequence folder names."""
        all_metrics: list[dict] = []
        for seq in sequences:
            logger.info("=== Evaluating sequence: %s ===", seq)
            metrics = self._run_sequence(Path(seq))
            if metrics is not None:
                all_metrics.append(metrics)

        if len(all_metrics) > 1:
            self._print_overall_results(all_metrics)

    # ------------------------------------------------------------------
    # Per-sequence loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_sequence(self, folder: Path) -> Optional[dict]:
        loader = self._build_loader(folder)
        if loader is None:
            return

        vis_dir: Optional[Path] = None
        if self.vis_base is not None:
            vis_dir = self.vis_base / folder.name / "imgs"
            vis_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Visualisation enabled → %s", vis_dir)

        self.metrics_evaluator.reset()

        is_event_model = self.model.model_type == ModelType.Events

        autocast_ctx = (
            torch.amp.autocast("cuda")
            if self.use_amp
            else torch.amp.autocast("cuda", enabled=False)
        )

        # Padding constants (replicate-pad events from 260×346 → 448×448) -----
        H_in, W_in = 260, 346
        target_size = 448
        pad_h = (target_size - H_in) // 2
        pad_w = (target_size - W_in) // 2
        # Format: (left, right, top, bottom)
        padding = (pad_w, target_size - W_in - pad_w, pad_h, target_size - H_in - pad_h)
        y_start, y_end = padding[2], target_size - padding[3]
        x_start, x_end = padding[0], target_size - padding[1]

        # Temporal holdover state — reset per sequence so sequences are independent
        last_valid_prediction: Optional[np.ndarray] = None
        count_invalid = 0
        count_holdover = 0
        total_saved = 0

        pbar = tqdm(loader, desc=f"Evaluating {folder.name}", unit="batch")

        for voxels, images, depths in pbar:
            voxels = voxels.to(self.device, non_blocking=True)
            images = images.to(self.device, non_blocking=True)
            depths = depths.to(self.device, non_blocking=True)

            with autocast_ctx:
                if is_event_model:
                    inp = torch.nn.functional.pad(voxels, padding, mode="replicate")
                else:
                    inp = images
                pred_padded = self.model(inp, {"upsample": True, "H": target_size, "W": target_size})

            # Crop back to original spatial resolution
            pred = pred_padded[:, :, y_start:y_end, x_start:x_end]
            images_cropped = images[:, :, y_start:y_end, x_start:x_end]

            pred_np = pred.detach().float().cpu().numpy()
            gt_np   = depths.detach().float().cpu().numpy()
            vox_np  = voxels.detach().cpu().numpy()  # kept on CPU for holdover check

            if pred_np.ndim == 4:
                pred_np = pred_np.squeeze(1)
            if gt_np.ndim == 4:
                gt_np = gt_np.squeeze(1)

            B = pred_np.shape[0]
            metric_preds = np.zeros_like(pred_np)
            metric_gts   = np.zeros_like(gt_np)
            valid_count  = 0

            for i in range(B):
                p = pred_np[i]
                t = gt_np[i]
                invalid_mask = (t <= 0) | (t < self.metrics_evaluator.min_depth) | (t > self.metrics_evaluator.max_depth)

                # Skip frames where every GT pixel is invalid
                if invalid_mask.all():
                    count_invalid += 1
                    continue

                # ----------------------------------------------------------
                # Temporal holdover for event-based models:
                # If the voxel grid is all-zero (no events fired during this
                # time window), reuse the last valid model prediction instead
                # of using the uninformative zero-voxel output.  This mirrors
                # the physical intuition that a static scene produces no events
                # and the depth estimate should not change.
                # ----------------------------------------------------------
                if is_event_model and np.abs(vox_np[i]).sum() == 0:
                    if last_valid_prediction is not None:
                        p = last_valid_prediction.copy()
                        count_holdover += 1
                        logger.debug(
                            "Zero-event frame at sample %d — applying holdover prediction.",
                            i,
                        )
                    else:
                        # No prior prediction available yet; skip this sample
                        # to avoid polluting metrics with an uninformative output.
                        logger.debug(
                            "Zero-event frame at sample %d with no prior prediction — skipping.",
                            i,
                        )
                        count_invalid += 1
                        continue
                else:
                    # Update the holdover buffer with this frame's prediction
                    last_valid_prediction = p.copy()

                # Clip to the valid evaluation range
                t = np.clip(t, self.metrics_evaluator.min_depth, self.metrics_evaluator.max_depth)
                p = np.clip(p, self.metrics_evaluator.min_depth, self.metrics_evaluator.max_depth)

                # Re-zero out originally-invalid GT pixels after clipping
                t[invalid_mask] = 0.0

                if vis_dir is not None:
                    ev = voxels[i] if is_event_model else None
                    self._save_visualization(
                        images_cropped[i], ev, t, p, total_saved, vis_dir
                    )
                    total_saved += 1

                # Build metric arrays (prediction also zeroed at invalid pixels)
                p_metric = p.copy()
                p_metric[invalid_mask] = 0.0
                metric_gts[i]   = t
                metric_preds[i] = p_metric
                valid_count += 1

            self.metrics_evaluator.update_batch(metric_preds, metric_gts, valid_count=valid_count)

            del voxels, images, depths, pred, pred_padded
            torch.cuda.empty_cache()

        # End-of-sequence reporting
        if count_invalid > 0:
            logger.warning(
                "Skipped %d sample(s) with all-invalid GT depth or no prior holdover.",
                count_invalid,
            )
        if is_event_model and count_holdover > 0:
            logger.info(
                "Applied temporal holdover on %d zero-event frame(s).", count_holdover
            )

        self._print_results(folder.name)

        if vis_dir is not None:
            self._create_video(vis_dir)

        return self.metrics_evaluator.get_metrics_summary()  

    # ------------------------------------------------------------------
    # DataLoader builder
    # ------------------------------------------------------------------

    def _build_loader(self, folder: Path) -> Optional[DataLoader]:
        """Find the dataset matching *folder* and wrap it in a DataLoader."""
        dataset = next(
            (dt for dt in self.datasets.get_datasets() if folder.name in str(dt.h5_path)),
            None,
        )
        if dataset is None:
            logger.error("No dataset found for sequence: %s", folder)
            return None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=(self.num_workers > 0),
        )

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _save_visualization(
        self,
        image_tensor: torch.Tensor,
        ev_tensor: Optional[torch.Tensor],
        gt_depth: np.ndarray,
        pred_depth: np.ndarray,
        idx: int,
        vis_dir: Path,
    ) -> None:
        """Save a side-by-side panel: RGB [| Events] | GT depth | Predicted depth."""
        rgb      = img_undo_normalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
        gt_vis   = vis_depth(gt_depth,   "magma") / 255.0
        pred_vis = vis_depth(pred_depth, "magma") / 255.0

        panels = [rgb]
        if ev_tensor is not None:
            ev_img = voxel_to_rgb_image(ev_tensor.cpu(), 1).numpy()
            if ev_img.max() > 1.0:
                ev_img = ev_img / 255.0
            panels.append(ev_img)
        panels += [gt_vis, pred_vis]

        save_path = vis_dir / f"eval_{idx:05d}.png"
        plt.imsave(str(save_path), np.hstack(panels))

    def _create_video(
        self,
        vis_dir: Path,
        output_name: str = "validation_video_depth.mp4",
        fps: int = 10,
    ) -> None:
        """Stitch saved PNG frames into an MP4, trying avc1 then mp4v codec."""
        video_path = vis_dir.parent / output_name
        logger.info("Stitching video → %s", video_path)

        frames = sorted(vis_dir.glob("*.png"))
        if not frames:
            logger.warning("No frames found — skipping video export.")
            return

        first = cv2.imread(str(frames[0]))
        if first is None:
            logger.error("Cannot read first frame: %s", frames[0])
            return

        h, w = first.shape[:2]

        writer: Optional[cv2.VideoWriter] = None
        for fourcc_tag in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
            candidate = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            if candidate.isOpened():
                writer = candidate
                logger.info("Using codec: %s", fourcc_tag)
                break
            candidate.release()

        if writer is None:
            logger.error("Could not open a VideoWriter — skipping video export.")
            return

        for frame_path in tqdm(frames, desc="Encoding video"):
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                writer.write(frame)
        writer.release()
        logger.info("Video ready: %s", video_path)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _print_results(self, seq_name: str) -> None:
        metrics = self.metrics_evaluator.get_metrics_summary()

        table_data = [
            ["Abs Rel",      f"{metrics['abs_rel']:.4f}"],
            ["Sq Rel",       f"{metrics['sq_rel']:.4f}"],
            ["RMSE",         f"{metrics['rmse']:.4f}"],
            ["RMSE log",     f"{metrics['rmse_log']:.4f}"],
            ["SILog",        f"{metrics['silog']:.4f}"],
            ["d1 (<1.25)",   f"{metrics['a1']:.4f}"],
            ["d2 (<1.25²)",  f"{metrics['a2']:.4f}"],
            ["d3 (<1.25³)",  f"{metrics['a3']:.4f}"],
            ["Err 10m",      f"{metrics.get('err_10m', 0.0):.4f}"],
            ["Err 20m",      f"{metrics.get('err_20m', 0.0):.4f}"],
            ["Err 30m",      f"{metrics.get('err_30m', 0.0):.4f}"],
        ]

        sep = "=" * 50
        output = "\n".join([
            f"\n{sep}",
            f"   DEPTH RESULTS — {seq_name}",
            sep,
            tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"),
            sep,
        ])

        logger.info(output)

        try:
            with self.results_txt_path.open("a") as fh:
                fh.write(output + "\n")
            logger.info("Results appended to %s", self.results_txt_path)
        except OSError as exc:
            logger.warning("Could not write results file: %s", exc)

    def _print_overall_results(self, all_metrics: list[dict]) -> None:
        """Compute and report mean metrics across all evaluated sequences."""
        if not all_metrics:
            return

        # Simple mean across sequences (each sequence weighted equally)
        keys = all_metrics[0].keys()
        mean_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}

        n = len(all_metrics)
        table_data = [
            ["Abs Rel",      f"{mean_metrics['abs_rel']:.4f}"],
            ["Sq Rel",       f"{mean_metrics['sq_rel']:.4f}"],
            ["RMSE",         f"{mean_metrics['rmse']:.4f}"],
            ["RMSE log",     f"{mean_metrics['rmse_log']:.4f}"],
            ["SILog",        f"{mean_metrics['silog']:.4f}"],
            ["d1 (<1.25)",   f"{mean_metrics['a1']:.4f}"],
            ["d2 (<1.25²)",  f"{mean_metrics['a2']:.4f}"],
            ["d3 (<1.25³)",  f"{mean_metrics['a3']:.4f}"],
            ["Err 10m",      f"{mean_metrics.get('err_10m', 0.0):.4f}"],
            ["Err 20m",      f"{mean_metrics.get('err_20m', 0.0):.4f}"],
            ["Err 30m",      f"{mean_metrics.get('err_30m', 0.0):.4f}"],
        ]

        sep = "=" * 50
        output = "\n".join([
            f"\n{sep}",
            f"   OVERALL RESULTS — mean across {n} sequence(s)",
            sep,
            tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"),
            sep,
        ])

        logger.info(output)

        try:
            with self.results_txt_path.open("a") as fh:
                fh.write(output + "\n")
            logger.info("Overall results appended to %s", self.results_txt_path)
        except OSError as exc:
            logger.warning("Could not write overall results: %s", exc)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REALM depth evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./realm/realm/configs/depth.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device string, e.g. 'cuda', 'cuda:1', 'cpu'.",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save per-frame PNG visualisations and stitch an MP4.",
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
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=[
            "outdoor_driving_day1",
            "outdoor_driving_night1",
            "outdoor_driving_night2",
            "outdoor_driving_night3",
        ],
        help="Sequence folder names to evaluate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = get_logger("validate_depth", level=args.log_level)


    validator = DepthValidator(args)
    validator.run(args.sequences)