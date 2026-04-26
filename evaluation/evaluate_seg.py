"""
validate_segmentation.py — segmentation evaluator for REALM models.

Usage:
    python validate_segmentation.py --config configs/segmentation.yaml
    python validate_segmentation.py --config configs/segmentation.yaml --save_vis --device cuda:1
    python validate_segmentation.py --config configs/segmentation.yaml --save_vis --fp32
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tabulate import tabulate
from tqdm import tqdm

from dataloaders.dataset_builder import create_REALM_dataloader
from dataloaders.semantic_labels import labels_11_Cityscapes
from dataloaders.utils_dataloaders import img_undo_normalize, voxel_to_rgb_image
from metrics.metrics_seg import MetricsSemseg
from realm.model import ModelType
from realm.model_factory import REALM_creator
from realm.utils.log import get_logger

# ---------------------------------------------------------------------------
# Deterministic / performance flags
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False  # Keep False for reproducible tile sizes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = get_logger("validate_segmentation", level=logging.INFO)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

class SegSettings:
    """Derived segmentation settings extracted from the Cityscapes label spec."""

    def __init__(
        self,
        semseg_num_classes: int,
        semseg_ignore_label: int,
        semseg_class_names: List[str],
        semseg_color_map: np.ndarray,
    ) -> None:
        self.semseg_num_classes = semseg_num_classes
        self.semseg_ignore_label = semseg_ignore_label
        self.semseg_class_names = semseg_class_names
        self.semseg_color_map = semseg_color_map


def build_settings_from_labels() -> SegSettings:
    """Derive class names and color map from ``labels_11_Cityscapes``."""
    train_id_map: dict = {}
    for label in labels_11_Cityscapes:
        if label.trainId == 255:
            continue
        train_id_map.setdefault(label.trainId, label)

    sorted_ids = sorted(train_id_map.keys())
    class_names = [train_id_map[tid].name for tid in sorted_ids]

    color_map = np.zeros((256, 3), dtype=np.uint8)
    for tid in sorted_ids:
        color_map[tid] = train_id_map[tid].color

    return SegSettings(
        semseg_num_classes=len(sorted_ids),
        semseg_ignore_label=255,
        semseg_class_names=class_names,
        semseg_color_map=color_map,
    )


# ---------------------------------------------------------------------------
# Config helpers
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
# Tile builder
# ---------------------------------------------------------------------------

def build_tiles(
    h_in: int,
    w_in: int,
    tile_size: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Build corner-anchored tiles of ``tile_size × tile_size`` that fully cover
    an ``h_in × w_in`` input.  For the default 480×640 → 448 case this gives
    exactly 4 tiles with two-pixel overlap bands.

    Returns a list of ``(y_start, y_end, x_start, x_end)`` tuples.
    """
    if tile_size > h_in or tile_size > w_in:
        raise ValueError(
            f"tile_size ({tile_size}) must be <= input dimensions ({h_in}×{w_in})"
        )
    tiles = [
        (0,          tile_size, 0,          tile_size),
        (0,          tile_size, w_in - tile_size, w_in),
        (h_in - tile_size, h_in, 0,          tile_size),
        (h_in - tile_size, h_in, w_in - tile_size, w_in),
    ]
    return tiles


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SegmentationValidator:
    """
    End-to-end segmentation evaluator for REALM models.

    Responsibilities
    ----------------
    * Tiled inference over arbitrary-resolution inputs
    * Zero-event holdover for event-camera modalities
    * mIoU / per-class IoU accumulation
    * Optional per-frame PNG visualizations + MP4 stitching
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)

        # Config ---------------------------------------------------------------
        self.cfg = load_config(args.config)
        data_cfg: dict = self.cfg.get("data", {})
        tile_cfg: dict = self.cfg.get("tiling", {})

        # Model ----------------------------------------------------------------
        logger.info("Building model...")
        self.model = REALM_creator(self.cfg).to(self.device)

        self.model.eval()

        # Label / metrics settings ---------------------------------------------
        self.settings = build_settings_from_labels()
        self.metrics_evaluator = MetricsSemseg(
            num_classes=self.settings.semseg_num_classes,
            ignore_label=self.settings.semseg_ignore_label,
            class_names=self.settings.semseg_class_names,
        )

        # Tiling ---------------------------------------------------------------
        self.h_in: int = tile_cfg.get("input_height", 480)
        self.w_in: int = tile_cfg.get("input_width", 640)
        self.tile_size: int = tile_cfg.get("tile_size", 448)
        self.tiles = build_tiles(self.h_in, self.w_in, self.tile_size)
        logger.info(
            "Tile strategy: %d tiles of %d×%d over %d×%d input",
            len(self.tiles), self.tile_size, self.tile_size, self.h_in, self.w_in,
        )

        # AMP ------------------------------------------------------------------
        self.use_amp: bool = not args.fp32
        if self.use_amp:
            logger.info("Using mixed-precision (AMP) inference.")
        else:
            logger.info("Using full-precision (FP32) inference.")

        # Dataloader -----------------------------------------------------------
        self.dataloader = create_REALM_dataloader(
            config=data_cfg.get("datasets_config", {}),
            mode="test",
            batch_size=data_cfg.get("batch_size", 32),
            num_workers=data_cfg.get("num_workers", 4),
            prefetch_factor=data_cfg.get("prefetch_factor", 2)
        )

        # Visualisation --------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_tag = f"seg_{self.model.model_type.value}_{timestamp}"
        self.output_dir = Path("results") / "segmentation" / run_tag
        self.results_txt_path = self.output_dir / "results.txt"
        self.vis_enabled = args.save_vis
        if self.vis_enabled:
            self.output_dir = self.output_dir / "frames"
            logger.info("Visualizations will be saved to: %s", self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self) -> dict:
        """Run the full evaluation loop and return the metrics summary dict."""
        self.metrics_evaluator.reset()
        pbar = tqdm(self.dataloader, desc="Evaluating", unit="batch")

        last_valid_mask: Optional[np.ndarray] = None
        total_saved = 0

        for batch in pbar:
            voxels, images, labels = (
                batch[0].to(self.device, non_blocking=True),
                batch[1].to(self.device, non_blocking=True),
                batch[2].to(self.device, non_blocking=True).long(),
            )
            B, _C, H, W = images.shape

            pred_masks_np, last_valid_mask = self._infer_batch(
                voxels, images, B, H, W, last_valid_mask
            )

            pred_tensor = torch.from_numpy(pred_masks_np)
            self.metrics_evaluator.update_batch(pred_tensor, labels.cpu())

            if self.vis_enabled:
                for i in range(B):
                    ev_tensor = voxels[i] if self.model.model_type == ModelType.Events else None
                    self._save_visualization(
                        images[i], ev_tensor, labels[i].cpu().numpy(),
                        pred_masks_np[i], total_saved,
                    )
                    total_saved += 1

            # Free GPU memory eagerly
            del voxels, images, labels
            torch.cuda.empty_cache()

        metrics = self._print_results()

        if self.vis_enabled:
            self._create_video()

        return metrics

    # ------------------------------------------------------------------
    # Tiled inference
    # ------------------------------------------------------------------

    def _infer_batch(
        self,
        voxels: torch.Tensor,
        images: torch.Tensor,
        B: int,
        H: int,
        W: int,
        last_valid_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run tiled forward pass and apply zero-event holdover.

        Returns ``(pred_masks_np, updated_last_valid_mask)``.
        """
        accum = torch.zeros(
            (B, self.settings.semseg_num_classes, H, W), device=self.device
        )
        count = torch.zeros((B, 1, H, W), device=self.device)

        autocast_ctx = torch.amp.autocast("cuda") if self.use_amp else torch.amp.autocast("cuda", enabled=False)

        with autocast_ctx:
            for y_s, y_e, x_s, x_e in self.tiles:
                tile_in = (
                    voxels[:, :, y_s:y_e, x_s:x_e]
                    if self.model.model_type == ModelType.Events
                    else images[:, :, y_s:y_e, x_s:x_e]
                )
                tile_opts = {
                    "B": B,
                    "upsample": True,
                    "H": self.tile_size,
                    "W": self.tile_size,
                }
                tile_out = self.model(tile_in, tile_opts)
                accum[:, :, y_s:y_e, x_s:x_e] += tile_out
                count[:, :, y_s:y_e, x_s:x_e] += 1.0
                del tile_out

            pred_masks = torch.argmax(accum / count, dim=1)

        del accum, count

        # Apply zero-event holdover on CPU to avoid blocking the GPU pipeline.
        pred_masks_np = pred_masks.cpu().numpy()
        voxels_np = voxels.cpu().numpy()

        for i in range(B):
            if (
                self.model.model_type == ModelType.Events
                and np.abs(voxels_np[i]).sum() == 0
                and last_valid_mask is not None
            ):
                pred_masks_np[i] = last_valid_mask.copy()
            else:
                last_valid_mask = pred_masks_np[i].copy()

        return pred_masks_np, last_valid_mask

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _decode_seg_map(self, label_mask: np.ndarray) -> np.ndarray:
        """Map label IDs → RGB colors.  Void (255) is rendered black."""
        safe = label_mask.copy()
        safe[safe == 255] = 0
        rgb = self.settings.semseg_color_map[safe].copy()
        rgb[label_mask == 255] = 0
        return rgb.astype(np.uint8)

    def _save_visualization(
        self,
        image_tensor: torch.Tensor,
        ev_tensor: Optional[torch.Tensor],
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        idx: int,
    ) -> None:
        """Save a side-by-side comparison: RGB [| Events] | GT | Prediction."""
        assert self.output_dir is not None

        rgb = (
            (img_undo_normalize(image_tensor) * 255)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        gt_vis = self._decode_seg_map(gt_mask)
        pred_vis = self._decode_seg_map(pred_mask)

        panels = [rgb]
        if ev_tensor is not None:
            ev_img = voxel_to_rgb_image(ev_tensor.cpu(), 1).numpy()
            if ev_img.dtype != np.uint8:
                ev_img = (ev_img * 255 if ev_img.max() <= 1.0 else ev_img).astype(np.uint8)
            panels.append(ev_img)
        panels += [gt_vis, pred_vis]

        combined_bgr = cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR)
        save_path = self.output_dir / f"eval_{idx:05d}.png"
        cv2.imwrite(str(save_path), combined_bgr)

    def _create_video(
        self,
        output_name: str = "validation_video_seg.mp4",
        fps: int = 10,
    ) -> None:
        """Stitch saved PNG frames into an MP4 using OpenCV."""
        assert self.output_dir is not None
        video_path = self.output_dir.parent / output_name
        logger.info("Stitching video → %s", video_path)

        frames = sorted(self.output_dir.glob("*.png"))
        if not frames:
            logger.warning("No frames found — skipping video export.")
            return

        first = cv2.imread(str(frames[0]))
        if first is None:
            logger.error("Cannot read first frame: %s", frames[0])
            return

        h, w = first.shape[:2]

        # Prefer H.264 (avc1); fall back to mp4v if unavailable.
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

        for frame_path in tqdm(frames, desc="Encoding video"):
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                writer.write(frame)
        writer.release()
        logger.info("Video ready: %s", video_path)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _print_results(self) -> dict:
        metrics = self.metrics_evaluator.get_metrics_summary()

        global_rows = [
            ["mIoU",     f"{metrics['mean_iou']:.4f}"],
            ["Accuracy", f"{metrics['acc']:.4f}"],
        ]
        per_class_rows = [
            [name, f"{metrics.get(name, 0.0):.4f}"]
            for name in self.settings.semseg_class_names
        ]

        sep = "=" * 50
        output = f"\n{sep}\n   FINAL SEGMENTATION RESULTS\n{sep}"
        output += "\n--- Global Metrics ---\n"
        output += tabulate(global_rows, headers=["Metric", "Value"], tablefmt="grid")
        output += "\n\n--- Per-class IoU ---\n"
        output += tabulate(per_class_rows, headers=["Class", "IoU"], tablefmt="simple")
        output += f"\n{sep}"
        logger.info(output)

        try:
            with self.results_txt_path.open("a") as fh:
                fh.write(output + "\n")
            logger.info("Results appended to %s", self.results_txt_path)
        except OSError as exc:
            logger.warning("Could not write results file: %s", exc)

        return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REALM segmentation evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./realm/realm/configs/segmentation.yaml",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = get_logger("validate_segmentation", level=args.log_level)

    validator = SegmentationValidator(args)
    validator.run()