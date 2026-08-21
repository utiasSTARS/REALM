"""
image_reconstruction.py — qualitative reconstruction test for the REALM reconstruction head.

Loads a REALM model (frozen encoder + trained reconstruction head, see
``realm.heads.image_recon_head.ImageReconHead``) and runs reconstruction on
VECtor event or RGB data.  Saves side-by-side panels:
    Events  model: [event preview  | reconstructed RGB]
    RGB     model: [input RGB      | reconstructed RGB]
    Hybrid  model: dispatched by channel count (same panels as above)

Usage:
    python evaluation/image_reconstruction.py --config realm/realm/configs/image_reconstruction_rgb.yaml --dataset_path datasets/VECtor/robot-fast
    python evaluation/image_reconstruction.py --config realm/realm/configs/image_reconstruction_rgb.yaml --dataset_path datasets/VECtor/robot-fast --device cuda:1 --fp32
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataloaders.utils_dataloaders import img_undo_normalize, voxel_to_rgb_image
from dataloaders.vector_dataset import build_vector_datasets
from realm.model import ModelType
from realm.model_factory import REALM_creator
from realm.utils.log import get_logger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

logger = get_logger("image_reconstruction", level=logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as fh:
        return yaml.safe_load(fh)


def center_crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop a (C, H, W) tensor to (C, size, size)."""
    _, h, w = tensor.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return tensor[:, y0:y0 + size, x0:x0 + size]


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] → (H, W, 3) uint8."""
    return (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DecoderVectorValidator:
    """
    Run a REALM reconstruction model on VECtor data and save reconstruction panels.

    Panel layout
    ------------
    Events model : event RGB preview  | reconstructed RGB
    RGB model    : input RGB (undone) | reconstructed RGB
    """

    CROP: int = 448

    def __init__(self, args: argparse.Namespace) -> None:
        self.args   = args
        self.device = torch.device(args.device)
        self.cfg    = load_config(args.config)
        self.cfg.setdefault("data", {})["dataset_path"] = args.dataset_path

        # REALM model (frozen encoder + trained reconstruction head) -----------
        logger.info("Loading model...")
        self.model = REALM_creator(self.cfg).to(self.device).eval()

        self.use_amp: bool = not args.fp32
        logger.info("Using %s inference.", "AMP" if self.use_amp else "FP32")

        # Output dirs ----------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        seq_name  = Path(args.dataset_path).name
        self.vis_dir = Path("results") / "image_reconstruction" / f"{seq_name}_{timestamp}" / "imgs"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving frames → %s", self.vis_dir)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self) -> None:
        is_event_model = self.model.model_type in (ModelType.Events, ModelType.Hybrid)

        # VECtor dataset returns (timestamp, voxel) for events,
        # (timestamp, voxel, image) for ee+rgb modes.
        mode     = "ee" if is_event_model else "ii"
        datasets = build_vector_datasets(
            dataset_path=self.args.dataset_path,
            mode=mode,
            ev_us=self.args.model_delta_t,
            cfg=self.cfg,
        )
        dataset = datasets[0]   # primary modality

        autocast_ctx = (
            torch.amp.autocast("cuda")
            if self.use_amp
            else torch.amp.autocast("cuda", enabled=False)
        )

        last_valid_recon: Optional[np.ndarray] = None
        last_valid_input: Optional[np.ndarray] = None
        total_saved = 0

        for item in tqdm(dataset, desc="Reconstructing", unit="frame"):
            timestamp  = item[0]
            raw_tensor = item[1]   # voxel for events, RGB tensor for images

            if raw_tensor is None:
                continue
            if isinstance(raw_tensor, np.ndarray):
                raw_tensor = torch.from_numpy(raw_tensor).float()

            if is_event_model:
                raw_tensor = center_crop(raw_tensor, self.CROP)
            else:
                raw_tensor = torch.nn.functional.interpolate(
                    raw_tensor.unsqueeze(0), size=(self.CROP, self.CROP),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)

            # Zero-event holdover for event models
            if is_event_model and np.abs(raw_tensor.numpy()).sum() == 0:
                if last_valid_recon is None:
                    continue
                panel = np.hstack([last_valid_input, last_valid_recon])
                self._save(panel, total_saved, float(timestamp))
                total_saved += 1
                continue

            inp = raw_tensor.unsqueeze(0).to(self.device)   # (1, C, H, W)

            with autocast_ctx:
                pred = self.model(inp)  # (1, out_channels, H, W)

            # Build the left panel (input preview)
            if is_event_model:
                left = voxel_to_rgb_image(raw_tensor.cpu(), 1.0).numpy()  # (H, W, 3) float
                if left.max() > 1.0:
                    left = left / 255.0
            else:
                left = to_uint8(img_undo_normalize(raw_tensor.cpu())) / 255.0  # (H, W, 3) float

            right = to_uint8(img_undo_normalize(pred[0].cpu())) / 255.0        # (H, W, 3) float

            last_valid_input = left.copy()
            last_valid_recon = right.copy()

            panel = np.hstack([left, right])
            self._save(panel, total_saved, float(timestamp))
            total_saved += 1

        logger.info("Done. %d frames saved.", total_saved)
        self._create_video()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _save(self, panel: np.ndarray, idx: int, timestamp: float) -> None:
        import matplotlib.pyplot as plt
        path = self.vis_dir / f"recon_{idx:05d}_t{timestamp:.3f}.png"
        plt.imsave(str(path), panel)

    def _create_video(self, fps: int = 10) -> None:
        frames = sorted(self.vis_dir.glob("*.png"))
        if not frames:
            return

        first = cv2.imread(str(frames[0]))
        if first is None:
            return

        h, w       = first.shape[:2]
        video_path = self.vis_dir.parent / "decoder_vector_video.mp4"
        writer: Optional[cv2.VideoWriter] = None

        for fourcc_tag in ("avc1", "mp4v"):
            candidate = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*fourcc_tag), fps, (w, h))
            if candidate.isOpened():
                writer = candidate
                logger.info("Using codec: %s", fourcc_tag)
                break
            candidate.release()

        if writer is None:
            logger.error("Could not open VideoWriter — skipping video.")
            return

        for fp in tqdm(frames, desc="Encoding video"):
            frame = cv2.imread(str(fp))
            if frame is not None:
                writer.write(frame)
        writer.release()
        logger.info("Video ready: %s", video_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the REALM reconstruction head on VECtor data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",        default="realm/realm/configs/image_reconstruction_rgb.yaml",
                        help="REALM YAML config, including the reconstruction head and its pretrained weights.")
    parser.add_argument("--dataset_path",  default="/home/viciopoli/datasets/TUM_VIE/VECtor/hdr_fast/",
                        help="Path to the VECtor sequence directory.")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--model_delta_t", type=float, default=33000.0,
                        help="Event window duration (µs).")
    parser.add_argument("--fp32",          action="store_true", help="Disable AMP.")
    parser.add_argument("--log_level",     default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger("image_reconstruction", level=args.log_level)

    validator = DecoderVectorValidator(args)
    validator.run()