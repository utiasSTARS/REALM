"""
validate_decoder_vector.py — qualitative reconstruction test for the trained ImageDecoder.

Loads a frozen REALM encoder + trained ImageDecoder and runs reconstruction on
VECtor event or RGB data.  Saves side-by-side panels:
    Events  model: [event preview  | reconstructed RGB]
    RGB     model: [input RGB      | reconstructed RGB]
    Hybrid  model: dispatched by channel count (same panels as above)

Usage:
    python validate_decoder_vector.py --config configs/encoder_only.yaml --checkpoint results/decoder/best_decoder.pt --dataset_path datasets/VECtor/robot-fast
    python validate_decoder_vector.py --config configs/encoder_only.yaml --checkpoint results/decoder/best_decoder.pt --dataset_path datasets/VECtor/robot-fast --device cuda:1 --fp32
"""

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from dataloaders.utils_dataloaders import img_undo_normalize, voxel_to_rgb_image
from dataloaders.vector_dataset import build_vector_datasets
from realm.model import ModelType
from realm.model_factory import REALM_creator
from realm.utils.log import get_logger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

logger = get_logger("validate_decoder_vector", level=logging.INFO)


# ---------------------------------------------------------------------------
# Decoder  (must match the architecture used during training)
# ---------------------------------------------------------------------------

class ImageDecoder(nn.Module):
    def __init__(self, num_tokens: int = 1024, token_dim: int = 768, target_size: int = 448):
        super().__init__()
        self.Ph = int(math.isqrt(num_tokens))
        assert self.Ph ** 2 == num_tokens, "num_tokens must be a perfect square"

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
            )

        up = lambda: nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.net = nn.Sequential(
            block(token_dim, 512),   up(),   # 32 → 64
            block(512, 256),         up(),   # 64 → 128
            block(256, 128),         up(),   # 128 → 256
            block(128, 64),
            nn.Upsample(size=target_size, mode="bilinear", align_corners=False),  # 256 → 448
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = patch_tokens.shape
        x = patch_tokens.reshape(B, self.Ph, self.Ph, C).permute(0, 3, 1, 2).contiguous()
        return self.net(x)


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
    Run the trained ImageDecoder on VECtor data and save reconstruction panels.

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

        # Encoder (frozen) -----------------------------------------------------
        logger.info("Loading encoder...")
        self.encoder = REALM_creator(self.cfg).to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Decoder --------------------------------------------------------------
        logger.info("Loading decoder from %s", args.checkpoint)
        self.decoder = ImageDecoder(num_tokens=1024, token_dim=768, target_size=self.CROP).to(self.device)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        total_params = count_parameters(self.decoder)
        print(f"Total trainable parameters: {total_params:,}")

        ckpt = torch.load(args.checkpoint, map_location=self.device)
        self.decoder.load_state_dict(ckpt["decoder"])
        self.decoder.eval()
        logger.info("Decoder restored from epoch %d (val loss %.4f)", ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")))

        self.use_amp: bool = not args.fp32
        logger.info("Using %s inference.", "AMP" if self.use_amp else "FP32")

        # Output dirs ----------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        seq_name  = Path(args.dataset_path).name
        self.vis_dir = Path("results") / "decoder_vector" / f"{seq_name}_{timestamp}" / "imgs"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving frames → %s", self.vis_dir)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self) -> None:
        is_event_model = self.encoder.model_type in (ModelType.Events, ModelType.Hybrid)

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
                patch_tokens = self.encoder(inp)["x_norm_patchtokens"]  # (1, 1024, 768)
                pred         = self.decoder(patch_tokens)                # (1, 3, H, W)

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
        description="Test trained ImageDecoder on VECtor data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",        default="realm/realm/configs/encoder_only.yaml",
                        help="REALM YAML config (must match the one used during training).")
    parser.add_argument("--checkpoint",    default="/home/viciopoli/checkpoints/mast3r/heads/img_recon_rgb.pth",
                        help="Path to best_decoder.pt saved by train_decoder.py.")
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
    logger = get_logger("validate_decoder_vector", level=args.log_level)

    validator = DecoderVectorValidator(args)
    validator.run()