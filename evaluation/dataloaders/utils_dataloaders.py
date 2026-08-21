import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2 as T

H5_CACHE_KWARGS = {
    'rdcc_nbytes': 1024 * 1024 * 4,  # 4MB chunk cache
    'rdcc_w0': 0.75,                 # Preemption policy (default 0.75 is balanced)
    'rdcc_nslots': 10007,            # Prime number for hash table
    'libver': 'latest',              # Use latest library version for speed
}

def voxel_to_rgb_image(
    voxel_grid_tensor: torch.Tensor,
    background_color: float = 0.0,
) -> np.ndarray:
    """
    Convert a Voxel Grid tensor (C, H, W) into an RGB image (H, W, 3).

    Red   = positive events
    Blue  = negative events
    Green = unused

    Args:
        voxel_grid_tensor: (C, H, W) tensor (float), voxel grid or stacked bins.
        background_color: background intensity in [0, 1].
                          Applied only where there are no events.

    Returns:
        np.ndarray: (H, W, 3) float image in [0, 1] for visualization.
    """
    # check if tensor is a voxel grid, that is the channels are more than 3
    if voxel_grid_tensor.shape[0] <= 3:
        print("Input tensor has 3 or fewer channels; expected voxel grid with multiple temporal bins.")
        return voxel_grid_tensor
    if not isinstance(voxel_grid_tensor, torch.Tensor):
        grid = torch.from_numpy(voxel_grid_tensor)
    else:
        grid = voxel_grid_tensor.detach().cpu()

    flat_grid = grid.sum(dim=0)

    # ---- Separate polarities --------------------------------------------
    pos = flat_grid>0.0
    neg = flat_grid<0.0

    # ---- Build RGB image -------------------------------------------------
    H, W = flat_grid.shape

    if background_color != 0.0:
        rgb = torch.ones((H, W, 3), dtype=torch.float32)    
    else:
        rgb = torch.zeros((H, W, 3), dtype=torch.float32)

    rgb[pos, :] = torch.tensor([0.0, 0.0, 1.0])    # Blue for positive events
    rgb[neg, :] = torch.tensor([1.0, 0.0, 0.0])    # Red for negative events

    return rgb

def make_event_preview(events_ori, H=None, W=None):
    """
    Generate red-blue event visualization.
    events: [N,4] array/tensor with [t,x,y,p]
    Red = negative events, Blue = positive events
    """
    # Convert torch to numpy if needed
    if isinstance(events_ori, torch.Tensor):
        events = events_ori.detach().cpu().numpy()
    else:
        events = events_ori.copy()

    if H is None or W is None:
        raise ValueError("H and W required")
    
    # Accumulate events
    sum_events = np.zeros((H, W), dtype=np.float32)
    xs = np.clip(events[:, 1].astype(np.int32), 0, W - 1)
    ys = np.clip(events[:, 2].astype(np.int32), 0, H - 1)
    ps = events[:, 3] * 2 - 1  # [0,1] -> [-1,1]
    np.add.at(sum_events, (ys, xs), ps)
    
    # Create red-blue visualization
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[sum_events > 0, 0] = 255  # Blue for positive
    img[sum_events < 0, 2] = 255  # Red for negative
    
    return img


def overlay_events_on_image(image, events, alpha=0.5):
    """
    Overlay events on image with transparency.
    
    Args:
        image: [H,W] grayscale or [H,W,3] RGB numpy array or torch tensor
        events: [N,4] array/tensor with [t,x,y,p]
        alpha: transparency of event overlay (0=invisible, 1=opaque)
    
    Returns:
        [H,W,3] RGB numpy array with events overlaid
    """
    if len(image.shape) == 4:
        image = image[0]  # Take first in batch if batched
    # Convert image to numpy RGB
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=2)
    else:
        image_rgb = image.copy()
    
    # Normalize to [0, 255]
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    else:
        image_rgb = image_rgb.astype(np.uint8)
    
    # check if img is C, H, W or H, W, C
    if image_rgb.shape[0] in (1, 3):
        image_rgb = np.transpose(image_rgb, (1, 2, 0))
    
    # if C is 1 convert to 3 channels
    if image_rgb.shape[2] == 1:
        image_rgb = np.concatenate([image_rgb]*3, axis=2)

    H, W, _ = image_rgb.shape
    
    # Generate event visualization
    event_img = make_event_preview(events, H=H, W=W)
    
    # Blend where events exist
    event_mask = (event_img.sum(axis=2) > 0)
    result = image_rgb.copy()
    result[event_mask] = (alpha * event_img[event_mask] + 
                          (1 - alpha) * image_rgb[event_mask]).astype(np.uint8)
    
    return result


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ZEROS = [m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

IMAGENET_NORM = True

def image_to_normalized_tensor(img) -> torch.Tensor:
    """
    Convert a PIL image / numpy array / torch tensor to a normalized torch.Tensor.
    Handles (C, H, W) and (B, C, H, W) shapes.
    """
    if not IMAGENET_NORM:
        if isinstance(img, torch.Tensor) and img.max() > 1.0:
            img = img / 255.0
        elif isinstance(img, np.ndarray) and img.max() > 1.0:
            img = img / 255.0
        return img

    # If input is a PIL image, convert to numpy array first
    arr = np.array([])
    if isinstance(img, Image.Image):
        arr = np.array(img)
    elif isinstance(img, np.ndarray):
        arr = img
    elif isinstance(img, torch.Tensor):
        # --- PATH USED BY EVENTSCAPE / EDS / M3ED ---
        t = img
        if t.dtype != torch.float32:
            t = t.to(torch.float32) 
        
        # Scale [0, 255] -> [0, 1]
        try:
            if t.max() > 1.0:
                t = t / 255.0
        except Exception:
            pass
            
        # --- FIX: Handle 4D Tensors (B, C, H, W) ---
        if t.ndim == 4:
            # Check if Channels are at dim 1 (B, C, H, W)
            if t.shape[1] in (1, 3): 
                tensor = t
            # Check if Channels are at dim 3 (B, H, W, C) -> Permute
            elif t.shape[3] in (1, 3):
                tensor = t.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 4D tensor shape: {t.shape}")

        # --- Handle 3D Tensors (C, H, W) ---
        elif t.ndim == 3:
            if t.shape[0] in (1, 3):
                tensor = t
            elif t.shape[2] in (1, 3):
                tensor = t.permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported 3D tensor shape: {t.shape}")
        else:
            raise ValueError(f"Unsupported tensor shape: {t.shape}")

        # Normalize
        # (3, 1, 1) broadcasts correctly to both (3, H, W) and (B, 3, H, W)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
        return (tensor - mean) / std

    # --- NUMPY PATH ---
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32)

    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(-1, 1, 1)

    img = (tensor - mean) / std

    return img

def img_undo_normalize(tensor: torch.Tensor, inplace: bool = False, to_pil: bool = False) -> torch.Tensor:
    # """
    # Undo ImageNet normalization on a tensor.
    # Input: torch.Tensor of shape (C, H, W)
    # Output: torch.Tensor of shape (C, H, W) with values in [0, 1]
    # """
    # mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    # std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    # tensor = tensor * std + mean
    # tensor = torch.clamp(tensor, 0.0, 1.0)
    # return tensor

    """
    Undo ImageNet mean/std normalization.

    Args:
        tensor: torch.Tensor with shape (C, H, W) in normalized space (i.e. after
                (x - mean) / std). Can also accept a single-batch tensor (1, C, H, W).
        inplace: whether to operate in-place on the provided tensor (default False).
        to_pil: if True, returns a PIL.Image in uint8 [0,255] format; otherwise returns
                a torch.float32 tensor clamped to [0,1] with shape (C, H, W).

    Returns:
        PIL.Image if to_pil=True, else torch.Tensor in [0,1], dtype torch.float32.
    """

    if not IMAGENET_NORM:
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    t = tensor
    # Accept batch of size 1
    squeezed_batch = False
    if t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
        squeezed_batch = True

    # Accept HWC as well as CHW
    if t.ndim == 3 and t.shape[2] in (1, 3):
        t = t.permute(2, 0, 1)

    if t.ndim != 3 or t.shape[0] not in (1, 3):
        raise ValueError(f"Unsupported tensor shape for undo_normalize: {tensor.shape}")

    if not inplace:
        t = t.clone()

    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device).view(-1, 1, 1)

    t = t * std + mean
    t = t.clamp(0.0, 1.0)

    if to_pil:
        arr = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    return t
