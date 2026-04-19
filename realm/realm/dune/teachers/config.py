from .dinov2.models.vision_transformer import vit_large as dinov2_vitlarge
from .vit_master import mast3r


TEACHER_CFG = {
    "dino2reg_vitlarge_14": {
        "loader": dinov2_vitlarge,
        "ckpt_path": "/path/to/dinov2reg/vitlarge14/checkpoint.pth",
        "ckpt_key": "",
        "num_features": 1024,
        "image_size": 518,
        "patch_size": 14,
        "num_register_tokens": 4,
        "init_values": 1,
        "token_types": ["cls", "patch"],
    },
    "mast3r_vitlarge_16": {
        "loader": mast3r,
        "ckpt_path": "/path/to/mast3r/vitlarge16/checkpoint.pth",
        "ckpt_key": None,
        "code_dir": "/path/to/mast3r/code",  # Path to the MAST3R code, downloaded by scripts/teachers/mast3r.sh
        "num_features": 1024,
        "image_size": 512,  # shortest side
        "patch_size": 16,
        "token_types": ["patch"],  # ignore the cls token
    },
    "multihmr_vitlarge_14_672": {
        "loader": dinov2_vitlarge,
        "ckpt_path": "/path/to/multihmr/vitlarge14_672/checkpoint.pth",
        "ckpt_key": "model_state_dict",
        "num_features": 1024,
        "image_size": 518,  # model has 37 ** 2 + 1 positional embeddings, but fine-tuned at resolution 672
        "patch_size": 14,
        "num_register_tokens": 0,
        "init_values": 1,
        "token_types": ["patch"],  # ignore the cls token
    },
}
