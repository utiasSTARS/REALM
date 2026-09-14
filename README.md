<a name="readme-top"></a>

<div align="center" style="background-color: #0e2841; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
  <img src="media/logo.png" alt="REALM Logo" width="400"/>
  <h1 style="color: #ffffff; margin-top: 20px;">RGB and Event Aligned Latent Manifold</h1>
</div>

<div align="center" style="margin: 22px 0 28px;">
  <a href="https://papers.starslab.ca/realm/" target="_blank" rel="noreferrer">
    <img src="https://img.shields.io/badge/Website-REALM-0A84FF?style=for-the-badge&logo=globe&logoColor=white" alt="Website" />
  </a>
  <a href="https://arxiv.org/abs/2605.00271" target="_blank" rel="noreferrer" style="margin-left: 8px; margin-right: 8px;">
    <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper" />
  </a>
  <a href="https://viciopoli-realm-demo.hf.space/" target="_blank" rel="noreferrer">
    <img src="https://img.shields.io/badge/Hugging%20Face-Demo-FFD21F?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Demo" />
  </a>
</div>

Welcome to **REALM**! This repository contains the implementation of REALM, for advanced computer vision tasks involving both traditional RGB and Event-based vision.


<div align="center">
    <img src="media/demo_realm.gif" alt="demo" >
</div>

If you use this code, please cite the following publication:

```bibtex
@inproceedings{polizzi_2026_realm,
      title={REALM: An RGB- and Event-Aligned Latent Manifold for Cross-Modal Perception}, 
      author={Vincenzo Polizzi and David B. Lindell and Jonathan Kelly},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2026}
}
```

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#-features">Features</a></li>
    <li>
      <a href="#️-installation">Installation</a>
      <ul>
        <li><a href="#1-create-a-conda-environment">Create a Conda Environment</a></li>
        <li><a href="#2-install-requirements">Install Requirements</a></li>
        <li><a href="#3-install-the-realm-package">Install the REALM Package</a></li>
      </ul>
    </li>
    <li>
      <a href="#-usage">Usage</a>
      <ul>
        <li><a href="#1-import-and-use-realm-in-your-code">Import and Use REALM</a></li>
        <li><a href="#2-running-evaluation-scripts">Running Evaluation Scripts</a></li>
      </ul>
    </li>
    <li><a href="#-license">License</a></li>
    <li><a href="#-acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

---

## Abstract

Event cameras provide several unique advantages over standard frame-based sensors, including high temporal resolution, low latency, and robustness to extreme lighting. However, existing learning-based approaches for event processing are typically confined to narrow, task-specific silos and lack the ability to generalize across modalities. 

We address this gap with **REALM**, a cross-modal framework that learns an **R**GB and **E**vent **A**ligned **L**atent **M**anifold by projecting event representations into the pretrained latent space of RGB foundation models. Instead of task-specific training, we leverage low-rank adaptation (LoRA) to bridge the modality gap, effectively unlocking the geometric and semantic priors of frozen RGB backbones for asynchronous event streams.

We demonstrate that **REALM** effectively maps events into the ViT-based foundation latent space. Our method allows us to perform downstream tasks like depth estimation and semantic segmentation by simply transferring linear heads trained on the RGB teacher. Most significantly, **REALM** enables the direct, zero-shot application of complex, frozen image-trained decoders, such as MASt3R, to raw event data. We demonstrate state-of-the-art performance in wide-baseline feature matching, significantly outperforming specialized architectures. Code and models are available upon acceptance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🚀 Features

- **Multi-Modal Matching**: 3D grounded matching pipelines for RGB-to-Event and Event-to-Event.
- **Advanced Architectures**: Seamlessly integrates RGB trained heads using [DUNE backbone](https://github.com/naver/dune).
- **Multiple Downstream Tasks**: Support for depth estimation, semantic segmentation, 3D reconstruction and matching.
- **Comprehensive Evaluation**: Extensive benchmarking and evaluation scripts.

## 🛠️ Installation

We highly recommend using Conda to manage your python environment. Follow the steps below to install all dependencies and the `realm` package.

### 1. Clone the Repository
First, clone the repository and navigate into the project directory:

```bash
git clone --recursive https://github.com/utiasSTARS/REALM.git
cd REALM
```

### 2. Create a Conda Environment

Create and activate a new Conda environment (we recommend Python 3.10+):

```bash
conda create -n realm python=3.10 -y
conda activate realm
conda install -y -c "nvidia/label/cuda-12.8.0" cuda-toolkit=12.8 cuda-nvcc=12.8
```

Check that the nvcc compiler is available and the CUDA version is 12.8:

```bash
nvcc --version # should show CUDA 12.8
```

### 3. Install Requirements

Install the dependencies from the `requirements.txt` file located at the root of the repository:

```bash
pip install -r requirements.txt
```

### 4. Install the REALM Package

Navigate into the `realm` directory and install the core package:

```bash
cd realm
pip install --no-build-isolation .
```

`--no-build-isolation` is required so the build can see the `torch`/CUDA toolkit you already installed in steps 2–3, which lets it compile the optional `curope` CUDA extension (used by the DUSt3R/CroCo backbone for fast rotary position embeddings). Without it, pip builds in an isolated environment with no `torch` available, so the extension is silently skipped and REALM falls back to a slower pure-PyTorch RoPE implementation — it still works, just slower.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 💡 Usage

### 1. Import and Use REALM in Your Code

After installation, you can import the REALM package in your Python code as follows:

```python
import torch
from realm import REALM_creator
from realm.utils import representation_factory
from realm.utils import Resize


# Initialize the REALM model
# see realm/realm/configs/ for example configurations
model = REALM_creator(config='path/to/config.yaml') 

# Random event generation assuming a camera resolution of 720p and 5 channels (e.g., x, y, timestamp, polarity, and one additional feature channel)
H, W = 720, 1280

# random x, y, timestamp, polarity
x = torch.randint(0, W, (1000,))  # x-coordinates of events
y = torch.randint(0, H, (1000,))  # y-coordinates of events
timestamp = torch.rand(1000) * 1e6  # timestamps in microseconds
polarity = torch.randint(0, 2, (1000,))  # polarity: 0 for negative events, 1 for positive events

# Create an event representation using the factory function
channels = 5
normalize = True  # Whether to normalize the input data
ev_repr = representation_factory(
    rep_type="voxel_grid", height=H, width=W,
    channels=channels, normalize=normalize,
)

# Example input (replace with actual data)
input_data = ev_repr(x, y, timestamp, polarity)  # shape: (C, H, W) where C is the number of channels

# resize to 448x448 for REALM input
resize = Resize((448, 448))
input_data = resize(input_data)  # shape: (C, 448, 448)

# Forward pass through the model
output = model(input_data.unsqueeze(0), {optional_options})  # Add batch dimension, shape: (1, C, H, W)
```

### 2. Running Evaluation Scripts

The `evaluation/` directory contains scripts for evaluating the performance of REALM on various tasks. You can run these scripts as follows:

```bash
python evaluation/evaluate_depth.py
python evaluation/evaluate_seg.py
python evaluation/evaluate_matching.py
```

To store the visualization of the results, pass the `--save_vis` flag, results will be saved under `results/`.

To run the image reconstruction demo (event or RGB input reconstructed back into an image via the `reconstruction` head), run:

```bash
python evaluation/image_reconstruction.py --config realm/realm/configs/image_reconstruction_rgb.yaml --dataset_path <path/to/VECtor/sequence>
python evaluation/image_reconstruction_llff.py --config realm/realm/configs/image_reconstruction.yaml --dataset_path <path/to/ev-deblurnerf/sequence>
```

To run a quick feature matching demo between any two inputs — RGB images and/or event voxel grids (`.npy`), in any combination (image↔image, event↔event, or image↔event) — run:

```bash
python evaluation/demo_matching.py --input1 <path/to/view1> --input2 <path/to/view2>
```

Run it with no arguments to try it on the bundled test files (an event voxel grid matched against an RGB image); the result is saved to `results/demo_matching.png`. See `python evaluation/demo_matching.py --help` for all options (config, device, target resolution, number of matches drawn, output path).

Expect to see something like the following image:
<div align="center">
    <img src="test/smoke_test_result.png" alt="demo" >
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📝 License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

You are free to share and adapt this material for **non-commercial purposes**, provided that you:

- give appropriate credit and cite the REALM paper,
- indicate if changes were made,
- distribute any derivative work under the same license.

© 2025 Space and Terrestrial Autonomous Robotic Systems (STARS) Lab, University of Toronto Institute for Aerospace Studies (UTIAS). All rights reserved.

### Third-party Components

The following components are included in this repository under their own respective licenses:

- **[DUSt3R](https://github.com/naver/dust3r)** (`thirdparty/dust3r/`) — please refer to the original repository for license details.
- **MASt3R head** (`heads/mast3r/`) — modified from [MASt3R](https://github.com/naver/mast3r); please refer to the original repository for license details.
- **[DUNE](https://github.com/naver/dune)** (`dune/`) — please refer to the original repository for license details.

Please ensure you comply with the respective licenses of these components when using or redistributing this software.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🙏 Acknowledgements

This code is based on some open-source repositories, including:
- [DUNE](https://github.com/naver/dune)
- [MASt3R](https://github.com/naver/mast3r)
- [DUSt3R](https://github.com/naver/dust3r)
- The work by the [Robotics and Perception Group](https://rpg.ifi.uzh.ch/) at the University of Zurich for the event-based vision datasets and evaluation scripts that make it possible to benchmark the performance of REALM on downstream tasks.

<p align="right">(<a href="#readme-top">back to top</a>)</p>