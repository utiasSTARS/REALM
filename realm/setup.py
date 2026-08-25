import sys
import subprocess
import os
from setuptools import setup, find_packages

def build_dust3r_extensions():
    # Note: thirdparty is now INSIDE the realm folder
    root_dir = os.path.dirname(os.path.abspath(__file__))
    thirdparty_dir = os.path.join(root_dir, 'realm', 'thirdparty')
    dust3r_dir = os.path.join(thirdparty_dir, 'dust3r')
    curope_dir = os.path.join(dust3r_dir, 'croco', 'models', 'curope')

    # 1. Clone submodules if missing
    if not os.path.exists(os.path.join(dust3r_dir, '.git')):
        print("Initializing submodules...")
        subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'], cwd=root_dir)

    # 2. Build C++ extensions inside the folder
    # Requires `pip install --no-build-isolation .` so this sees the torch/CUDA
    # toolkit already installed in the active environment. Best-effort: REALM
    # falls back to a pure-PyTorch RoPE implementation if this doesn't run.
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if os.path.exists(curope_dir):
            print("Building the curope C++ extension...")
            subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=curope_dir)
    except ImportError:
        print("PyTorch is not installed. Skipping curope extension build.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: failed to build the curope extension ({e}). "
              "Continuing without it; REALM will fall back to a slower pure-PyTorch RoPE implementation.")

# Run the build step
build_dust3r_extensions()

def read_requirements():
    # Reuse the pinned versions from the repo-level requirements.txt so that
    # `pip install .` and `pip install git+...#subdirectory=realm` pull in the
    # full, version-matched dependency set instead of just bare torch/numpy.
    root_dir = os.path.dirname(os.path.abspath(__file__))
    req_path = os.path.join(root_dir, os.pardir, 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('--'):
                    continue
                requirements.append(line)
    return requirements or ['torch', 'numpy']

# Run standard setup
setup(
    name="realm",
    version="0.1",
    packages=find_packages(),
    include_package_data=True, # This tells pip to read the MANIFEST.in file
    install_requires=read_requirements(),
)
