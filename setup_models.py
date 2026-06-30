#!/usr/bin/env python3
"""Download ESMFold model from Hugging Face (~7.9 GB, required for 3D validation).

Run once before first use:
    python setup_models.py

The pipeline works without this model (ESMFold validation is skipped),
but the full 9-step pipeline including 3D structure validation requires it.
"""

import os
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "ESMFold"
MODEL_FILE = MODEL_DIR / "pytorch_model.bin"

HF_REPO = "facebook/esmfold_v1"

def main():
    if MODEL_FILE.exists():
        size_gb = os.path.getsize(MODEL_FILE) / 1e9
        print(f"ESMFold model already exists: {MODEL_FILE} ({size_gb:.1f} GB)")
        return

    print(f"Downloading ESMFold from Hugging Face ({HF_REPO})...")
    print(f"This will download ~7.9 GB. Please wait...")
    print()

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.safetensors"],
        )
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.safetensors"],
        )

    if MODEL_FILE.exists():
        size_gb = os.path.getsize(MODEL_FILE) / 1e9
        print(f"Done. Model downloaded: {size_gb:.1f} GB")
    else:
        print("Download may have failed. Check your internet connection.")
        print(f"Alternatively, manually place pytorch_model.bin in {MODEL_DIR}")

if __name__ == "__main__":
    main()
