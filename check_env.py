"""Phase 1 checkpoint: every team member runs this and shares the full output.

Usage (inside the activated virtual environment, from the repo root):
    python check_env.py
"""
import importlib
import platform
import sys

PACKAGES = [
    "numpy", "pandas", "pyarrow", "sklearn", "lightgbm", "torch",
    "river", "scipy", "shap", "nfstream", "streamlit", "plotly",
    "joblib", "yaml", "pytest",
]

print("=== System ===")
print("OS        :", platform.platform())
print("Python    :", sys.version.split()[0])
print("Machine   :", platform.machine())

try:
    import psutil  # optional
    print("RAM (GB)  :", round(psutil.virtual_memory().total / 1e9, 1))
except ImportError:
    print("RAM (GB)  : install psutil to show this (pip install psutil)")

print("\n=== Packages ===")
missing = []
for name in PACKAGES:
    try:
        mod = importlib.import_module(name)
        print(f"{name:10s}: {getattr(mod, '__version__', 'installed')}")
    except Exception as exc:  # ImportError or a broken install
        missing.append(name)
        print(f"{name:10s}: MISSING ({type(exc).__name__})")

print("\n=== GPU ===")
try:
    import torch
    if torch.cuda.is_available():
        print("CUDA GPU  :", torch.cuda.get_device_name(0))
    else:
        print("CUDA GPU  : none (CPU only is fine for LightGBM; use Colab/Kaggle for the deep model)")
except Exception:
    print("CUDA GPU  : could not check (torch missing)")

print("\n=== Reproducibility ===")
import numpy as np
rng = np.random.default_rng(42)
checksum = round(float(rng.normal(size=1000).sum()), 6)
print("Seeded checksum:", checksum)
print("Everyone's checksum must be identical.")

print("\nRESULT:", "PASS" if not missing else f"FAIL - missing: {', '.join(missing)}")