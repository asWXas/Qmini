"""Build a pinned MinkowskiEngine with local PyTorch 2.11 / CUDA 13 compatibility fixes.

Uses the current Python environment. Never changes its torch/numpy versions or
uninstalls a global package. Build sources and wheels stay in the user cache.
"""

import argparse
import ctypes.util
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

REVISION = "02fc608bea4c0549b0a7b00ca1bf15dee4a0b228"
URL = "https://github.com/NVIDIA/MinkowskiEngine.git"


def run(command, **kwargs):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path.home() / ".cache/qmini" / f"MinkowskiEngine-{REVISION[:12]}"
    )
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--arch", help="TORCH_CUDA_ARCH_LIST; default detected GPU capability, e.g. 12.0")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    import torch

    if torch.version.cuda is None or not torch.version.cuda.startswith("13.0"):
        parser.error("This installer is pinned for torch CUDA 13.0; see README for other environments")
    if not ctypes.util.find_library("blas"):
        parser.error("System BLAS is missing; install its development headers/libraries before building")
    if not shutil.which("uv") or not shutil.which("git"):
        parser.error("uv and git must be available on PATH")
    arch = args.arch
    if arch is None:
        if not torch.cuda.is_available():
            parser.error("No GPU detected: specify --arch for the target device")
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "ninja==1.13.2",
            "wheel==0.48.0",
            "nvidia-cuda-nvcc==13.0.88",
            "nvidia-cuda-crt==13.0.88",
            "nvidia-cuda-cccl==13.0.85",
            "nvidia-nvvm==13.0.88",
        ]
    )
    toolkit = Path(importlib.metadata.distribution("nvidia-cuda-nvcc").locate_file("nvidia/cu13")).resolve()
    if not (toolkit / "lib64").exists():
        (toolkit / "lib64").symlink_to("lib", target_is_directory=True)
    for name in ("cudart", "cusparse", "cublas"):
        link = toolkit / "lib" / f"lib{name}.so"
        if not link.exists():
            candidates = sorted((toolkit / "lib").glob(f"lib{name}.so.*"))
            if not candidates:
                parser.error(f"Torch CUDA runtime dependency lib{name} is missing")
            link.symlink_to(candidates[0].name)
    source = args.source.resolve()
    if not source.exists():
        source.mkdir(parents=True)
        run(["git", "init", str(source)])
        run(["git", "fetch", "--depth", "1", URL, REVISION], cwd=source)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != REVISION:
        parser.error("Source revision mismatch; choose a new cache directory")
    patch = Path(__file__).with_name("minkowski_cuda13.patch")
    applied = (
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=source,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if not applied:
        run(["git", "apply", "--check", str(patch)], cwd=source)
        run(["git", "apply", str(patch)], cwd=source)
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "CUDA_HOME": str(toolkit),
        "TORCH_CUDA_ARCH_LIST": arch,
        "MAX_JOBS": str(args.jobs),
        "CPATH": str(toolkit / "include/cccl") + os.pathsep + os.environ.get("CPATH", ""),
    }
    run([sys.executable, "setup.py", "bdist_wheel", "--force_cuda", "--blas=blas"], cwd=source, env=env)
    wheels = sorted(
        (p for p in (source / "dist").glob("*.whl") if p.name.lower().startswith("minkowskiengine-")),
        key=lambda p: p.stat().st_mtime,
    )
    if not wheels:
        parser.error("Build did not produce a wheel")
    run(["uv", "pip", "install", "--python", sys.executable, "--no-deps", "--reinstall", str(wheels[-1])])
    run(
        [
            sys.executable,
            "-c",
            "import torch, MinkowskiEngine as ME; "
            "print('MinkowskiEngine', ME.__version__, 'CUDA', torch.cuda.is_available())",
        ],
        env=env,
    )


if __name__ == "__main__":
    main()
