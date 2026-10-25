"""Pretrained networks, published as GitHub release assets.

``fetch("reliance")`` returns a local directory holding the trained network,
downloading and verifying it on first use; ``fetch("comparison")`` does the same
for the model-comparison network. Set ``RELYFLOW_WEIGHTS_DIR`` to a directory of
``<name>/`` training outputs to use local networks instead (for example offline).
Downloads are cached in ``RELYFLOW_CACHE`` (default ``~/.cache/relyflow``).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = "https://github.com/DanielPorter/relyflow/releases/download/{tag}/{asset}"
NETWORK_FILES = ("approximator.keras", "classifier.keras")
CHECKPOINT_FILES = (*NETWORK_FILES, "config.json", "history.json")
COMPARISON = "comparison"

# Network name (a model, or "comparison") -> (release tag, asset name, SHA-256 of the asset).
REGISTRY: dict[str, tuple[str, str, str]] = {
    "advisory": (
        "weights-v1",
        "relyflow-advisory-weights-v1.zip",
        "af228a9663848bf08b0853d62a9e554ce1ba8d221968169c860af0a07b971287",
    ),
    "reliance": (
        "weights-v1",
        "relyflow-reliance-weights-v1.zip",
        "47e1c25fcbafe5ed10ce421f0d4a72ed9ee4561523e568bf32cebcf42bf9f304",
    ),
    "learning": (
        "weights-v1",
        "relyflow-learning-weights-v1.zip",
        "3a708d147fbc9a81a019a5ab7beab5f7ae04fe38390d565917e8a29f21bcf303",
    ),
    COMPARISON: (
        "weights-v2",
        "relyflow-comparison-weights-v2.zip",
        "1da28edeafe05d68140a0cea9a9afff754966859511fc58a5c33962accdb2345",
    ),
}


def cache_dir() -> Path:
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "relyflow"
    return Path(os.environ.get("RELYFLOW_CACHE", default))


def fetch(name: str) -> Path:
    """Directory holding the pretrained network ``name`` (a model, or ``"comparison"``)."""
    local = os.environ.get("RELYFLOW_WEIGHTS_DIR")
    if local:
        directory = Path(local) / name
        if not _has_network(directory):
            raise FileNotFoundError(f"RELYFLOW_WEIGHTS_DIR has no network at {directory}")
        return directory
    if name not in REGISTRY:
        raise KeyError(f"no pretrained network registered for {name!r}; have {sorted(REGISTRY)}")

    tag, asset, sha256 = REGISTRY[name]
    directory = cache_dir() / tag / name
    if _has_network(directory):
        return directory

    url = RELEASE_URL.format(tag=tag, asset=asset)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / asset
        try:
            with urllib.request.urlopen(url) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
        except OSError as error:
            raise RuntimeError(
                f"Could not download {url} ({error}). If the repository is private or you "
                "are offline, set RELYFLOW_WEIGHTS_DIR to a directory of trained networks."
            ) from error
        digest = _sha256(archive)
        if digest != sha256:
            raise RuntimeError(f"{asset} failed its integrity check (SHA-256 {digest})")
        staging = Path(tmp) / "extracted"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        directory.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, directory, dirs_exist_ok=True)
    return directory


def package(checkpoint: str | Path, destination: str | Path) -> str:
    """Zip a training output directory for release; returns the archive's SHA-256."""
    checkpoint, destination = Path(checkpoint), Path(destination)
    if not _has_network(checkpoint):
        raise FileNotFoundError(f"no trained network in {checkpoint}")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in CHECKPOINT_FILES:
            if (checkpoint / name).exists():
                zf.write(checkpoint / name, arcname=name)
    return _sha256(destination)


def _has_network(directory: Path) -> bool:
    return any((directory / name).exists() for name in NETWORK_FILES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    """Package training outputs for a release and print the matching REGISTRY entries.

    ``python -m relyflow.inference package checkpoints dist/weights --tag weights-v1``
    """
    import argparse

    from relyflow.models import MODELS

    parser = argparse.ArgumentParser(
        prog="python -m relyflow.inference package", description=main.__doc__.splitlines()[0]
    )
    parser.add_argument(
        "checkpoints", help="directory containing <model>/ and comparison/ training outputs"
    )
    parser.add_argument("out", help="directory for the release archives")
    parser.add_argument("--tag", required=True, help="GitHub release tag, e.g. weights-v1")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("REGISTRY = {")
    for name in (*MODELS, COMPARISON):
        checkpoint = Path(args.checkpoints) / name
        if not _has_network(checkpoint):
            continue
        asset = f"relyflow-{name}-{args.tag}.zip"
        digest = package(checkpoint, out / asset)
        print(f'    "{name}": ("{args.tag}", "{asset}", "{digest}"),')
    print("}")
