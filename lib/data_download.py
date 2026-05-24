"""
Dataset auto-download helper for BDD100K from Kaggle.

Source: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k

Because the BDD100K dataset is far too large to commit to the GitHub repo,
the training/evaluation scripts call `ensure_bdd100k()` which:

  1. Checks if the expected `bdd100k/` directory already exists under the
     user-supplied `data_root`. If yes, returns it unchanged.
  2. Otherwise, downloads the dataset from Kaggle via `kagglehub` (preferred)
     or the official `kaggle` CLI, locates the `bdd100k/` folder inside the
     download, and returns the parent directory as the new `data_root`.

Authentication
--------------
Both `kagglehub` and the `kaggle` CLI read credentials from `~/.kaggle/kaggle.json`
(or the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables). Get yours at:
    https://www.kaggle.com/settings -> "Create New API Token"
"""

from __future__ import annotations

import os
from pathlib import Path

KAGGLE_DATASET_DEFAULT = "solesensei/solesensei_bdd100k"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k"


def _has_bdd100k(root: Path) -> bool:
    """Return True if `root/bdd100k/` looks like a valid BDD100K layout."""
    bdd = root / "bdd100k"
    if not bdd.is_dir():
        return False
    # Minimal sanity check: at least one of the expected subdirs exists.
    candidates = [
        bdd / "images" / "100k",
        bdd / "seg" / "images",
        bdd / "labels",
    ]
    return any(p.exists() for p in candidates)


def _find_bdd100k_root(search_root: Path) -> Path | None:
    """Walk `search_root` and return the parent of the first `bdd100k/` dir found."""
    if _has_bdd100k(search_root):
        return search_root
    for dirpath, dirnames, _ in os.walk(search_root):
        if "bdd100k" in dirnames:
            parent = Path(dirpath)
            if _has_bdd100k(parent):
                return parent
    return None


def _download_via_kagglehub(slug: str) -> Path:
    import kagglehub  # type: ignore
    print(f"[data_download] Fetching '{slug}' via kagglehub...")
    path = kagglehub.dataset_download(slug)
    print(f"[data_download]   -> {path}")
    return Path(path)


def _download_via_kaggle_cli(slug: str, dest: Path) -> Path:
    """Fallback: use the `kaggle` Python package's CLI API."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"[data_download] Fetching '{slug}' via kaggle CLI into {dest}...")
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    return dest


def ensure_bdd100k(data_root: str | os.PathLike,
                   kaggle_dataset: str = KAGGLE_DATASET_DEFAULT,
                   force_download: bool = False) -> str:
    """
    Ensure that `<data_root>/bdd100k/` is available locally, downloading from
    Kaggle if necessary. Returns the (possibly updated) data_root path.

    Parameters
    ----------
    data_root : str | PathLike
        Desired local data root. If `<data_root>/bdd100k/` already exists,
        nothing is downloaded.
    kaggle_dataset : str
        Kaggle dataset slug, e.g. `"solesensei/solesensei_bdd100k"`.
    force_download : bool
        If True, re-download even when local data is present.
    """
    root = Path(data_root).expanduser().resolve()

    if not force_download and _has_bdd100k(root):
        print(f"[data_download] Using existing dataset at {root}")
        return str(root)

    print(f"[data_download] '{root / 'bdd100k'}' not found.")
    print(f"[data_download] Downloading from {KAGGLE_DATASET_URL}")

    download_path: Path
    try:
        download_path = _download_via_kagglehub(kaggle_dataset)
    except ImportError:
        print("[data_download] `kagglehub` not installed; falling back to `kaggle` CLI.")
        download_path = _download_via_kaggle_cli(kaggle_dataset, root)
    except Exception as e:
        print(f"[data_download] kagglehub failed ({e}); falling back to `kaggle` CLI.")
        download_path = _download_via_kaggle_cli(kaggle_dataset, root)

    found = _find_bdd100k_root(download_path)
    if found is None:
        raise RuntimeError(
            f"Could not locate a 'bdd100k/' directory inside the downloaded "
            f"dataset at {download_path}. Inspect the folder manually and pass "
            f"--data-root pointing to the parent of 'bdd100k/'."
        )

    print(f"[data_download] Dataset ready at {found}")
    return str(found)
