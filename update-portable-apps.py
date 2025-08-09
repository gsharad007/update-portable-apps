# mypy: strict
"""grab_portables_strict.py
An aggressively-typed, modular and chatty rewrite of the original
"grab-portables" script.  Runs on Python ≥3.10.

Key design points
=================
* **PEP-561 style typing** everywhere (``mypy --strict`` passes).
* **Dataclass-driven config** (``apps.json`` → ``List[AppConfig]``).
* **Dedicated exception hierarchy** so callers can distinguish errors.
* **Structured logging** (console + log-file) at DEBUG level by default.
* **Retry / back-off** for flaky HTTP downloads.
* **Assertions** for critical invariants (fail fast in dev / CI).
* **Graceful degradation** - one app's failure won't stop the batch.
* **Single-file** so you can still «just drop it into a USB».

3rd-party deps: ``requests``, ``tqdm``, ``rich``, ``py7zr``
Install once:
```
pip install -U requests tqdm rich py7zr
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, List, Optional, Tuple

import requests
import py7zr
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------

LOG_FILE: Final[str] = "grab-portables.log"
DEFAULT_CFG: Final[str] = "apps.json"
DEFAULT_RETRIES: Final[int] = 3
CHUNK: Final[int] = 8192

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger: Final[logging.Logger] = logging.getLogger(__name__)
console: Final[Console] = Console()

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GrabPortablesError(Exception):
    """Base-class for all script exceptions."""


class ConfigError(GrabPortablesError):
    pass


class AssetNotFoundError(GrabPortablesError):
    pass


class DownloadError(GrabPortablesError):
    pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AppConfig:
    name: str
    install_dir: str
    github_repo: Optional[str] = None  # e.g. "pbatard/rufus"
    asset_regex: Optional[str] = None  # e.g. "rufus-[0-9.]+p?\\.exe$"
    url: Optional[str] = None  # direct-download URL

    # --- validation ------------------------------------------------------- #
    def __post_init__(self) -> None:  # type: ignore[override]
        if self.github_repo is None and self.url is None:
            raise ConfigError(f"{self.name}: either github_repo or url is required")
        if self.github_repo and self.url:
            raise ConfigError(
                f"{self.name}: define *either* github_repo or url, not both"
            )
        if self.github_repo and not self.asset_regex:
            raise ConfigError(
                f"{self.name}: asset_regex required when github_repo is set"
            )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def newest_github_asset(repo: str, pattern: str) -> Tuple[str, str]:
    """Return (tag, asset_url) for the latest release whose asset name matches regex."""
    logger.debug("Fetching GitHub release: %s", repo)
    api_url: str = f"https://api.github.com/repos/{repo}/releases/latest"
    r = requests.get(api_url, timeout=30)
    if r.status_code != 200:
        raise AssetNotFoundError(
            f"GitHub API failed for {repo}: {r.status_code} {r.text[:200]}"
        )
    data = r.json()
    tag: str = data.get("tag_name", "")

    for asset in data.get("assets", []):
        name: str = asset.get("name", "")
        if re.search(pattern, name, flags=re.I):
            logger.debug("Matched asset %s for %s", name, repo)
            return tag, asset["browser_download_url"]

    raise AssetNotFoundError(f"No asset in {repo} matched /{pattern}/")


@contextmanager
def download_temp(url: str, retries: int = DEFAULT_RETRIES) -> Iterator[Path]:
    """Stream *url* into a temp-file with retries; yield its Path."""
    tmp: Optional[Path] = None
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            logger.debug("[%d/%d] Downloading %s", attempt, retries, url)
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            filename: str = Path(resp.url).name or "download.bin"

            # create an empty temp file path (works cross‑platform)
            fd, tmp_name = tempfile.mkstemp(prefix="grab_", suffix=filename)
            os.close(fd)
            tmp = Path(tmp_name)

            total: int = int(resp.headers.get("content-length", 0))

            with (
                tmp.open("wb") as fh,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                    leave=False,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))
            assert tmp.exists() and tmp.stat().st_size > 0, "Empty download?"
            yield tmp
            return  # success - exit generator
        except Exception as exc:
            last_exc = exc
            logger.exception("Download attempt %d failed: %s", attempt, exc)
            time.sleep(2**attempt)  # exponential backoff
        finally:
            if tmp and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    logger.warning("Temp cleanup failed for %s", tmp)
    raise DownloadError(f"All retries failed for {url}") from last_exc


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract *archive* (zip / tar / 7z) into *dest* directory."""
    logger.debug("Extracting %s to %s", archive, dest)
    dest.mkdir(parents=True, exist_ok=True)

    suffix: str = archive.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif suffix in {".tar", ".gz", ".bz2", ".xz"} or archive.name.endswith(".tar.xz"):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    elif suffix == ".7z":
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(dest)
    else:
        # not an archive - copy
        shutil.copy2(archive, dest / archive.name)


def prompt_yes_no(question: str, default: bool = True) -> bool:
    default_txt: str = "Y/n" if default else "y/N"
    ans = input(f"{question} ({default_txt}) ").strip().lower()
    if not ans:
        return default
    return ans in {"y", "yes"}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_app(cfg: AppConfig) -> None:
    logger.info("Processing %s", cfg.name)

    target_root: Path = Path(cfg.install_dir).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    if cfg.github_repo:
        tag, url = newest_github_asset(cfg.github_repo, cfg.asset_regex or ".*")
        version_name: str = f"{cfg.name}_{tag}"
    else:
        url = cfg.url  # type: ignore[assignment]
        tag = "direct"
        version_name = cfg.name

    dest_sub: Path = target_root / version_name
    logger.debug("Destination folder: %s", dest_sub)

    # prune older versions
    older: List[Path] = [
        p
        for p in target_root.iterdir()
        if p.is_dir() and p.name.startswith(cfg.name) and p != dest_sub
    ]
    if older and prompt_yes_no(f"Delete older versions of {cfg.name}?", default=True):
        for p in older:
            logger.debug("Deleting old version: %s", p)
            shutil.rmtree(p, ignore_errors=True)

    if dest_sub.exists():
        logger.info("%s already up-to-date (tag %s)", cfg.name, tag)
        return

    # download + extract
    assert url, "URL should be resolved by now"
    with download_temp(url) as tmp:
        extract_archive(tmp, dest_sub)

    console.log(f"[bold green]✔ {cfg.name}[/] installed → {dest_sub}")


# ---------------------------------------------------------------------------
# CLI / entry-point
# ---------------------------------------------------------------------------


def load_config(cfg_path: Path) -> List[AppConfig]:
    logger.debug("Loading config %s", cfg_path)
    if not cfg_path.exists():
        raise ConfigError(f"Missing config file: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ConfigError("apps.json must be a list of objects")
    configs: List[AppConfig] = [AppConfig(**item) for item in raw]
    logger.info("Loaded %d app definitions", len(configs))
    return configs


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download & install portable apps declared in apps.json"
    )
    p.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CFG,
        type=Path,
        help="Path to apps.json config",
    )
    return p


def main(argv: List[str] | None = None) -> None:  # noqa: D401
    """Program entry-point."""
    args = build_arg_parser().parse_args(argv)
    try:
        configs = load_config(args.config)
        for cfg in configs:
            try:
                process_app(cfg)
            except GrabPortablesError as app_exc:
                logger.error("%s failed: %s", cfg.name, app_exc, exc_info=True)
    except GrabPortablesError as exc:
        logger.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
