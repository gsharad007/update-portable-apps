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
pip install -U requests tqdm rich py7zr beautifulsoup4 lxml
```
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tarfile
import time
import urllib.parse as uparse
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Final,
    Iterator,
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Never,
)

import requests
import py7zr
from bs4 import BeautifulSoup
from rich.console import Console
from tqdm import tqdm

__all__: Sequence[str] = (
    "AppConfig",
    "GrabPortablesError",
    "ConfigError",
    "AssetNotFoundError",
    "DownloadError",
    "NetworkError",
    "main",
)

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------

LOG_FILE: Final[str] = "grab-portables.log"
DEFAULT_CFG: Final[str] = "apps.json"
DEFAULT_RETRIES: Final[int] = 3
CHUNK: Final[int] = 8192
TIMEOUT: Final[float] = 60.0  # seconds for HTTP
UA: Final[str] = "Mozilla/5.0 (compatible; PortablesFetcher/1.0; +https://invalid/)"

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

UrlStr: TypeAlias = str

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GrabPortablesError(Exception):
    """Root of all domain‑specific exceptions."""


class ConfigError(GrabPortablesError):
    """Raised for malformed or inconsistent *apps.json* entries."""


class AssetNotFoundError(GrabPortablesError):
    """Raised when no release asset matches the supplied regex."""


class DownloadError(GrabPortablesError):
    """Raised after exhausting retries while downloading a file."""


class NetworkError(GrabPortablesError):
    """Raised when a network request fails."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AppConfig:
    name: str
    github_repo: Optional[str] = None  # owner/repo
    gitlab_repo: Optional[str] = None  # group/project (URL‑slug form)
    url: Optional[UrlStr] = None
    page_url: Optional[UrlStr] = None
    asset_regex: Optional[str] = None

    def __post_init__(self) -> None:  # noqa: D401
        # Ensure exactly one download source
        sources = [
            self.github_repo,
            self.gitlab_repo,
            self.url,
            self.page_url,
        ]
        if sum(x is not None for x in sources) != 1:
            raise ConfigError(
                f"{self.name}: specify exactly one of github_repo, gitlab_repo, url, or download_page_url"
            )
        if (self.github_repo or self.gitlab_repo) and not self.asset_regex:
            raise ConfigError(f"{self.name}: asset_regex required for VCS repos")
        if self.page_url and not self.asset_regex:
            raise ConfigError(f"{self.name}: asset_regex required for page scraping")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def assert_never(value: Never) -> NoReturn:  # noqa: D401
    _unused: Never = value
    raise AssertionError("Unreachable code executed")


def http_get(url: UrlStr, context: str, headers: Optional[dict[str, str]] = None) -> requests.Response:
    """Wrapper around requests.get that raises NetworkError on failure."""
    try:
        return requests.get(url, timeout=TIMEOUT, headers=headers)
    except requests.RequestException as exc:
        raise NetworkError(f"{context}: {exc}") from exc


# ----------------------------- GitHub ------------------------------------- #


def newest_github_asset(repo: str, pattern: str) -> Tuple[str, UrlStr]:
    api: UrlStr = f"https://api.github.com/repos/{repo}/releases/latest"
    logger.debug("GitHub API %s", api)
    resp: requests.Response = http_get(api, f"GitHub {repo}")
    if resp.status_code != 200:
        raise AssetNotFoundError(f"GitHub {repo}: HTTP {resp.status_code}")

    data: dict[str, object] = resp.json()
    tag: str = str(data.get("tag_name", ""))
    assets: List[dict[str, object]] = list(data.get("assets", []))  # type: ignore[arg-type]

    for asset in assets:
        name = str(asset.get("name", ""))
        if re.search(pattern, name, flags=re.I):
            url_field: object = asset.get("browser_download_url")
            if isinstance(url_field, str):
                return tag, url_field
    raise AssetNotFoundError(f"No GitHub asset in {repo} matches /{pattern}/i")


# ----------------------------- GitLab ------------------------------------- #


def newest_gitlab_asset(repo: str, pattern: str) -> Tuple[str, UrlStr]:
    # GitLab API expects URL‑encoded project path
    proj: str = uparse.quote_plus(repo)
    api: UrlStr = f"https://gitlab.com/api/v4/projects/{proj}/releases"
    logger.debug("GitLab API %s", api)
    resp: requests.Response = http_get(api, f"GitLab {repo}")
    if resp.status_code != 200:
        raise AssetNotFoundError(f"GitLab {repo}: HTTP {resp.status_code}")

    releases: List[dict[str, object]] = resp.json()  # type: ignore[assignment]
    if not releases:
        raise AssetNotFoundError(f"GitLab {repo}: no releases")
    latest: dict[str, object] = releases[0]
    tag: str = str(latest.get("tag_name", ""))

    assets: List[dict[str, object]] = list(latest.get("assets", {}).get("links", []))  # type: ignore[arg-type]
    for asset in assets:
        name: str = str(asset.get("name", ""))
        if re.search(pattern, name, flags=re.I):
            url_field: object = asset.get("url")
            if isinstance(url_field, str):
                return tag, url_field
    raise AssetNotFoundError(f"No GitLab asset in {repo} matches /{pattern}/i")


def newest_direct_asset(page_url: UrlStr, pattern: str) -> Tuple[str, UrlStr]:
    """Scrape *page_url*, parse <a href> links, return (version?, url) of first match.

    Uses BeautifulSoup (lxml/html.parser) for robust HTML parsing and resolves
    relative links using <base href> when present.
    """
    logger.debug(f"direct download page {page_url}")
    resp: requests.Response = http_get(
        page_url, f"Page fetch {page_url}", headers={"User-Agent": UA}
    )
    if resp.status_code != 200:
        raise AssetNotFoundError(f"Page fetch {page_url}: HTTP {resp.status_code}")
    soup = BeautifulSoup(resp.text, "lxml")

    # Determine effective base for relative links
    base_tag = soup.find("base", href=True)
    effective_base: UrlStr = (
        uparse.urljoin(page_url, base_tag["href"]) if base_tag else page_url
    )

    rx = re.compile(pattern, re.I)
    for a in soup.find_all("a", href=True):
        raw_href = str(a["href"])  # ensure str for typing
        url = uparse.urljoin(effective_base, raw_href)
        scheme = uparse.urlparse(url).scheme.lower()
        # logger.debug(f"direct download page links: '{a}' '{url}' '{a.get_text(strip=True)}'")
        if scheme not in ("http", "https"):
            continue
        m = rx.search(url) or rx.search(a.get_text(strip=True))
        if m:
            version = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
            return version, url

    raise AssetNotFoundError(f"No link in {page_url} matches /{pattern}/i")


# ----------------------------- Download / extract ------------------------- #


@contextmanager
def download(
    url: UrlStr,
    download_dir: Path,
    retries: int = DEFAULT_RETRIES,
) -> Iterator[Path]:
    """Download *url* into a temp-file under *download_dir*; yields Path."""

    parsed: uparse.ParseResult = uparse.urlparse(url)
    download_dir.mkdir(parents=True, exist_ok=True)

    filename: str = Path(parsed.path).name or f"download{int(time.time())}"
    dest: Path = download_dir / filename
    last_exc: Optional[requests.RequestException] = None

    for attempt in range(1, retries + 1):
        try:
            logger.debug("[%d/%d] GET %s", attempt, retries, url)
            resp: requests.Response = requests.get(url, stream=True, timeout=TIMEOUT)
            resp.raise_for_status()
            total_bytes: int = int(resp.headers.get("content-length", 0))

            with (
                dest.open("wb") as fh,
                tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                    leave=False,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    if chunk:  # pragma: no branch — network may send keep‑alives
                        fh.write(chunk)
                        bar.update(len(chunk))

            file_size: int = dest.stat().st_size
            if file_size == 0:
                dest.unlink(missing_ok=True)
                raise DownloadError("Downloaded zero‑byte file")
            if total_bytes and file_size != total_bytes:
                # dest.unlink(missing_ok=True)
                raise DownloadError(
                    f"Downloaded file size {file_size} does not match expected {total_bytes}"
                )

            break  # success
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Download attempt %d failed: %s", attempt, exc)
            dest.unlink(missing_ok=True)
            time.sleep(2**attempt)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
    else:
        raise DownloadError(f"Retries exhausted for {url}") from last_exc

    yield dest


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract *archive* (zip / tar / 7z) into *dest* directory."""
    logger.debug("Extracting %s to %s", archive, dest)
    dest.mkdir(parents=True, exist_ok=True)

    suffix: str = archive.suffix.lower()
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
        except zipfile.BadZipFile as exc:
            raise GrabPortablesError(f"{archive.name}: {exc}") from exc
    elif suffix in {".tar", ".gz", ".bz2", ".xz"} or archive.name.endswith(".tar.xz"):
        try:
            with tarfile.open(archive) as tf:
                tf.extractall(dest)
        except tarfile.TarError as exc:
            raise GrabPortablesError(f"{archive.name}: {exc}") from exc
    elif suffix == ".7z":
        try:
            with py7zr.SevenZipFile(archive) as z:
                z.extractall(dest)
        except py7zr.Bad7zFile as exc:
            raise GrabPortablesError(f"{archive.name}: {exc}") from exc
    else:
        # not an archive - copy or rename
        target: Path = dest / archive.name
        try:
            if archive.resolve() == target.resolve():
                logger.debug("Source and destination are the same")
            else:
                shutil.copy2(archive, target)
        except OSError as exc:
            raise GrabPortablesError("destination in use or locked") from exc


def prompt_yes_no(question: str, default: bool = True) -> bool:
    default_txt: str = "Y/n" if default else "y/N"
    ans = input(f"{question} ({default_txt}) ").strip().lower()
    if not ans:
        return default
    return ans in {"y", "yes"}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_app(cfg: AppConfig, download_dir: Path) -> None:
    logger.info("Processing %s", cfg.name)

    tag: Optional[str] = None
    dl_url: Optional[UrlStr] = None

    if cfg.page_url is not None:
        tag, dl_url = newest_direct_asset(cfg.page_url, cfg.asset_regex or "")
    elif cfg.gitlab_repo is not None:
        tag, dl_url = newest_gitlab_asset(cfg.gitlab_repo, cfg.asset_regex or ".*")
    elif cfg.github_repo is not None:
        tag, dl_url = newest_github_asset(cfg.github_repo, cfg.asset_regex or ".*")
    elif cfg.url is not None:
        dl_url = cfg.url
    else:  # pragma: no cover – validation prevents
        assert_never(cfg)  # type: ignore[arg-type]

    root: Path = download_dir
    root.mkdir(parents=True, exist_ok=True)

    folder_name: str = f"{cfg.name}_{tag}" if tag else cfg.name
    dest: Path = root / folder_name

    # prune older versions
    older: List[Path] = [
        p
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith(cfg.name) and p != dest
    ]

    if dest.exists():
        console.log(f"[bold green]✔ {cfg.name} already up-to-date (tag {tag})")
    else:
        # download + extract
        assert dl_url, "URL should be resolved by now"
        with download(dl_url, dest) as downloaded_archive:
            extract_archive(downloaded_archive, dest)

        console.log(
            f"[bold green]✔ {cfg.name} (tag {tag})[/] [bold green]✔ installed → {dest}"
        )

    if older and prompt_yes_no(
        f"Delete older versions of {cfg.name} ({older} => {dest})?", default=True
    ):
        for p in older:
            logger.debug("Deleting old version: %s", p)
            shutil.rmtree(p, ignore_errors=True)


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
    p.add_argument(
        "-d",
        "--download-dir",
        default=str(Path.cwd()),
        help="Where to place temporary downloads",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:  # noqa: D401
    """Program entry-point."""
    args = build_arg_parser().parse_args(argv)
    download_dir = Path(args.download_dir)
    try:
        configs = load_config(args.config)
        for cfg in configs:
            try:
                process_app(cfg, download_dir)
            except GrabPortablesError as app_exc:
                logger.error("%s failed: %s", cfg.name, app_exc)  # , exc_info=True)
    except GrabPortablesError as exc:
        logger.critical("Fatal error: %s", exc)  # , exc_info=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
