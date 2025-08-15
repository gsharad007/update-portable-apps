"""grab_portables_strict.py
An aggressively-typed, modular and chatty rewrite of the original
"grab-portables" script.  Runs on Python ≥3.10.

Key design points
=================
* **PEP-561 style typing** everywhere (``mypy --strict`` passes).
* **Dataclass-driven config** (``apps.json`` → ``List[AppConfig]``).
* **Dedicated exception hierarchy** so callers can distinguish errors.
* **Structured logging** (console + log-file) at DEBUG level by default.
* **Assertions** for critical invariants (fail fast in dev / CI).
* **Graceful degradation** - one app's failure won't stop the batch.
* **Single-file** so you can still «just drop it into a USB».

3rd-party deps: ``requests``, ``httpx``, ``tqdm``, ``rich``, ``py7zr``, ``json5``
Install once:
```
pip install -U requests httpx tqdm rich py7zr beautifulsoup4 lxml json5
```
"""

from __future__ import annotations

import argparse
import json5
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
    Iterable,
    Generator,
    List,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Never,
    cast,
)

import requests
import httpx
import py7zr
from bs4 import BeautifulSoup, Tag
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
    """Root of all domain-specific exceptions."""


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
    gitlab_repo: Optional[str] = None  # group/project (URL-slug form)
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


def http_get(
    url: UrlStr, context: str, headers: Optional[dict[str, str]] = None
) -> requests.Response:
    """
    Wrapper around requests.get that raises NetworkError on failure.

    Parameters:
        url (UrlStr): The URL to fetch.
        context (str): Context description for error messages.
        headers (Optional[dict[str, str]]): HTTP headers to include in the request.

    Returns:
        requests.Response: The HTTP response object.

    Raises:
        NetworkError: If the request fails.
    """
    try:
        response: requests.Response = requests.get(
            url, timeout=TIMEOUT, headers=headers
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise NetworkError(
                f"{context}: HTTP error {response.status_code} for {url}: {exc}"
            ) from exc
        if response.status_code != 200:
            raise AssetNotFoundError(
                f"{context}: HTTP {response.status_code} for {url}"
            )
        return response
    except requests.RequestException as exc:
        raise NetworkError(f"{context}: {exc}") from exc


# ----------------------------- GitHub ------------------------------------- #


def newest_github_asset(repo: str, pattern: str) -> Tuple[str, UrlStr]:
    api: UrlStr = f"https://api.github.com/repos/{repo}/releases/latest"
    logger.debug("GitHub API %s", api)
    resp: requests.Response = http_get(api, f"GitHub {repo}")

    data: dict[str, object] = resp.json()
    tag: str = str(data.get("tag_name", ""))
    assets_iter: Iterable[dict[str, object]] = cast(
        Iterable[dict[str, object]], data.get("assets", [])
    )
    assets: List[dict[str, object]] = list(assets_iter)

    for asset in assets:
        name = str(asset.get("name", ""))
        if re.search(pattern, name, flags=re.I):
            url_field: object = asset.get("browser_download_url")
            if isinstance(url_field, str):
                return tag, url_field
    raise AssetNotFoundError(f"No GitHub asset in {repo} matches /{pattern}/i")


# ----------------------------- GitLab ------------------------------------- #


def newest_gitlab_asset(repo: str, pattern: str) -> Tuple[str, UrlStr]:
    # GitLab API expects URL-encoded project path
    proj: str = uparse.quote_plus(repo)
    api: UrlStr = f"https://gitlab.com/api/v4/projects/{proj}/releases"
    logger.debug("GitLab API %s", api)
    resp: requests.Response = http_get(api, f"GitLab {repo}")

    releases: List[dict[str, object]] = cast(List[dict[str, object]], resp.json())
    if not releases:
        raise AssetNotFoundError(f"GitLab {repo}: no releases")
    latest: dict[str, object] = releases[0]
    tag: str = str(latest.get("tag_name", ""))

    assets_dict: dict[str, object] = cast(dict[str, object], latest.get("assets", {}))
    links_iter: Iterable[dict[str, object]] = cast(
        Iterable[dict[str, object]], assets_dict.get("links", [])
    )
    assets: List[dict[str, object]] = list(links_iter)
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

    soup = BeautifulSoup(resp.text, "lxml")

    # Determine effective base for relative links
    base_tag = soup.find("base", href=True)
    if isinstance(base_tag, Tag):
        href_val = base_tag.get("href")
        base_href: str = str(href_val) if href_val else ""
        effective_base: UrlStr = uparse.urljoin(page_url, base_href)
    else:
        effective_base = page_url

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


def _filename_from_response(response: httpx.Response) -> str:
    """Derive a filename from *response* headers or URL."""
    cd: Optional[str] = response.headers.get("Content-Disposition")
    if cd is not None:
        match: Optional[re.Match[str]] = re.search(r"filename=\"?([^\";]+)\"?", cd)
        if match:
            return match.group(1)

    name: str = Path(uparse.urlparse(str(response.url)).path).name
    return name or f"download{int(time.time())}"


@contextmanager
def download(url: UrlStr, download_dir: Path) -> Generator[Path, None, None]:
    """Download *url* into *download_dir*; yields Path."""
    download_dir.mkdir(parents=True, exist_ok=True)

    initial_name: str = (
        Path(uparse.urlparse(url).path).name or f"download{int(time.time())}"
    )
    dest: Path = download_dir / initial_name
    resume_pos: int = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {"User-Agent": UA}
    if resume_pos:
        headers["Range"] = f"bytes={resume_pos}-"

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code not in {200, 206}:
                    dest.unlink(missing_ok=True)
                    raise DownloadError(f"HTTP {response.status_code} for {url}")
                final_name: str = _filename_from_response(response)
                if final_name != dest.name:
                    dest = dest.with_name(final_name)
                    resume_pos = dest.stat().st_size if dest.exists() else 0
                ctype: str = response.headers.get("Content-Type", "")
                if "text/html" in ctype:
                    dest.unlink(missing_ok=True)
                    raise DownloadError("expected binary content, got HTML")
                total: int = int(response.headers.get("Content-Length", "0"))
                if resume_pos and response.status_code == 206:
                    content_range: Optional[str] = response.headers.get("Content-Range")
                    if content_range and "/" in content_range:
                        total = int(content_range.split("/")[-1])
                    else:
                        total += resume_pos
                elif resume_pos:
                    total += resume_pos
                mode: str = "ab" if resume_pos else "wb"
                with (
                    open(dest, mode) as file,
                    tqdm(
                        unit="B",
                        unit_scale=True,
                        desc=dest.name,
                        leave=False,
                        total=total or None,
                        initial=resume_pos,
                    ) as bar,
                ):
                    for chunk in response.iter_bytes(65_536):
                        file.write(chunk)
                        bar.update(len(chunk))
        except httpx.HTTPError as exc:  # pragma: no cover - network errors
            dest.unlink(missing_ok=True)
            raise DownloadError(str(exc)) from exc

    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise DownloadError("Downloaded zero-byte file")

    try:
        yield dest
    except Exception:
        dest.unlink(missing_ok=True)
        raise


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
        with download(dl_url, dest) as downloaded_archive:  # type: Path
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


def _parse_config(text: str, cfg_path: Path) -> List[AppConfig]:
    """Convert JSON text into ``AppConfig`` objects.

    Parameters
    ----------
    text:
        Raw JSON configuration.
    cfg_path:
        Path to the configuration file, used only for error messages.

    Returns
    -------
    list[AppConfig]
        Parsed app configuration entries.

    Raises
    ------
    ConfigError
        If the JSON is invalid or entries cannot be mapped to ``AppConfig``.
    """

    try:
        raw: object = json5.loads(text)
    except json5.JSON5DecodeError as exc:  # noqa: PERF203 - re-raised with context
        line = getattr(exc, "lineno", "?")
        col = getattr(exc, "colno", "?")
        raise ConfigError(
            f"{cfg_path}: JSON decode error at line {line} column {col}"
        ) from exc

    if not isinstance(raw, list):
        raise ConfigError(f"{cfg_path}: root must be a list of objects")

    configs: List[AppConfig] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"{cfg_path} entry #{idx}: expected object")
        try:
            configs.append(AppConfig(**item))
        except TypeError as exc:  # noqa: PERF203 - provide context
            raise ConfigError(f"{cfg_path} entry #{idx}: {exc}") from exc

    return configs


def load_config(cfg_path: Path) -> List[AppConfig]:
    """Read ``apps.json`` from ``cfg_path`` and parse ``AppConfig`` entries."""

    logger.debug("Loading config %s", cfg_path)
    if not cfg_path.exists():
        raise ConfigError(f"Missing config file: {cfg_path}")

    text: str = cfg_path.read_text(encoding="utf-8")
    configs: List[AppConfig] = _parse_config(text, cfg_path)

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
