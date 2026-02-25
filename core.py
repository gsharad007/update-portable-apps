"""core.py — shared building blocks for update-portable-apps.py and update-installers.py.

All logic lives here as small, composable functions.  The two scripts are thin
wrappers that supply a single ``download_fn`` and call :func:`run_phases`.

3rd-party deps: ``requests``, ``httpx``, ``tqdm``, ``rich``, ``json5``,
``beautifulsoup4``, ``lxml``.  Optional: ``py7zr``, ``requests_html``,
``lxml_html_clean`` (headless scraping / 7z extraction).
"""

from __future__ import annotations

import argparse
import io
import json5
import logging
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse as uparse
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.util import find_spec
from pathlib import Path
from typing import (
    Callable,
    Final,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    cast,
)

import httpx
import requests
from bs4 import BeautifulSoup, Tag
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

__all__: Sequence[str] = (
    "AppError", "ConfigError", "AssetNotFoundError", "DownloadError",
    "NetworkError", "UserQuit",
    "AppStatus", "AppConfig", "CheckResult", "DownloadResult",
    "DownloadFn",
    # network
    "http_get", "fetch_html", "render_html",
    # scraping
    "scrape_links", "newest_page_asset",
    # VCS
    "newest_github_asset", "newest_gitlab_asset",
    # package managers
    "newest_winget_asset", "newest_choco_asset",
    # source dispatch
    "fetch_latest",
    # redirect
    "follow_redirects",
    # download helpers
    "filename_from_response", "resume_state", "download_total", "write_stream",
    # download context manager
    "download_file",
    # extraction
    "extract_archive",
    # folder / version
    "dest_name", "older_versions", "make_check_result", "failed_check",
    # phases
    "check_one", "check_all",
    # config
    "parse_config_text", "load_config",
    # input
    "safe_input", "yes_no",
    # UI
    "print_check_table", "updatable", "parse_indices", "prompt_specific",
    "prompt_selection",
    # orchestration
    "run_downloads", "run_cleanup", "print_summary", "run_phases",
    # CLI
    "arg_parser",
    # globals
    "console", "logger",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEOUT: Final[float] = 60.0
HEADLESS_WAIT: Final[float] = 2.0
UA: Final[str] = "Mozilla/5.0 (compatible; AppUpdater/1.0; +https://invalid/)"

UrlStr: TypeAlias = str
DownloadFn: TypeAlias = Callable[["CheckResult", Path], "DownloadResult"]

console: Final[Console] = Console()
logger: Final[logging.Logger] = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Root of all domain-specific exceptions."""


class ConfigError(AppError):
    """Raised for malformed or inconsistent config entries."""


class AssetNotFoundError(AppError):
    """Raised when no release asset matches the supplied regex."""


class DownloadError(AppError):
    """Raised when a download fails."""


class NetworkError(AppError):
    """Raised when a network request fails."""


class UserQuit(SystemExit):
    """Raised when the user chooses to quit."""

    def __init__(self) -> None:
        super().__init__(0)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AppStatus(Enum):
    UP_TO_DATE = auto()
    UPDATE_AVAILABLE = auto()
    CHECK_FAILED = auto()


@dataclass(slots=True, frozen=True)
class AppConfig:
    name: str
    github_repo: Optional[str] = None
    gitlab_repo: Optional[str] = None
    url: Optional[UrlStr] = None
    page_url: Optional[UrlStr] = None
    asset_regex: Optional[str] = None
    referer: Optional[str] = None  # Required by some CDNs (e.g. AMD drivers.amd.com)
    winget_id: Optional[str] = None  # e.g. "7zip.7zip", "Google.Chrome"
    choco_id: Optional[str] = None   # e.g. "7zip", "googlechrome"

    def __post_init__(self) -> None:
        sources = [
            self.github_repo, self.gitlab_repo, self.url,
            self.page_url, self.winget_id, self.choco_id,
        ]
        if sum(x is not None for x in sources) != 1:
            raise ConfigError(
                f"{self.name}: specify exactly one of github_repo, gitlab_repo, "
                f"url, page_url, winget_id, or choco_id"
            )
        if (self.github_repo or self.gitlab_repo) and not self.asset_regex:
            raise ConfigError(f"{self.name}: asset_regex required for VCS repos")
        if self.page_url and not self.asset_regex:
            raise ConfigError(f"{self.name}: asset_regex required for page scraping")


@dataclass(slots=True, frozen=True)
class CheckResult:
    cfg: AppConfig
    status: AppStatus
    tag: Optional[str] = None
    download_url: Optional[UrlStr] = None
    dest_folder: Optional[Path] = None
    current_folder: Optional[Path] = None
    older_folders: tuple[Path, ...] = field(default_factory=tuple)
    error_message: Optional[str] = None


@dataclass(slots=True)
class DownloadResult:
    check: CheckResult
    success: bool
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def http_get(
    url: UrlStr,
    context: str,
    headers: Optional[dict[str, str]] = None,
) -> requests.Response:
    """GET *url*, raise ``NetworkError`` on any failure."""
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=headers)
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            raise NetworkError(
                f"{context}: HTTP {r.status_code} for {url}: {exc}"
            ) from exc
        return r
    except requests.RequestException as exc:
        raise NetworkError(f"{context}: {exc}") from exc


def fetch_html(url: UrlStr) -> str:
    """Fetch *url* and return the response body as text."""
    return http_get(url, f"Page fetch {url}", headers={"User-Agent": UA}).text


def render_html(url: UrlStr) -> str:
    """Render *url* with a headless browser and return the final HTML."""
    if not (find_spec("requests_html") and find_spec("lxml_html_clean")):
        raise NetworkError(
            "headless scraping requires requests_html and lxml_html_clean"
        )
    from requests_html import HTML, HTMLResponse, HTMLSession  # type: ignore[import]

    with HTMLSession() as session:
        resp: HTMLResponse = session.get(
            url, headers={"User-Agent": UA}, timeout=TIMEOUT
        )
        html_obj: HTML = resp.html
        try:
            html_obj.render(timeout=TIMEOUT, sleep=HEADLESS_WAIT)
        except Exception as exc:
            raise NetworkError(f"Headless fetch {url}: {exc}") from exc
        return str(html_obj.html)


# ---------------------------------------------------------------------------
# Page scraping
# ---------------------------------------------------------------------------


def _page_base_url(soup: BeautifulSoup, page_url: UrlStr) -> UrlStr:
    """Return the effective base URL: <base href> resolved against *page_url*."""
    base_tag = soup.find("base", href=True)
    if isinstance(base_tag, Tag):
        href = str(base_tag.get("href") or "")
        return uparse.urljoin(page_url, href)
    return page_url


def _first_link_match(
    soup: BeautifulSoup, base: UrlStr, pattern: str
) -> Optional[Tuple[str, UrlStr]]:
    """Scan ``<a href>`` tags and return ``(version, url)`` for the first match."""
    rx = re.compile(pattern, re.I)
    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        url = uparse.urljoin(base, str(a["href"]))
        if uparse.urlparse(url).scheme.lower() not in ("http", "https"):
            continue
        m = rx.search(url) or rx.search(a.get_text(strip=True))
        if m:
            version = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
            return version, url
    return None


def scrape_links(
    html: str, page_url: UrlStr, pattern: str
) -> Optional[Tuple[str, UrlStr]]:
    """Parse *html* and return the first link matching *pattern*, or ``None``."""
    soup = BeautifulSoup(html, "lxml")
    base = _page_base_url(soup, page_url)
    return _first_link_match(soup, base, pattern)


def newest_page_asset(page_url: UrlStr, pattern: str) -> Tuple[str, UrlStr]:
    """Fetch *page_url*, extract the first link matching *pattern*.

    Falls back to a headless browser if the simple fetch yields no match.
    """
    logger.debug("Page asset lookup: %s", page_url)
    result = scrape_links(fetch_html(page_url), page_url, pattern)
    if result:
        return result
    logger.debug("Simple scrape failed, retrying with headless browser: %s", page_url)
    result = scrape_links(render_html(page_url), page_url, pattern)
    if result:
        return result
    raise AssetNotFoundError(f"No link in {page_url} matches /{pattern}/i")


# ---------------------------------------------------------------------------
# VCS asset lookup
# ---------------------------------------------------------------------------


def _github_latest_release(repo: str) -> Tuple[str, List[dict[str, object]]]:
    """Return ``(tag, assets)`` for the latest GitHub release of *repo*."""
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    logger.debug("GitHub API: %s", api)
    data: dict[str, object] = http_get(api, f"GitHub {repo}").json()
    tag = str(data.get("tag_name", ""))
    assets = list(cast(Iterable[dict[str, object]], data.get("assets", [])))
    return tag, assets


def _gitlab_latest_release(repo: str) -> Tuple[str, List[dict[str, object]]]:
    """Return ``(tag, asset_links)`` for the latest GitLab release of *repo*."""
    api = f"https://gitlab.com/api/v4/projects/{uparse.quote_plus(repo)}/releases"
    logger.debug("GitLab API: %s", api)
    releases: List[dict[str, object]] = cast(
        List[dict[str, object]], http_get(api, f"GitLab {repo}").json()
    )
    if not releases:
        raise AssetNotFoundError(f"GitLab {repo}: no releases found")
    latest = releases[0]
    tag = str(latest.get("tag_name", ""))
    links = list(
        cast(
            Iterable[dict[str, object]],
            cast(dict[str, object], latest.get("assets", {})).get("links", []),
        )
    )
    return tag, links


def _match_asset_url(
    assets: List[dict[str, object]], pattern: str, url_key: str
) -> UrlStr:
    """Return the download URL of the first asset whose name matches *pattern*."""
    rx = re.compile(pattern, re.I)
    for asset in assets:
        if rx.search(str(asset.get("name", ""))):
            u = asset.get(url_key)
            if isinstance(u, str):
                return u
    raise AssetNotFoundError(f"No asset matches /{pattern}/i")


def newest_github_asset(repo: str, pattern: str) -> Tuple[str, UrlStr]:
    """Return ``(tag, download_url)`` for the latest matching GitHub asset."""
    tag, assets = _github_latest_release(repo)
    return tag, _match_asset_url(assets, pattern, "browser_download_url")


def newest_gitlab_asset(repo: str, pattern: str) -> Tuple[str, UrlStr]:
    """Return ``(tag, download_url)`` for the latest matching GitLab asset."""
    tag, assets = _gitlab_latest_release(repo)
    return tag, _match_asset_url(assets, pattern, "url")


# ----------------------------- winget -------------------------------------- #


def newest_winget_asset(
    pkg_id: str, arch: str = "x64"
) -> Tuple[str, UrlStr]:
    """Return ``(version, installer_url)`` from the winget-pkgs manifest on GitHub.

    Fetches the manifest YAML from the ``microsoft/winget-pkgs`` repo on
    GitHub and extracts the first ``InstallerUrl`` that matches *arch*.
    No YAML library is needed — we parse with simple regexes.
    """
    # Package IDs look like "Publisher.Name"; manifest path uses the first char.
    parts = pkg_id.split(".")
    if len(parts) < 2:
        raise ConfigError(f"winget_id must be Publisher.Name, got {pkg_id!r}")
    first_char = parts[0][0].lower()
    manifest_dir = "/".join(parts)

    # List version directories via GitHub API
    api = (
        f"https://api.github.com/repos/microsoft/winget-pkgs"
        f"/contents/manifests/{first_char}/{manifest_dir}"
    )
    resp = http_get(api, f"winget {pkg_id}")
    entries: list[dict[str, object]] = cast(list[dict[str, object]], resp.json())

    # Pick the highest version directory (semantic sort by splitting on '.')
    versions: list[str] = [
        str(e.get("name", ""))
        for e in entries
        if e.get("type") == "dir"
    ]
    if not versions:
        raise AssetNotFoundError(f"winget {pkg_id}: no version directories found")

    def _ver_key(v: str) -> list[int]:
        parts_list: list[int] = []
        for p in re.split(r"[.\-]", v):
            try:
                parts_list.append(int(p))
            except ValueError:
                parts_list.append(0)
        return parts_list

    latest = max(versions, key=_ver_key)
    logger.debug("winget %s: latest version %s", pkg_id, latest)

    # Fetch the installer manifest YAML
    yaml_name = f"{pkg_id}.installer.yaml"
    raw_url = (
        f"https://raw.githubusercontent.com/microsoft/winget-pkgs"
        f"/master/manifests/{first_char}/{manifest_dir}/{latest}/{yaml_name}"
    )
    yaml_text = fetch_html(raw_url)

    # Parse InstallerUrl entries paired with their Architecture.
    # The YAML structure has Installers: as a list of mappings.
    # We look for Architecture + InstallerUrl pairs.
    installer_blocks = re.split(r"(?m)^- ", yaml_text)
    for block in installer_blocks:
        arch_match = re.search(r"Architecture:\s*(\S+)", block)
        url_match = re.search(r"InstallerUrl:\s*(\S+)", block)
        if arch_match and url_match and arch_match.group(1).lower() == arch.lower():
            return latest, url_match.group(1)

    # Fallback: first InstallerUrl in the file regardless of arch
    fallback = re.search(r"InstallerUrl:\s*(\S+)", yaml_text)
    if fallback:
        return latest, fallback.group(1)

    raise AssetNotFoundError(f"winget {pkg_id} v{latest}: no InstallerUrl found")


# ----------------------------- chocolatey --------------------------------- #


def newest_choco_asset(pkg_id: str) -> Tuple[str, UrlStr]:
    """Return ``(version, download_url)`` by inspecting a Chocolatey package.

    Downloads the ``.nupkg`` for *pkg_id*, extracts the install script, and
    parses the ``$url64bit`` / ``$url64`` / ``$url`` variable to find the
    actual download URL.
    """
    # Step 1: query latest version from the Chocolatey v2 API (OData).
    odata = (
        f"https://community.chocolatey.org/api/v2/Packages()"
        f"?$filter=Id%20eq%20%27{pkg_id}%27%20and%20IsLatestVersion"
        f"&$select=Version"
    )
    resp = http_get(odata, f"choco {pkg_id}")
    version_match = re.search(
        r"<d:Version[^>]*>(.*?)</d:Version>", resp.text
    )
    if not version_match:
        raise AssetNotFoundError(f"choco {pkg_id}: no version in API response")
    version = version_match.group(1)
    logger.debug("choco %s: latest version %s", pkg_id, version)

    # Step 2: download the .nupkg (which is a ZIP).
    nupkg_url = f"https://community.chocolatey.org/api/v2/package/{pkg_id}/{version}"
    nupkg_resp = requests.get(
        nupkg_url, timeout=TIMEOUT, headers={"User-Agent": UA}
    )
    if nupkg_resp.status_code != 200:
        raise NetworkError(f"choco {pkg_id}: HTTP {nupkg_resp.status_code} for nupkg")

    # Step 3: extract chocolateyInstall.ps1 from the zip.
    try:
        with zipfile.ZipFile(io.BytesIO(nupkg_resp.content)) as zf:
            script: Optional[str] = None
            for name in zf.namelist():
                if name.lower().endswith("chocolateyinstall.ps1"):
                    script = zf.read(name).decode("utf-8-sig", errors="replace")
                    break
    except zipfile.BadZipFile as exc:
        raise AssetNotFoundError(f"choco {pkg_id}: bad nupkg: {exc}") from exc

    if script is None:
        raise AssetNotFoundError(f"choco {pkg_id}: no chocolateyInstall.ps1 in nupkg")

    # Step 4: parse download URL from the PowerShell script.
    # Common patterns: $url64bit = '...', $url64 = '...', $url = '...'
    for var_pattern in (
        r"\$url64(?:bit)?\s*=\s*['\"]([^'\"]+)['\"]",
        r"['\"]?url64(?:bit)?['\"]?\s*=\s*['\"]([^'\"]+)['\"]",
        r"\$url\s*=\s*['\"]([^'\"]+)['\"]",
        r"['\"]?url['\"]?\s*=\s*['\"]([^'\"]+)['\"]",
    ):
        m = re.search(var_pattern, script, re.I)
        if m:
            dl_url = m.group(1)
            if dl_url.startswith("http"):
                return version, dl_url

    raise AssetNotFoundError(
        f"choco {pkg_id} v{version}: could not extract download URL from install script"
    )


# ---------------------------------------------------------------------------
# Source dispatch
# ---------------------------------------------------------------------------


def fetch_latest(cfg: AppConfig) -> Tuple[Optional[str], UrlStr]:
    """Resolve the latest download URL (and optional version tag) for *cfg*."""
    if cfg.page_url is not None:
        return newest_page_asset(cfg.page_url, cfg.asset_regex or "")
    if cfg.gitlab_repo is not None:
        return newest_gitlab_asset(cfg.gitlab_repo, cfg.asset_regex or ".*")
    if cfg.github_repo is not None:
        return newest_github_asset(cfg.github_repo, cfg.asset_regex or ".*")
    if cfg.winget_id is not None:
        return newest_winget_asset(cfg.winget_id)
    if cfg.choco_id is not None:
        return newest_choco_asset(cfg.choco_id)
    if cfg.url is not None:
        return None, cfg.url
    raise AssertionError(f"AppConfig {cfg.name!r} has no source (should not happen)")


# ---------------------------------------------------------------------------
# HTML redirect following
# ---------------------------------------------------------------------------


def _meta_refresh_url(soup: BeautifulSoup, base: UrlStr) -> Optional[UrlStr]:
    """Extract the redirect URL from a ``<meta http-equiv=refresh>`` tag."""
    meta = soup.find("meta", attrs={"http-equiv": re.compile("^refresh$", re.I)})
    if not isinstance(meta, Tag):
        return None
    m = re.search(r"url=([^;]+)", str(meta.get("content", "")), re.I)
    return uparse.urljoin(base, m.group(1).strip()) if m else None


def _html_redirect_step(url: UrlStr, headers: dict[str, str]) -> Optional[UrlStr]:
    """If *url* returns an HTML page, resolve one redirect step; else ``None``."""
    try:
        head = requests.head(url, headers=headers, allow_redirects=True, timeout=TIMEOUT)
    except requests.RequestException:
        return None
    if "text/html" not in head.headers.get("Content-Type", "").lower():
        return None
    soup = BeautifulSoup(
        http_get(url, f"Indirect fetch {url}", headers=headers).text, "lxml"
    )
    next_url = _meta_refresh_url(soup, url)
    if next_url:
        return next_url
    anchor = soup.find("a", href=True)
    return uparse.urljoin(url, str(anchor["href"])) if isinstance(anchor, Tag) else None


def follow_redirects(url: UrlStr) -> UrlStr:
    """Chase up to five HTML redirect hops and return the final binary URL."""
    headers = {"User-Agent": UA}
    current = url
    for _ in range(5):
        nxt = _html_redirect_step(current, headers)
        if nxt is None or nxt == current:
            break
        current = nxt
    return current


# ---------------------------------------------------------------------------
# Download filename detection
# ---------------------------------------------------------------------------


def _filename_from_content_disposition(cd: str) -> Optional[str]:
    """Extract a filename from a ``Content-Disposition`` header value."""
    m = re.search(r'filename="?([^";]+)"?', cd)
    return m.group(1) if m else None


def _filename_from_url(url: UrlStr) -> str:
    """Derive a fallback filename from the URL path."""
    return Path(uparse.urlparse(url).path).name or f"download{int(time.time())}"


def filename_from_response(response: httpx.Response) -> str:
    """Best-effort filename: Content-Disposition → URL path → timestamp."""
    cd = response.headers.get("Content-Disposition")
    if cd:
        name = _filename_from_content_disposition(cd)
        if name:
            return name
    return _filename_from_url(str(response.url))


# ---------------------------------------------------------------------------
# Download resume state
# ---------------------------------------------------------------------------


def resume_state(dest: Path) -> Tuple[int, dict[str, str]]:
    """Return ``(bytes_already_on_disk, request_headers)`` for resumable download."""
    size = dest.stat().st_size if dest.exists() else 0
    hdrs: dict[str, str] = {"User-Agent": UA}
    if size:
        hdrs["Range"] = f"bytes={size}-"
    return size, hdrs


def download_total(response: httpx.Response, resume_pos: int) -> int:
    """Calculate the expected total file size from response headers."""
    total = int(response.headers.get("Content-Length", "0"))
    if resume_pos and response.status_code == 206:
        cr = response.headers.get("Content-Range", "")
        total = int(cr.split("/")[-1]) if "/" in cr else total + resume_pos
    elif resume_pos:
        total += resume_pos
    return total


# ---------------------------------------------------------------------------
# Download stream writing
# ---------------------------------------------------------------------------


def write_stream(
    response: httpx.Response, dest: Path, resume_pos: int, total: int
) -> None:
    """Stream *response* body into *dest*, showing a tqdm progress bar."""
    mode = "ab" if resume_pos else "wb"
    with open(dest, mode) as fh, tqdm(
        unit="B",
        unit_scale=True,
        desc=dest.name,
        leave=False,
        total=total or None,
        initial=resume_pos,
    ) as bar:
        for chunk in response.iter_bytes(65_536):
            fh.write(chunk)
            bar.update(len(chunk))


# ---------------------------------------------------------------------------
# Download context manager
# ---------------------------------------------------------------------------


@contextmanager
def download_file(
    url: UrlStr,
    dest_dir: Path,
    referer: Optional[str] = None,
) -> Generator[Path, None, None]:
    """Download *url* into *dest_dir*, yielding the resulting ``Path``.

    Supports HTTP range-resume, Content-Disposition filenames, and HTML
    redirect chains.  Deletes the partial file on any exception inside the
    ``with`` block.

    *referer*, when set, is sent as an HTTP ``Referer`` header (required by
    some CDNs such as AMD's ``drivers.amd.com``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved = follow_redirects(url)
    if resolved != url:
        logger.debug("HTML redirect: %s -> %s", url, resolved)

    dest = dest_dir / _filename_from_url(resolved)
    resume_pos, req_headers = resume_state(dest)
    if referer:
        req_headers["Referer"] = referer

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            with client.stream("GET", resolved, headers=req_headers) as response:
                if response.status_code not in {200, 206}:
                    dest.unlink(missing_ok=True)
                    raise DownloadError(f"HTTP {response.status_code} for {resolved}")
                # Resolve the true filename now that we have response headers.
                final_name = filename_from_response(response)
                if final_name != dest.name:
                    dest = dest_dir / final_name
                    resume_pos = dest.stat().st_size if dest.exists() else 0
                if "text/html" in response.headers.get("Content-Type", ""):
                    dest.unlink(missing_ok=True)
                    raise DownloadError("expected binary content, got HTML")
                write_stream(
                    response, dest, resume_pos, download_total(response, resume_pos)
                )
        except httpx.HTTPError as exc:
            dest.unlink(missing_ok=True)
            raise DownloadError(str(exc)) from exc

    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise DownloadError("Downloaded zero-byte file")

    try:
        yield dest
    except Exception:
        try:
            dest.unlink(missing_ok=True)
        except PermissionError:
            logger.warning("Could not delete %s (file in use)", dest)
        raise


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------


def _find_7z_cmd() -> Optional[str]:
    """Locate 7z on PATH or in standard Windows install directories."""
    for cmd in ("7z", "7za", "7zz"):
        if shutil.which(cmd):
            return cmd
    if sys.platform == "win32":
        for prog_dir in (
            Path("C:/Program Files/7-Zip"),
            Path("C:/Program Files (x86)/7-Zip"),
        ):
            exe = prog_dir / "7z.exe"
            if exe.exists():
                return str(exe)
    return None


def _run_7z_cli(cmd: str, archive: Path, dest: Path) -> None:
    """Extract *archive* to *dest* using the 7z command-line tool.

    Raises ``AppError`` if the process exits non-zero.
    """
    result = subprocess.run(
        [cmd, "x", str(archive), f"-o{dest}", "-y", "-bso0", "-bsp0"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AppError(
            f"{archive.name}: 7z CLI failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )


def _extract_zip(archive: Path, dest: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise AppError(f"{archive.name}: {exc}") from exc


def _extract_tar(archive: Path, dest: Path) -> None:
    try:
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    except tarfile.TarError as exc:
        raise AppError(f"{archive.name}: {exc}") from exc


def _extract_7z(archive: Path, dest: Path) -> None:
    """Extract a .7z file, falling back to the 7z CLI if py7zr fails."""
    try:
        import py7zr  # type: ignore[import]  # optional dep

        with py7zr.SevenZipFile(archive) as z:
            z.extractall(dest)
    except Exception as exc:
        cmd = _find_7z_cmd()
        if cmd:
            logger.debug("py7zr failed (%s); retrying with 7z CLI", exc)
            _run_7z_cli(cmd, archive, dest)
        else:
            raise AppError(
                f"{archive.name}: py7zr failed ({exc}) and 7z CLI not available"
            ) from exc


def _copy_plain_file(archive: Path, dest: Path) -> None:
    """Copy a non-archive file into *dest* (treating it as its own 'extraction')."""
    target = dest / archive.name
    if archive.resolve() == target.resolve():
        logger.debug("Source and destination are the same, skipping copy")
        return
    try:
        shutil.copy2(archive, target)
    except OSError as exc:
        raise AppError("destination in use or locked") from exc


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract *archive* (zip / tar / 7z / plain file) into *dest*."""
    logger.debug("Extracting %s -> %s", archive, dest)
    dest.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        _extract_zip(archive, dest)
    elif suffix in {".tar", ".gz", ".bz2", ".xz"} or archive.name.endswith(".tar.xz"):
        _extract_tar(archive, dest)
    elif suffix == ".7z":
        _extract_7z(archive, dest)
    else:
        _copy_plain_file(archive, dest)


# ---------------------------------------------------------------------------
# Folder / version management
# ---------------------------------------------------------------------------


def dest_name(app_name: str, tag: Optional[str]) -> str:
    """Build the destination folder name: ``'{name}_{tag}'`` or just ``name``."""
    return f"{app_name}_{tag}" if tag else app_name


def older_versions(root: Path, app_name: str, dest: Path) -> tuple[Path, ...]:
    """Return existing version folders for *app_name* that are not *dest*."""
    if not root.exists():
        return ()
    return tuple(
        p
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith(app_name) and p != dest
    )


def make_check_result(
    cfg: AppConfig, tag: Optional[str], url: UrlStr, root: Path
) -> CheckResult:
    """Build a ``CheckResult`` from a resolved tag + URL."""
    dest = root / dest_name(cfg.name, tag)
    older = older_versions(root, cfg.name, dest)
    if dest.exists():
        status = AppStatus.UP_TO_DATE
        current: Optional[Path] = dest
    else:
        status = AppStatus.UPDATE_AVAILABLE
        current = older[0] if older else None
    return CheckResult(
        cfg=cfg,
        status=status,
        tag=tag,
        download_url=url,
        dest_folder=dest,
        current_folder=current,
        older_folders=older,
    )


def failed_check(cfg: AppConfig, exc: Exception) -> CheckResult:
    """Build a ``CheckResult`` representing a failed version check."""
    logger.error("%s check failed: %s", cfg.name, exc)
    return CheckResult(
        cfg=cfg, status=AppStatus.CHECK_FAILED, error_message=str(exc)
    )


# ---------------------------------------------------------------------------
# Phase 1: check
# ---------------------------------------------------------------------------


def check_one(cfg: AppConfig, root: Path) -> CheckResult:
    """Resolve the latest release for *cfg* and return a ``CheckResult``."""
    try:
        tag, url = fetch_latest(cfg)
        return make_check_result(cfg, tag, url, root)
    except AppError as exc:
        return failed_check(cfg, exc)


def check_all(configs: List[AppConfig], root: Path) -> List[CheckResult]:
    """Run :func:`check_one` for every config entry, showing a live status."""
    results: List[CheckResult] = []
    with console.status("[bold cyan]Checking for updates...") as status:
        for i, cfg in enumerate(configs, 1):
            status.update(
                f"[bold cyan]Checking {cfg.name} ({i}/{len(configs)})..."
            )
            results.append(check_one(cfg, root))
    return results


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


_FIELD_REMAP: dict[str, str] = {
    "regex": "asset_regex",      # portable_regex / installer_regex → asset_regex
}


def _normalize_entry(item: dict[str, object], kind: str) -> Optional[dict[str, object]]:
    """Map ``{kind}_*`` fields to their base names; drop the other kind's fields.

    Returns ``None`` when the entry does not carry any fields for *kind*.
    """
    other = "installer" if kind == "portable" else "portable"
    normalized: dict[str, object] = {}
    has_kind_fields = False
    for k, v in item.items():
        if k.startswith(f"{kind}_"):
            base = k[len(kind) + 1:]
            normalized[_FIELD_REMAP.get(base, base)] = v
            has_kind_fields = True
        elif not k.startswith(f"{other}_"):
            normalized[k] = v
    return normalized if has_kind_fields else None


def _decode_json(text: str, cfg_path: Path) -> list:
    """Decode JSON5 text; raise ``ConfigError`` on parse failure."""
    try:
        raw = json5.loads(text)
    except json5.JSON5DecodeError as exc:
        line = getattr(exc, "lineno", "?")
        raise ConfigError(
            f"{cfg_path}: JSON error at line {line}: {exc}"
        ) from exc
    if not isinstance(raw, list):
        raise ConfigError(f"{cfg_path}: root must be a JSON array")
    return raw


def _build_app_config(entry: dict, idx: int, cfg_path: Path) -> AppConfig:
    """Construct an ``AppConfig`` from a normalised dict; raise ``ConfigError`` on bad fields."""
    try:
        return AppConfig(**entry)
    except TypeError as exc:
        raise ConfigError(f"{cfg_path} entry #{idx}: {exc}") from exc


def parse_config_text(text: str, cfg_path: Path, kind: str) -> List[AppConfig]:
    """Parse JSON5 text into a list of ``AppConfig`` objects for *kind*."""
    configs: List[AppConfig] = []
    for idx, item in enumerate(_decode_json(text, cfg_path), 1):
        if not isinstance(item, dict):
            raise ConfigError(f"{cfg_path} entry #{idx}: expected an object")
        normalized = _normalize_entry(item, kind)
        if normalized is not None:
            configs.append(_build_app_config(normalized, idx, cfg_path))
    return configs


def load_config(cfg_path: Path, kind: str) -> List[AppConfig]:
    """Read *cfg_path* and return ``AppConfig`` entries for *kind*."""
    logger.debug("Loading config %s (kind=%s)", cfg_path, kind)
    if not cfg_path.exists():
        raise ConfigError(f"Missing config file: {cfg_path}")
    configs = parse_config_text(cfg_path.read_text(encoding="utf-8"), cfg_path, kind)
    logger.info("Loaded %d %s definition(s)", len(configs), kind)
    return configs


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def safe_input(prompt: str) -> str:
    """Read a line of input; raise ``UserQuit`` on Ctrl+C, EOF, or 'q'/'quit'."""
    try:
        ans = input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold]Aborted.[/bold]")
        raise UserQuit()
    if ans.lower() in ("q", "quit"):
        console.print("[bold]Quitting.[/bold]")
        raise UserQuit()
    return ans


def yes_no(question: str, default: bool = True) -> bool:
    """Ask a yes/no question; return the answer as a bool."""
    hint = "Y/n/q" if default else "y/N/q"
    ans = safe_input(f"{question} ({hint}) ").lower()
    return default if not ans else ans in ("y", "yes")


# ---------------------------------------------------------------------------
# UI display
# ---------------------------------------------------------------------------


def _format_current_version(result: CheckResult) -> str:
    """Extract the version string from the current folder name (strip app prefix)."""
    if result.current_folder is None:
        return "---"
    prefix = result.cfg.name + "_"
    name = result.current_folder.name
    return name[len(prefix):] if name.startswith(prefix) else name


def _format_status_cell(result: CheckResult) -> str:
    """Format the status column for a ``CheckResult`` row."""
    if result.status == AppStatus.UP_TO_DATE:
        return "[green]Up to date[/green]"
    if result.status == AppStatus.UPDATE_AVAILABLE:
        return "[yellow]Update available[/yellow]"
    msg = (result.error_message or "unknown")[:60]
    return f"[red]Error: {msg}[/red]"


def build_check_table(results: List[CheckResult], title: str) -> Table:
    """Build a Rich table summarising *results*."""
    t = Table(title=title, show_lines=True)
    t.add_column("#", style="dim", width=4, justify="right")
    t.add_column("App", style="bold")
    t.add_column("Current", justify="center")
    t.add_column("Available", justify="center")
    t.add_column("Status", justify="center")
    for i, r in enumerate(results, 1):
        t.add_row(
            str(i),
            r.cfg.name,
            _format_current_version(r),
            r.tag or "---",
            _format_status_cell(r),
        )
    return t


def print_check_table(results: List[CheckResult], title: str) -> None:
    """Print the summary table to the console."""
    console.print(build_check_table(results, title))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def updatable(results: List[CheckResult]) -> List[CheckResult]:
    """Filter *results* to only those with an available update."""
    return [r for r in results if r.status == AppStatus.UPDATE_AVAILABLE]


def parse_indices(raw: str, max_n: int) -> Tuple[List[int], List[str]]:
    """Parse comma/space-separated 1-based indices.

    Returns ``(valid_zero_based_indices, warning_strings)``.
    """
    good: List[int] = []
    warns: List[str] = []
    for tok in re.split(r"[,\s]+", raw):
        if not tok:
            continue
        try:
            i = int(tok) - 1
            if 0 <= i < max_n:
                good.append(i)
            else:
                warns.append(f"out of range: {tok}")
        except ValueError:
            warns.append(f"not a number: {tok}")
    return good, warns


def prompt_specific(items: List[CheckResult]) -> List[CheckResult]:
    """Print a numbered list and let the user pick entries by number."""
    console.print("\nAvailable updates:")
    for i, r in enumerate(items, 1):
        console.print(f"  [bold]{i}[/bold]. {r.cfg.name}  ({r.tag or '?'})")
    raw = safe_input("\nEnter numbers (e.g. 1,3,5) or q to quit: ")
    if not raw:
        return []
    indices, warns = parse_indices(raw, len(items))
    for w in warns:
        console.print(f"[yellow]Ignored – {w}[/yellow]")
    return [items[i] for i in indices]


def prompt_selection(
    results: List[CheckResult],
    table_title: str,
    item_label: str,
) -> List[CheckResult]:
    """Show the summary table, then ask the user what to download."""
    print_check_table(results, table_title)
    available = updatable(results)
    if not available:
        console.print(f"\n[bold green]All {item_label} are up to date!")
        return []
    console.print(f"\n[bold]{len(available)} update(s) available.[/bold]")
    console.print(
        "  [bold]A[/bold] = Download all   "
        "[bold]S[/bold] = Select specific   "
        "[bold]N[/bold] = Skip   "
        "[bold]Q[/bold] = Quit"
    )
    choice = safe_input("\nYour choice [A/s/n/q]: ").lower()
    if choice in ("n", "no", "none"):
        return []
    if choice in ("s", "select"):
        return prompt_specific(available)
    return available


# ---------------------------------------------------------------------------
# Download orchestration
# ---------------------------------------------------------------------------


def _print_download_result(result: DownloadResult, verb: str) -> None:
    """Print a one-line success or failure message for a completed download."""
    check = result.check
    if result.success:
        console.print(
            f"[bold green]  {verb} {check.cfg.name} ({check.tag})"
            f" -> {check.dest_folder}"
        )
    else:
        console.print(
            f"[bold red]  Failed: {check.cfg.name}: {result.error_message}"
        )


def run_downloads(
    selected: List[CheckResult],
    download_fn: DownloadFn,
    aux_dir: Path,
    verb: str,
) -> List[DownloadResult]:
    """Call *download_fn* for each selected app; print progress and results."""
    results: List[DownloadResult] = []
    for i, check in enumerate(selected, 1):
        console.rule(
            f"[bold blue]Downloading {check.cfg.name} ({i}/{len(selected)})"
        )
        result = download_fn(check, aux_dir)
        results.append(result)
        _print_download_result(result, verb)
    return results


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def _collect_cleanable(
    results: List[DownloadResult],
) -> List[Tuple[str, Optional[Path], tuple[Path, ...]]]:
    """Return entries that succeeded and have old version folders to remove."""
    return [
        (dr.check.cfg.name, dr.check.dest_folder, dr.check.older_folders)
        for dr in results
        if dr.success and dr.check.older_folders
    ]


def _build_cleanup_table(
    entries: List[Tuple[str, Optional[Path], tuple[Path, ...]]],
) -> Table:
    """Build a Rich table listing old version folders."""
    t = Table(show_lines=False)
    t.add_column("App", style="bold")
    t.add_column("Old Folder(s)")
    t.add_column("New Folder", style="green")
    for name, dest, olders in entries:
        old_names = ", ".join(p.name for p in olders)
        dest_name_str = dest.name if dest else "?"
        t.add_row(name, old_names, dest_name_str)
    return t


def _delete_old_folders(
    entries: List[Tuple[str, Optional[Path], tuple[Path, ...]]],
) -> None:
    """Delete all old version folders listed in *entries*."""
    for _name, _dest, olders in entries:
        for p in olders:
            logger.debug("Deleting old version: %s", p)
            shutil.rmtree(p, ignore_errors=True)


def run_cleanup(results: List[DownloadResult]) -> None:
    """Offer to delete old version folders for successfully updated apps."""
    entries = _collect_cleanable(results)
    if not entries:
        return
    console.print("\n[bold]Old versions that can be removed:[/bold]")
    console.print(_build_cleanup_table(entries))
    if yes_no("\nDelete all old versions listed above?"):
        _delete_old_folders(entries)
        console.print("[green]Old versions deleted.[/green]")
    else:
        console.print("Old versions kept.")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------


def print_summary(
    dl_results: List[DownloadResult],
    all_results: List[CheckResult],
    dl_verb: str,
) -> None:
    """Print session-end statistics."""
    up_to_date = sum(1 for r in all_results if r.status == AppStatus.UP_TO_DATE)
    check_failed = sum(1 for r in all_results if r.status == AppStatus.CHECK_FAILED)
    ok = sum(1 for d in dl_results if d.success)
    failed = sum(1 for d in dl_results if not d.success)

    console.print("\n[bold]Session Summary:[/bold]")
    console.print(f"  Already up to date: {up_to_date}")
    if ok:
        console.print(f"  [green]{dl_verb}: {ok}[/green]")
    if failed:
        console.print(f"  [red]Failed to download: {failed}[/red]")
    if check_failed:
        console.print(f"  [red]Version check failed: {check_failed}[/red]")


# ---------------------------------------------------------------------------
# Workflow runner
# ---------------------------------------------------------------------------


def run_phases(
    configs: List[AppConfig],
    dl_dir: Path,
    download_fn: DownloadFn,
    *,
    table_title: str,
    item_label: str,
    dl_verb: str,
    aux_dir: Optional[Path] = None,
    cleanup_aux: bool = False,
) -> None:
    """Run the full update workflow (check → select → download → cleanup → summary).

    Parameters
    ----------
    configs:
        App definitions to process.
    dl_dir:
        Root directory where versioned app folders are created.
    download_fn:
        Callable ``(check, aux_dir) -> DownloadResult`` supplied by the caller.
        For portables this downloads to *aux_dir* then extracts; for installers
        it downloads directly to ``check.dest_folder``.
    table_title:
        Header for the Rich summary table.
    item_label:
        Human label used in "All {item_label} are up to date" messages.
    dl_verb:
        Past-tense verb for success messages ("Installed" / "Downloaded").
    aux_dir:
        Auxiliary directory passed to *download_fn* (e.g. a temp staging dir).
        Defaults to *dl_dir*.
    cleanup_aux:
        When ``True``, delete *aux_dir* after all downloads complete.
    """
    _aux = aux_dir if aux_dir is not None else dl_dir

    console.rule("[bold cyan]Phase 1: Checking for updates")
    all_results = check_all(configs, dl_dir)

    console.rule("[bold cyan]Phase 2: Review & Select")
    selected = prompt_selection(all_results, table_title, item_label)

    if not selected:
        console.print("[bold]Nothing to download.[/bold]")
        print_summary([], all_results, dl_verb)
        return

    console.rule("[bold cyan]Phase 3: Downloading")
    dl_results = run_downloads(selected, download_fn, _aux, dl_verb)

    if cleanup_aux and aux_dir is not None and aux_dir.exists():
        shutil.rmtree(aux_dir, ignore_errors=True)

    console.rule("[bold cyan]Phase 4: Cleanup")
    run_cleanup(dl_results)

    print_summary(dl_results, all_results, dl_verb)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def arg_parser(description: str) -> argparse.ArgumentParser:
    """Build a standard argument parser with ``--config`` and ``--download-dir``."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "-c", "--config",
        default="apps.json",
        type=Path,
        help="Path to apps.json (default: %(default)s)",
    )
    p.add_argument(
        "-d", "--download-dir",
        default=str(Path.cwd()),
        help="Root directory for versioned app folders (default: cwd)",
    )
    return p
