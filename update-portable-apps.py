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
* **Batched workflow** - checks all apps first, lets user choose, then
  downloads and offers cleanup.

3rd-party deps: ``requests``, ``httpx``, ``tqdm``, ``rich``, ``py7zr``, ``json5``,
``beautifulsoup4``, ``lxml`` and ``requests_html`` + ``lxml_html_clean`` (optional
for JS-driven pages).

Install once:
```
pip install -U requests httpx tqdm rich py7zr beautifulsoup4 lxml json5
pip install -U requests-html lxml_html_clean  # optional headless support
```
"""

from __future__ import annotations

import argparse
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
from rich.table import Table
from tqdm import tqdm

__all__: Sequence[str] = (
    "AppConfig",
    "AppStatus",
    "CheckResult",
    "DownloadResult",
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
HEADLESS_WAIT: Final[float] = 2.0  # seconds for JS to execute

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


class UserQuit(SystemExit):
    """Raised when the user chooses to quit."""

    def __init__(self) -> None:
        super().__init__(0)


class AppStatus(Enum):
    """Outcome of the check phase for a single app."""

    UP_TO_DATE = auto()
    UPDATE_AVAILABLE = auto()
    CHECK_FAILED = auto()


# ---------------------------------------------------------------------------
# Data model
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


@dataclass(slots=True, frozen=True)
class CheckResult:
    """Result of resolving the latest version for one app."""

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
    """Result of downloading and extracting one app."""

    check: CheckResult
    success: bool
    error_message: Optional[str] = None


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


def _render_with_headless(page_url: UrlStr) -> str:
    """Render ``page_url`` using a headless browser and return HTML."""
    if not (find_spec("requests_html") and find_spec("lxml_html_clean")):
        raise NetworkError(
            "Headless scraping requires requests_html and lxml_html_clean"
        )

    from requests_html import HTML, HTMLResponse, HTMLSession

    with HTMLSession() as session:
        resp: HTMLResponse = session.get(
            page_url, headers={"User-Agent": UA}, timeout=TIMEOUT
        )
        html_obj: HTML = resp.html
        try:
            html_obj.render(timeout=TIMEOUT, sleep=HEADLESS_WAIT)
        except Exception as exc:  # pragma: no cover - network/browser
            raise NetworkError(f"Headless fetch {page_url}: {exc}") from exc
        html: str = html_obj.html
    return html


def newest_direct_asset(page_url: UrlStr, pattern: str) -> Tuple[str, UrlStr]:
    """Scrape *page_url*, parse ``<a href>`` links and return ``(version?, url)``.

    First attempts a simple ``requests`` fetch.  If no matching link is found,
    falls back to a headless browser via ``requests_html`` to render dynamic
    content before parsing.
    """

    def _extract(html: str) -> Optional[Tuple[str, UrlStr]]:
        soup = BeautifulSoup(html, "lxml")

        base_tag = soup.find("base", href=True)
        if isinstance(base_tag, Tag):
            href_val = base_tag.get("href")
            base_href: str = str(href_val) if href_val else ""
            effective_base: UrlStr = uparse.urljoin(page_url, base_href)
        else:
            effective_base = page_url

        rx: re.Pattern[str] = re.compile(pattern, re.I)
        for a in soup.find_all("a", href=True):
            if not isinstance(a, Tag):
                continue
            raw_href = str(a["href"])
            url = uparse.urljoin(effective_base, raw_href)
            scheme = uparse.urlparse(url).scheme.lower()
            if scheme not in ("http", "https"):
                continue
            match = rx.search(url) or rx.search(a.get_text(strip=True))
            if match:
                version = match.group(1) if match.lastindex and match.lastindex >= 1 else ""
                return version, url
        return None

    logger.debug(f"direct download page {page_url}")
    resp: requests.Response = http_get(
        page_url, f"Page fetch {page_url}", headers={"User-Agent": UA}
    )
    result = _extract(resp.text)
    if result:
        return result

    logger.debug("simple scrape failed, trying headless browser for %s", page_url)
    html = _render_with_headless(page_url)
    result = _extract(html)
    if result:
        return result

    raise AssetNotFoundError(f"No link in {page_url} matches /{pattern}/i")


# ----------------------------- Download / extract ------------------------- #


def _follow_html_redirect(url: UrlStr) -> UrlStr:
    """Resolve HTML indirections like ``<meta refresh>`` or lone links."""

    headers: dict[str, str] = {"User-Agent": UA}
    current: UrlStr = url
    for _ in range(5):
        try:
            head = requests.head(
                current, headers=headers, allow_redirects=True, timeout=TIMEOUT
            )
        except requests.RequestException:
            break
        ctype: str = head.headers.get("Content-Type", "")
        if "text/html" not in ctype.lower():
            return current
        resp: requests.Response = http_get(
            current, f"Indirect fetch {current}", headers=headers
        )
        soup = BeautifulSoup(resp.text, "lxml")
        meta = soup.find("meta", attrs={"http-equiv": re.compile("^refresh$", re.I)})
        if isinstance(meta, Tag):
            content_attr = meta.get("content", "")
            match = re.search(r"url=([^;]+)", str(content_attr), flags=re.I)
            if match:
                current = uparse.urljoin(current, match.group(1).strip())
                continue
        anchor = soup.find("a", href=True)
        if isinstance(anchor, Tag):
            current = uparse.urljoin(current, str(anchor["href"]))
            continue
        break
    return current


def _filename_from_response(response: httpx.Response) -> str:
    """Derive a filename from *response* headers or URL."""
    cd: Optional[str] = response.headers.get("Content-Disposition")
    if cd is not None:
        match: Optional[re.Match[str]] = re.search(
            r"filename=\"?([^\";]+)\"?", cd
        )
        if match:
            return match.group(1)

    name: str = Path(uparse.urlparse(str(response.url)).path).name
    return name or f"download{int(time.time())}"


@contextmanager
def download(url: UrlStr, download_dir: Path) -> Generator[Path, None, None]:
    """Download *url* into *download_dir*; yields Path."""
    download_dir.mkdir(parents=True, exist_ok=True)

    resolved: UrlStr = _follow_html_redirect(url)
    if resolved != url:
        logger.debug("resolved indirect download %s -> %s", url, resolved)

    initial_name: str = Path(uparse.urlparse(resolved).path).name or f"download{int(time.time())}"
    dest: Path = download_dir / initial_name
    resume_pos: int = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {"User-Agent": UA}
    if resume_pos:
        headers["Range"] = f"bytes={resume_pos}-"

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            with client.stream("GET", resolved, headers=headers) as response:
                if response.status_code not in {200, 206}:
                    dest.unlink(missing_ok=True)
                    raise DownloadError(f"HTTP {response.status_code} for {resolved}")
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
                with open(dest, mode) as file, tqdm(
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                    total=total or None,
                    initial=resume_pos,
                ) as bar:
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
        try:
            dest.unlink(missing_ok=True)
        except PermissionError:
            logger.warning("Could not delete %s (file in use)", dest)
        raise


def _find_7z() -> Optional[str]:
    """Locate the 7z command-line executable."""
    for cmd in ("7z", "7za", "7zz"):
        if shutil.which(cmd) is not None:
            return cmd
    # Check standard Windows install locations
    if sys.platform == "win32":
        for prog_dir in (
            Path("C:/Program Files/7-Zip"),
            Path("C:/Program Files (x86)/7-Zip"),
        ):
            exe = prog_dir / "7z.exe"
            if exe.exists():
                return str(exe)
    return None


def _extract_7z_cli(archive: Path, dest: Path) -> bool:
    """Try extracting a .7z archive using the 7z command-line tool.

    Returns ``True`` on success, ``False`` if 7z is not available.
    Raises ``GrabPortablesError`` if 7z is available but extraction fails.
    """
    cmd = _find_7z()
    if cmd is None:
        return False
    result = subprocess.run(
        [cmd, "x", str(archive), f"-o{dest}", "-y", "-bso0", "-bsp0"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    raise GrabPortablesError(
        f"{archive.name}: 7z CLI failed (exit {result.returncode}): "
        f"{result.stderr.strip()}"
    )


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
        except (py7zr.Bad7zFile, Exception) as exc:
            # py7zr doesn't support some compression methods (e.g. BCJ2).
            # Fall back to the 7z CLI if available.
            if _extract_7z_cli(archive, dest):
                logger.debug("py7zr failed (%s), used 7z CLI instead", exc)
            else:
                raise GrabPortablesError(
                    f"{archive.name}: py7zr failed ({exc}) and 7z CLI not available"
                ) from exc
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


def _safe_input(prompt: str) -> str:
    """Read input, raising ``UserQuit`` on Ctrl+C or 'q'/'quit'."""
    try:
        ans: str = input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold]Aborted.[/bold]")
        raise UserQuit()
    if ans.lower() in ("q", "quit"):
        console.print("[bold]Quitting.[/bold]")
        raise UserQuit()
    return ans


def prompt_yes_no(question: str, default: bool = True) -> bool:
    default_txt: str = "Y/n/q" if default else "y/N/q"
    ans = _safe_input(f"{question} ({default_txt}) ").lower()
    if not ans:
        return default
    return ans in {"y", "yes"}


# ---------------------------------------------------------------------------
# Phase 1: Check for updates
# ---------------------------------------------------------------------------


def resolve_latest(cfg: AppConfig, root: Path) -> CheckResult:
    """Resolve the newest version of *cfg* and classify its update status."""
    try:
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
        else:  # pragma: no cover - validation prevents
            assert_never(cast(Never, cfg))

        folder_name: str = f"{cfg.name}_{tag}" if tag else cfg.name
        dest: Path = root / folder_name

        older: tuple[Path, ...] = ()
        if root.exists():
            older = tuple(
                p
                for p in root.iterdir()
                if p.is_dir() and p.name.startswith(cfg.name) and p != dest
            )

        current: Optional[Path]
        if dest.exists():
            status = AppStatus.UP_TO_DATE
            current = dest
        else:
            status = AppStatus.UPDATE_AVAILABLE
            current = older[0] if older else None

        return CheckResult(
            cfg=cfg,
            status=status,
            tag=tag,
            download_url=dl_url,
            dest_folder=dest,
            current_folder=current,
            older_folders=older,
        )
    except GrabPortablesError as exc:
        logger.error("%s check failed: %s", cfg.name, exc)
        return CheckResult(
            cfg=cfg,
            status=AppStatus.CHECK_FAILED,
            error_message=str(exc),
        )


def check_all(configs: List[AppConfig], root: Path) -> List[CheckResult]:
    """Check all apps for available updates."""
    results: List[CheckResult] = []
    with console.status("[bold cyan]Checking for updates...") as status:
        for i, cfg in enumerate(configs, 1):
            status.update(f"[bold cyan]Checking {cfg.name} ({i}/{len(configs)})...")
            results.append(resolve_latest(cfg, root))
    return results


# ---------------------------------------------------------------------------
# Phase 2: User selection
# ---------------------------------------------------------------------------


def display_summary_table(results: List[CheckResult]) -> None:
    """Display a rich table summarizing check results."""
    table = Table(title="Portable Apps Update Summary", show_lines=True)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("App", style="bold")
    table.add_column("Current", justify="center")
    table.add_column("Available", justify="center")
    table.add_column("Status", justify="center")

    for idx, r in enumerate(results, 1):
        current_ver: str = "---"
        if r.current_folder is not None:
            prefix = r.cfg.name + "_"
            if r.current_folder.name.startswith(prefix):
                current_ver = r.current_folder.name[len(prefix):]
            else:
                current_ver = r.current_folder.name

        avail_ver: str = r.tag if r.tag else "---"

        if r.status == AppStatus.UP_TO_DATE:
            status_str = "[green]Up to date[/green]"
        elif r.status == AppStatus.UPDATE_AVAILABLE:
            status_str = "[yellow]Update available[/yellow]"
        else:
            msg = r.error_message or "unknown"
            status_str = f"[red]Error: {msg[:60]}[/red]"

        table.add_row(str(idx), r.cfg.name, current_ver, avail_ver, status_str)

    console.print(table)


def _prompt_specific_selection(updatable: List[CheckResult]) -> List[CheckResult]:
    """Show numbered list and let user pick by entering numbers."""
    console.print("\nAvailable updates:")
    for i, r in enumerate(updatable, 1):
        console.print(f"  [bold]{i}[/bold]. {r.cfg.name}  ({r.tag or '?'})")

    raw: str = _safe_input(
        "\nEnter numbers separated by commas/spaces (e.g. 1,3,5), or q to quit: "
    )

    if not raw:
        return []

    selected: List[CheckResult] = []
    for token in re.split(r"[,\s]+", raw):
        try:
            idx = int(token) - 1
            if 0 <= idx < len(updatable):
                selected.append(updatable[idx])
            else:
                console.print(f"[yellow]Ignoring out-of-range number: {token}[/yellow]")
        except ValueError:
            console.print(f"[yellow]Ignoring invalid input: {token}[/yellow]")

    return selected


def prompt_selection(results: List[CheckResult]) -> List[CheckResult]:
    """Present summary and let user choose which apps to download."""
    display_summary_table(results)

    updatable: List[CheckResult] = [
        r for r in results if r.status == AppStatus.UPDATE_AVAILABLE
    ]

    if not updatable:
        console.print("\n[bold green]All apps are up to date!")
        return []

    console.print(
        f"\n[bold]{len(updatable)} update(s) available.[/bold]"
    )
    console.print("  [bold]A[/bold] = Download all updates")
    console.print("  [bold]S[/bold] = Select specific apps")
    console.print("  [bold]N[/bold] = Skip all downloads")
    console.print("  [bold]Q[/bold] = Quit")

    choice: str = _safe_input("\nYour choice [A/s/n/q]: ").lower()

    if choice in ("n", "no", "none"):
        return []
    if choice in ("s", "select"):
        return _prompt_specific_selection(updatable)
    # Default: download all
    return updatable


# ---------------------------------------------------------------------------
# Phase 3+4: Download & Extract
# ---------------------------------------------------------------------------


def download_and_extract(check: CheckResult, temp_dir: Path) -> DownloadResult:
    """Download and extract a single app.

    Uses the existing ``download()`` context manager with extraction inside the
    ``with`` block so archive cleanup semantics are preserved.
    """
    assert check.download_url is not None, f"{check.cfg.name}: no download URL"
    assert check.dest_folder is not None, f"{check.cfg.name}: no dest folder"

    try:
        with download(check.download_url, temp_dir) as archive:
            extract_archive(archive, check.dest_folder)
            return DownloadResult(check=check, success=True)
    except GrabPortablesError as exc:
        logger.error("%s download/extract failed: %s", check.cfg.name, exc)
        return DownloadResult(
            check=check, success=False, error_message=str(exc)
        )


def download_selected(
    selected: List[CheckResult], download_dir: Path
) -> List[DownloadResult]:
    """Download and extract all selected apps."""
    results: List[DownloadResult] = []
    temp_dl_dir: Path = download_dir / ".downloads"

    for i, check in enumerate(selected, 1):
        console.rule(f"[bold blue]Downloading {check.cfg.name} ({i}/{len(selected)})")
        result = download_and_extract(check, temp_dl_dir)
        results.append(result)

        if result.success:
            console.print(
                f"[bold green]  Installed {check.cfg.name} ({check.tag})"
                f" -> {check.dest_folder}"
            )
        else:
            console.print(
                f"[bold red]  Failed: {check.cfg.name}: {result.error_message}"
            )

    # Clean up temp downloads directory
    if temp_dl_dir.exists():
        shutil.rmtree(temp_dl_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Phase 5: Cleanup old versions
# ---------------------------------------------------------------------------


def cleanup_old_versions(download_results: List[DownloadResult]) -> None:
    """Offer to delete old versions of successfully updated apps."""
    cleanable: List[tuple[str, Optional[Path], tuple[Path, ...]]] = []
    for dr in download_results:
        if dr.success and dr.check.older_folders:
            cleanable.append(
                (dr.check.cfg.name, dr.check.dest_folder, dr.check.older_folders)
            )

    if not cleanable:
        return

    console.print("\n[bold]Old versions that can be removed:[/bold]")
    table = Table(show_lines=False)
    table.add_column("App", style="bold")
    table.add_column("Old Folder(s)")
    table.add_column("New Folder", style="green")
    for name, dest, olders in cleanable:
        old_names = ", ".join(p.name for p in olders)
        table.add_row(name, old_names, dest.name if dest else "?")
    console.print(table)

    if prompt_yes_no("\nDelete all old versions listed above?", default=True):
        for _name, _dest, olders in cleanable:
            for p in olders:
                logger.debug("Deleting old version: %s", p)
                shutil.rmtree(p, ignore_errors=True)
        console.print("[green]Old versions deleted.[/green]")
    else:
        console.print("Old versions kept.")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------


def display_final_summary(
    download_results: List[DownloadResult],
    all_results: List[CheckResult],
) -> None:
    """Print a final summary of the session."""
    up_to_date = sum(1 for r in all_results if r.status == AppStatus.UP_TO_DATE)
    check_failed = sum(1 for r in all_results if r.status == AppStatus.CHECK_FAILED)
    downloaded_ok = sum(1 for d in download_results if d.success)
    downloaded_fail = sum(1 for d in download_results if not d.success)

    console.print("\n[bold]Session Summary:[/bold]")
    console.print(f"  Already up to date: {up_to_date}")
    if downloaded_ok:
        console.print(f"  [green]Successfully updated: {downloaded_ok}[/green]")
    if downloaded_fail:
        console.print(f"  [red]Failed to update: {downloaded_fail}[/red]")
    if check_failed:
        console.print(f"  [red]Version check failed: {check_failed}[/red]")


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
        configs: List[AppConfig] = load_config(args.config)
    except GrabPortablesError as exc:
        logger.critical("Fatal error: %s", exc)
        sys.exit(1)

    # Phase 1: Check all apps for updates
    console.rule("[bold cyan]Phase 1: Checking for updates")
    all_results: List[CheckResult] = check_all(configs, download_dir)

    # Phase 2: User selection
    console.rule("[bold cyan]Phase 2: Review & Select")
    selected: List[CheckResult] = prompt_selection(all_results)

    if not selected:
        console.print("[bold]Nothing to download.[/bold]")
        display_final_summary([], all_results)
        return

    # Phase 3+4: Download and extract
    console.rule("[bold cyan]Phase 3: Downloading & Installing")
    download_results: List[DownloadResult] = download_selected(selected, download_dir)

    # Phase 5: Cleanup old versions
    console.rule("[bold cyan]Phase 4: Cleanup")
    cleanup_old_versions(download_results)

    # Final summary
    display_final_summary(download_results, all_results)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold]Aborted.[/bold]")
        sys.exit(0)
