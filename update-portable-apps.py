"""update-portable-apps.py
Download and extract portable apps declared in ``apps.json``.
Runs on Python >=3.10.

Archives are extracted into versioned folders under --download-dir;
a temporary staging directory (.downloads/) is used and cleaned up
after each run.

3rd-party deps: ``requests``, ``httpx``, ``tqdm``, ``rich``, ``json5``,
``beautifulsoup4``, ``lxml``.  Optional: ``py7zr``, ``requests_html``,
``lxml_html_clean`` (headless scraping / 7z extraction).

Install once:
```
pip install -U requests httpx tqdm rich beautifulsoup4 lxml json5
pip install -U requests-html lxml_html_clean  # optional headless support
```
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

from core import (
    AppError,
    CheckResult,
    DownloadResult,
    arg_parser,
    download_file,
    extract_archive,
    load_config,
    run_phases,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = "grab-portables.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Backwards-compatibility alias used by any external code that imports this.
GrabPortablesError = AppError


# ---------------------------------------------------------------------------
# Portable-specific download function
# ---------------------------------------------------------------------------


def _download_and_extract(check: CheckResult, temp_dir: Path) -> DownloadResult:
    """Download *check*'s archive into *temp_dir*, then extract to dest_folder."""
    assert check.download_url is not None, f"{check.cfg.name}: no download URL"
    assert check.dest_folder is not None, f"{check.cfg.name}: no dest folder"
    try:
        with download_file(check.download_url, temp_dir, referer=check.cfg.referer) as archive:
            extract_archive(archive, check.dest_folder)
        return DownloadResult(check=check, success=True)
    except AppError as exc:
        return DownloadResult(check=check, success=False, error_message=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Program entry-point."""
    p = arg_parser("Download & install portable apps declared in apps.json")
    p.set_defaults(download_dir=str(Path.cwd() / "PortableApps"))
    args = p.parse_args(argv)

    dl_dir = Path(args.download_dir)

    try:
        configs = load_config(Path(args.config), kind="portable")
    except AppError as exc:
        logging.getLogger(__name__).critical("Fatal error: %s", exc)
        sys.exit(1)

    run_phases(
        configs,
        dl_dir,
        download_fn=_download_and_extract,
        table_title="Portable Apps Update Summary",
        item_label="apps",
        dl_verb="Installed",
        aux_dir=dl_dir / ".downloads",
        cleanup_aux=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        from core import console
        console.print("\n[bold]Aborted.[/bold]")
        sys.exit(0)
