"""update-installers.py
Download installer (non-portable) versions of apps declared in
``apps.json``.  Runs on Python >=3.10.

Downloaded files are kept as-is inside ``{name}_{tag}/`` folders;
no archive extraction is performed.

3rd-party deps: ``requests``, ``httpx``, ``tqdm``, ``rich``, ``json5``,
``beautifulsoup4``, ``lxml``.  Optional: ``requests_html``,
``lxml_html_clean`` (headless scraping).

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
    load_config,
    run_phases,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = "grab-installers.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Backwards-compatibility alias.
GrabInstallersError = AppError


# ---------------------------------------------------------------------------
# Installer-specific download function (no extraction)
# ---------------------------------------------------------------------------


def _download_only(check: CheckResult, _aux: Path) -> DownloadResult:
    """Download *check*'s file directly into its dest_folder; no extraction."""
    assert check.download_url is not None, f"{check.cfg.name}: no download URL"
    assert check.dest_folder is not None, f"{check.cfg.name}: no dest folder"
    try:
        with download_file(check.download_url, check.dest_folder):
            pass  # file stays on normal exit
        return DownloadResult(check=check, success=True)
    except AppError as exc:
        return DownloadResult(check=check, success=False, error_message=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Program entry-point."""
    args = arg_parser(
        "Download installer apps declared in apps.json"
    ).parse_args(argv)

    dl_dir = Path(args.download_dir)

    try:
        configs = load_config(Path(args.config), kind="installer")
    except AppError as exc:
        logging.getLogger(__name__).critical("Fatal error: %s", exc)
        sys.exit(1)

    run_phases(
        configs,
        dl_dir,
        download_fn=_download_only,
        table_title="Installers Update Summary",
        item_label="installers",
        dl_verb="Downloaded",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        from core import console
        console.print("\n[bold]Aborted.[/bold]")
        sys.exit(0)
