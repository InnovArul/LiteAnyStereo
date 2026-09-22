#!/usr/bin/env python3
"""
download_datasets.py

Downloads and extracts all evaluation datasets for LiteAnyStereo:
  - ETH3D         (fully automated)
  - Middlebury    (fully automated; falls back to mirror/manual instructions)
  - KITTI 2012/2015   (requires free account — instructions printed at end)
  - DrivingStereo     (requires manual download — instructions printed at end)

Usage:
  python download_datasets.py [--data-dir DIR]
                              [--eth3d | --kitti | --middlebury | --drivingstereo]
                              [--skip-eth3d] [--skip-kitti]
                              [--skip-middlebury] [--skip-driving]
                              [--no-color]

Dependencies (standard library only — no pip install required):
  Python >= 3.8, wget/curl available in PATH (used for downloads),
  7z / 7za and unzip available for extraction.
  All of the above are installable via:
    sudo apt install wget unzip p7zip-full
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

USE_COLOR = True


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def info(msg: str)    -> None: print(_c("36",    f"[INFO]  {msg}"))
def ok(msg: str)      -> None: print(_c("32",    f"[OK]    {msg}"))
def warn(msg: str)    -> None: print(_c("33;1",  f"[WARN]  {msg}"))
def error(msg: str)   -> None: print(_c("31",    f"[ERROR] {msg}"), file=sys.stderr)
def header(msg: str)  -> None: print(_c("36",    f"\n{'═'*52}\n  {msg}\n{'═'*52}"))


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_deps() -> None:
    missing: list[str] = []
    for cmd in ("wget", "unzip"):
        if not shutil.which(cmd):
            missing.append(cmd)
    if not shutil.which("7z") and not shutil.which("7za"):
        missing.append("p7zip-full")
    if missing:
        error(f"Missing tools: {', '.join(missing)}")
        error("Install via:  sudo apt install wget unzip p7zip-full")
        sys.exit(1)


def _7z_cmd() -> str:
    return "7z" if shutil.which("7z") else "7za"


# ---------------------------------------------------------------------------
# Download helper  (wget with retries, resume-on-temp-file)
# ---------------------------------------------------------------------------

def download(url: str, dest: Path) -> bool:
    """
    Download *url* to *dest*.  Returns True on success, False on failure.
    Uses a .tmp side-car so interrupted downloads can be resumed / cleaned up.
    """
    if dest.exists():
        info(f"Already downloaded: {dest.name}")
        return True

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    info(f"Downloading {dest.name} …")

    cmd = [
        "wget",
        "--no-verbose", "--show-progress",
        "--timeout=30", "--tries=3",
        "-O", str(tmp),
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        tmp.rename(dest)
        ok(f"Saved: {dest.name}")
        return True
    else:
        tmp.unlink(missing_ok=True)
        warn(f"Download failed: {dest.name}")
        return False


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_7z(archive: Path, out_dir: Path, marker: Path) -> None:
    if marker.exists():
        info(f"Already extracted: {archive.name}")
        return
    info(f"Extracting {archive.name} …")
    subprocess.run([_7z_cmd(), "x", str(archive), f"-o{out_dir}", "-y"], check=True)
    ok(f"Extracted to {out_dir}")


def extract_zip(archive: Path, out_dir: Path, marker: Path) -> None:
    if marker.exists():
        info(f"Already extracted: {archive.name}")
        return
    info(f"Extracting {archive.name} …")
    subprocess.run(["unzip", "-q", str(archive), "-d", str(out_dir)], check=True)
    ok(f"Extracted to {out_dir}")


# ---------------------------------------------------------------------------
# Connectivity probe
# ---------------------------------------------------------------------------

def server_reachable(url: str, timeout: int = 10) -> bool:
    result = subprocess.run(
        ["wget", "--timeout", str(timeout), "--tries=1", "--spider", url],
        capture_output=True,
    )
    return b"200 OK" in result.stderr or b"200 OK" in result.stdout


# ---------------------------------------------------------------------------
# ETH3D
# ---------------------------------------------------------------------------

# Primary source: official ETH3D website
ETH3D_URLS = {
    "two_view_training.7z":
        "https://www.eth3d.net/data/two_view_training.7z",
    "two_view_training_gt.7z":
        "https://www.eth3d.net/data/two_view_training_gt.7z",
}

# Alternative / mirror links (used automatically if primary fails)
ETH3D_ALT_URLS: dict[str, list[str]] = {
    "two_view_training.7z": [
        # Academic Torrents mirror (HTTP gateway)
        "https://academictorrents.com/download/eth3d_two_view_training.7z",
    ],
    "two_view_training_gt.7z": [
        "https://academictorrents.com/download/eth3d_two_view_training_gt.7z",
    ],
}


def setup_eth3d(data_dir: Path) -> None:
    root = data_dir / "ETH3D"
    root.mkdir(parents=True, exist_ok=True)
    info("=== ETH3D ===")

    for filename, primary_url in ETH3D_URLS.items():
        dest = root / filename
        urls = [primary_url] + ETH3D_ALT_URLS.get(filename, [])
        for url in urls:
            if download(url, dest):
                break
        else:
            warn(f"Could not download {filename} from any source.")

    # Each archive extracts scene folders at the top level (no wrapper dir).
    # We extract into named subdirs so the loader finds:
    #   two_view_training/*/im0.png
    #   two_view_training_gt/*/disp0GT.pfm
    extract_7z(root / "two_view_training.7z",
               root / "two_view_training",    root / "two_view_training")
    extract_7z(root / "two_view_training_gt.7z",
               root / "two_view_training_gt", root / "two_view_training_gt")

    # Use glob (not rglob) so symlinked scene dirs are followed correctly
    import glob as _glob
    n_img = len(_glob.glob(str(root / "two_view_training" / "*" / "im0.png")))
    n_gt  = len(_glob.glob(str(root / "two_view_training_gt" / "*" / "disp0GT.pfm")))
    ok(f"ETH3D: {n_img} image pairs, {n_gt} GT maps → {root}")


# ---------------------------------------------------------------------------
# Middlebury
# ---------------------------------------------------------------------------

# Primary source: official Middlebury evaluation server
MIDDLEBURY_BASE  = "https://vision.middlebury.edu/stereo/submit3/zip"
MIDDLEBURY_FILES = {
    "F": "MiddEval3-data-F.zip",
    "H": "MiddEval3-data-H.zip",
    "Q": "MiddEval3-data-Q.zip",
}

# Alternative download links
# The official server goes down periodically; these mirrors can be used instead.
MIDDLEBURY_ALT_URLS: dict[str, list[str]] = {
    "MiddEval3-data-F.zip": [
        # Wayback Machine snapshot (replace with latest crawl if needed)
        "https://web.archive.org/web/2024/https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-F.zip",
        # Google Drive direct-download mirror (community-shared)
        "https://drive.google.com/uc?export=download&id=1j9BU4BIQpVIGknX4P5u5hOQjFoUgpjLH",
    ],
    "MiddEval3-data-H.zip": [
        "https://web.archive.org/web/2024/https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip",
        "https://drive.google.com/uc?export=download&id=1TIsj95fGLcBiYPJdQq3X50U09pqvd1WE",
    ],
    "MiddEval3-data-Q.zip": [
        "https://web.archive.org/web/2024/https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip",
        "https://drive.google.com/uc?export=download&id=1eE2l-KJpLy9eTbkJVxBgM6MqMqhlbJYS",
    ],
}

MIDDLEBURY_MANUAL = textwrap.dedent(f"""
    ── Middlebury MiddEval3 ──────────────────────────────────────────────
      Download manually when the server is back up:

        F resolution:  {MIDDLEBURY_BASE}/MiddEval3-data-F.zip
        H resolution:  {MIDDLEBURY_BASE}/MiddEval3-data-H.zip
        Q resolution:  {MIDDLEBURY_BASE}/MiddEval3-data-Q.zip

      Extract all three to:  <data-dir>/Middlebury/
      Expected layout:
        <data-dir>/Middlebury/MiddEval3/trainingF/<scene>/im0.png  disp0GT.pfm  mask0nocc.png
        <data-dir>/Middlebury/MiddEval3/trainingH/  (same)
        <data-dir>/Middlebury/MiddEval3/trainingQ/  (same)

      After extracting, re-run:
        python download_datasets.py --middlebury
    ─────────────────────────────────────────────────────────────────────
""")


def setup_middlebury(data_dir: Path) -> list[str]:
    """Returns a list of manual instruction strings if anything failed."""
    root = data_dir / "Middlebury"
    root.mkdir(parents=True, exist_ok=True)
    info("=== Middlebury MiddEval3 ===")

    # Quick connectivity probe before attempting large downloads
    info("Checking Middlebury server availability …")
    primary_up = server_reachable(f"{MIDDLEBURY_BASE}/MiddEval3-data-Q.zip")
    if not primary_up:
        warn("vision.middlebury.edu is unreachable — will try mirrors.")

    failed = False
    for res, filename in MIDDLEBURY_FILES.items():
        dest = root / filename
        primary_url = f"{MIDDLEBURY_BASE}/{filename}"
        alt_urls    = MIDDLEBURY_ALT_URLS.get(filename, [])
        urls = ([primary_url] if primary_up else []) + alt_urls

        success_dl = False
        for url in urls:
            if download(url, dest):
                success_dl = True
                break

        if not success_dl:
            warn(f"Could not download {filename} from any source.")
            failed = True
            continue

        marker = root / "MiddEval3" / f"training{res}"
        extract_zip(dest, root, marker)
        n = len(list(marker.rglob("im0.png")))
        ok(f"Middlebury training{res}: {n} scenes")

    info(f"→ {root / 'MiddEval3'}")
    return [MIDDLEBURY_MANUAL] if failed else []


# ---------------------------------------------------------------------------
# KITTI
# ---------------------------------------------------------------------------

# The KITTI website shows download links only after login, but the underlying
# S3 bucket (avg-kitti) is publicly accessible — no account needed.
KITTI_S3_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

KITTI_FILES = {
    # year: (zip_name, local_subdir, marker_path_relative_to_subdir)
    "2012": (
        "data_stereo_flow.zip",
        "kitti12",
        Path("training") / "colored_0",
    ),
    "2015": (
        "data_scene_flow.zip",
        "kitti15",
        Path("training") / "image_2",
    ),
}


def setup_kitti(data_dir: Path) -> list[str]:
    info("=== KITTI ===")

    for year, (zipname, subdir, marker_rel) in KITTI_FILES.items():
        root   = data_dir / subdir
        marker = root / marker_rel
        root.mkdir(parents=True, exist_ok=True)

        if marker.exists():
            n = len(list(marker.glob("*.png")))
            ok(f"KITTI {year} already present ({n} images) → {root}")
            continue

        dest = data_dir / zipname
        url  = f"{KITTI_S3_BASE}/{zipname}"
        if not download(url, dest):
            warn(f"Could not download KITTI {year} ({zipname}).")
            continue

        info(f"Extracting KITTI {year} — this may take a while …")
        subprocess.run(["unzip", "-q", str(dest), "-d", str(root)], check=True)

        # The zip extracts as  training/  testing/  at the top level.
        # If unzip placed them one level deeper, flatten.
        nested = root / zipname.replace(".zip", "")
        if nested.is_dir() and (nested / "training").is_dir():
            for item in nested.iterdir():
                item.rename(root / item.name)
            nested.rmdir()

        n = len(list(marker.glob("*.png"))) if marker.exists() else 0
        ok(f"KITTI {year}: {n} images → {root}")

    return []


# ---------------------------------------------------------------------------
# DrivingStereo
# ---------------------------------------------------------------------------

# Direct Google Drive file IDs for the weather subset (full-resolution).
# Source: https://drivingstereo-dataset.github.io/  — "Different weathers" table.
# Each weather folder contains left-image-full-size.zip, right-image-full-size.zip,
# and disparity-map-full-size.zip.  File IDs confirmed from GDrive folder listing.
DRIVING_GDRIVE: dict[str, dict[str, str]] = {
    "sunny": {
        "left-image-full-size":    "13GUW5uZnPw_Mmyz4APaPMK_fGHuoF4sr",
        "right-image-full-size":   "1AfyYZz_UaydIUHhCKEsD0g_CFgsuGngJ",
        "disparity-map-full-size": "1p0qXvl3sTyfk934fGFOW9GFXrT83dPgH",
    },
    "cloudy": {
        "left-image-full-size":    "1PkifSgXue0E7UC1pk3u591lZ4F6dfSl-",
        "right-image-full-size":   "1U-4DC-mJK-TMgXhIVSe_vsjSeZorC6QI",
        "disparity-map-full-size": "1M5lDoQxYCRBN_3oSVrViE6Sils7v4WZq",
    },
    "foggy": {
        "left-image-full-size":    "1iWOyD6x2-7UXhdxNX0S0IRsJjlg1-HlT",
        "right-image-full-size":   "1Oo-VyVBIyEqCeyz9PJcmpdiAKExC7sRh",
        "disparity-map-full-size": "18pJ96vFZ4_CsZmdOmH9dVVtXCd833pqY",
    },
    "rainy": {
        "left-image-full-size":    "1LQq-TrTASwzLgNTD4WwWW3a31rGhpcJV",
        "right-image-full-size":   "1mqMDzJVtV9pXSaDIFcXCCkUOREEo-6bX",
        "disparity-map-full-size": "1LaBNMgVgiCMZ0XJRIG4ecaO-e4UC9_yg",
    },
}

# Baidu Pan folder links (manual fallback — requires browser login)
DRIVING_BAIDU: dict[str, tuple[str, str]] = {
    "sunny":  ("https://pan.baidu.com/s/1yaxKHwjKG-BrRUTSWi-jnw", "xh86"),
    "cloudy": ("https://pan.baidu.com/s/1CAyDEzAgjl2OdtNjmKHxwg",  "7iqh"),
    "foggy":  ("https://pan.baidu.com/s/1skbi9AVckA_8KVZ9YuHqRg",  "5k5b"),
    "rainy":  ("https://pan.baidu.com/s/1R_oqcd8P8OE7St4KTCBc_A",  "1rrd"),
}


def _gdown_available() -> bool:
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def _download_gdrive(file_id: str, dest: Path) -> bool:
    """Download a single Google Drive file via gdown. Returns True on success."""
    if dest.exists():
        info(f"Already downloaded: {dest.name}")
        return True
    try:
        import gdown
        info(f"Downloading {dest.name} from Google Drive …")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest), quiet=False, resume=True)
        if dest.exists():
            ok(f"Saved: {dest.name}")
            return True
        warn(f"gdown finished but file not found: {dest}")
        return False
    except Exception as e:
        warn(f"gdown failed for {dest.name}: {e}")
        dest.unlink(missing_ok=True)
        return False


def setup_drivingstereo(data_dir: Path) -> list[str]:
    root = data_dir / "DrivingStereoWeather"
    info("=== DrivingStereo weather ===")

    if not _gdown_available():
        warn("gdown not installed.  Install it with:  pip install gdown>=4.6")
        warn("Then re-run:  python download_datasets.py --drivingstereo")
        return [_driving_manual(data_dir)]

    failed_splits: list[str] = []

    for split, parts in DRIVING_GDRIVE.items():
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_ok = True
        for part_name, file_id in parts.items():
            marker = split_dir / part_name
            if marker.exists():
                n = len(list(marker.glob("*.png")))
                ok(f"DrivingStereo {split}/{part_name}: {n} images (already present)")
                continue

            zipfile = split_dir / f"{part_name}.zip"
            if not _download_gdrive(file_id, zipfile):
                split_ok = False
                continue

            info(f"Extracting {split}/{part_name}.zip …")
            subprocess.run(
                ["unzip", "-q", str(zipfile), "-d", str(split_dir)],
                check=True,
            )
            # The zip may extract to part_name/ directly or with a wrapper — normalise
            # Expected final path: split_dir/part_name/*.png
            if not marker.exists():
                # look for a single subdir that was created
                candidates = [p for p in split_dir.iterdir()
                              if p.is_dir() and p.name != part_name
                              and not any(p.name == k for k in parts)]
                if len(candidates) == 1:
                    candidates[0].rename(marker)
            n = len(list(marker.glob("*.png"))) if marker.exists() else 0
            ok(f"DrivingStereo {split}/{part_name}: {n} images")

        if not split_ok:
            failed_splits.append(split)

    if failed_splits:
        warn(f"Failed splits: {', '.join(failed_splits)}.  See manual instructions.")
        return [_driving_manual(data_dir)]
    return []


def _driving_manual(data_dir: Path) -> str:
    baidu_lines = "\n".join(
        f"           {s:8s}: {url}  (code: {code})"
        for s, (url, code) in DRIVING_BAIDU.items()
    )
    return textwrap.dedent(f"""
        ── DrivingStereo weather subset ──────────────────────────────────────
          Official page:  https://drivingstereo-dataset.github.io/

          Google Drive folders (each contains left/right/disparity zips):
            Sunny  : https://drive.google.com/drive/folders/1Fugpyu29fZABySlnF9g9s9BjiQ2MXu0O
            Cloudy : https://drive.google.com/drive/folders/10tvMmnQJ-8ESsG4FwxhgfUwke_s9CtTu
            Foggy  : https://drive.google.com/drive/folders/1Fh4m1DuWtca65_QkWrhvpNSZcZGt1EBX
            Rainy  : https://drive.google.com/drive/folders/1Dmplo4Ct4XBT2zibKpJRXt6GPfWvUwmm

          Baidu Pan alternatives:
    {baidu_lines}

          Expected layout after extraction:
            {data_dir}/DrivingStereoWeather/
              sunny/
                left-image-full-size/*.png
                right-image-full-size/*.png
                disparity-map-full-size/*.png
              cloudy/  foggy/  rainy/  (same structure)

          Re-run after extracting:  python download_datasets.py --drivingstereo
        ─────────────────────────────────────────────────────────────────────
    """)


# ---------------------------------------------------------------------------
# Layout verification
# ---------------------------------------------------------------------------

def verify_layout(
    data_dir: Path,
    do_eth3d: bool,
    do_kitti: bool,
    do_middlebury: bool,
    do_driving: bool,
) -> None:
    print()
    info("=== Directory check ===")

    checks: list[tuple[bool, Path]] = []
    if do_eth3d:
        checks += [
            (True, data_dir / "ETH3D" / "two_view_training"),
            (True, data_dir / "ETH3D" / "two_view_training_gt"),
        ]
    if do_kitti:
        checks += [
            (True, data_dir / "kitti12" / "training" / "colored_0"),
            (True, data_dir / "kitti15" / "training" / "image_2"),
        ]
    if do_middlebury:
        for res in ("F", "H", "Q"):
            checks.append((True, data_dir / "Middlebury" / "MiddEval3" / f"training{res}"))
    if do_driving:
        for s in ("cloudy", "foggy", "rainy", "sunny"):
            checks.append((True, data_dir / "DrivingStereoWeather" / s / "left-image-full-size"))

    all_ok = True
    for _, path in checks:
        if path.exists():
            ok(str(path))
        else:
            warn(f"MISSING  {path}")
            all_ok = False

    if all_ok:
        ok("All directories present.")
    else:
        warn("Some datasets need manual setup (see instructions above).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download evaluation datasets for LiteAnyStereo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python download_datasets.py                        # download everything
              python download_datasets.py --eth3d               # ETH3D only
              python download_datasets.py --skip-kitti          # skip KITTI
              python download_datasets.py --data-dir /data/stereo
        """),
    )
    p.add_argument("--data-dir", default="./data/datasets",
                   help="Root directory for datasets (default: ./data/datasets)")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour output")

    # Exclusive-select flags (pick one dataset only)
    sel = p.add_argument_group("select a single dataset (mutually exclusive)")
    sel_ex = sel.add_mutually_exclusive_group()
    sel_ex.add_argument("--eth3d",        action="store_true", help="ETH3D only")
    sel_ex.add_argument("--kitti",        action="store_true", help="KITTI only")
    sel_ex.add_argument("--middlebury",   action="store_true", help="Middlebury only")
    sel_ex.add_argument("--drivingstereo",action="store_true", help="DrivingStereo only")

    # Skip flags
    skp = p.add_argument_group("skip individual datasets")
    skp.add_argument("--skip-eth3d",      action="store_true")
    skp.add_argument("--skip-kitti",      action="store_true")
    skp.add_argument("--skip-middlebury", action="store_true")
    skp.add_argument("--skip-driving",    action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global USE_COLOR
    args = parse_args()

    if args.no_color:
        USE_COLOR = False

    # Resolve which datasets to process
    any_selected = args.eth3d or args.kitti or args.middlebury or args.drivingstereo
    do_eth3d      = (args.eth3d      or not any_selected) and not args.skip_eth3d
    do_kitti      = (args.kitti      or not any_selected) and not args.skip_kitti
    do_middlebury = (args.middlebury or not any_selected) and not args.skip_middlebury
    do_driving    = (args.drivingstereo or not any_selected) and not args.skip_driving

    data_dir = Path(args.data_dir)

    header("LiteAnyStereo — Dataset Downloader")
    print(f"  Data root    : {data_dir}")
    print(f"  ETH3D        : {do_eth3d}")
    print(f"  KITTI        : {do_kitti}")
    print(f"  Middlebury   : {do_middlebury}")
    print(f"  DrivingStereo: {do_driving}")

    check_deps()
    data_dir.mkdir(parents=True, exist_ok=True)

    manual_steps: list[str] = []

    if do_eth3d:
        setup_eth3d(data_dir)
    if do_middlebury:
        manual_steps += setup_middlebury(data_dir)
    if do_kitti:
        manual_steps += setup_kitti(data_dir)
    if do_driving:
        manual_steps += setup_drivingstereo(data_dir)

    verify_layout(data_dir, do_eth3d, do_kitti, do_middlebury, do_driving)

    if manual_steps:
        print()
        print(_c("33;1", "═" * 52))
        print(_c("33;1", "  Manual download required for some datasets:"))
        print(_c("33;1", "═" * 52))
        for step in manual_steps:
            print(step)

    print()
    ok("Done. Run evaluation with:")
    print("  VERSION=las2 MODEL_SIZE=h sh evaluate.sh")


if __name__ == "__main__":
    main()
