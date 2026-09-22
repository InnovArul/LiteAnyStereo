#!/usr/bin/env python3
"""
download_datasets.py

Downloads and extracts all evaluation datasets for LiteAnyStereo:
  - ETH3D             (fully automated)
  - Middlebury        (fully automated; falls back to manual instructions)
  - KITTI 2012/2015   (fully automated via public S3 bucket)
  - DrivingStereo     (requires gdown: pip install gdown>=4.6)

Usage:
  python download_datasets.py [--data-dir DIR]
                              [--eth3d | --kitti | --middlebury | --drivingstereo]
                              [--skip-eth3d] [--skip-kitti]
                              [--skip-middlebury] [--skip-driving]
                              [--no-color]

  # Extract-only mode (skip downloads, just fix folder structure):
  python download_datasets.py --extract-only

Dependencies:
  Python >= 3.8, wget, unzip, 7z/7za in PATH.
    sudo apt install wget unzip p7zip-full
  For DrivingStereo:
    pip install gdown>=4.6
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


def info(msg: str)   -> None: print(_c("36",   f"[INFO]  {msg}"))
def ok(msg: str)     -> None: print(_c("32",   f"[OK]    {msg}"))
def warn(msg: str)   -> None: print(_c("33;1", f"[WARN]  {msg}"))
def error(msg: str)  -> None: print(_c("31",   f"[ERROR] {msg}"), file=sys.stderr)
def header(msg: str) -> None: print(_c("36",   f"\n{'═'*52}\n  {msg}\n{'═'*52}"))


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
# Download helper
# ---------------------------------------------------------------------------

def download(url: str, dest: Path) -> bool:
    """
    Download url to dest. Skips if dest already exists (archive already
    downloaded). Uses a .tmp sidecar so interrupted downloads can be resumed.
    Returns True on success.
    """
    if dest.exists():
        info(f"Already downloaded: {dest.name}  (skipping)")
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

def extract_7z_to(archive: Path, out_dir: Path) -> None:
    """
    Extract a .7z archive into out_dir, placing contents directly there.

    Uses a sibling temp directory and then moves the results into out_dir.
    This avoids issues with some 7z builds ignoring the -o flag when the
    archive path is relative, and guarantees no symlinks are created.
    """
    info(f"Extracting {archive.name} into {out_dir} …")
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = archive.parent / (archive.stem + "_extract_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [_7z_cmd(), "x", str(archive.resolve()), f"-o{tmp_dir.resolve()}", "-y"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            warn(f"7z reported an error:\n{result.stderr.strip()}")

        # Move every top-level item from tmp_dir into out_dir
        moved = 0
        for item in tmp_dir.iterdir():
            dest = out_dir / item.name
            if dest.exists():
                warn(f"Destination already exists, skipping: {dest}")
                continue
            item.rename(dest)
            moved += 1
        ok(f"Extracted {moved} item(s) → {out_dir}")
    finally:
        # Clean up temp dir (should be empty after moves)
        try:
            tmp_dir.rmdir()
        except OSError:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_zip_to(archive: Path, out_dir: Path) -> None:
    """Extract a .zip archive into out_dir."""
    info(f"Extracting {archive.name} into {out_dir} …")
    subprocess.run(["unzip", "-q", str(archive), "-d", str(out_dir)], check=True)
    ok(f"Extracted → {out_dir}")


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

ETH3D_URLS = {
    "two_view_training.7z":    "https://www.eth3d.net/data/two_view_training.7z",
    "two_view_training_gt.7z": "https://www.eth3d.net/data/two_view_training_gt.7z",
}


def setup_eth3d(data_dir: Path, extract_only: bool = False) -> None:
    root = data_dir / "ETH3D"
    root.mkdir(parents=True, exist_ok=True)
    header("ETH3D")

    for filename, url in ETH3D_URLS.items():
        dest = root / filename
        if not extract_only:
            download(url, dest)
        elif not dest.exists():
            warn(f"Archive not found (--extract-only): {dest}")
            continue

        # Determine target subdir: strip .7z to get the folder name
        subdir_name = filename.replace(".7z", "")
        out_dir = root / subdir_name

        # Check if already extracted: look for any .pfm or im0.png inside
        import glob as _glob
        existing = (
            _glob.glob(str(out_dir / "*" / "disp0GT.pfm")) +
            _glob.glob(str(out_dir / "*" / "im0.png"))
        )
        if existing:
            ok(f"Already extracted: {subdir_name}/ ({len(existing)} files found)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        extract_7z_to(dest, out_dir)

    # Verify
    import glob as _glob
    n_img = len(_glob.glob(str(root / "two_view_training" / "*" / "im0.png")))
    n_gt  = len(_glob.glob(str(root / "two_view_training_gt" / "*" / "disp0GT.pfm")))
    ok(f"ETH3D: {n_img} image pairs, {n_gt} GT disparity maps")
    if n_gt == 0:
        warn("No GT maps found in two_view_training_gt/ — check extraction above.")


# ---------------------------------------------------------------------------
# Middlebury
# ---------------------------------------------------------------------------

MIDDLEBURY_BASE    = "https://vision.middlebury.edu/stereo/submit3/zip"
MIDDLEBURY_WAYBACK = "https://web.archive.org/web/2024/https://vision.middlebury.edu/stereo/submit3/zip"

# MiddEval3-data-*.zip  → images only (trainingX + testX, NO ground truth)
# MiddEval3-GT0-*.zip   → ground truth disparity + masks for trainingX scenes
# Each entry lists URLs in priority order; first successful download wins.
MIDDLEBURY_FILES: dict[str, dict[str, list[str]]] = {
    "F": {
        "data": [
            f"{MIDDLEBURY_BASE}/MiddEval3-data-F.zip",
            f"{MIDDLEBURY_WAYBACK}/MiddEval3-data-F.zip",
        ],
        "gt": [
            f"{MIDDLEBURY_BASE}/MiddEval3-GT0-F.zip",
            f"{MIDDLEBURY_WAYBACK}/MiddEval3-GT0-F.zip",
        ],
    },
    "H": {
        "data": [
            f"{MIDDLEBURY_BASE}/MiddEval3-data-H.zip",
            f"{MIDDLEBURY_WAYBACK}/MiddEval3-data-H.zip",
        ],
        "gt": [
            f"{MIDDLEBURY_BASE}/MiddEval3-GT0-H.zip",
            f"{MIDDLEBURY_WAYBACK}/MiddEval3-GT0-H.zip",
        ],
    },
    "Q": {
        "data": [
            f"{MIDDLEBURY_BASE}/MiddEval3-data-Q.zip",
            f"{MIDDLEBURY_WAYBACK}/MiddEval3-data-Q.zip",
        ],
        "gt": [
            f"{MIDDLEBURY_BASE}/MiddEval3-GT0-Q.zip",
            f"{MIDDLEBURY_WAYBACK}/MiddEval3-GT0-Q.zip",
        ],
    },
}

MIDDLEBURY_MANUAL = textwrap.dedent(f"""
    ── Middlebury MiddEval3 ──────────────────────────────────────────────
      Primary:   {MIDDLEBURY_BASE}/MiddEval3-GT0-F.zip  (and -H, -Q)
      Mirror:    {MIDDLEBURY_WAYBACK}/MiddEval3-GT0-F.zip  (and -H, -Q)

      Extract all GT zips to:  <data-dir>/Middlebury/
      Expected layout:
        <data-dir>/Middlebury/MiddEval3/trainingF/<scene>/disp0GT.pfm
        <data-dir>/Middlebury/MiddEval3/trainingF/<scene>/mask0nocc.png
        (same for trainingH and trainingQ)

      Re-run after extracting:
        python download_datasets.py --middlebury --extract-only
    ─────────────────────────────────────────────────────────────────────
""")


def setup_middlebury(data_dir: Path, extract_only: bool = False) -> list[str]:
    root = data_dir / "Middlebury"
    root.mkdir(parents=True, exist_ok=True)
    header("Middlebury MiddEval3")

    failed = False
    for res, kinds in MIDDLEBURY_FILES.items():
        for kind, urls in kinds.items():
            filename = Path(urls[0]).name
            dest = root / filename
            if not extract_only:
                if not dest.exists():
                    downloaded = False
                    for url in urls:
                        info(f"Trying {url} …")
                        if download(url, dest):
                            downloaded = True
                            break
                    if not downloaded:
                        warn(f"Could not download {filename} from any source.")
                        failed = True
                        continue
            else:
                if not dest.exists():
                    warn(f"Archive not found (--extract-only): {dest}")
                    failed = True
                    continue

            # Check if already extracted
            marker = root / "MiddEval3" / f"training{res}"
            import glob as _glob
            if kind == "data":
                if _glob.glob(str(marker / "*" / "im0.png")):
                    ok(f"Already extracted: MiddEval3/training{res} images")
                    continue
            elif kind == "gt":
                if _glob.glob(str(marker / "*" / "disp0GT.pfm")):
                    ok(f"Already extracted: MiddEval3/training{res} GT")
                    continue

            extract_zip_to(dest, root)

        # Verify
        import glob as _glob
        marker = root / "MiddEval3" / f"training{res}"
        n_img = len(_glob.glob(str(marker / "*" / "im0.png")))
        n_gt  = len(_glob.glob(str(marker / "*" / "disp0GT.pfm")))
        ok(f"Middlebury training{res}: {n_img} image pairs, {n_gt} GT maps")
        if n_gt == 0:
            warn(f"No GT found for training{res} — MiddEval3-GT0-{res}.zip may be missing.")
            failed = True

    return [MIDDLEBURY_MANUAL] if failed else []


# ---------------------------------------------------------------------------
# KITTI
# ---------------------------------------------------------------------------

KITTI_S3_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

KITTI_FILES = {
    "2012": ("data_stereo_flow.zip",  "kitti12", Path("training") / "colored_0"),
    "2015": ("data_scene_flow.zip",   "kitti15", Path("training") / "image_2"),
}


def setup_kitti(data_dir: Path, extract_only: bool = False) -> list[str]:
    header("KITTI")

    for year, (zipname, subdir, marker_rel) in KITTI_FILES.items():
        root   = data_dir / subdir
        marker = root / marker_rel
        root.mkdir(parents=True, exist_ok=True)

        if marker.exists() and list(marker.glob("*.png")):
            n = len(list(marker.glob("*.png")))
            ok(f"KITTI {year} already extracted ({n} images)")
            continue

        dest = data_dir / zipname
        if not extract_only:
            if not download(f"{KITTI_S3_BASE}/{zipname}", dest):
                warn(f"Could not download KITTI {year}.")
                continue
        else:
            if not dest.exists():
                warn(f"Archive not found (--extract-only): {dest}")
                continue

        info(f"Extracting KITTI {year} — this may take a while …")
        subprocess.run(["unzip", "-q", str(dest), "-d", str(root)], check=True)

        # Flatten if unzip added a wrapper directory
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
    if dest.exists():
        info(f"Already downloaded: {dest.name}  (skipping)")
        return True
    try:
        import gdown
        info(f"Downloading {dest.name} from Google Drive …")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            str(dest), quiet=False, resume=True,
        )
        if dest.exists():
            ok(f"Saved: {dest.name}")
            return True
        warn(f"gdown finished but file not found: {dest}")
        return False
    except Exception as e:
        warn(f"gdown failed for {dest.name}: {e}")
        dest.unlink(missing_ok=True)
        return False


def _flatten_driving_part(split_dir: Path, part_name: str) -> None:
    """
    The DrivingStereo zips extract as:
      split/part-name/part-name/*.png   (double-nested)
    Expected layout:
      split/part-name/*.png             (single level)
    This moves the inner directory's contents up one level.
    """
    outer = split_dir / part_name
    inner = outer / part_name

    if not inner.is_dir():
        return  # already flat

    info(f"Flattening {part_name}/ (removing extra nesting) …")
    # Move all files from inner up to outer
    for f in inner.iterdir():
        f.rename(outer / f.name)
    inner.rmdir()
    n = len(list(outer.glob("*.png")))
    ok(f"Flattened {part_name}/: {n} files")


def setup_drivingstereo(data_dir: Path, extract_only: bool = False) -> list[str]:
    root = data_dir / "DrivingStereoWeather"
    header("DrivingStereo weather")

    # --- Extract-only or already-downloaded: just fix folder structure ---
    # Check if any zips are already extracted (even with wrong nesting)
    any_zips = list(root.rglob("*.zip")) if root.exists() else []

    if extract_only or any_zips:
        # Fix nesting for everything that's already been extracted
        fixed_something = False
        for split in DRIVING_GDRIVE:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for part_name in DRIVING_GDRIVE[split]:
                inner = split_dir / part_name / part_name
                if inner.is_dir():
                    _flatten_driving_part(split_dir, part_name)
                    fixed_something = True
        if fixed_something or extract_only:
            _verify_driving(root)
            if extract_only:
                return []

    if not _gdown_available():
        warn("gdown not installed.  Install with:  pip install gdown>=4.6")
        return [_driving_manual(data_dir)]

    failed_splits: list[str] = []

    for split, parts in DRIVING_GDRIVE.items():
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_ok = True

        for part_name, file_id in parts.items():
            final_dir = split_dir / part_name
            # Already correctly extracted?
            if final_dir.is_dir() and list(final_dir.glob("*.png")):
                n = len(list(final_dir.glob("*.png")))
                ok(f"Already extracted: {split}/{part_name}  ({n} files)")
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
            # Fix double-nesting produced by these zips
            _flatten_driving_part(split_dir, part_name)

        if not split_ok:
            failed_splits.append(split)

    _verify_driving(root)

    if failed_splits:
        warn(f"Failed splits: {', '.join(failed_splits)}")
        return [_driving_manual(data_dir)]
    return []


def _verify_driving(root: Path) -> None:
    for split in DRIVING_GDRIVE:
        for part_name in DRIVING_GDRIVE[split]:
            d = root / split / part_name
            if d.is_dir():
                n = len(list(d.glob("*.png")))
                status = ok if n > 0 else warn
                status(f"DrivingStereo {split}/{part_name}: {n} files")
            else:
                warn(f"DrivingStereo {split}/{part_name}: MISSING")


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
              sunny/left-image-full-size/*.png
              sunny/right-image-full-size/*.png
              sunny/disparity-map-full-size/*.png
              (cloudy/ foggy/ rainy/ same structure)

          Re-run after manual download/extract:
            python download_datasets.py --drivingstereo --extract-only
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
    import glob as _glob
    print()
    info("=== Final layout check ===")

    all_ok = True

    if do_eth3d:
        for pattern, label in [
            ("ETH3D/two_view_training/*/im0.png",      "ETH3D images"),
            ("ETH3D/two_view_training_gt/*/disp0GT.pfm", "ETH3D GT"),
        ]:
            n = len(_glob.glob(str(data_dir / pattern)))
            if n > 0:
                ok(f"{label}: {n} files")
            else:
                warn(f"{label}: 0 files — {data_dir / pattern.split('*')[0]}")
                all_ok = False

    if do_kitti:
        for path in [
            data_dir / "kitti12" / "training" / "colored_0",
            data_dir / "kitti15" / "training" / "image_2",
        ]:
            n = len(list(path.glob("*.png"))) if path.exists() else 0
            (ok if n > 0 else warn)(f"KITTI {path.parent.parent.name}: {n} images  ({path})")
            if n == 0:
                all_ok = False

    if do_middlebury:
        for res in ("F", "H", "Q"):
            base = data_dir / "Middlebury" / "MiddEval3" / f"training{res}"
            n_img = len(_glob.glob(str(base / "*" / "im0.png")))
            n_gt  = len(_glob.glob(str(base / "*" / "disp0GT.pfm")))
            (ok if n_img > 0 and n_gt > 0 else warn)(
                f"Middlebury training{res}: {n_img} images, {n_gt} GT maps"
            )
            if n_img == 0 or n_gt == 0:
                all_ok = False

    if do_driving:
        for split in DRIVING_GDRIVE:
            for part in ("left-image-full-size", "disparity-map-full-size"):
                p = data_dir / "DrivingStereoWeather" / split / part
                n = len(list(p.glob("*.png"))) if p.exists() else 0
                (ok if n > 0 else warn)(f"DrivingStereo {split}/{part}: {n} files")
                if n == 0:
                    all_ok = False

    print()
    if all_ok:
        ok("All datasets look good.")
    else:
        warn("Some datasets are incomplete — see warnings above.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download/extract evaluation datasets for LiteAnyStereo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python download_datasets.py                         # everything
              python download_datasets.py --eth3d                 # ETH3D only
              python download_datasets.py --extract-only          # fix folder structure only
              python download_datasets.py --drivingstereo --extract-only  # flatten DrivingStereo
              python download_datasets.py --data-dir /data/stereo
        """),
    )
    p.add_argument("--data-dir", default="./data/datasets",
                   help="Root directory for datasets (default: ./data/datasets)")
    p.add_argument("--extract-only", action="store_true",
                   help="Skip all downloads; only extract/fix already-downloaded archives")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colour output")

    sel = p.add_argument_group("select a single dataset (mutually exclusive)")
    sel_ex = sel.add_mutually_exclusive_group()
    sel_ex.add_argument("--eth3d",         action="store_true")
    sel_ex.add_argument("--kitti",         action="store_true")
    sel_ex.add_argument("--middlebury",    action="store_true")
    sel_ex.add_argument("--drivingstereo", action="store_true")

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

    any_selected = args.eth3d or args.kitti or args.middlebury or args.drivingstereo
    do_eth3d      = (args.eth3d         or not any_selected) and not args.skip_eth3d
    do_kitti      = (args.kitti         or not any_selected) and not args.skip_kitti
    do_middlebury = (args.middlebury    or not any_selected) and not args.skip_middlebury
    do_driving    = (args.drivingstereo or not any_selected) and not args.skip_driving

    data_dir = Path(args.data_dir)
    ext_only = args.extract_only

    header("LiteAnyStereo — Dataset Setup")
    print(f"  Data root     : {data_dir}")
    print(f"  Extract-only  : {ext_only}")
    print(f"  ETH3D         : {do_eth3d}")
    print(f"  KITTI         : {do_kitti}")
    print(f"  Middlebury    : {do_middlebury}")
    print(f"  DrivingStereo : {do_driving}")

    if not ext_only:
        check_deps()
    data_dir.mkdir(parents=True, exist_ok=True)

    manual_steps: list[str] = []

    if do_eth3d:
        setup_eth3d(data_dir, extract_only=ext_only)
    if do_middlebury:
        manual_steps += setup_middlebury(data_dir, extract_only=ext_only)
    if do_kitti:
        manual_steps += setup_kitti(data_dir, extract_only=ext_only)
    if do_driving:
        manual_steps += setup_drivingstereo(data_dir, extract_only=ext_only)

    verify_layout(data_dir, do_eth3d, do_kitti, do_middlebury, do_driving)

    if manual_steps:
        print()
        print(_c("33;1", "═" * 52))
        print(_c("33;1", "  Manual steps required:"))
        print(_c("33;1", "═" * 52))
        for step in manual_steps:
            print(step)

    print()
    ok("Done.  Run evaluation with:")
    print("  VERSION=las2 MODEL_SIZE=h sh evaluate.sh")


if __name__ == "__main__":
    main()
