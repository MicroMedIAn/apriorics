#!/usr/bin/env python3
"""
Convert AtlasPatch .h5 patch-coordinate files into PathAIA-compatible CSV files.

AtlasPatch h5 layout (per slide, written at <output>/patches/<stem>.h5):
    file attrs:
        patch_size          int   patch size at the target magnification
        patch_size_level0   int   patch footprint rescaled to level 0
        num_patches         int   number of patches
        wsi_path            str   source WSI path
        ... (other metadata attrs)
    datasets:
        coords              (N, 5) int32, columns = (x, y, read_w, read_h, level)
        passports           (N,)   S160

PathAIA Patch CSV row (see pathaia.util.types.Patch.to_csv_row):
    id, global_id, x, y, level, dx, dy, size_x, size_y, parent

Mapping (AtlasPatch -> PathAIA Patch field):
    slidename   <- stem of the h5 file (or `filename` file attr if present)
    position    <- (coords[:, 0], coords[:, 1])            # level-0 (x, y)
    level       <- coords[:, 4]
    size        <- (coords[:, 2], coords[:, 3])            # (read_w, read_h) at `level`
    size_0      <- (patch_size_level0, patch_size_level0)  # level-0 footprint
    id          <- row index as a string ("0", "1", ...)
    parent      <- "None"  (AtlasPatch has no parent concept)

Usage:
    python atlaspatch_to_pathaia_csv.py INPUT OUTPUT
    python atlaspatch_to_pathaia_csv.py slide.h5 out_dir/
    python atlaspatch_to_pathaia_csv.py patches_dir/ out_dir/ --recursive
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Iterator

try:
    import h5py
except ImportError:
    sys.exit("Missing dependency 'h5py'. Install it with:  pip install h5py")


# PathAIA Patch.to_csv_row() field order — keep this exact.
CSV_FIELDS = [
    "id",
    "global_id",
    "x",
    "y",
    "level",
    "dx",
    "dy",
    "size_x",
    "size_y",
    "parent",
]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def iter_h5_files(input_path: Path, recursive: bool) -> Iterator[Path]:
    """Yield every .h5 (and .hdf5) file under `input_path`.

    If `input_path` is a file, yield it (regardless of extension, but warn
    if it doesn't look like an h5 file). If it's a directory, walk it.
    """
    if input_path.is_file():
        if input_path.suffix.lower() not in (".h5", ".hdf5"):
            print(
                f"WARNING: {input_path} does not have a .h5/.hdf5 extension; "
                "trying to read it anyway.",
                file=sys.stderr,
            )
        yield input_path
        return

    if input_path.is_dir():
        globber = "**/*" if recursive else "*"
        found = sorted(
            p
            for p in input_path.glob(globber)
            if p.is_file() and p.suffix.lower() in (".h5", ".hdf5")
        )
        if not found:
            print(
                f"WARNING: no .h5 files found under {input_path}"
                f"{' (recursive)' if recursive else ''}.",
                file=sys.stderr,
            )
        yield from found
        return

    sys.exit(f"ERROR: input path does not exist: {input_path}")


# --------------------------------------------------------------------------- #
# Core conversion
# --------------------------------------------------------------------------- #
def resolve_slidename(h5_path: Path, attrs) -> str:
    """Prefer the `filename` file attribute (AtlasPatch sets it to slide.path.name);
    otherwise fall back to the h5 file stem."""
    filename_attr = attrs.get("filename")
    if isinstance(filename_attr, bytes):
        filename_attr = filename_attr.decode("utf-8", errors="replace")
    if filename_attr:
        # `filename` is the slide's basename (e.g. "slide.svs"); use its stem.
        return Path(str(filename_attr)).stem
    return h5_path.stem


def convert_one(h5_path: Path, output_dir: Path) -> Path:
    """Convert a single AtlasPatch h5 file into a PathAIA-style CSV.

    Returns the path of the written CSV file."""
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise ValueError(
                f"{h5_path}: no 'coords' dataset — not an AtlasPatch patch file."
            )

        coords = f["coords"][...]  # (N, 5) int32
        if coords.ndim != 2 or coords.shape[1] < 5:
            raise ValueError(
                f"{h5_path}: unexpected 'coords' shape {coords.shape}; "
                "expected (N, 5) with columns (x, y, read_w, read_h, level)."
            )

        attrs = dict(f.attrs)

        # `patch_size_level0` is the level-0 footprint of each patch.
        # Required by AtlasPatch's own reader (REQUIRED_PATCH_FILE_ATTRS).
        if "patch_size_level0" not in attrs:
            raise ValueError(
                f"{h5_path}: missing required file attribute 'patch_size_level0'. "
                "Older AtlasPatch versions may not write it; upgrade or set it manually."
            )
        patch_size_level0 = int(attrs["patch_size_level0"])

        slidename = resolve_slidename(h5_path, attrs)

        # Build rows.
        rows = []
        for i, (x, y, read_w, read_h, level) in enumerate(
            coords[:, :5].astype(int, copy=False)
        ):
            patch_id = str(i)
            rows.append(
                {
                    "id": patch_id,
                    "global_id": slidename + patch_id,
                    "x": int(x),
                    "y": int(y),
                    "level": int(level),
                    # size_0 = level-0 footprint (square in AtlasPatch)
                    "dx": patch_size_level0,
                    "dy": patch_size_level0,
                    # size = what was actually read at `level`
                    "size_x": int(read_w),
                    "size_y": int(read_h),
                    "parent": "None",
                }
            )

    # Write CSV.
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{h5_path.stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlaspatch_to_pathaia_csv",
        description=(
            "Convert AtlasPatch .h5 patch files into PathAIA-compatible CSV files. "
            "INPUT may be a single .h5 file or a directory containing .h5 files. "
            "OUTPUT must be a directory (created if missing)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s slide.h5 out_csv/\n"
            "  %(prog)s patches/ out_csv/\n"
            "  %(prog)s patches/ out_csv/ --recursive\n"
            "  %(prog)s patches/ out_csv/ --overwrite\n"
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a single .h5 file or a directory containing .h5 files.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory where CSV files will be written (created if missing).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when INPUT is a folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files in the output directory (default: skip).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    input_path: Path = args.input
    output_dir: Path = args.output

    if output_dir.is_file():
        sys.exit(f"ERROR: output path is a file, not a directory: {output_dir}")

    h5_files = list(iter_h5_files(input_path, args.recursive))
    if not h5_files:
        sys.exit("ERROR: no .h5 files to process.")

    n_ok = n_skip = n_fail = 0
    for h5_path in h5_files:
        try:
            csv_path = output_dir / f"{h5_path.stem}.csv"
            if csv_path.exists() and not args.overwrite:
                print(f"SKIP   {h5_path} -> {csv_path} (already exists)")
                n_skip += 1
                continue

            convert_one(h5_path, output_dir)
            print(f"OK     {h5_path} -> {csv_path}")
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 — we want to keep going
            print(f"FAIL   {h5_path}: {exc}", file=sys.stderr)
            n_fail += 1

    print(
        f"\nDone: {n_ok} converted, {n_skip} skipped, {n_fail} failed "
        f"(out of {len(h5_files)} h5 file(s)).",
        file=sys.stderr,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
