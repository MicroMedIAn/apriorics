import csv
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.util.paths import get_files
from pathaia.util.types import Patch, Slide
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from apriorics.dataset_preparation import filter_thumbnail_mask_extraction

parser = ArgumentParser(prog="Generates the PathAIA patch CSVs for slides.")
parser.add_argument(
    "--slidefolder",
    type=Path,
    help="Input folder containing slide svs files.",
    required=True,
)
parser.add_argument(
    "--maskfolder",
    type=Path,
    help="Input folder containing mask tif files. Optional.",
)
parser.add_argument(
    "--outfolder",
    type=Path,
    help=(
        "Target output folder. Actual output folder will be "
        "outfolder/{patch_size}_{level}/patch_csvs."
    ),
    required=True,
)
parser.add_argument(
    "--recurse",
    "-r",
    action="store_true",
    help="Specify to recurse through slidefolder when looking for svs files. Optional.",
)
parser.add_argument(
    "--ihc_type",
    help="Name of the IHC.",
    required=True,
)
parser.add_argument(
    "--slide_extension",
    default=".svs",
    help="File extension of slide files. Default .svs.",
)
parser.add_argument(
    "--patch_size",
    type=int,
    default=1024,
    help="Size of the (square) patches to extract. Default 1024.",
)
parser.add_argument(
    "--level",
    type=int,
    default=0,
    help="Pyramid level to extract patches on. Default 0.",
)
parser.add_argument(
    "--overlap",
    type=float,
    default=0,
    help="Part of the patches that should overlap. Default 0.",
)
parser.add_argument(
    "--filter_pos",
    type=int,
    default=0,
    help="Minimum number of positive pixels in mask to keep patch. Default 0.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Specify to overwrite existing csvs. Optional.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    help="Number of workers to use for processing. Defaults to all available workers.",
)
parser.add_argument(
    "--export_geojson",
    action="store_true",
    help="Specify to save geojson representation of patch extractions. Optional.",
)
parser.add_argument("--regfolder", type=Path)

if __name__ == "__main__":
    args = parser.parse_known_args()[0]
    input_files = get_files(
        args.slidefolder / "HE",
        extensions=args.slide_extension,
        recurse=args.recurse,
    )
    input_files.sort(key=lambda x: x.stem)

    outfolder = args.outfolder / f"{args.patch_size}_{args.level}/patch_csvs"
    if not outfolder.exists():
        outfolder.mkdir(parents=True)

    geojsonfolder = args.outfolder / f"{args.patch_size}_{args.level}/patch_geojsons"
    if args.export_geojson and not geojsonfolder.exists():
        geojsonfolder.mkdir()

    interval = -int(args.overlap * args.patch_size)

    def write_patches(in_file_path):
        out_file_path = outfolder / in_file_path.with_suffix(".csv").name
        if not args.overwrite and out_file_path.exists():
            return

        slide = Slide(in_file_path, backend="cucim")

        if args.maskfolder is not None:
            mask_path = args.maskfolder / in_file_path.relative_to(
                args.slidefolder / "HE"
            ).with_suffix(".gpkg")
            if not mask_path.exists():
                return
        else:
            mask_path = None

        if args.regfolder is not None:
            reg_path = args.regfolder / f"{in_file_path.stem}.geojson"
            if not reg_path.exists():
                return
            gdf = gpd.read_file(reg_path)
            gdf = gdf.loc[gdf["geometry"].is_valid]
        else:
            gdf = None

        print(in_file_path.stem)

        patches = slide_rois_no_image(
            slide,
            args.level,
            psize=args.patch_size,
            interval=interval,
            slide_filters=[filter_thumbnail_mask_extraction],
            thumb_size=2000,
        )

        pols = []
        with open(out_file_path, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=Patch.get_fields() + ["n_pos"])
            writer.writeheader()
            for patch in patches:
                x, y = patch.position
                w, h = patch.size
                pol = box(x, y, x + w, y + h)
                if gdf is not None:
                    if not gdf["geometry"].contains(pol).any():
                        continue

                if mask_path is not None:
                    mask = gpd.read_file(mask_path, engine="pyogrio", bbox=pol)[
                        "geometry"
                    ]
                    n_pos = mask.area.sum()
                else:
                    n_pos = None

                row = patch.to_csv_row()
                row["n_pos"] = n_pos
                if n_pos is None or n_pos >= args.filter_pos:
                    writer.writerow(row)
                    if args.export_geojson:
                        pols.append(pol)

        if args.export_geojson:
            pols = unary_union(pols)
            if isinstance(pols, Polygon):
                pols = MultiPolygon([pols])
            gpd.GeoSeries(pols.geoms).to_file(
                geojsonfolder / in_file_path.with_suffix(".geojson").name
            )

    with Pool(processes=args.num_workers) as pool:
        pool.map(write_patches, input_files)
        pool.close()
        pool.join()
