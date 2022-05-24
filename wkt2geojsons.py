import json
from argparse import ArgumentParser
from pathlib import Path

import geopandas
from pathaia.util.paths import get_files
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon

parser = ArgumentParser("Transforms wkt files into geojson format.")
parser.add_argument(
    "--wktfolder", type=Path, help="Input folder containing wkt files.", required=True
)
parser.add_argument(
    "--geojsonfolder", type=Path, help="Output folder for geojson files.", required=True
)
parser.add_argument(
    "--recurse",
    action="store_true",
    help="Specify to recurse through slidefolder when looking for svs files. Optional.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    wktfiles = get_files(args.wktfolder, extensions=".wkt", recurse=args.recurse)

    for wktfile in wktfiles:
        print(wktfile.stem)
        with open(wktfile, "r") as f:
            polygons = wkt.load(f)
        if isinstance(polygons, Polygon):
            polygons = MultiPolygon(polygons=[polygons])

        with open(
            args.geojsonfolder
            / wktfile.relative_to(args.wktfolder).with_suffix(".geojson"),
            "w",
        ) as f:
            json.dump(geopandas.GeoSeries(polygons.geoms).__geo_interface__, f)
