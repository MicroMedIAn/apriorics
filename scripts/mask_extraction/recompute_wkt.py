import json
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import geopandas
import numpy as np
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.util.paths import get_files
from pathaia.util.types import Slide
from shapely.affinity import translate
from shapely.ops import unary_union

from apriorics.polygons import mask_to_polygons_layer

parser = ArgumentParser()
parser.add_argument("--maskfolder", type=Path)
parser.add_argument("--slidefolder", type=Path)
parser.add_argument("--wktfolder", type=Path)
parser.add_argument("--geojsonfolder", type=Path)
parser.add_argument("--num-workers", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    maskfiles = get_files(args.maskfolder, extensions=".tif")
    maskfiles.sort(key=lambda x: x.stem)
    slidefiles = maskfiles.map(lambda x: args.slidefolder / f"{x.stem}.svs")

    for maskfile, slidefile in zip(maskfiles, slidefiles):
        slidename = slidefile.stem
        outfile = args.wktfolder / f"{slidefile.stem}.wkt"
        if outfile.exists():
            continue
        print(slidename)

        slide_mask = Slide(maskfile, backend="cucim")
        slide = Slide(slidefile, backend="cucim")
        w, h = slide.dimensions

        def correct_mask(patch):
            mask_reg = np.asarray(
                slide_mask.read_region(patch.position, patch.level, patch.size).convert(
                    "1"
                )
            )
            if not mask_reg.sum():
                return

            polygons = mask_to_polygons_layer(mask_reg, angle_th=0, distance_th=0)
            polygons = translate(polygons, *patch.position)
            return polygons

        patches = slide_rois_no_image(
            slide, 0, 1000, interval=-100, thumb_size=5000, slide_filters=["full"]
        )
        print("Computing mask...")
        with Pool(processes=args.num_workers) as pool:
            all_polygons = pool.map(correct_mask, patches)
            pool.close()
            pool.join()
        all_polygons = [x for x in all_polygons if x is not None]
        all_polygons = unary_union(all_polygons)

        with open(outfile, "w") as f:
            f.write(all_polygons.wkt)

        with open(args.geojsonfolder/f"{slidefile.stem}.geojson", "w") as f:
            json.dump(geopandas.GeoSeries(all_polygons.geoms).__geo_interface__, f)
