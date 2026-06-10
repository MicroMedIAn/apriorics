import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import geopandas as gpd
    from pathaia.util.types import Slide
    import cv2
    import numpy as np
    from skimage.morphology import remove_small_objects, remove_small_holes
    from apriorics.polygons import mask_to_polygons_layer
    from shapely.affinity import scale
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from shapely import make_valid
    from pathlib import Path
    import json

    return (
        Path,
        Slide,
        cv2,
        gpd,
        json,
        make_valid,
        mask_to_polygons_layer,
        np,
        remove_small_holes,
        remove_small_objects,
        scale,
        unary_union,
    )


@app.cell
def _(cv2, np, remove_small_holes, remove_small_objects):
    def get_adip_mask(slide):
        level = slide.level_count - 1
        w, h = slide.level_dimensions[level]
        thumb = np.array(slide.read_region((0, 0), level, (w, h)).convert("RGB"))
        thumb_gray = cv2.GaussianBlur(
            cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY), (9, 9), 0
        )
        mask = remove_small_objects(
            remove_small_holes((thumb_gray > 230) & (thumb_gray < 247)), min_size=10
        )
        return mask

    return (get_adip_mask,)


@app.cell
def _(
    Slide,
    get_adip_mask,
    gpd,
    make_valid,
    mask_to_polygons_layer,
    scale,
    unary_union,
):
    def get_adip_gs(slide_path, ihc):
        slide = Slide(slide_path)
        mask = get_adip_mask(slide)
        t_h, t_w = mask.shape
        w, h = slide.dimensions
        dsr = w / t_w
        tissue_gs = gpd.read_file(
            f"/media/AprioricsSlides/tissue_geojsons/{ihc}/{slide_path.stem.split('-')[0]}.geojson"
        )["geometry"].make_valid()
        pols = mask_to_polygons_layer(mask, 0, 0)
        pols = scale(pols, dsr, dsr, origin=(0, 0, 0))
        pols = make_valid(unary_union(pols))
        adip_gs = tissue_gs.intersection(pols).polygonize()
        return adip_gs

    return (get_adip_gs,)


@app.cell
def _(json):
    with open("ihc_mapping.json") as f:
        ihc_mapping = json.load(f)
    return (ihc_mapping,)


@app.cell
def _(Path, get_adip_gs, ihc_mapping):
    for ihc, ihc_n in ihc_mapping["HE"].items():
        ihc_fold = Path(f"/media/AprioricsSlides/adip_geojsons/{ihc}")
        ihc_fold.mkdir(exist_ok=True)
        for fp in Path("/media/AprioricsSlides/").glob(f"*-{ihc_n}_*.svs"):
            out_fp = ihc_fold / f"{fp.stem.split('-')[0]}.geojson"
            if not out_fp.exists():
                adip_gs = get_adip_gs(fp, ihc)
                adip_gs.to_file(out_fp)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
