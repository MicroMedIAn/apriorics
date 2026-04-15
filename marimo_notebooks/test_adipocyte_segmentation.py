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
    from pathlib import Path

    return (
        Path,
        Slide,
        cv2,
        gpd,
        mask_to_polygons_layer,
        mo,
        np,
        remove_small_holes,
        remove_small_objects,
        scale,
        unary_union,
    )


@app.cell
def _(Slide):
    slide = Slide("dvc/P40ColIV/data/slides/HE/21I000005.svs")
    return (slide,)


@app.cell
def _(mo, thumb_gray):
    mo.image(thumb_gray)
    return


@app.cell
def _(cv2, mo, np, remove_small_holes, remove_small_objects, slide):
    t_w, t_h = slide.level_dimensions[3]
    thumb = np.array(slide.read_region((0, 0), 3, (t_w, t_h)).convert("RGB"))
    thumb_gray = cv2.GaussianBlur(
        cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY), (9, 9), 0
    )
    mask = remove_small_objects(
        remove_small_holes((thumb_gray > 230) & (thumb_gray < 247)), min_size=10
    )
    mo.image(mask.astype(int))
    return (thumb_gray,)


@app.cell
def _(cv2, np, remove_small_holes, remove_small_objects):
    def get_adip_mask(slide):
        thumb = np.array(slide.get_thumbnail((1000, 1000)).convert("RGB"))
        thumb_gray = cv2.GaussianBlur(
            cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY), (5, 5), 0
        )
        mask = remove_small_objects(
            remove_small_holes((thumb_gray > 230) & (thumb_gray < 249)), min_size=10
        )
        return mask

    return (get_adip_mask,)


@app.cell
def _(Slide, get_adip_mask, gpd, mask_to_polygons_layer, scale, unary_union):
    def get_adip_gs(slide_path):
        slide = Slide(slide_path)
        mask = get_adip_mask(slide)
        t_h, t_w = mask.shape
        w, h = slide.dimensions
        dsr = w / t_w
        tissue_gs = gpd.read_file(
            f"/media/AprioricsSlides/tissue_geojsons/P40ColIV/{slide_path.stem}.geojson"
        )["geometry"]
        pols = mask_to_polygons_layer(mask, 0, 0)
        pols = scale(pols, dsr, dsr, origin=(0, 0, 0))
        pols = unary_union(pols)
        adip_gs = tissue_gs.intersection(pols).polygonize()
        return adip_gs

    return (get_adip_gs,)


@app.cell
def _(Path, get_adip_gs):
    for fp in Path("dvc/P40ColIV/data/slides/HE/").glob("*.svs"):
        adip_gs = get_adip_gs(fp)
        adip_gs.to_file(
            f"/media/AprioricsSlides/adip_geojsons/P40ColIV/{fp.stem}.geojson"
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
