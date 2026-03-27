import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from openslide import OpenSlide as Slide
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.morphology import (
        binary_dilation,
        disk,
        remove_small_objects,
        remove_small_holes,
        label,
    )
    import cv2
    from apriorics.polygons import mask_to_polygons_layer
    from pathaia.patches import slide_rois_no_image
    from shapely.affinity import scale, translate
    from shapely.geometry import box, Polygon, MultiPolygon
    from shapely.ops import unary_union
    import geopandas as gpd
    import multiprocessing as mp
    import json
    from datetime import datetime
    from pathlib import Path
    return (
        MultiPolygon,
        Path,
        Polygon,
        Slide,
        binary_dilation,
        box,
        cv2,
        datetime,
        disk,
        gpd,
        json,
        mask_to_polygons_layer,
        mp,
        np,
        remove_small_objects,
        scale,
        slide_rois_no_image,
        translate,
        unary_union,
    )


@app.cell
def _(binary_dilation, cv2, disk, np, remove_small_objects):
    def flood_mask(img, mask, n=40):
        # labels, n = label(mask, return_num=True)    
        ii, jj = np.nonzero(mask)
        out = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        #for k in range(1, n+1):
            #ii, jj = np.nonzero(labels==k)
        for i, j in zip(ii, jj):        
            # m = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
            cv2.floodFill(
                img,
                out,
                (j, i),
                newVal=(0, 0, 0),
                loDiff=(n, n, n),
                upDiff=(n, n, n),
                flags=4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY,
            )
            # out |= m > 0
        return out[1:-1, 1:-1] > 0

    def get_mask_ink(img):
        mask1 = np.ones(img.shape[:2], dtype=bool)
        ranges1 = [(56, 96), (75, 119), (120, 170)]
        for c, r in enumerate(ranges1):
            a, b = r
            mask1 &= (img[..., c] >= a) & (img[..., c] <= b)
        mask2 = np.ones(img.shape[:2], dtype=bool)
        ranges2 = [(38, 55), (40, 70), (75, 115)]
        for c, r in enumerate(ranges2):
            a, b = r
            mask2 &= (img[..., c] >= a) & (img[..., c] <= b)    
        mask = remove_small_objects(mask1 | mask2, min_size=3)
        if not mask.any():
            return mask
        mask = binary_dilation(
            flood_mask(img, mask, 15), footprint=disk(50)
        )
        return mask
    return (get_mask_ink,)


@app.cell
def _(box, gpd, slide_rois_no_image):
    def get_patch_iter(slide, psize, level, interval, tissue_path):
        tissue_gs = gpd.read_file(
            tissue_path,
            engine="pyogrio",
            use_arrow=True,
        )["geometry"]
        for patch in slide_rois_no_image(
            slide,
            level,
            (psize, psize),
            (interval, interval),
            thumb_size=5000,
        ):
            if not tissue_gs.intersects(
                box(*patch.position, *(patch.position + patch.size_0))
            ).any():
                continue    
            yield patch
    return (get_patch_iter,)


@app.cell
def _(
    MultiPolygon,
    Polygon,
    get_mask_ink,
    mask_to_polygons_layer,
    np,
    scale,
    translate,
):
    def pool_init_func(arg_slide):
        global slide
        slide = arg_slide

    def get_full_mask_ink(patch):
        img = np.array(
            slide.read_region(patch.position, patch.level, patch.size).convert(
                "RGB"
            )
        )
        dsr = slide.level_downsamples[patch.level]
        mask = get_mask_ink(img)
        if not mask.any():
            return
        pols = mask_to_polygons_layer(mask, 0, 0)
        pols = translate(pols, *(patch.position / dsr))
        pols = scale(pols, dsr, dsr, origin=(0, 0, 0))
        pols = MultiPolygon([Polygon(pol.exterior) for pol in pols.geoms])
        return pols
    return get_full_mask_ink, pool_init_func


@app.cell
def _(
    Path,
    Slide,
    datetime,
    get_full_mask_ink,
    get_patch_iter,
    gpd,
    mp,
    pool_init_func,
    unary_union,
):
    def get_slide_ink_mask(slide_path, ihc):
        slide = Slide(str(slide_path))
        slidename = slide_path.name.split("-")[0]
        out_path = Path(
            f"/media/AprioricsSlides/ink_geojsons/{ihc}/{slidename}.geojson"
        )
        if out_path.exists():
            return
        print(datetime.now().isoformat(), slidename)
        level = 1
        psize = 2_000
        interval = -int(0.1 * psize)
        tissue_path = (
            f"/media/AprioricsSlides/tissue_geojsons/{ihc}/{slidename}.geojson"
        )
        with mp.Pool(
            processes=30, initializer=pool_init_func, initargs=(slide,)
        ) as pool:
            patch_iter = get_patch_iter(slide, psize, level, interval, tissue_path)
            all_pols = pool.map(get_full_mask_ink, patch_iter)
            pool.close()
            pool.join()
        all_pols = [pols for pols in all_pols if pols is not None]
        all_pols = unary_union(all_pols)
        gpd.GeoSeries(all_pols).to_file(out_path)
    return (get_slide_ink_mask,)


@app.cell
def _(json):
    with open("ihc_mapping.json") as f:
        ihc_mapping = json.load(f)
    return (ihc_mapping,)


@app.cell
def _(Path, get_slide_ink_mask, ihc_mapping):
    ihc_order = ["P40ColIV", "CD3CD20", "CD163", "AE1AE3", "PHH3", "ERGPodoplanine", "ERGCaldesmone", "INI1", "EMD"]
    for ihc in ihc_order:
        print(ihc)
        ihc_n = ihc_mapping["HE"][ihc]
        for slide_path in Path("/media/AprioricsSlides/").glob(f"*-{ihc_n}_*.svs"):
            Path(f"/media/AprioricsSlides/ink_geojsons/{ihc}").mkdir(exist_ok=True)
            get_slide_ink_mask(slide_path, ihc)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
