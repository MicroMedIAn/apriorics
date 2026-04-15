import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from pathaia.util.types import Slide
    from skimage.color import rgb2hsv
    from skimage.morphology import remove_small_holes, remove_small_objects
    import marimo as mo
    import cv2
    from apriorics.polygons import mask_to_polygons_layer
    import geopandas as gpd

    return (
        Slide,
        cv2,
        gpd,
        mask_to_polygons_layer,
        mo,
        np,
        remove_small_holes,
        remove_small_objects,
        rgb2hsv,
    )


@app.cell
def _(Slide, np):
    slide = Slide("/media/AprioricsSlides/21I000162-1-06-20_110602.svs")
    x, y = 44323, 24110
    ihc = np.array(slide.read_region((x, y), 0, (5000, 5000)).convert("RGB"))
    return ihc, x, y


@app.cell
def _(cv2, np):
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

    return (flood_mask,)


@app.cell
def _(flood_mask, ihc, remove_small_holes, remove_small_objects, rgb2hsv):
    ihc_h = rgb2hsv(ihc)
    ihc_s = ihc_h[:, :, 1]
    ihc_v = ihc_h[:, :, 2]
    ihc_h = ihc_h[:, :, 0]
    mask_ihc_r = remove_small_objects(
        flood_mask(
            ihc,
            remove_small_holes(
                ((ihc_h < 5 / 360) | (ihc_h > 230 / 360))
                & (ihc_s > 0.3)
                & (ihc_v > 0.4),
                area_threshold=200,
            ),
        ),
        min_size=200,
    )
    return (mask_ihc_r,)


@app.cell
def _(mask_ihc_r, mo):
    mo.image(mask_ihc_r.astype(float))
    return


@app.cell
def _(gpd, mask_ihc_r, mask_to_polygons_layer, x, y):
    pols = mask_to_polygons_layer(mask_ihc_r, 0, 0)
    gs = gpd.GeoSeries(pols).translate(x, y)
    return (gs,)


@app.cell
def _(gs):
    gs.to_file("/media/AprioricsSlides/gt_geojsons/ColIV/21I000162.geojson")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
