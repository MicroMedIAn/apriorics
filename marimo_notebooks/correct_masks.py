import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notebook to delete blue ink from existing masks and writing them back as gpkg and geojson
    """)
    return


@app.cell
def _():
    import marimo as mo
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.enums import MergeAlg
    import numpy as np
    from openslide import OpenSlide
    from pathaia.patches import regular_grid
    import matplotlib.pyplot as plt
    import cv2
    from shapely.ops import unary_union
    from pathlib import Path
    import logging
    from shutil import move
    from pyogrio.errors import DataSourceError

    return (
        DataSourceError,
        MergeAlg,
        Path,
        cv2,
        gpd,
        logging,
        mo,
        move,
        np,
        plt,
        rasterize,
        unary_union,
    )


@app.cell
def _(logging):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    LOGGER = logging.getLogger()
    return (LOGGER,)


@app.cell
def _(Path):
    in_geojson_folder = Path("/media/AprioricsSlides/geojsons/")
    in_ink_folder = Path("/media/AprioricsSlides/ink_geojsons/")
    out_tmp_folder = Path.home() / ".tmp"
    out_tmp_folder.mkdir()
    out_geojson_folder = Path("/media/AprioricsSlides/gt_geojsons/")
    out_gpkg_folder = Path("/media/AprioricsSlides/gt_gpkgs/")
    return (
        in_geojson_folder,
        in_ink_folder,
        out_geojson_folder,
        out_gpkg_folder,
        out_tmp_folder,
    )


@app.cell
def _(
    DataSourceError,
    LOGGER,
    gpd,
    in_geojson_folder,
    in_ink_folder,
    move,
    out_geojson_folder,
    out_gpkg_folder,
    out_tmp_folder,
    unary_union,
):
    for ihc_folder in in_geojson_folder.iterdir():
        if not ihc_folder.is_dir():
            continue

        out_geojson_ihc_folder = out_geojson_folder / ihc_folder.name
        out_geojson_ihc_folder.mkdir(parents=True, exist_ok=True)
        out_gpkg_ihc_folder = out_gpkg_folder / ihc_folder.name
        out_gpkg_ihc_folder.mkdir(parents=True, exist_ok=True)

        for annot_path in ihc_folder.glob("*.geojson"):
            tmp_out_geojson = out_tmp_folder / annot_path.name
            tmp_out_gpkg = out_tmp_folder / f"{annot_path.stem}.gpkg"
            out_geojson = out_geojson_ihc_folder / tmp_out_geojson.name
            out_gpkg = out_gpkg_ihc_folder / tmp_out_gpkg.name
            if out_geojson.exists() and out_gpkg.exists():
                continue
            LOGGER.info(annot_path.relative_to(in_geojson_folder))

            # Sometimes a DataSourceError occurs when trying to directly open the geojson file (usually when too big)
            try:
                annot_gs = gpd.read_file(annot_path, engine="pyogrio")["geometry"]
            except DataSourceError:
                with annot_path.open() as f:
                    annot_gs = gpd.read_file(
                        f, engine="pyogrio", driver="GeoJSON"
                    )["geometry"]

            ink_gs = gpd.read_file(
                in_ink_folder / annot_path.relative_to(in_geojson_folder)
            )["geometry"]

            # Exclude all annotations that intersect with an ink polygon
            annot_gs = annot_gs.loc[annot_gs.disjoint(unary_union(ink_gs.values))]

            # Need to use a tmp folder because directly writing gpkg files to AprioricsSlides does not work
            annot_gs.to_file(tmp_out_geojson)
            annot_gs.to_file(tmp_out_gpkg)

            move(tmp_out_geojson, out_geojson)
            move(tmp_out_gpkg, out_gpkg)

    out_tmp_folder.rmdir()
    return


@app.cell
def _(MergeAlg, gpd, plt, rasterize):
    x, y = 32700, 49764
    _gs = gpd.read_file("21I000043.gpkg", bbox=(x, y, x+512, y+512), engine="pyogrio")["geometry"]
    # _gs = _gs.translate(-x, -y)
    _mask_test = rasterize(_gs.values, out_shape=(256, 256), transform=(2, 0, x, 0, 2, y), merge_alg=MergeAlg.replace)
    plt.imshow(_mask_test)
    return x, y


@app.cell
def _(gs):
    gs.get_coordinates(index_parts=True).index
    return


@app.cell
def _(cv2, gpd, np, plt):

    x, y = 32700, 49764
    _gs = gpd.read_file("21I000043.gpkg", bbox=(x, y, x+256, y+256), engine="pyogrio")["geometry"]
    # _gs = _gs.translate(-x, -y)
    _mask_test = cv2.fillPoly(np.zeros((256, 256), dtype=np.uint8), )
    plt.imshow(_mask_test)
    return x, y


@app.cell
def _(mask_slide, np, plt, x, y):
    _mask = np.array(mask_slide.read_region((x, y), 0, (256, 256)))
    plt.imshow(_mask)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
