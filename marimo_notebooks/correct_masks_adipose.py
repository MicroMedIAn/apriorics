import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notebook to delete adipocytes ink from existing masks and writing them back as gpkg and geojson
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

    return DataSourceError, Path, gpd, logging, mo, move, unary_union


@app.cell
def _(logging):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    LOGGER = logging.getLogger()
    return (LOGGER,)


@app.cell
def _(Path):
    in_gpkg_folder = Path("/media/AprioricsSlides/gt_gpkgs/")
    in_adip_folder = Path("/media/AprioricsSlides/adip_geojsons/")
    out_tmp_folder = Path.home() / ".tmp"
    out_tmp_folder.mkdir(exist_ok=True)
    out_geojson_folder = Path("/media/AprioricsSlides/gt_geojsons_new/")
    out_gpkg_folder = Path("/media/AprioricsSlides/gt_gpkgs_new/")
    return (
        in_adip_folder,
        in_gpkg_folder,
        out_geojson_folder,
        out_gpkg_folder,
        out_tmp_folder,
    )


@app.cell
def _(
    DataSourceError,
    LOGGER,
    gpd,
    in_adip_folder,
    in_gpkg_folder,
    move,
    out_geojson_folder,
    out_gpkg_folder,
    out_tmp_folder,
    unary_union,
):
    for ihc_folder in in_gpkg_folder.iterdir():
        if not ihc_folder.is_dir():
            continue

        out_geojson_ihc_folder = out_geojson_folder / ihc_folder.name
        out_geojson_ihc_folder.mkdir(parents=True, exist_ok=True)
        out_gpkg_ihc_folder = out_gpkg_folder / ihc_folder.name
        out_gpkg_ihc_folder.mkdir(parents=True, exist_ok=True)

        for annot_path in ihc_folder.glob("*.gpkg"):
            tmp_out_geojson = out_tmp_folder / f"{annot_path.stem}.geojson"
            tmp_out_gpkg = out_tmp_folder / annot_path.name
            out_geojson = out_geojson_ihc_folder / tmp_out_geojson.name
            out_gpkg = out_gpkg_ihc_folder / tmp_out_gpkg.name
            if out_geojson.exists() and out_gpkg.exists():
                continue
            LOGGER.info(annot_path.relative_to(in_gpkg_folder))

            # Sometimes a DataSourceError occurs when trying to directly open the geojson file (usually when too big)
            try:
                annot_gs = gpd.read_file(annot_path, engine="pyogrio")["geometry"]
            except DataSourceError:
                with annot_path.open() as f:
                    annot_gs = gpd.read_file(
                        f, engine="pyogrio", driver="GeoJSON"
                    )["geometry"]

            adip_gs = gpd.read_file(
                in_adip_folder / annot_path.relative_to(in_gpkg_folder).with_suffix(".geojson")
            )["geometry"]

            # Exclude all annotations that intersect with an ink polygon
            annot_gs = annot_gs.loc[annot_gs.disjoint(unary_union(adip_gs.values))]

            # Need to use a tmp folder because directly writing gpkg files to AprioricsSlides does not work
            annot_gs.to_file(tmp_out_geojson)
            annot_gs.to_file(tmp_out_gpkg)

            move(tmp_out_geojson, out_geojson)
            move(tmp_out_gpkg, out_gpkg)

    out_tmp_folder.rmdir()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
