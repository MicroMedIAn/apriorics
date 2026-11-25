import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import geopandas as gpd
    from shapely.geometry import MultiPolygon

    return (gpd,)


@app.cell
def _():
    import json
    with open("/data_fast/anapath/tumor_cell_fraction/cellvit/22T047837-1-0-1/cell_detection_corrected.geojson") as f:
        d = json.load(f)
        print(len(d["features"]), d["features"][0])
    return


@app.cell
def _(gpd):
    nuc_gdf = gpd.read_file(
        "/data_fast/anapath/tumor_cell_fraction/cellvit/22T047837-1-0-1/cell_detection_corrected.geojson",
        engine="pyogrio",
    )
    print(nuc_gdf.shape)
    return (nuc_gdf,)


@app.cell
def _(nuc_gdf):
    nuc_gdf
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
