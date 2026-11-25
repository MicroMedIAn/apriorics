import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from shutil import copyfile
    from pathlib import Path
    import torch
    from pathaia.util.types import Slide, Patch
    import numpy as np

    return Path, Slide, copyfile, np, pd, torch


@app.cell
def _(pd):
    df = pd.read_pickle("~/primacode_data.pkl")
    file_list = [fp for path_list in df["HE_pathes"].values for fp in path_list ]
    return (file_list,)


@app.cell
def _(file_list):
    file_list
    return


@app.cell
def _(Path, copyfile, file_list):
    for fp in file_list:
        if fp.startswith("/data"):
            fp = fp.replace("/data/prima_code/24-004_PRIMACODE", "/mnt/imagin/SELVES Janick/24-004.PRIMACODE")
        fp = Path(fp)
        out_fp = Path("/data_fast/anapath/prima_code/slides/") / fp.name
        if not out_fp.exists():
            copyfile(fp, out_fp)
    return


@app.cell
def _(Slide):
    slide = Slide("/data_fast/anapath/prima_code/slides/HE/23F000436-1-01-3_122229.svs")
    h, w = slide.dimensions
    t_h = int(1000 * min(1, h/w))
    t_w = int(1000 * min(1, w/h))
    return t_h, t_w


@app.cell
def _(np, t_h, t_w):
    mask = np.zeros((t_h, t_w))
    return


@app.cell
def _(torch):
    coords = torch.load("/data_fast/anapath/prima_code/virchow_tiles/0/23F000436/coords.pt")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
