import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from argparse import ArgumentParser
    from pathlib import Path
    import marimo as mo
    import pandas as pd
    import json
    import numpy as np

    return ArgumentParser, Path, json, mo, np, pd


@app.cell
def _(mo):
    file_list_ui = mo.ui.file_browser("file_lists", label="File list path", multiple=False)
    n_ui = mo.ui.number(0, 1100, 10, label="Desired number of files in list")
    qc_ui = mo.ui.file_browser("..", multiple=False, filetypes=[".xlsx"], label="QC path")
    mo.vstack([
        file_list_ui,
        n_ui,
        qc_ui
    ])
    return file_list_ui, n_ui, qc_ui


@app.cell
def _(ArgumentParser, Path, file_list_ui, n_ui, qc_ui):
    parser = ArgumentParser()
    parser.add_argument("--file_list_path", type=Path, default=file_list_ui.value[0].path)
    parser.add_argument("-n", type=int, default=n_ui.value)
    parser.add_argument("--splits_path", type=Path, default="splits.csv")
    parser.add_argument("--mapping_path", type=Path, default="ihc_mapping.json")
    parser.add_argument("--qc_path", type=Path, default=qc_ui.value[0].path)
    return (parser,)


@app.cell
def _(parser):
    args = parser.parse_args()
    return (args,)


@app.cell
def _(args, pd):
    splits_df = pd.read_csv(args.splits_path)
    return (splits_df,)


@app.cell
def _(args, json):
    ihc = args.file_list_path.name
    with open(args.mapping_path) as _f:
        ihc_n = json.load(_f)["HE"][ihc]
    return ihc, ihc_n


@app.cell
def _(args, ihc, ihc_n, pd):
    qc_df = pd.read_excel(args.qc_path, sheet_name=f"{ihc_n}-{ihc}", engine="openpyxl")
    forbidden = set(qc_df.loc[qc_df["QC"] == 0, "slide"].values)
    return (forbidden,)


@app.cell
def _(args, forbidden):
    if args.file_list_path.exists():
        with open(args.file_list_path) as _f:
            existing_cohort = {r.rstrip("\n") for r in _f.readlines() if r}
    else:
        existing_cohort = None
    existing_cohort = existing_cohort - forbidden
    return (existing_cohort,)


@app.cell
def _(args):
    total = args.n
    n_test = int(0.1*total)
    n_val = int(0.2*total)
    n_train = total-n_val-n_test
    return n_test, n_train, n_val, total


@app.cell
def _(existing_cohort, n_test, n_train, n_val, splits_df):
    n_test_new = n_test - len(
        splits_df.loc[
            splits_df["slide"].isin(existing_cohort)
            & (splits_df["split"] == "test")
        ]
    )
    n_val_new = n_val - len(
        splits_df.loc[
            splits_df["slide"].isin(existing_cohort) & (splits_df["split"] == "0")
        ]
    )
    n_train_new = n_train - len(
        splits_df.loc[
            splits_df["slide"].isin(existing_cohort)
            & ~splits_df["split"].isin(("0", "test"))
        ]
    )
    return n_test_new, n_train_new, n_val_new


@app.cell
def _(
    existing_cohort,
    forbidden,
    n_test_new,
    n_train_new,
    n_val_new,
    np,
    splits_df,
):
    new_train = set(
        np.random.choice(
            splits_df.loc[
                ~splits_df["slide"].isin(existing_cohort | forbidden)
                & ~splits_df["split"].isin(("0", "test")),
                "slide",
            ],
            size=n_train_new,
            replace=False,
        )
    )
    new_val = set(
        np.random.choice(
            splits_df.loc[
                ~splits_df["slide"].isin(existing_cohort | forbidden)
                & (splits_df["split"] == "0"),
                "slide",
            ],
            size=n_val_new,
            replace=False,
        )
    )
    new_test = set(
        np.random.choice(
            splits_df.loc[
                ~splits_df["slide"].isin(existing_cohort | forbidden)
                & (splits_df["split"] == "test"),
                "slide",
            ],
            size=n_test_new,
            replace=False,
        )
    )
    return new_test, new_train, new_val


@app.cell
def _(existing_cohort, new_test, new_train, new_val, total):
    assert len(existing_cohort|new_train|new_test|new_val) == total
    return


@app.cell
def _(args, existing_cohort, new_test, new_train, new_val):
    with open(args.file_list_path, "w") as _f:
        _f.write(
            "\n".join(sorted(existing_cohort | new_train | new_test | new_val))
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
