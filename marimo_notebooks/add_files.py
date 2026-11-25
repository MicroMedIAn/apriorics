import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    import numpy as np

    return np, pd


@app.cell
def _(pd):
    df = pd.read_csv("splits.csv")
    return (df,)


@app.cell
def _():
    with open("file_lists/AE1AE3") as _f:
        existing_cohort = set(_f.read().strip().split("\n"))
    return (existing_cohort,)


@app.cell
def _(existing_cohort):
    existing_cohort
    return


@app.cell
def _():
    total = 250
    n_test = int(0.1*total)
    n_val = int(0.2*total)
    n_train = total-n_val-n_test
    return n_test, n_train, n_val


@app.cell
def _(df, existing_cohort, n_test, n_train, n_val):
    n_test_new = n_test - len(df.loc[df["slide"].isin(existing_cohort) & (df["split"] == "test")])
    n_val_new = n_val - len(df.loc[df["slide"].isin(existing_cohort) & (df["split"] == "0")])
    n_train_new = n_train - len(df.loc[df["slide"].isin(existing_cohort) & ~df["split"].isin(("0", "test"))])
    return n_test_new, n_train_new, n_val_new


@app.cell
def _(n_test, n_test_new, n_train, n_train_new, n_val, n_val_new):
    n_test_new, n_test, n_val_new, n_val, n_train_new, n_train
    return


@app.cell
def _(df, existing_cohort, n_test_new, n_train_new, n_val_new, np):
    new_train = set(np.random.choice(df.loc[~df["slide"].isin(existing_cohort) & ~df["split"].isin(("0", "test")), "slide"], size=n_train_new, replace=False))
    new_val = set(np.random.choice(df.loc[~df["slide"].isin(existing_cohort) & (df["split"] == "0"), "slide"], size=n_val_new, replace=False))
    new_test = set(np.random.choice(df.loc[~df["slide"].isin(existing_cohort) & (df["split"] == "test"), "slide"], size=n_test_new, replace=False))
    return new_test, new_train, new_val


@app.cell
def _(existing_cohort, new_test, new_train, new_val):
    len(existing_cohort|new_train|new_val|new_test)
    return


@app.cell
def _(existing_cohort, new_test, new_train, new_val):
    with open("file_lists/AE1AE3", "w") as _f:
        _f.write("\n".join(sorted(existing_cohort|new_train|new_val|new_test)))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
