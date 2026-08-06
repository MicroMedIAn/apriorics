import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import comet_ml
    import seaborn as sns
    import dvc.api
    import polars as pl
    from collections import defaultdict

    return comet_ml, defaultdict, dvc, pl


@app.cell
def _(dvc, pl):
    df = (
        pl.DataFrame(dvc.api.exp_show(repo="dvc/P40ColIV/"))
        .filter(
            (pl.col("typ") == "branch_commit") & (pl.col("State") == "Success")
        )
        .select("Experiment", "rev")
    )
    df
    return (df,)


@app.cell
def _(comet_ml):
    api = comet_ml.api.API()
    return (api,)


@app.cell
def _(api, defaultdict, df, pl):
    data = defaultdict(list)
    for exp_name in df["Experiment"]:
        exp = api.get(workspace="apriorics", project_name="apriorics", experiment=exp_name)
        data["Experiment"].append(exp_name)
        data["version"].append(exp.key)
        for param in ("p_pos", "model", "loss"):
            data[param].append(exp.get_parameters_summary(param)["valueCurrent"])
        for metric in ("AUROC", "AUPRC", "BinaryRecall", "BinaryPrecision", "DiceScore", "BinaryJaccardIndex"):
            try:
                data[metric].append(float(exp.get_metrics_summary(metric)["valueCurrent"]))
            except ValueError:
                data[metric].append(None)
    res_df = pl.DataFrame(data).cast({"p_pos": float})
    return (res_df,)


@app.cell
def _(pl, res_df):
    res_df.filter(
        (pl.col("AUPRC") >= res_df["AUPRC"].top_k(3).to_numpy()[-1])
        | (pl.col("BinaryRecall") >= res_df["BinaryRecall"].top_k(3).to_numpy()[-1])
        | (pl.col("BinaryPrecision") >= res_df["BinaryPrecision"].top_k(3).to_numpy()[-1])
        | (pl.col("DiceScore") >= res_df["DiceScore"].top_k(3).to_numpy()[-1])
        | (pl.col("BinaryJaccardIndex") >= res_df["BinaryJaccardIndex"].top_k(3).to_numpy()[-1])
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
