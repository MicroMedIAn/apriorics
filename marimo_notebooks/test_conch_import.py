import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from segmentation_models_pytorch import create_model

    return (create_model,)


@app.cell
def _(create_model):
    model = create_model("dpt", encoder_name="tu-hf-hub:MahmoodLab/CONCH-Z", img_size=224, encoder_weights="True")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
