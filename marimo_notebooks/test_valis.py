import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from valis import registration
    from valis.micro_rigid_registrar import MicroRigidRegistrar
    import marimo as mo
    import numpy as np

    return MicroRigidRegistrar, np, registration


@app.cell
def _():
    slide_src_dir = "./marimo_notebooks/data/valis/in_slides"
    results_dst_dir = "./marimo_notebooks/data/valis/registered_slides"
    micro_reg_fraction = 0.25
    return micro_reg_fraction, results_dst_dir, slide_src_dir


@app.cell
def _(MicroRigidRegistrar, registration, results_dst_dir, slide_src_dir):
    registrar = registration.Valis(
        slide_src_dir,
        results_dst_dir,
        img_list=[
            #"./marimo_notebooks/data/valis/in_slides/21I000005-1-16-2_143842.svs",
            "./marimo_notebooks/data/valis/in_slides/21I000005-1-16-1_135140.svs",
            "./marimo_notebooks/data/valis/in_slides/21I000005-1-16-13_111619.svs",
        ],
        micro_rigid_registrar_cls=MicroRigidRegistrar,
        crop_for_rigid_reg=False,
        reference_img_f="./marimo_notebooks/data/valis/in_slides/21I000005-1-16-1_135140.svs"
    )
    return (registrar,)


@app.cell
def _(registrar):
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    return (error_df,)


@app.cell
def _(error_df):
    error_df
    return


@app.cell
def _(micro_reg_fraction, np, registrar):
    img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
    min_max_size = np.min([np.max(d) for d in img_dims])
    img_areas = [np.multiply(*d) for d in img_dims]
    max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
    micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)
    return (micro_reg_size,)


@app.cell
def _(micro_reg_size, registrar):
    micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)
    return (micro_error,)


@app.cell
def _(micro_error):
    micro_error
    return


@app.cell
def _(registrar):
    registrar.warp_and_save_slides("./marimo_notebooks/data/valis/registered_slides/in_slides/")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
