from argparse import ArgumentParser
from pathlib import Path

import geopandas as gpd
import numpy as np
import yaml
from shapely import MultiPolygon

IHCS = [
    "AE1AE3",
    "CD163",
    "CD3CD20",
    "EMD",
    "ERGCaldes",
    "ERGPodo",
    "INI1",
    "P40ColIV",
    "PHH3",
]

parser = ArgumentParser(
    prog=(
        "Train a segmentation model for a specific IHC. To train on multiple gpus, "
        "should be called as `horovodrun -np n_gpus python train_segmentation.py "
        "--horovod`."
    )
)
parser.add_argument(
    "--ihc_type",
    choices=IHCS,
    help=f"Name of the IHC to train for. Must be one of {', '.join(IHCS)}.",
    required=True,
)
parser.add_argument("--evalfolder", type=Path, help="Evaluate folder.", required=True)
parser.add_argument(
    "--gtfolder", type=Path, help="Ground truth geojson folder.", required=True
)
parser.add_argument(
    "--hovernetfolder", type=Path, help="Hovernet folder.", required=True
)
parser.add_argument("--trainfolder", type=Path, help="Train folder.", required=True)
parser.add_argument(
    "--regfolder", type=Path, help="Folder with registration geojsons.", required=True
)
parser.add_argument(
    "--patch_size", type=int, help="Patch size used for eval.", required=True
)
parser.add_argument("--level", type=int, help="Patch extraction level.", default=0)
parser.add_argument(
    "--hash_file",
    help="Yaml file containing the hash (=version) of the model to load weights from.",
    required=True,
)
parser.add_argument(
    "--iou_threshold",
    type=float,
    default=0.5,
    help="Minimum iou of objects to keep. Default 50%.",
)
parser.add_argument("--fold", default="0", help="Fold used for validation. Default 0.")


if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    with open(args.hash_file, "r") as f:
        version = yaml.safe_load(f)[args.fold]

    evalfolder = args.evalfolder / args.ihc_type / version / "geojsons"
    gtfolder = args.gtfolder / args.ihc_type / "HE"
    hovernetfolder = args.hovernetfolder / args.ihc_type
    patchgjfolder = (
        args.trainfolder
        / args.ihc_type
        / f"{args.patch_size}_{args.level}/patch_geojsons"
    )
    regfolder = args.regfolder / args.ihc_type / "HE"

    fpfolder = (
        args.evalfolder
        / args.ihc_type
        / version
        / f"geojsons_fp_{int(args.iou_threshold*100)}"
    )
    fnfolder = (
        args.evalfolder
        / args.ihc_type
        / version
        / f"geojsons_fn_{int(args.iou_threshold*100)}"
    )
    metfolder = args.evalfolder / args.ihc_type / version / "metrics_hover"

    fpfolder.mkdir(exist_ok=True)
    fnfolder.mkdir(exist_ok=True)
    metfolder.mkdir(exist_ok=True)

    precs = []
    recs = []
    for evalpath in evalfolder.glob("*.geojson"):
        print(evalpath.stem)
        reg_gs = gpd.read_file(regfolder / evalpath.name)["geometry"]
        patch_gs = gpd.read_file(patchgjfolder / evalpath.name)["geometry"]
        included_gs = patch_gs.intersection(MultiPolygon(reg_gs.values.tolist()))

        print("Loading eval data...")
        eval_gs = gpd.read_file(evalpath, mask=included_gs)["geometry"]

        print("Loading GT data...")
        gt_gs = gpd.read_file(gtfolder / evalpath.name, mask=included_gs)["geometry"]

        print("Loading hovernet data...")
        hover_gs = gpd.read_file(
            hovernetfolder / evalpath.with_suffix(".gpkg").name, engine="pyogrio"
        )["geometry"]
        print("Merging hovernet and eval...")
        eval_idx, hover_idx = hover_gs.sindex.query(eval_gs)
        ious = (
            hover_gs.loc[hover_idx]
            .intersection(eval_gs.loc[eval_idx], align=False)
            .area
            / hover_gs.loc[hover_idx].area
        )
        hover_idx = hover_idx[ious > 0.2]
        eval_gs = hover_gs.loc[np.unique(hover_idx)]
        eval_gs.reset_index(inplace=True, drop=True)

        print("Computing...")
        gt_idx, eval_idx = eval_gs.sindex.query(gt_gs)
        ious = (
            eval_gs.loc[eval_idx].intersection(gt_gs.loc[gt_idx], align=False).area
            / eval_gs.loc[eval_idx].union(gt_gs.loc[gt_idx], False).area
        )
        tp_eval_idx = eval_idx[ious >= args.iou_threshold]
        tp_gt_idx = gt_idx[ious >= args.iou_threshold]

        fp_idx = ~eval_gs.index.isin(tp_eval_idx)
        fn_idx = ~gt_gs.index.isin(tp_gt_idx)

        fp_gs = eval_gs.loc[fp_idx]
        fn_gs = gt_gs.loc[fn_idx]

        fp_gs.to_file(fpfolder / evalpath.name)
        fn_gs.to_file(fnfolder / evalpath.name)

        tp_eval = len(np.unique(tp_eval_idx))
        tp_gt = len(np.unique(tp_gt_idx))

        prec = tp_eval / (tp_eval + fp_idx.sum() + 1e-7)
        rec = tp_gt / (tp_gt + fn_idx.sum() + 1e-7)
        recs.append(rec)
        precs.append(prec)

        with open(metfolder / f"{evalpath.stem}.txt", "w") as f:
            f.write(f"VPP: {prec*100:.2f}%\n")
            f.write(f"Sensitivity: {rec*100:.2f}%\n")

    prec = np.mean(precs)
    rec = np.mean(recs)
    with open(metfolder.parent / "average_metrics_hover.txt", "w") as f:
        f.write(f"Mean VPP: {prec*100:.2f}%\n")
        f.write(f"Mean Sensitivity: {rec*100:.2f}%\n")

    (Path.cwd() / ".fp_fn_done").touch()
