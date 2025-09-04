import json
import warnings
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import geopandas
import numpy as np
import torch
from albumentations import Crop
from pathaia.util.paths import get_files
from pytorch_lightning.utilities.seed import seed_everything
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from timm import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from apriorics.data import TestDataset
from apriorics.masks import flood_full_mask
from apriorics.model_components.normalization import group_norm
from apriorics.plmodules import BasicClassificationModule, BasicSegmentationModule
from apriorics.polygons import mask_to_polygons_layer
from apriorics.transforms import ToTensor

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

parser = ArgumentParser()
parser.add_argument(
    "--model",
    help=(
        "Model to use for training. If unet, can be formatted as unet/encoder to "
        "specify a specific encoder. Must be one of unet, med_t, logo, axalunet, gated."
    ),
    required=True,
)
parser.add_argument(
    "--ihc_type",
    choices=IHCS,
    help=f"Name of the IHC to train for. Must be one of {', '.join(IHCS)}.",
    required=True,
)
parser.add_argument(
    "--outfolder", type=Path, help="Output folder for geojsons.", required=True
)
parser.add_argument(
    "--trainfolder",
    type=Path,
    help="Folder containing all train files.",
    required=True,
)
parser.add_argument(
    "--slidefolder",
    type=Path,
    help="Input folder containing svs slide files.",
    required=True,
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="GPU index to use. Default 0.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help=(
        "Batch size for training. effective batch size is multiplied by the number of"
        " gpus. Default 8."
    ),
)
parser.add_argument(
    "--patch_size",
    type=int,
    default=1024,
    help="Size of the patches used for training. Default 1024.",
)
parser.add_argument(
    "--base_size",
    type=int,
    default=1024,
    help=(
        "Size of the patches used before crop for training. Must be greater or equal "
        "to patch_size. Default 1024."
    ),
)
parser.add_argument(
    "--level", type=int, default=0, help="WSI level for patch extraction. Default 0."
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers to use for data loading. Default 0 (only main process).",
)
parser.add_argument(
    "--group_norm",
    action="store_true",
    help="Specify to use group norm instead of batch norm in model. Optional.",
)
parser.add_argument(
    "--version",
    help="Hash (=version) of the model to load weights from.",
    required=True,
)
parser.add_argument(
    "--seed",
    type=int,
    help=(
        "Specify seed for RNG. Can also be set using PL_GLOBAL_SEED environment "
        "variable. Optional."
    ),
)
parser.add_argument(
    "--min_area",
    type=int,
    default=50,
    help="Minimum area of objects to keep. Default 50.",
)
parser.add_argument(
    "--max_area",
    type=int,
    default=5000,
    help="Minimum area of objects to keep. Default 50.",
)
parser.add_argument(
    "--slide_extension",
    default=".svs",
    help="File extension of slide files. Default .svs.",
)
parser.add_argument("--classif_model")
parser.add_argument("--classif_version")
parser.add_argument("--flood_mask", action="store_true")
parser.add_argument(
    "--thr", type=float, help="Threshold for predictions to be considered positive."
)
parser.add_argument(
    "--iou_threshold",
    type=float,
    default=0.5,
    help="Threshold for IOU between hoverfast nuc and prediction.",
)
parser.add_argument("--hoverfastfolder", type=Path)
parser.add_argument("--odd_even", type=int)


if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    seed_everything(seed=args.seed, workers=True)

    trainfolder = args.trainfolder / args.ihc_type
    patch_csv_folder = args.outfolder / f"{args.base_size}_{args.level}/patch_csvs"
    slidefolder = args.slidefolder
    logfolder = args.trainfolder / "logs"

    version = args.version

    patches_paths = get_files(
        patch_csv_folder, extensions=".csv", recurse=False
    ).sorted(key=lambda x: x.stem)
    if args.hoverfastfolder is not None:
        patches_paths = patches_paths.filter(
            lambda x: (args.hoverfastfolder / f"{x.stem}.gpkg").exists()
        )
    if args.odd_even is not None:
        if args.odd_even == 0:
            patches_paths = patches_paths[::2]
        else:
            patches_paths = patches_paths[1::2]
    slide_paths = patches_paths.map(
        lambda x: slidefolder / x.with_suffix(args.slide_extension).name
    )

    model = args.model.split("/")
    if model[0] == "unet":
        encoder_name = model[1]
    else:
        encoder_name = None

    device = torch.device(f"cuda:{args.gpu}")
    model = create_model(
        model[0],
        encoder_name=encoder_name,
        pretrained=True,
        img_size=args.patch_size,
        num_classes=1,
        norm_layer=group_norm if args.group_norm else torch.nn.BatchNorm2d,
    ).eval()
    model.requires_grad_(False)

    model = BasicSegmentationModule(
        model,
        loss=None,
        dl_lengths=(0, 0),
        lr=0,
        wd=0,
    ).to(device)

    ckpt_path = logfolder / f"apriorics/{version}/checkpoints/last.ckpt"
    checkpoint = torch.load(ckpt_path)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)

    if args.classif_model is not None and args.classif_version is not None:
        clf = create_model(
            args.classif_model, num_classes=1, norm_layer=torch.nn.BatchNorm2d
        ).eval()
        clf.requires_grad_(False)

        clf = BasicClassificationModule(
            clf, loss=None, dl_lengths=(0, 0), lr=0, wd=0
        ).to(device)

        ckpt_path = (
            logfolder / f"apriorics/{args.classif_version}/checkpoints/last.ckpt"
        )
        checkpoint = torch.load(ckpt_path)
        missing, unexpected = clf.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
    else:
        clf = None

    suffix = "_clf" if clf is not None else ""
    outfolder = (
        args.outfolder
        / args.ihc_type
        / f"{version}{suffix}"
        / f"geojsons_{args.thr}_{args.iou_threshold}"
    )
    if not outfolder.exists():
        outfolder.mkdir(parents=True)

    if args.patch_size < args.base_size:
        interval = int(0.3 * args.patch_size)
        max_coord = args.base_size - args.patch_size
        crops = []
        for x in range(0, max_coord + 1, interval):
            for y in range(0, max_coord + 1, interval):
                crops.append((x, y, x + args.patch_size, y + args.patch_size))
            if max_coord % interval != 0:
                crops.append((x, max_coord, x + args.patch_size, args.base_size))
                crops.append((max_coord, x, args.base_size, x + args.patch_size))
        if max_coord % interval != 0:
            crops.append((max_coord, max_coord, args.base_size, args.base_size))
    else:
        crops = [(0, 0, args.base_size, args.base_size)]

    for slide_path, patches_path in zip(slide_paths, patches_paths):
        if (outfolder / f"{slide_path.stem}.geojson").exists():
            continue
        print(slide_path.stem)
        polygons = []
        for crop in crops:
            print(crop)
            ds = TestDataset(
                slide_path,
                patches_path,
                transforms=[Crop(*crop), ToTensor()],
            )
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            for batch_idx, x in (
                pbar := tqdm(
                    enumerate(dl),
                    total=len(dl),
                )
            ):
                x = x.to(device)
                y_hat = torch.sigmoid(model(x))
                if clf is not None:
                    y_hat *= torch.sigmoid(clf(x))[:, None, None]
                x = np.ascontiguousarray(
                    (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255),
                    dtype=np.uint8,
                )
                y_hat = y_hat.cpu().numpy() > args.thr

                def _get_polygons(k_mask):
                    k, mask = k_mask

                    if args.flood_mask:
                        img = x[k]
                        mask = flood_full_mask(
                            img, mask, n=20, area_threshold=args.area_threshold
                        )

                    if not mask.sum():
                        return

                    idx = batch_idx * args.batch_size + k
                    patch = ds.patches[idx]
                    polygon = mask_to_polygons_layer(mask, angle_th=0, distance_th=0)

                    if isinstance(polygon, Polygon):
                        polygon = MultiPolygon([polygon])

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gs = geopandas.GeoSeries(polygon.geoms)
                        x0 = patch.position.x + crop[0]
                        y0 = patch.position.y + crop[1]
                        gs = gs.translate(
                            xoff=x0,
                            yoff=y0,
                        )
                        gs = gs.loc[gs.area > args.min_area]
                        polygon = MultiPolygon(gs.values)
                        if args.hoverfastfolder is not None:
                            nuc_gs = geopandas.read_file(
                                args.hoverfastfolder / f"{slide_path.stem}.gpkg",
                                bbox=(
                                    x0,
                                    y0,
                                    x0 + args.patch_size,
                                    y0 + args.patch_size,
                                ),
                            )["geometry"]
                            inter = nuc_gs.intersection(polygon).area
                            iou = inter / (nuc_gs.area + 1e-7)
                            gs = nuc_gs.loc[iou > args.iou_threshold]
                    return gs

                with Pool(processes=args.num_workers) as pool:
                    all_gs = pool.map(_get_polygons, enumerate(y_hat))
                    pool.close()
                    pool.join()
                for gs in all_gs:
                    if gs is not None:
                        polygons.extend(
                            filter(lambda x: isinstance(x, Polygon), gs.values)
                        )
                pbar.set_postfix({"pols": str(len(polygons))})
        polygons = unary_union(polygons)
        if isinstance(polygons, Polygon):
            polygons = MultiPolygon(polygons=[polygons])

        with open(outfolder / f"{slide_path.stem}.geojson", "w") as f:
            json.dump(geopandas.GeoSeries(polygons.geoms).__geo_interface__, f)
