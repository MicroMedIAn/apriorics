#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_folder_raster.py — réécriture de predict_folder_old.py.

Principe
--------
Le pipeline historique vectorisait chaque masque de tuile
(mask_to_polygons_layer) puis fusionnait tous les polygones à la fin
(parallel_union) : sur les vraies données c'est cette fusion finale qui
bloquait (slide_1 : 8 h ; slide_2 : > 2 jours sans finir).

Ici, les masques prédits sont fusionnés AU NIVEAU RASTER dans un tableau
global de la slide (SlideMaskFuser, cf. mask_fusion_slide.py), puis le masque
complet est vectorisé UNE SEULE FOIS en fin de slide : les polygones de sortie
sont disjoints par construction (composantes connexes + trous), aucune union
GEOS n'est jamais nécessaire.

Ce qui disparaît par rapport à predict_folder_old.py
----------------------------------------------------
* multiprocessing.Pool : SUPPRIMÉ. Il était recréé à chaque batch (coûteux),
  crashait avec le défaut --num_workers 0 (Pool(processes=0) lève ValueError)
  et ne servait qu'au post-traitement, qui est désormais trivial (OU logique
  dans le processus parent, ~ms par batch). L'inférence reste parallèle via le
  DataLoader (num_workers), qui est le vrai goulot d'E/S.
* mask_to_polygons_layer par tuile : remplacé par add_tile (OU logique).
* parallel_union / grid_size : plus aucune union vectorielle.
* lecture du gpkg hoverfast à chaque masque : remplacée par UNE lecture par
  slide + déduplication exacte des noyaux (WKT).
* conversion image numpy (x -> uint8) à chaque batch : faite seulement si
  --flood_mask est actif.

Gestion de --hoverfastfolder
----------------------------
En fin de slide, les zones prédites fusionnées sont vectorisées puis filtrées
par aire (min_area) ; chaque noyau unique du gpkg reçoit son ratio de
couverture inter(noyau, zones_predites) / aire(noyau) — même formule que
l'original (+1e-7, seuil iou_threshold) — calculé par intersections
vectorisées (STRtree, sans union GEOS puisque les composantes sont
disjointes). Les noyaux conservés sont écrits tels quels (pas d'union : les
doublons issus des crops recouvrants sont déjà éliminés par déduplication).

Différences de comportement assumées (toutes dans le sens "plus fidèle")
------------------------------------------------------------------------
* hoverfast : le ratio est calculé contre les zones fusionnées de TOUTE la
  slide (union des crops) et non contre les polygones du seul masque courant :
  un noyau couvert à cheval sur deux crops n'est plus perdu aux frontières.
  Le résultat est un surensemble strict de l'ancien filtrage.
* min_area s'applique aux composantes FUSIONNÉES (et non aux morceaux découpés
  aux frontières de tuiles) : plus de "bites" artificielles dans les zones.
* les noyaux MultiPolygon ne sont plus silencieusement supprimés.
* --flood_mask : l'argument --area_threshold manquant (AttributeError dans
  l'original) est ajouté.
* --thr est désormais requis (l'original crashait s'il manquait).

Sortie
------
* sans hoverfast : composantes connexes des zones prédites (avec trous),
  coordonnées pixels du niveau analysé (même référentiel que l'original).
* avec hoverfast : noyaux uniques dont le ratio de couverture dépasse le seuil.
* --save_mask écrit le masque global brut en GeoTIFF (debug).

Prérequis : mask_fusion_slide.py dans le même dossier que ce script.
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from affine import Affine
from albumentations import Crop
from lightning_fabric.utilities.seed import seed_everything
from pathaia.util.paths import get_files
from timm import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.morphology import remove_small_holes

from apriorics.data import TestDataset
from apriorics.masks import flood_full_mask
from apriorics.model_components.normalization import group_norm
from apriorics.plmodules import BasicClassificationModule, BasicSegmentationModule
from apriorics.transforms import ToTensor
from apriorics.mask_fusion_slide import (  # noqa: E402
    SlideMaskFuser,
    build_crops,
    coverage_areas,
    mask_to_polygons,
)

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
    help=(
        "Number of workers for the DataLoader (lecture des slides). Le Pool de "
        "post-traitement de l'ancienne version n'existe plus. Default 0."
    ),
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
    help=(
        "Minimum area of merged components to keep, in pixels of the analysis "
        "level. Default 50."
    ),
)
parser.add_argument(
    "--max_area",
    type=int,
    default=5000,
    help="Unused (kept for CLI compatibility, as in the original script).",
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
    "--area_threshold",
    type=int,
    required=True
)
parser.add_argument(
    "--thr",
    type=float,
    required=True,
    help="Threshold for predictions to be considered positive.",
)
parser.add_argument(
    "--iou_threshold",
    type=float,
    default=0.5,
    help="Threshold for IOU between hoverfast nuc and prediction.",
)
parser.add_argument("--hoverfastfolder", type=Path)
parser.add_argument("--odd_even", type=int)
parser.add_argument(
    "--mask_mode",
    choices=["auto", "ram", "disque"],
    default="auto",
    help=(
        "Stockage du masque global de la slide. auto = ram si W*H tient dans "
        "--mask_ram_gb Go, sinon memmap disque (jamais de saturation RAM). "
        "Default auto."
    ),
)
parser.add_argument(
    "--mask_ram_gb",
    type=float,
    default=4.0,
    help="Limite (Go) du masque global en RAM en mode auto. Default 4.",
)
parser.add_argument(
    "--connectivity",
    type=int,
    choices=[4, 8],
    default=8,
    help=(
        "Connectivité de la vectorisation finale. 8 = plus fidèle à l'union "
        "GEOS historique (deux zones qui ne se touchent que par un coin sont "
        "fusionnées). Default 8."
    ),
)
parser.add_argument(
    "--save_mask",
    action="store_true",
    help="Écrire le masque global brut de chaque slide en GeoTIFF (debug).",
)


def _save_output(geoms, path: Path) -> None:
    """Écrit les géométries en GeoJSON ; FeatureCollection vide si rien."""
    if len(geoms):
        gpd.GeoSeries(list(geoms)).to_file(path)
        print(f"  -> {path} ({len(geoms)} entités)")
    else:
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
        print(f"  -> {path} (vide)")


def _save_mask_debug(mask, path: Path, x_min: float, y_min: float) -> None:
    """GeoTIFF du masque global brut (debug, réutilisable)."""
    import rasterio

    h, w = mask.shape
    tr = Affine(1.0, 0.0, float(x_min), 0.0, 1.0, float(y_min))
    with rasterio.open(str(path), "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="uint8", transform=tr, compress="deflate",
                       predictor=2, tiled=True) as dst:
        dst.write(np.asarray(mask), indexes=1)
    print(f"  masque debug : {path}")


def _process_slide(slide_path: Path, patches_path: Path, crops, args,
                   model, clf, device, outfolder: Path) -> None:
    out_path = outfolder / f"{slide_path.stem}.geojson"
    if out_path.exists():
        return
    print(slide_path.stem)

    fuser = None
    x_min = y_min = 0.0
    px_total = 0
    raw_path = None
    try:
        for crop in crops:
            ds = TestDataset(
                slide_path,
                patches_path,
                transforms=[Crop(*crop), ToTensor()],
            )
            if fuser is None:
                if not ds.patches:
                    print("  aucun patch dans le csv -> geojson vide")
                    _save_output([], out_path)
                    return
                # extent du masque global = emprise des patches de base_size
                xs0 = [p.position.x for p in ds.patches]
                ys0 = [p.position.y for p in ds.patches]
                x_min, y_min = float(min(xs0)), float(min(ys0))
                W = int(round(max(xs0) + args.base_size - x_min))
                H = int(round(max(ys0) + args.base_size - y_min))
                mode = args.mask_mode
                if mode == "auto":
                    mode = "ram" if W * H <= args.mask_ram_gb * 1e9 else "disque"
                if mode == "disque":
                    raw_path = outfolder / f".mask_{slide_path.stem}.raw"
                    fuser = SlideMaskFuser(W, H, mode="disque", path=str(raw_path))
                else:
                    fuser = SlideMaskFuser(W, H)
                print(f"  masque global {W} x {H} ({mode}), "
                      f"origine=({x_min:.0f}, {y_min:.0f})")

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
                y_hat = y_hat.cpu().numpy() > args.thr

                # conversion image uniquement si flood_full_mask en a besoin
                if args.flood_mask:
                    x_np = np.ascontiguousarray(
                        (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255),
                        dtype=np.uint8,
                    )

                for k in range(y_hat.shape[0]):
                    mask = y_hat[k]
                    if args.flood_mask:
                        mask = flood_full_mask(
                            x_np[k], mask, n=20, area_threshold=args.area_threshold
                        )
                    if not mask.any():
                        continue
                    mask = remove_small_holes(mask, area_threshold=args.area_threshold)
                    idx = batch_idx * args.batch_size + k
                    patch = ds.patches[idx]
                    x0 = int(round(patch.position.x + crop[0] - x_min))
                    y0 = int(round(patch.position.y + crop[1] - y_min))
                    fuser.add_tile(x0, y0, mask)
                    px_total += int(mask.sum())
                pbar.set_postfix({"px_pred": px_total})

        if fuser is None:  # aucune tuile traitée (csv vide sur tous les crops)
            _save_output([], out_path)
            return

        fuser.flush()
        print(f"  pixels prédits : {px_total} ; vectorisation du masque global...")
        polys = mask_to_polygons(
            fuser.mask, origin=(x_min, y_min), connectivity=args.connectivity
        )
        if args.save_mask:
            _save_mask_debug(fuser.mask, outfolder / f"{slide_path.stem}_mask.tif",
                             x_min, y_min)

        # min_area sur les composantes fusionnées (plus de bites de bordures)
        polys = [p for p in polys if p.area > args.min_area]
        print(f"  {len(polys)} composantes (aire > {args.min_area})")

        if args.hoverfastfolder is not None:
            # lecture UNIQUE du gpkg + déduplication exacte (les crops
            # recouvrants lisaient les mêmes noyaux plusieurs fois)
            nuc = gpd.read_file(
                args.hoverfastfolder / f"{slide_path.stem}.gpkg"
            )["geometry"]
            # déduplication exacte des noyaux (WKT vectorisé) : les crops
            # recouvrants lisaient plusieurs fois les mêmes noyaux
            wkts = nuc.to_wkt() if hasattr(nuc, "to_wkt") else nuc.apply(
                lambda g: g.wkt
            )
            nuc_arr = np.asarray(nuc.loc[~wkts.duplicated()].values, dtype=object)
            print(f"  {len(nuc_arr)} noyaux uniques dans le gpkg")
            if len(polys) and len(nuc_arr):
                # ratio = inter(noyau, zones prédites fusionnées)/aire(noyau),
                # même formule que l'original ; les composantes prédites sont
                # disjointes -> somme des intersections = intersection avec
                # l'union, sans aucun calcul d'union GEOS
                inter = coverage_areas(polys, list(nuc_arr))
                ratio = inter / (np.array([g.area for g in nuc_arr]) + 1e-7)
                keep = list(nuc_arr[ratio > args.iou_threshold])
            else:
                keep = []
            print(f"  {len(keep)} noyaux conservés (ratio > {args.iou_threshold})")
            _save_output(keep, out_path)
        else:
            _save_output(polys, out_path)
    finally:
        if raw_path is not None:
            try:
                del fuser
            except Exception:
                pass
            Path(raw_path).unlink(missing_ok=True)


if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    seed_everything(seed=args.seed, workers=True)

    trainfolder = args.trainfolder
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
    if len(model) > 1:
        encoder_name = "/".join(model[1:])
    else:
        encoder_name = None

    model = create_model(
        model[0],
        encoder_name=encoder_name,
        pretrained=True,
        img_size=args.patch_size,
        num_classes=1,
        norm_layer=group_norm if args.group_norm else torch.nn.BatchNorm2d,
    ).eval()
    model.requires_grad_(False)

    device = torch.device(f"cuda:{args.gpu}")
    model = BasicSegmentationModule(
        model,
        loss=None,
        dl_lengths=(0, 0),
        lr=0,
        wd=0,
    ).to(device)

    ckpt_path = logfolder / f"apriorics/{version}/checkpoints/last.ckpt"
    checkpoint = torch.load(ckpt_path, weights_only=False)
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

    crops = build_crops(args.patch_size, args.base_size)
    print(f"{len(crops)} crop(s) par patch de base_size")

    for slide_path, patches_path in zip(slide_paths, patches_paths):
        _process_slide(slide_path, patches_path, crops, args, model, clf,
                       device, outfolder)
