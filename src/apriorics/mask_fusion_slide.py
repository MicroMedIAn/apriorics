#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_fusion_slide.py
====================

Fusion des masques de prediction AU NIVEAU RASTER (slide entiere), puis
UNE SEULE vectorisation finale. Tout l'union vectorielle disparait.

Pipeline actuel (le goulot) :

    (par tuile)  NN -> masque -> mask_to_polygons_layer -> polygones
    (fin)        union vectorielle de tous les polygones
                 -> slide_1 : 28 760 s (~8 h) ; slide_2 : > 2 jours sans finir

Pipeline propose :

    (par tuile)  NN -> masque -> OU logique dans le tableau global de la slide
    (fin)        rasterio.features.shapes -> polygones DISJOINTS par
                 construction (composantes connexes + trous), donc
                 AUCUNE union vectorielle, donc plus de reduction finale.

Pourquoi il n'y a plus d'union a faire
--------------------------------------
La vectorisation finale transforme le masque binaire complet de la slide en
composantes connexes : chaque polygone de sortie correspond a une region
connectee, avec ses trous. Deux polygones de sortie ne se chevauchent jamais
et ne se touchent pas : les decoupes aux frontieres de tuiles ont disparu
AVANT la vectorisation, au lieu d'etre recolles apres coup en vectoriel.

Le tableau global de la slide (uint8, 1 octet/pixel) peut vivre :
  * en RAM      (mode="ram")    : W*H octets.
  * sur disque  (mode="disque") : numpy.memmap — le noyau ne garde en RAM que
    les pages chaudes (cache page, reclaims sous pression memoire) -> la RAM
    ne sature jamais, quelle que soit la taille de la slide.

Accumulation des tuiles
-----------------------
* Par defaut (shared=False) : les workers renvoient le masque de tuile au
  parent qui fait le OU logique. Aucun probleme de concurrence, transfert
  ~ w*h octets par tuile (negligeable pour des tuiles 1024-4096 px).
* shared=True : le tableau global est alloue dans multiprocessing.Array
  (c_ubyte, lock=False) AVANT le fork (exactement l'idee de
  multiprocessing.sharedctypes) et chaque worker ecrit SA zone directement :
  zero transfert. Necessite des tuiles disjointes (sinon shared_lock=True).
  Le mode "disque" est nativement partage (memmap MAP_SHARED) : shared=True
  marche sans Array supplementaire.

Integration dans predict_folder_old.py
--------------------------------------
    from mask_fusion_slide import SlideMaskFuser, run_tiles, finalize_and_save

    fuser = SlideMaskFuser(slide_w, slide_h)            # mode "ram"
    # Si W*H > ~4 Go : fuser = SlideMaskFuser(slide_w, slide_h, mode="disque",
    #                                        path="masque_slide.raw")

    def predict_tile(x, y, w, h):
        ...        # votre inference NN existante, renvoie un masque (h, w)

    tiles = [(x, y, w, h), ...]
    run_tiles(fuser, tiles, predict_tile, workers=30)
    # variante zero-copie : run_tiles(fuser, tiles, predict_tile,
    #                                 workers=30, shared=True)

    polys = finalize_and_save(fuser, "slide_polygons.geojson", simplify=0.5)

Dans la boucle de tuiles, remplacer l'appel mask_to_polygons_layer(mask) par
fuser.add_tile(x, y, mask) ; supprimer l'appel final union_parallele(...).
Si le NN sort des polygones plutot qu'un masque :
    fuser.add_tile_polygons(x, y, polys_tuile, w, h)   # coordonnees tuile
    fuser.add_tile_polygons(x, y, polys_slide, w, h, coords="slide")

Conventions et options
----------------------
* Sortie : x vers la droite, y vers le BAS (convention image, meme ordre que
  les indices du masque — identique au transforme identite de rasterio, donc
  compatible avec les coordonnees pixel de vos tuiles). pixel_size et origin
  permettent de repasser en unites physiques.
* connectivite=4 (defaut) : deux zones qui ne se touchent que par un coin
  restent separees ; connectivite=8 : elles sont fusionnees.
* simplify (Douglas-Peucker, tolerance en pixels) lisse l'effet d'escalier du
  raster : 0.5-1.0 px divise le nombre de sommets par ~10 sans changer l'aire
  de facon perceptible.
* Auto-rogner : une tuile qui deborde du bord de slide est rognee
  automatiquement (les masques partiels de bord de slide sont acceptes).
* Si vous avez deja les masques de tuiles sauves sur disque, run_tiles accepte
  aussi des tuiles (x, y, masque) : aucun worker n'est lance, le parent
  accumulate directement.

Budget RAM du masque global (uint8, 1 octet/pixel) :

    10 000 x 10 000  ->   100 Mo        40 000 x 40 000 -> 1.6 Go
    20 000 x 20 000  ->   400 Mo        60 000 x 60 000 -> 3.6 Go
    30 000 x 30 000  ->   900 Mo       100 000 x 100 000 -> 10 Go (mode disque)

Verification : python mask_fusion_slide.py --demo
              python mask_fusion_slide.py --info
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from ctypes import c_ubyte

import numpy as np

try:
    from affine import Affine
    from rasterio import features as _rio_features
    from rasterio.features import rasterize as _rio_rasterize
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _HAS_RASTERIO = False

from shapely.geometry import mapping as _shp_mapping
from shapely.geometry import shape as _shp_shape

LOG_TAG = "[raster_fusion]"


def log(msg: str) -> None:
    print(f"{LOG_TAG} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Helpers masques
# ---------------------------------------------------------------------------

def _as_u8(m) -> np.ndarray:
    """Normalise un masque 2D en uint8 0/1 contigu."""
    a = np.asarray(m)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    elif a.dtype != np.uint8:
        a = (a > 0).astype(np.uint8)
    if a.ndim != 2:
        raise ValueError(f"masque 2D attendu, shape={a.shape}")
    return np.ascontiguousarray(a)


def _clip_window(x, y, w, h, W, H):
    """Renvoie la fenetre (x0, y0, x1, y1) rognee aux bornes de la slide."""
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x) + int(w), W), min(int(y) + int(h), H)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Accumulateur du masque global
# ---------------------------------------------------------------------------

class SlideMaskFuser:
    """Accumulateur du masque binaire global de la slide.

    mode="ram"    : ndarray uint8 en memoire (W*H octets).
    mode="disque" : numpy.memmap uint8 sur disque ; la RAM n'heberge que le
                    cache page (reclamable par le noyau) -> pas de saturation.
    """

    def __init__(self, width: int, height: int, mode: str = "ram",
                 path: str | None = None):
        if not _HAS_RASTERIO:
            raise ImportError("rasterio est requis : pip install rasterio")
        self.width = int(width)
        self.height = int(height)
        self.mode = mode
        self.path = path
        self.n_tiles = 0
        self._warned_clip = False
        self._shared_arr = None
        if mode == "ram":
            mo = self.width * self.height / 1e6
            log(f"masque global RAM {self.width} x {self.height} uint8 = {mo:.0f} Mo")
            self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        elif mode == "disque":
            if path is None:
                raise ValueError("mode='disque' : fournir path=<fichier .raw>")
            mo = self.width * self.height / 1e6
            log(f"masque global DISQUE {self.width} x {self.height} uint8 = "
                f"{mo:.0f} Mo -> {path}")
            self.mask = np.memmap(path, dtype=np.uint8, mode="w+",
                                  shape=(self.height, self.width))
            self.mask[:] = 0
            self.mask.flush()
        else:
            raise ValueError(f"mode inconnu : {mode!r} (ram|disque)")

    # -- ajout de tuiles ----------------------------------------------------

    def add_tile(self, x: int, y: int, tile_mask) -> None:
        """OU logique du masque de tuile dans le masque global (cote parent).

        Une tuile qui deborde du bord de slide est rognee automatiquement.
        """
        tm = _as_u8(tile_mask)
        x0, y0, x1, y1 = _clip_window(x, y, tm.shape[1], tm.shape[0],
                                      self.width, self.height)
        if x1 <= x0 or y1 <= y0:
            return
        if (x0, y0, x1, y1) != (int(x), int(y), int(x) + tm.shape[1],
                                int(y) + tm.shape[0]) and not self._warned_clip:
            self._warned_clip = True
            log("tuiles rognees aux bords de slide (message une seule fois)")
        region = self.mask[y0:y1, x0:x1]
        np.maximum(region, tm[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x)],
                   out=region)
        self.n_tiles += 1
        if self.mode == "disque" and self.n_tiles % 16 == 0:
            self.mask.flush()

    def add_tile_polygons(self, x, y, polygons, tile_w, tile_h,
                          coords: str = "local") -> None:
        """Variante : rastérise des polygones de la tuile dans le masque global.

        coords="local" : les polygones sont en coordonnees tuile (0,0 = coin
        haut-gauche de la tuile).
        coords="slide" : les polygones sont deja en coordonnees slide
        (ils seront translates de (-x, -y) avant rasterisation).
        """
        if not polygons:
            self.n_tiles += 1
            return
        if coords == "slide":
            from shapely.affinity import translate
            polygons = [translate(p, xoff=-float(x), yoff=-float(y))
                        for p in polygons]
        tr = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        rast = _rio_rasterize(
            [(p, 1) for p in polygons],
            out_shape=(int(tile_h), int(tile_w)),
            transform=tr, fill=0, all_touched=False, dtype="uint8")
        self.add_tile(int(x), int(y), rast)

    # -- partage inter-processus --------------------------------------------

    def enable_shared(self) -> None:
        """Remplace le ndarray RAM par un multiprocessing.Array partage.

        A appeler AVANT la creation du Pool (les enfants heritent du bloc
        memoire partage par fork). En mode "disque" le memmap est deja
        partage (MAP_SHARED) : rien a faire.
        """
        if self.mode != "ram":
            log("mode disque : memmap deja partage entre processus (MAP_SHARED)")
            return
        if self._shared_arr is None:
            from multiprocessing.sharedctypes import RawArray
            old = self.mask
            arr = RawArray(c_ubyte, self.width * self.height)
            self._shared_arr = arr
            self.mask = np.ctypeslib.as_array(arr).reshape(
                self.height, self.width)
            del old
            mo = self.width * self.height / 1e6
            log(f"tableau partage alloue (multiprocessing.Array) : {mo:.0f} Mo")

    def flush(self) -> None:
        if self.mode == "disque":
            self.mask.flush()


# ---------------------------------------------------------------------------
# Vectorisation finale
# ---------------------------------------------------------------------------

def mask_to_polygons(mask, pixel_size: float = 1.0, origin=(0.0, 0.0),
                     connectivity: int = 4, y_down: bool = True):
    """Vectorisation unique du masque binaire -> polygones disjoints (trous inclus).

    Les polygones de sortie sont les composantes connexes du masque :
    disjoints par construction, aucune union necessaire.
    Fonctionne sur ndarray RAM comme sur numpy.memmap (sans copie du masque).

    NB : GDALPolygonize polygonise TOUTES les classes de valeurs, le fond (0)
    compris. On passe donc l'avant-plan lui-meme comme masque de validite GDAL
    pour ne jamais polygoniser le fond (sur une slide dense : des millions de
    poches de fond jetees immediatement = temps et GC pour rien).
    """
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8)
    ps = float(pixel_size)
    ox, oy = float(origin[0]), float(origin[1])
    if y_down:
        tr = Affine(ps, 0.0, ox, 0.0, ps, oy)
    else:
        tr = Affine(ps, 0.0, ox, 0.0, -ps, oy + m.shape[0] * ps)
    # Masque de validite GDAL = l'avant-plan lui-meme. uint8 0/1 est accepte
    # nativement par rasterio (doc : "Must evaluate to bool or uint8"), donc
    # AUCUNE copie pleine-taille, crucial pour les grands masques memmap.
    # D'eventuelles valeurs > 1 seraient polygonisees puis ecartees par le
    # filtre ci-dessous (impossible via add_tile, qui n'ecrit que 0/1).
    polys = []
    n_bg = 0
    t0 = time.time()
    for geom, val in _rio_features.shapes(m, mask=m,
                                          connectivity=int(connectivity),
                                          transform=tr):
        if int(val) == 1:
            polys.append(_shp_shape(geom))
        else:
            n_bg += 1
    dt = time.time() - t0
    log(f"vectorisation raster -> polygones (fond exclus via masque GDAL) : "
        f"{len(polys)} polygones, connexite={connectivity} ({dt:.1f} s)")
    if n_bg:
        log(f"regions de valeur != 1 ignorees : {n_bg} (masque non binaire ?)")
    return polys


def _vertex_count(polys) -> int:
    n = 0
    for p in polys:
        n += len(p.exterior.coords)
        for r in p.interiors:
            n += len(r.coords)
    return n


def _simplify_polys(polys, tol: float):
    try:
        import shapely
        arr = np.array(polys, dtype=object)
        return list(shapely.simplify(arr, float(tol), preserve_topology=True))
    except Exception:
        return [p.simplify(float(tol), preserve_topology=True) for p in polys]


def coverage_areas(base_polys, query_geoms):
    """Aire d'intersection de chaque geometrie avec l'union de base_polys.

    base_polys doit etre une liste de polygones DISJOINTS (c'est le cas de la
    sortie de mask_to_polygons) : la somme des intersections avec chaque
    polygone est alors exactement l'intersection avec l'union, sans aucun
    calcul d'union GEOS. Retourne un tableau float64 de len(query_geoms).
    """
    import shapely
    base = np.asarray(list(base_polys), dtype=object)
    query = np.asarray(list(query_geoms), dtype=object)
    out = np.zeros(len(query), dtype=np.float64)
    if len(base) == 0 or len(query) == 0:
        return out
    tree = shapely.STRtree(base)
    pairs = tree.query(query, predicate="intersects")
    if pairs.size:
        inter = shapely.intersection(base[pairs[1]], query[pairs[0]])
        np.add.at(out, pairs[0], shapely.area(inter))
    return out


def build_crops(patch_size: int, base_size: int):
    """Portage exact de la grille de crops de predict_folder_old.py.

    patch_size < base_size : fenetres glissantes d'interval int(0.3*patch_size)
    dans chaque patch de base_size (fusion par union des predictions
    recouvrantes). patch_size == base_size : un seul crop plein patch.
    """
    if patch_size < base_size:
        interval = int(0.3 * patch_size)
        max_coord = base_size - patch_size
        crops = []
        for x in range(0, max_coord + 1, interval):
            for y in range(0, max_coord + 1, interval):
                crops.append((x, y, x + patch_size, y + patch_size))
            if max_coord % interval != 0:
                crops.append((x, max_coord, x + patch_size, base_size))
                crops.append((max_coord, x, base_size, x + patch_size))
        if max_coord % interval != 0:
            crops.append((max_coord, max_coord, base_size, base_size))
        return crops
    return [(0, 0, base_size, base_size)]


def finalize_and_save(fuser: SlideMaskFuser, out_geojson: str | None = None,
                      pixel_size: float = 1.0, origin=(0.0, 0.0),
                      connectivity: int = 4, simplify: float = 0.0,
                      y_down: bool = True, save_mask_tif: str | None = None):
    """Vectorise le masque global, simplifie au besoin, ecrit le GeoJSON.

    Retourne la liste des polygones (shapely), disjoints par construction.
    """
    fuser.flush()
    if fuser.mode == "ram":
        log(f"pixels allumes : {int(fuser.mask.sum())}")
    polys = mask_to_polygons(fuser.mask, pixel_size=pixel_size, origin=origin,
                             connectivity=connectivity, y_down=y_down)
    if not polys:
        log("masque vide : aucun polygone")
        return []
    area = math.fsum(p.area for p in polys)
    nv = _vertex_count(polys)
    log(f"aire totale : {area:.0f} px^2 ; sommets : {nv}")
    if simplify and simplify > 0:
        t0 = time.time()
        polys = _simplify_polys(polys, simplify)
        nv2 = _vertex_count(polys)
        log(f"simplify(tol={simplify} px) : sommets {nv} -> {nv2} "
            f"({time.time() - t0:.1f} s)")
    if save_mask_tif:
        _save_mask_tif(fuser.mask, save_mask_tif, pixel_size, origin, y_down)
    if out_geojson:
        _save_geojson(polys, out_geojson)
    return polys


def _save_geojson(polys, path: str) -> None:
    feats = [{"type": "Feature",
              "properties": {"id": i, "area": float(p.area)},
              "geometry": _shp_mapping(p)} for i, p in enumerate(polys)]
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    mo = os.path.getsize(path) / 1e6
    log(f"GeoJSON ecrit : {path} ({len(polys)} entites, {mo:.1f} Mo)")


def _save_mask_tif(mask, path: str, pixel_size: float, origin, y_down: bool) -> None:
    """Sauvegarde du masque global en GeoTIFF compresse (debug / reutilisation)."""
    import rasterio
    h, w = mask.shape
    ps = float(pixel_size)
    if y_down:
        tr = Affine(ps, 0.0, float(origin[0]), 0.0, ps, float(origin[1]))
    else:
        tr = Affine(ps, 0.0, float(origin[0]), 0.0, -ps,
                    float(origin[1]) + h * ps)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="uint8", transform=tr, compress="deflate",
                       predictor=2, tiled=True) as dst:
        dst.write(np.asarray(mask), indexes=1)
    log(f"masque GeoTIFF ecrit : {path}")


# ---------------------------------------------------------------------------
# Orchestration parallele des tuiles
# ---------------------------------------------------------------------------

_G = {}  # etat herite par fork : predict, view, lock, W, H


def _init_worker(view, lock, predict_fn, W, H):
    global _G
    _G = {"view": view, "lock": lock, "predict": predict_fn, "W": W, "H": H}


def _predict_and_clip(x, y, w, h):
    m = _as_u8(_G["predict"](int(x), int(y), int(w), int(h)))
    if m.shape != (int(h), int(w)):
        raise ValueError(f"tuile ({x},{y}) : le masque renvoye {m.shape} != "
                         f"{(int(h), int(w))}")
    x0, y0, x1, y1 = _clip_window(x, y, w, h, _G["W"], _G["H"])
    if x1 <= x0 or y1 <= y0:
        return None, 0, 0
    return m[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x)], x0, y0


def _task_shared(item):
    """Worker : predict puis ecriture directe dans le tableau partage."""
    i, (x, y, w, h) = item
    mc, x0, y0 = _predict_and_clip(x, y, w, h)
    if mc is None:
        return i
    region = _G["view"][y0:y0 + mc.shape[0], x0:x0 + mc.shape[1]]
    if _G["lock"] is not None:
        with _G["lock"]:
            np.maximum(region, mc, out=region)
    else:
        np.maximum(region, mc, out=region)
    return i


def _task_parent(item):
    """Worker : predict puis renvoi du masque (le parent fait le OU)."""
    i, (x, y, w, h) = item
    mc, x0, y0 = _predict_and_clip(x, y, w, h)
    if mc is None:
        return (i, None, 0, 0)
    return (i, mc, x0, y0)


def _tiles_overlap(tiles) -> bool:
    ts = sorted(tiles, key=lambda t: (t[0], t[1]))
    n = len(ts)
    for i in range(n):
        x1, y1, w1, h1 = ts[i]
        x1_max = x1 + w1
        y1_max = y1 + h1
        for j in range(i + 1, n):
            x2, y2, w2, h2 = ts[j]
            if x2 >= x1_max:
                break
            if x1 < x2 + w2 and y1 < y2 + h2 and y2 < y1_max:
                return True
    return False


def run_tiles(fuser: SlideMaskFuser, tiles, predict_fn=None, workers: int = 8,
              start_method: str = "fork", shared: bool = False,
              shared_lock: bool = False, log_every: int = 25,
              chunksize: int = 1) -> None:
    """Exeecute la prediction par tuile et alimente le masque global.

    tiles : liste de (x, y, w, h) [predict_fn appelé dans les workers]
            ou de (x, y, masque)  [masques deja disponibles, pas de Pool]
    predict_fn(x, y, w, h) -> masque uint8/bool de forme (h, w).

    shared=False (defaut) : les workers renvoient les masques, le parent fait
        le OU. Robuste quel que soit le chevauchement des tuiles.
    shared=True : tableau global partage (mp.Array ou memmap) herite par fork,
        chaque worker ecrit SA zone : zero transfert. Les tuiles doivent etre
        disjointes ; sinon passer shared_lock=True.
    """
    if not _HAS_RASTERIO:
        raise ImportError("rasterio est requis : pip install rasterio")
    tiles = list(tiles)
    n = len(tiles)
    if n == 0:
        log("aucune tuile")
        return
    if all(len(t) == 3 for t in tiles):
        # masques deja disponibles : accumulation directe, pas de Pool
        t0 = time.time()
        done = 0
        for x, y, m in tiles:
            fuser.add_tile(x, y, m)
            done += 1
            if log_every and (done % log_every == 0 or done == n):
                rate = done / max(time.time() - t0, 1e-9)
                log(f"tuiles accumulees : {done}/{n} ({rate:.1f}/s)")
        return

    if predict_fn is None:
        raise ValueError("tuiles (x, y, w, h) : fournir predict_fn")

    if shared:
        fuser.enable_shared()
        if not shared_lock and _tiles_overlap([(t[0], t[1], t[2], t[3])
                                               for t in tiles]):
            raise ValueError("tuiles chevauchantes : utiliser shared_lock=True "
                             "ou shared=False")

    ctx = mp.get_context(start_method)
    t0 = time.time()
    mode_acc = "partagee (ecriture worker)" if shared else "cote parent"
    log(f"n={n} tuiles, workers={workers}, start_method={start_method}, "
        f"accumulation={mode_acc}")
    init = _init_worker
    initargs = (fuser.mask if shared else None,
                mp.Lock() if (shared and shared_lock) else None,
                predict_fn, fuser.width, fuser.height)
    with ctx.Pool(workers, initializer=init, initargs=initargs) as pool:
        task = _task_shared if shared else _task_parent
        done = 0
        it = pool.imap_unordered(task, enumerate(tiles), chunksize)
        if shared:
            for _i in it:
                done += 1
                if log_every and (done % log_every == 0 or done == n):
                    rate = done / max(time.time() - t0, 1e-9)
                    log(f"tuiles predict : {done}/{n} ({rate:.1f}/s)")
        else:
            for _i, mc, x0, y0 in it:
                if mc is not None:
                    fuser.add_tile(x0, y0, mc)
                done += 1
                if log_every and (done % log_every == 0 or done == n):
                    rate = done / max(time.time() - t0, 1e-9)
                    log(f"tuiles predict : {done}/{n} ({rate:.1f}/s)")
    fuser.flush()
    dt = time.time() - t0
    extra = ""
    if fuser.mode == "ram":
        extra = f", pixels allumes : {int(fuser.mask.sum())}"
    log(f"tuiles terminees : {dt:.1f} s{extra}")


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

def _demo(tile: int = 1024, gx: int = 4, gy: int = 3, workers: int = 4) -> int:
    """Preuve de bout en bout sur une slide synthetique : 3 modes compares."""
    W, H = tile * gx, tile * gy
    log(f"DEMO slide {W} x {H}, {gx * gy} tuiles de {tile}^2, workers={workers}")

    def predict(x, y, w, h):
        m = np.zeros((h, w), dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        X = xx + int(x)
        Y = yy + int(y)
        # cercles r=580 centres sur chaque noeud de la grille de tuiles :
        # chevauchements entre tuiles voisines + trous enclosure (512<r<591)
        for cxx in range(0, W + 1, tile):
            for cyy in range(0, H + 1, tile):
                m[((X - cxx) ** 2 + (Y - cyy) ** 2) <= 580 ** 2] = 1
        return m

    tiles = [(gx_ * tile, gy_ * tile, tile, tile)
             for gy_ in range(gy) for gx_ in range(gx)]

    # mode 1 : RAM, accumulation cote parent
    f1 = SlideMaskFuser(W, H)
    run_tiles(f1, tiles, predict, workers=workers, log_every=0)
    p1 = mask_to_polygons(f1.mask)

    # mode 2 : RAM partagee (mp.Array + fork)
    f2 = SlideMaskFuser(W, H)
    run_tiles(f2, tiles, predict, workers=workers, shared=True, log_every=0)
    p2 = mask_to_polygons(f2.mask)

    # mode 3 : disque (memmap, partage natif)
    tmpdir = tempfile.mkdtemp(prefix="raster_fusion_demo_")
    f3 = SlideMaskFuser(W, H, mode="disque",
                        path=os.path.join(tmpdir, "masque_demo.raw"))
    run_tiles(f3, tiles, predict, workers=workers, shared=True, log_every=0)
    p3 = mask_to_polygons(f3.mask)

    ok = True
    ok &= np.array_equal(f1.mask, f2.mask)
    ok &= np.array_equal(f1.mask, np.asarray(f3.mask))
    ok &= len(p1) == len(p2) == len(p3)
    a1, a2, a3 = (math.fsum(p.area for p in pp) for pp in (p1, p2, p3))
    ok &= abs(a1 - a2) < 1e-6 * max(a1, 1) and abs(a1 - a3) < 1e-6 * max(a1, 1)
    holes = sum(len(p.interiors) for p in p1)
    log(f"polygones : {len(p1)} ; trous : {holes} ; aire : {a1:.0f} px^2")
    log(f"masques identiques (ram vs partagé vs disque) : {bool(ok)}")
    if not ok:
        print("DEMO : ECHEC", file=sys.stderr)
        return 1
    print("DEMO : OK")
    return 0


def _info() -> None:
    print("Budget RAM du masque global uint8 (1 octet/pixel) :")
    for k in (4, 10, 20, 30, 40, 60, 80, 100):
        s = k * 1000
        mo = s * s / 1e6
        go = mo / 1000
        print(f"  {s:>7} x {s:<7} -> {mo:8.0f} Mo ({go:.1f} Go)"
              + ("  <- mode disque conseille" if go >= 4 else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fusion raster des masques de tuiles au niveau de la "
                    "slide, puis vectorisation unique (aucune union vectorielle).")
    ap.add_argument("--demo", action="store_true",
                    help="auto-test synthetique de bout en bout")
    ap.add_argument("--info", action="store_true",
                    help="budget RAM du masque global selon la taille de slide")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tile", type=int, default=1024)
    args = ap.parse_args(argv)
    if args.info:
        _info()
        return 0
    if args.demo:
        return _demo(tile=args.tile, workers=args.workers)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
