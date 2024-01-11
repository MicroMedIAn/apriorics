import json
from os import PathLike
from typing import Optional

import numpy as np
from albumentations.augmentations.crops.functional import get_center_crop_coords
from nptyping import Int, NDArray, Number, Shape
from pathaia.util.basic import ifnone
from pathaia.util.types import NDBoolMask
from rasterio import features
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon, shape


def get_reduced_coords(
    coords: NDArray[Shape["*, 2"], Number], angle_th: float, distance_th: float
) -> NDArray[Shape["*, 2"], Int]:
    r"""
    Given polygon vertices coordinates, deletes those that are too close or that form
    a too small angle.

    Args:
        coords: array of coordinates.
        angle_th: minimum angle (in degrees) formed by 3 consecutives vertices. If the
            angle is too small, the middle vertex will be deleted.
        distance_th: minimum distance between vertices. If 2 consecutive sides of the
            polygon are too small, the middle vertex will be deleted.

    Returns:
        Array of polygon coordinates with small and flat sides pruned.
    """
    vector_rep = np.diff(coords, axis=0)
    angle_th_rad = np.deg2rad(angle_th)
    points_removed = [0]
    while len(points_removed):
        points_removed = list()
        for i in range(len(vector_rep) - 1):
            if len(coords) - len(points_removed) == 3:
                break
            v01 = vector_rep[i]
            v12 = vector_rep[i + 1]
            d01 = np.linalg.norm(v01)
            d12 = np.linalg.norm(v12)
            if d01 < distance_th and d12 < distance_th:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
                continue
            angle = np.arccos(np.dot(v01, v12) / (d01 * d12))
            if angle < angle_th_rad:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
        coords = np.delete(coords, points_removed, axis=0)
        vector_rep = np.diff(coords, axis=0)
    return coords.astype(int)


def reduce_polygon(
    polygon: Polygon, angle_th: float = 0, distance_th: float = 0
) -> Polygon:
    r"""
    Given a :class:`shapely.geometry.Polygon`, delete vertices that create small or
    flat sides on the interior and on the exterior.

    Args:
        polygon: input polygon.
        angle_th: minimum angle (in degrees) formed by 3 consecutives vertices. If the
            angle is too small, the middle vertex will be deleted.
        distance_th: minimum distance between vertices. If 2 consecutive sides of the
            polygon are too small, the middle vertex will be deleted.

    Returns:
        Reduced polygon.
    """
    ext_poly_coords = get_reduced_coords(
        np.array(polygon.exterior.coords[:]), angle_th, distance_th
    )
    interior_coords = [
        get_reduced_coords(np.array(interior.coords[:]), angle_th, distance_th)
        for interior in polygon.interiors
    ]
    return Polygon(ext_poly_coords, interior_coords)


def mask_to_polygons_layer(
    mask: NDBoolMask, angle_th: float = 2, distance_th: float = 3
) -> MultiPolygon:
    """
    Convert mask array into :class:`shapely.geometry.MultiPolygon`.

    Args:
        mask: input mask array.

    Returns:
        :class:`shapely.geometry.MultiPolygon` where polygons are extracted from
        positive areas in the mask.
    """
    all_polygons = []
    for sh, _ in features.shapes(mask.astype(np.int16), mask=(mask > 0)):
        all_polygons.append(
            reduce_polygon(shape(sh), angle_th=angle_th, distance_th=distance_th)
        )

    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == "Polygon":
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def hovernet_to_wkt(
    infile: PathLike, outfile: PathLike, slide_height: Optional[int] = None
):
    """
    Take a hovernet json output file and convert it to a wkt file. Optionally convert
    coordinates to cytomine format.

    Args:
        infile: path to input hovernet json file.
        outfile: path to output wkt file.
        slide_height: height of the slide on which hovernet was used. If given,
            coordinates will be converted to cytomine format (`y = slide_height - y`).
    """
    with open(infile, "r") as f:
        hovernet_dict = json.load(f)
    polygons = []
    for k in hovernet_dict["nuc"]:
        polygon = Polygon(hovernet_dict["nuc"][k]["contour"])
        if polygon.is_valid:
            polygons.append(polygon)
    polygons = MultiPolygon(polygons)
    if slide_height is not None:
        polygons = affine_transform(polygons, [1, 0, 0, -1, 0, slide_height])

    with open(outfile, "w") as f:
        f.write(polygons.wkt)


def hovernet_to_geojson(
    infile: PathLike,
    outfile: PathLike,
    crop_size: Optional[int] = None,
    xoff: Optional[int] = None,
    yoff: Optional[int] = None,
):
    """
    Take a hovernet json output file and convert it to a geojson file. Optionally crop
    the center of the polygons.

    Args:
        infile: path to input hovernet json file.
        outfile: path to output wkt file.
        crop_size: Size of the crop zone.
    """
    type_info = {
        0: ["none", [0, 0, 0]],
        1: ["tumoral", [255, 0, 0]],
        2: ["immunitaire", [0, 255, 0]],
        3: ["conjonctif", [0, 0, 255]],
        4: ["nécrose", [255, 255, 0]],
        5: ["épithélial", [255, 165, 0]],
    }
    with open(infile, "r") as f:
        hovernet_dict = json.load(f)
    geojson_feats = []
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    for k in hovernet_dict["nuc"]:
        nuc = hovernet_dict["nuc"][k]
        coords = np.array(nuc["contour"])
        bbox = np.array(nuc["bbox"])
        offset = np.array([[ifnone(xoff, 0), ifnone(yoff, 0)]])
        coords += offset
        bbox += offset
        xmin, ymin = np.minimum((xmin, ymin), bbox[0])
        xmax, ymax = np.maximum((xmax, ymax), bbox[1])
        label, color = type_info[nuc["type"]]
        feat = {
            "id": k,
            "type": "Feature",
            "properties": {
                "objectType": "annotation",
                "classification": {"name": label, "color": color},
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords.tolist() + [coords[0].tolist()]],
                "bbox": bbox.flatten().tolist(),
            },
        }
        geojson_feats.append(feat)

    if crop_size is not None:
        x0, y0, x1, y1 = get_center_crop_coords(
            ymax - ymin, xmax - xmin, crop_size, crop_size
        )
        xmin += x0
        ymin += y0
        xmax += x1
        ymax += y1
        geojson_feats = [
            feat
            for feat in geojson_feats
            if feat["geometry"]["bbox"][0] >= xmin
            and feat["geometry"]["bbox"][1] >= ymin
            and feat["geometry"]["bbox"][2] <= xmax
            and feat["geometry"]["bbox"][3] <= ymax
        ]

    geojson_dict = {"type": "FeatureCollection", "features": geojson_feats}
    with open(outfile, "w") as f:
        json.dump(geojson_dict, f)
