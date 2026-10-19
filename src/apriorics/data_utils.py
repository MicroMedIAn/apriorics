import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance


def reverse_mapping(type_id, mapping):
    slide_type = "HE" if type_id < 13 else "IHC"
    ihc_types = []
    for ihc_type, id in mapping[slide_type].items():
        if id == type_id:
            ihc_types.append(ihc_type)
    if ihc_types:
        return slide_type, ihc_types
    raise ValueError


def get_info_from_filename(filename, mapping):
    split_name = filename.split("-")
    block = split_name[0]
    id = int(split_name[-1].split("_")[0])
    slide_type, ihc_types= reverse_mapping(id, mapping)

    return {"block": block, "slide_type": slide_type, "ihc_types": ihc_types}


def compute_dist(mask, resolution=None, dtype=None):
    res = np.zeros_like(mask, dtype=dtype)
    posmask = mask.astype(bool)
    if posmask.any():
        negmask = ~posmask
        res[:] = (
            eucl_distance(negmask, sampling=resolution) * negmask
            - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        )
    return res
