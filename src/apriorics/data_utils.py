import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance


def reverse_mapping(type_id, mapping):
    slide_type = "HE" if type_id < 13 else "IHC"
    for ihc_type, id in mapping[slide_type].items():
        if id == type_id:
            return slide_type, ihc_type
    raise ValueError


def get_info_from_filename(filename, mapping):
    split_name = filename.split("-")
    block = split_name[0]
    id = int(split_name[-1].split("_")[0])
    slide_type, ihc_type = reverse_mapping(id, mapping)

    return {"block": block, "slide_type": slide_type, "ihc_type": ihc_type}


def compute_dist(mask, resolution=None, dtype=None):
    res = np.zeros_like(mask, dtype=dtype)
    posmask = mask.astype(bool)
    if posmask.any():
        negmask = ~posmask
        res[:] = (
            eucl_distance(negmask, sampling=resolution) * negmask
            - (eucl_distance(posmask, resolution=resolution) - 1) * posmask
        )
    return res
