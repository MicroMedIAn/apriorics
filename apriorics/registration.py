from numbers import Number
from os import PathLike
from skimage.color import rgb2hed
import cv2
from skimage.util import img_as_ubyte, img_as_float
import numpy as np
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_closing,
    binary_dilation,
)
from skimage.measure import label
from subprocess import run
from pathlib import Path
from skimage.io import imsave
from pathaia.util.types import Coord
from .masks import get_dab_mask, get_tissue_mask
from pathaia.util.types import Slide, NDBoolMask, Patch, NDByteGrayImage, NDByteImage
from nptyping import NDArray
from typing import Any, Callable, List, Sequence, Tuple, Union


def get_dot_mask(
    slide: Slide, thumb_level: int = 3, min_val: int = 60, max_val: int = 90
) -> NDBoolMask:
    r"""
    Given a slide, computes a thumbnail mask where gray fixing dots are segmented.

    Args:
        thumb_level: pyramid level on which to extract the thumbnail.
        min_val: min value to filter on RGB channels.
        max_val: max value to filter on RGB channels.

    Returns:
        Mask array with fixing dots segmented.
    """
    thumb = np.asarray(
        slide.read_region(
            (0, 0),
            thumb_level,
            slide.level_dimensions[thumb_level],
        )
    )
    mask = np.ones(thumb.shape[:2], dtype=bool)
    for i in range(3):
        mask = mask & (min_val < thumb[:, :, i]) & (thumb[:, :, i] < max_val)
        mask = remove_small_holes(remove_small_objects(mask), area_threshold=500)
    return mask


def get_angle(v1: NDArray[(2,), Number], v2: NDArray[(2,), Number]) -> float:
    r"""
    Get angle between two 2D vectors.

    Args:
        v1: first input vector.
        v2: second input vector.

    Returns:
        Angle in radian between -pi and +pi.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.dot(v1, v2)
    angle = np.arccos(dot) * np.sign((v2 - v1)[1])
    return angle


def get_sort_key(
    vertices: Sequence[Tuple[Number, Number]]
) -> Callable[[Tuple[Number, Number]], float]:
    r"""
    Computes a key function that aims at sorting vertices by increasing angle around
    their centroid.

    Args:
        vertices: array of vertices coordinates.

    Returns:
        Function that takes a vertex as input and returns the angle between the (1, 0)
        horizontal unit vector and the vector between the vertices' centroid and the
        input vertex.
    """
    center = np.array(vertices).mean(0)

    def _key(vertex):
        vertex = np.array(vertex)
        vector = vertex - center
        return get_angle(np.array([1, 0]), vector)

    return _key


def get_vertices(mask: NDBoolMask) -> List[Tuple[int, int]]:
    r"""
    Given a fixing dot mask (obtained using :func:`get_dot_mask`), get list of all dots'
    centroid coordinates, sorted in trigonometric order.

    Args:
        mask: input dot mask array.

    Returns:
        List of vertex coordinates as tuples.
    """
    labels = label(mask)
    vertices = []
    for i in range(1, labels.max() + 1):
        ii, jj = (labels == i).nonzero()
        vertices.append((int(jj.mean()), int(ii.mean())))
    vertices.sort()
    while len(vertices) > 4:
        vertices.pop()
    vertices.sort(key=get_sort_key(vertices))
    return vertices


def get_rotation(
    fixed_vert: NDArray[(Any, 2), int], moving_vert: NDArray[(Any, 2), int]
) -> NDArray[(3, 3), float]:
    r"""
    Given 2 lists of vertices coordinates sorted in trigonometric order, get the
    rotation transform that aligns them.

    Args:
        fixed_vert: first vertices coordinates array considered as fixed for the
            rotation.
        moving_vert:  first vertices coordinates array considered as moving for the
            rotation.

    Returns:
        2D rotation 3x3 matrix.
    """
    angle = get_angle(moving_vert[1] - moving_vert[0], fixed_vert[1] - fixed_vert[0])
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def get_scale(
    fixed_vert: NDArray[(Any, 2), int], moving_vert: NDArray[(Any, 2), int]
) -> NDArray[(3, 3), float]:
    r"""
    Given 2 lists of vertices coordinates sorted in trigonometric order, get the
    scale transform that aligns them.

    Args:
        fixed_vert: first vertices coordinates array considered as fixed for the
            scale.
        moving_vert:  first vertices coordinates array considered as moving for the
            scale.

    Returns:
        2D scale 3x3 matrix.
    """
    x_moving_min, y_moving_min = moving_vert.min(0)
    x_moving_max, y_moving_max = moving_vert.max(0)
    x_fixed_min, y_fixed_min = fixed_vert.min(0)
    x_fixed_max, y_fixed_max = fixed_vert.max(0)
    if x_moving_max - x_moving_min < 5 or x_fixed_max - x_fixed_min < 5:
        scale_y = (y_fixed_max - y_fixed_min) / (y_moving_max - y_moving_min)
        scale_x = scale_y
    elif y_moving_max - y_moving_min < 5 or y_fixed_max - y_fixed_min < 5:
        scale_x = (x_fixed_max - x_fixed_min) / (x_moving_max - x_moving_min)
        scale_y = scale_x
    else:
        scale_x = (x_fixed_max - x_fixed_min) / (x_moving_max - x_moving_min)
        scale_y = (y_fixed_max - y_fixed_min) / (y_moving_max - y_moving_min)
    return np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])


def get_translation(
    fixed_vert: NDArray[(Any, 2), int], moving_vert: NDArray[(Any, 2), int]
) -> NDArray[(3, 3), float]:
    r"""
    Given 2 lists of vertices coordinates sorted in trigonometric order, get the
    translation transform that aligns them.

    Args:
        fixed_vert: first vertices coordinates array considered as fixed for the
            translation.
        moving_vert:  first vertices coordinates array considered as moving for the
            translation.

    Returns:
        2D translation 3x3 matrix.
    """
    xoff, yoff = fixed_vert.min(0) - moving_vert.min(0)
    return np.array([[1, 0, xoff], [0, 1, yoff], [0, 0, 1]])


def equalize_vert_lengths(
    l1: List[Tuple[Number, Number]], l2: List[Tuple[Number, Number]]
):
    r"""
    Given 2 lists of vertices coordinates, remove items from the longest one that are
    the farthest from all points from the smalles one.

    Args:
        l1: first list of vertices coordinates.
        l2: second list of vertices coordinates.
    """
    l1, l2 = sorted([l1, l2], key=len)
    sorted_l2 = sorted(
        l2, key=lambda x: min([(x[1] - y[1]) ** 2 + (x[0] - y[0]) ** 2 for y in l1])
    )
    while len(l2) > len(l1):
        to_remove = sorted_l2.pop()
        l2.remove(to_remove)


def get_affine_transform(
    fixed: NDBoolMask, moving: NDBoolMask
) -> Tuple[NDArray[(3, 3), float], NDArray[(3, 3), float], NDArray[(3, 3), float]]:
    r"""
    Given 2 dot masks, get the affine transform (as rotation, scale and translation) to
    apply to the moving one to align it with the fixed one.

    Args:
        fixed: input fixed mask.
        moving: input moving mask.

    Returns:
        Tuple containing the rotation, the scale and the translation 3x3 matrices.
    """
    fixed_vert = get_vertices(fixed)
    moving_vert = get_vertices(moving)
    if len(fixed_vert) != len(moving_vert):
        equalize_vert_lengths(fixed_vert, moving_vert)
    fixed_vert = np.array(fixed_vert)
    moving_vert = np.array(moving_vert)
    rot = get_rotation(fixed_vert, moving_vert)
    reg_vert = np.concatenate(
        (moving_vert, [[1] for _ in range(len(fixed_vert))]), axis=1
    )
    reg_vert = (rot @ reg_vert.T).T
    scale = get_scale(fixed_vert, reg_vert[:, :2])
    reg_vert = (scale @ reg_vert.T).T
    trans = get_translation(fixed_vert, reg_vert[:, :2])
    return rot, scale, trans


def get_coord_transform(
    slide_he: Slide, slide_ihc: Slide
) -> Callable[[int, int], Coord]:
    r"""
    Given an H&E slide and an immunohistochemistry slide, get a transform function that
    registers coordinates from the H&E slide into the IHC slide.

    Args:
        slide_he: input H&E slide.
        slide_ihc: input IHC slide.

    Returns:
        A function that takes coordinates from the H&E slide as input and returns the
        corresponding coords in the IHC slide.
    """
    thumb_level = slide_he.level_count - 1
    mask_he = get_dot_mask(slide_he, thumb_level=thumb_level)
    mask_ihc = get_dot_mask(slide_ihc, thumb_level=thumb_level)
    rot, scale, trans = get_affine_transform(mask_ihc, mask_he)
    dsr = slide_he.shape[0] / mask_he.shape[0]
    trans[:2, 2] *= dsr
    affine = trans @ scale @ rot

    def _transform(x, y):
        x1, y1, _ = (affine @ np.array([x, y, 1]).T).T
        return Coord(x1, y1)

    return _transform


def get_input_images(
    slide: Slide, patch: Patch, h_min: float = 0.017, h_max: float = 0.11
) -> Tuple[NDByteImage, NDByteGrayImage, NDByteGrayImage]:
    r"""
    Return patches from a slide that are used for registration:

    Args:
        slide: input slide.
        patch: input pathaia patch object.
        h_min: minimum hematoxylin value for standardization.
        h_max: maximum hematoxylin value for standardization.

    Returns:
        3-tuple containing patch as RGB image, grayscale image and with only Hematoxylin
        channel.
    """
    img = np.asarray(
        slide.read_region(patch.position, patch.level, patch.size).convert("RGB")
    )
    img_G = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_H = rgb2hed(img)[:, :, 0]
    img_H = img_as_ubyte(((img_H - h_min) / (h_max - h_min)).clip(0, 1))
    return img, img_G, img_H


def has_enough_tissue(
    img: NDByteImage, blacktol: int = 0, whitetol: int = 247, area_thr: float = 0.99
) -> bool:
    r"""
    Check if grayscale image contains enough tissue by filtering black and white pixels.

    Args:
        img: input grayscale image.
        blacktol: minimum accepted pixel value.
        whitetol: maximum accepted pixel value.
        area_thr: minimum fraction of pixels that must be positive to consider the img
            to have enough tissue.

    Returns:
        True if the image is considered to have enough tissue, False otherwise.
    """
    size = img.size
    area_thr = area_thr * size
    mask = get_tissue_mask(img, blacktol=blacktol, whitetol=whitetol)
    return mask.sum() > area_thr


def get_binary_op(op_name: str) -> Union[binary_closing, binary_dilation, None]:
    r"""
    Given a binary operation name, returns the corresponding scikit-image function.

    Args:
        op_name: name of the operation, either "closing", "dilation" or "none".

    Returns:
        If "closing", :func:`~skimage.morphology.binary_closing`; if "dilation",
        :func:`~skimage.morphology.binary_dilation`; else `None`.
    """
    if op_name == "closing":
        return binary_closing
    elif op_name == "dilation":
        return binary_dilation
    else:
        return None


def equalize_contrasts(
    he_H: NDByteGrayImage,
    ihc_H: NDByteGrayImage,
    he_G: NDByteGrayImage,
    ihc_G: NDByteGrayImage,
    whitetol: int = 220,
    low_percentile: float = 5,
    high_percentile: float = 95,
) -> Tuple[NDByteGrayImage, NDByteGrayImage]:
    r"""
    Equalizes the contrasts from H&E and IHC images that were converted into H space.

    Args:
        he_H: input H&E image in H space.
        ihc_H: input IHC image in H space.
        he_G: input H&E image in grayscale.
        ihc_G: input IHC image in grayscale.
        whitetol: value from grayscale image above which all values in the H image will
            be zeroed.
        low_percentile: percentile from the IHC_H image under which all values from both
            H images will be zeroed.
        low_percentile: percentile from the IHC_H image under which all values from both
            H images will be 255.

    Returns:
        Tuple containg H&E and IHC H images with adjusted contrasts.
    """
    he_H = img_as_float(he_H)
    ihc_H = img_as_float(ihc_H)
    ihc_min = np.percentile(ihc_H, low_percentile)
    ihc_max = np.percentile(ihc_H, high_percentile)
    print(ihc_min, ihc_max)
    for img, img_g in zip((he_H, ihc_H), (he_G, ihc_G)):
        img[:] = ((img - ihc_min) / (ihc_max - ihc_min)).clip(0, 1)
        img[img_g > whitetol] = 0
    return img_as_ubyte(he_H), img_as_ubyte(ihc_H)


def convert_to_nifti(path: PathLike):
    r"""
    Runs c2d on the input image to turn into
    `HistoReg <https://github.com/CBICA/HistoReg>`_ compatible nifti.

    Args:
        path: path to input image file.
    """
    path = Path(path)
    base_dir = path.parent
    path = Path(f"/data/inputs/{path.name}")
    c2d_cmd = (
        f"docker run -v {base_dir}:/data/inputs historeg "
        f"c2d -mcs {path} -foreach -orient LP -spacing 1x1mm -origin 0x0mm -endfor -omc"
        f" {path.with_suffix('.nii.gz')}"
    )
    run(c2d_cmd.split())


def register(
    base_path: PathLike,
    he_H_path: PathLike,
    ihc_H_path: PathLike,
    he_path: PathLike,
    ihc_path: PathLike,
    reg_path: PathLike,
    iterations: int = 20000,
    resample: int = 4,
):
    r"""
    Registers IHC H image into H&E H image using
    `HistoReg <https://github.com/CBICA/HistoReg>`_.

    Args:
        base_path: root path for all other files.
        he_H_path: relative path to H&E H image file.
        ihc_H_path: relative path to IHC H image file.
        he_path: relative path to H&E image file saved as nifti.
        ihc_path: relative path to IHC image file saved as nifti.
        iterations: number of iterations for initial rigid search.
        resample: percentage of the full resolution the images will be resampled to,
            used for computation.
    """
    he_H_path, ihc_H_path, he_path, ihc_path, reg_path = map(
        lambda x: Path("/data") / x.relative_to(base_path),
        (he_H_path, ihc_H_path, he_path, ihc_path, reg_path),
    )
    historeg_cmd = (
        f"docker run -v {base_path}:/data historeg "
        f"HistoReg -i {iterations} -r {resample} -s1 6 -s2 8 -f {he_H_path} -m "
        f"{ihc_H_path} -o /data/historeg"
    )
    run(historeg_cmd.split())

    tfm_path = Path(
        f"/data/historeg/{ihc_H_path.stem}_registered_to_{he_H_path.stem}"
        "/metrics/full_resolution"
    )
    greedy_cmd = (
        f"docker run -v {base_path}:/data historeg "
        f"greedy -d 2 -rf {he_path} -rm {ihc_path} {reg_path} -r "
        f"{tfm_path/'big_warp.nii.gz'} {tfm_path/'Affine.mat'}"
    )
    run(greedy_cmd.split())

    c2d_cmd = (
        f"docker run -v {base_path}:/data historeg "
        f"c2d -mcs {reg_path} -foreach -type uchar -endfor -omc "
        f"{reg_path.with_suffix('').with_suffix('.png')}"
    )
    run(c2d_cmd.split())


def full_registration(
    slide_he: Slide,
    slide_ihc: Slide,
    patch_he: Patch,
    patch_ihc: Patch,
    base_path: PathLike,
    dab_thr: float = 0.03,
    object_min_size: int = 1000,
    iterations: int = 20000,
) -> bool:
    r"""
    Perform full registration process on patches from an IHC slide and a H&E slide.

    Args:
        slide_he: input H&E slide.
        slide_ihc: input IHC slide.
        patch_he: input H&E patch (fixed for the registration).
        patch_ihc: input IHC patch (moving for the registration).
        base_path: root path for all other files.
        dab_thr: minimum value to use for DAB thresholding.
        object_min_size: the smallest allowable object size to check if registration
            needs to be performed.
        iterations: number of iterations for initial rigid search.

    Return:
        True if registration was sucesfully performed, False otherwise.
    """
    print(f"HE: {patch_he.position} / IHC: {patch_ihc.position}")
    he_H_path = base_path / "he_H.png"
    ihc_H_path = base_path / "ihc_H.png"
    he_path = base_path / "he.png"
    ihc_path = base_path / "ihc.png"
    reg_path = base_path / "ihc_warped.png"

    he, he_G, he_H = get_input_images(slide_he, patch_he)
    ihc, ihc_G, ihc_H = get_input_images(slide_ihc, patch_ihc)

    if not (has_enough_tissue(he_G, whitetol=255, area_thr=0.7)):
        print("Patch doesn't contain enough tissue, skipping.")
        return False

    mask = get_dab_mask(ihc, dab_thr=dab_thr, object_min_size=object_min_size)

    if not mask.sum():
        print("Mask would be empty, skipping.")
        return False

    # he_H, ihc_H = equalize_contrasts(he_H, ihc_H, he_G, ihc_G)

    imsave(he_H_path, he_H)
    imsave(ihc_H_path, ihc_H)

    imsave(he_path, he)
    imsave(ihc_path, ihc)
    convert_to_nifti(he_path)
    convert_to_nifti(ihc_path)

    print("Starting registration...")

    register(
        base_path,
        he_H_path,
        ihc_H_path,
        he_path.with_suffix(".nii.gz"),
        ihc_path.with_suffix(".nii.gz"),
        reg_path.with_suffix(".nii.gz"),
        resample=20,
        iterations=iterations,
    )

    print("Registration done...")

    return True
