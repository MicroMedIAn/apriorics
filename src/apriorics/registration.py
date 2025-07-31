from os import PathLike
from pathlib import Path
from typing import Callable, Tuple, Union

import cv2
import docker
import itk
import numpy as np
from pathaia.util.types import (
    Coord,
    NDByteGrayImage,
    NDByteImage,
    Patch,
    Slide,
)
from skimage.color import rgb2hed
from skimage.io import imsave
from skimage.morphology import binary_closing, binary_dilation
from skimage.registration import phase_cross_correlation
from skimage.util import img_as_float, img_as_ubyte

from apriorics.masks import get_dab_mask, get_tissue_mask


def get_thumbnail(slide: Slide, thumb_level: int = 3) -> NDByteImage:
    thumb = np.asarray(
        slide.read_region(
            (0, 0),
            thumb_level,
            slide.level_dimensions[thumb_level],
        ).convert("RGB")
    )
    return thumb


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
    thumb_level = min(slide_he.level_count, slide_ihc.level_count) - 1
    thumb_he = get_thumbnail(slide_he, thumb_level=thumb_level)
    thumb_ihc = get_thumbnail(slide_ihc, thumb_level=thumb_level)
    thumb_he_g = cv2.cvtColor(thumb_he, cv2.COLOR_RGB2GRAY)
    thumb_ihc_g = cv2.cvtColor(thumb_ihc, cv2.COLOR_RGB2GRAY)

    he_h, he_w = thumb_he_g.shape
    ihc_h, ihc_w = thumb_ihc_g.shape
    thumb_he_g = np.pad(
        thumb_he_g, ((0, max(0, ihc_h - he_h)), (0, max(0, ihc_w - he_w)))
    )
    thumb_ihc_g = np.pad(
        thumb_ihc_g, ((0, max(0, he_h - ihc_h)), (0, max(0, he_w - ihc_w)))
    )
    affine, _, _ = phase_cross_correlation(thumb_ihc_g, thumb_he_g)
    dsr = slide_he.dimensions[1] / thumb_he.shape[0]
    affine = np.array(affine, dtype=int) * dsr

    def _transform(x, y):
        y1, x1 = affine + np.array([y, x])
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


def convert_to_nifti(base_path: PathLike, path: PathLike, container=None):
    r"""
    Runs c2d on the input image to turn into
    `HistoReg <https://github.com/CBICA/HistoReg>`_ compatible nifti.

    Args:
        path: path to input image file.
    """
    path = Path(path)
    path = Path(f"/data/{path.relative_to(base_path)}")

    if container is None:
        client = docker.from_env()
        container = client.containers.create(
            "historeg",
            "/bin/bash",
            tty=True,
            stdin_open=True,
            auto_remove=False,
            volumes=[f"{base_path}:/data"],
        )
        container.start()
    c2d_cmd = (
        f"c2d -mcs {path} -foreach -orient LP -spacing 1x1mm -origin 0x0mm -endfor -omc"
        f" {path.with_suffix('.nii.gz')}"
    )
    with open(base_path / "log", "ab") as f:
        # f.write(f"{datetime.now()} - converting {path.name} to nifti...\n")
        res = container.exec_run(c2d_cmd, stream=True)
        for chunk in res.output:
            f.write(chunk)
        # f.write(f"{datetime.now()} - conversion finished.\n")

    return container


def run_historeg_preproc(img, resample=4):
    img = itk.image_view_from_array(img)
    h, w = img.shape
    img.SetDirection(np.array([[-1, 0], [0, -1]]))

    smooth = int(100 / (2 * resample))
    img = itk.recursive_gaussian_image_filter(img, sigma=smooth)
    res_w, res_h = int(resample / 100 * w), int(resample / 100 * h)
    scale = res_h / h
    input_spacing = itk.spacing(img)
    input_origin = itk.origin(img)
    Dimension = img.GetImageDimension()
    output_spacing = [input_spacing[d] / scale for d in range(Dimension)]
    output_origin = [
        input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d])
        for d in range(Dimension)
    ]
    img = itk.resample_image_filter(
        img,
        size=(res_w, res_h),
        output_spacing=output_spacing,
        output_origin=output_origin,
        output_direction=img.GetDirection(),
    )

    img.SetOrigin((0, 0))
    img.SetSpacing((1, 1))

    kernel = int(res_h / 40)
    arr = np.zeros((res_h, res_w), dtype=np.uint8)
    arr[:kernel, :kernel] = 1
    arr[:kernel, -kernel:] = 1
    arr[-kernel:, :kernel] = 1
    arr[-kernel:, -kernel:] = 1
    mean = (itk.array_view_from_image(img) * arr).mean().item()
    std = (itk.array_view_from_image(img) * arr).std().item()

    pad_size = 4 * kernel
    padded_w, padded_h = res_w + 2 * pad_size, res_h + 2 * pad_size
    mask = itk.image_view_from_array(np.ones((res_h, res_w), dtype=np.uint8))
    mask.SetDirection(np.array([[-1, 0], [0, -1]]))
    mask = itk.constant_pad_image_filter(
        mask,
        pad_lower_bound=(pad_size, pad_size),
        pad_upper_bound=(pad_size, pad_size),
        constant=0,
    )

    mask = itk.image_view_from_array(1 - itk.array_view_from_image(mask))
    mask.SetDirection(np.array([[-1, 0], [0, -1]]))
    empty_mask = itk.image_view_from_array(
        np.zeros((padded_h, padded_w), dtype=np.uint8)
    )
    empty_mask.SetDirection(np.array([[-1, 0], [0, -1]]))
    empty_mask = itk.additive_gaussian_noise_image_filter(
        empty_mask, mean=mean, standard_deviation=std
    )
    mask = itk.multiply_image_filter(mask, empty_mask)

    img = itk.constant_pad_image_filter(
        img,
        pad_lower_bound=(pad_size, pad_size),
        pad_upper_bound=(pad_size, pad_size),
        constant=0,
    )
    mask.SetLargestPossibleRegion(img.GetLargestPossibleRegion())
    img_arr = itk.array_view_from_image(img)
    img_arr += itk.array_view_from_image(mask)
    img.SetOrigin((0, 0))
    w, h = itk.size(img)
    region = itk.ImageRegion[2]((w, h))
    img.SetLargestPossibleRegion(region)
    img.SetRequestedRegionToLargestPossibleRegion()
    img.SetBufferedRegion(region)
    img.Update()

    return img


def run_historeg_postproc(warp, affine, pad_size, full_size, small_size, resample=4):
    warp_arr = itk.array_view_from_image(warp)
    warp = itk.image_from_array(
        warp_arr[pad_size:-pad_size, pad_size:-pad_size], is_vector=True
    )

    scale = full_size[1] / small_size
    input_spacing = itk.spacing(warp)
    input_origin = itk.origin(warp)
    output_spacing = [input_spacing[d] / scale for d in range(2)]
    output_origin = [
        input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d]) for d in range(2)
    ]

    warp = itk.resample_image_filter(
        warp,
        size=full_size,
        output_spacing=output_spacing,
        output_origin=output_origin,
        output_direction=warp.GetDirection(),
    )

    warp.SetOrigin((0, 0))
    warp.SetSpacing((1, 1))
    warp_arr = itk.array_view_from_image(warp)
    warp_arr *= int(100 / resample)

    affine[:2, 2] *= int(100 / resample)
    return warp, affine


def register(
    base_path: PathLike,
    he_H_path: PathLike,
    ihc_H_path: PathLike,
    he_path: PathLike,
    ihc_path: PathLike,
    reg_path: PathLike,
    patch_size: Coord,
    container=None,
    iterations: int = 20000,
    resample: int = 4,
    threads=0,
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

    (base_path / "historeg").mkdir(exist_ok=True)

    if container is None:
        client = docker.from_env()
        container = client.containers.create(
            "historeg",
            "/bin/bash",
            tty=True,
            stdin_open=True,
            auto_remove=False,
            volumes=[f"{base_path.absolute()}:/data"],
        )
        container.start()

    with open(base_path / "log", "ab") as f:
        small_size = int((resample / 100) * patch_size.y)
        kernel = int(small_size / 40)
        offset = int((small_size + 4 * kernel) / 10)
        pad_size = 4 * kernel

        affine_cmd = (
            f"greedy -a -m NCC {kernel}x{kernel} -n 100x50x10 -threads {threads} "
            f"-search {iterations} 5 {offset} -i {he_H_path} {ihc_H_path} -o "
            "/data/historeg/small_affine.mat"
        )
        res = container.exec_run(affine_cmd, stream=True)
        for chunk in res.output:
            f.write(chunk)

        diffeo_cmd = (
            f"greedy -d 2 -it /data/historeg/small_affine.mat -threads {threads} -m NCC"
            f" {kernel}x{kernel} -n 100x50x10 -s 6vox 8vox -i {he_H_path} {ihc_H_path} "
            "-o /data/historeg/small_warp.nii.gz"
        )
        res = container.exec_run(diffeo_cmd, stream=True)
        for chunk in res.output:
            f.write(chunk)

        warp = itk.imread(str(base_path / "historeg/small_warp.nii.gz"))
        warp = itk.image_view_from_array(
            itk.array_view_from_image(warp), is_vector=True
        )
        with open(base_path / "historeg/small_affine.mat") as f1:
            affine = f1.read()
            affine = [
                [float(x) for x in row.strip().split(" ")]
                for row in affine.strip().split("\n")
            ]
            affine = np.array(affine)

        warp, affine = run_historeg_postproc(
            warp, affine, pad_size, patch_size, small_size, resample=resample
        )

        itk.imwrite(warp, base_path / "historeg/big_warp.nii.gz")

        with open(base_path / "historeg/big_affine.mat", "w") as f1:
            f1.write(
                "\n".join(
                    [" ".join([f"{x:.6f}" for x in row]) for row in affine.tolist()]
                )
            )

        greedy_cmd = (
            f"greedy -d 2 -threads {threads} -rf {he_path} -rm {ihc_path} {reg_path} -r"
            f" /data/historeg/big_warp.nii.gz /data/historeg/big_affine.mat"
        )
        # f.write(f"{datetime.now()} - Starting Greedy...\n")
        res = container.exec_run(greedy_cmd, stream=True)
        for chunk in res.output:
            f.write(chunk)
        # f.write(f"{datetime.now()} - Greedy finished.\n")

        c2d_cmd = (
            f"c2d -mcs {reg_path} -foreach -type uchar -endfor -omc "
            f"{reg_path.with_suffix('').with_suffix('.png')}"
        )
        # f.write(f"{datetime.now()} - Starting c2d...\n")
        res = container.exec_run(c2d_cmd, stream=True)
        for chunk in res.output:
            f.write(chunk)
        # f.write(f"{datetime.now()} - c2d finished.\n")

    return container


def full_registration(
    slide_he: Slide,
    slide_ihc: Slide,
    patch_he: Patch,
    patch_ihc: Patch,
    base_path: PathLike,
    dab_thr: float = 0.03,
    object_min_size: int = 1000,
    iterations: int = 20000,
    threads=0,
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
    pid = base_path.name
    print(f"[{pid}] HE: {patch_he.position} / IHC: {patch_ihc.position}")

    if not base_path.exists():
        base_path.mkdir()

    he_H_path = base_path / "he_H.nii.gz"
    ihc_H_path = base_path / "ihc_H.nii.gz"
    he_path = base_path / "he.nii.gz"
    ihc_path = base_path / "ihc.nii.gz"
    reg_path = base_path / "ihc_warped.nii.gz"

    he, he_G, he_H = get_input_images(slide_he, patch_he)
    ihc, ihc_G, ihc_H = get_input_images(slide_ihc, patch_ihc)

    if not (
        has_enough_tissue(he_G, whitetol=247, area_thr=0.05)
        and has_enough_tissue(ihc_G, whitetol=247, area_thr=0.05)
    ):
        print(f"[{pid}] Patch doesn't contain enough tissue, skipping.")
        return False

    mask = get_dab_mask(ihc, dab_thr=dab_thr, object_min_size=object_min_size)

    if mask.sum() < object_min_size:
        print(f"[{pid}] Mask would be empty, skipping.")
        return False

    # he_H, ihc_H = equalize_contrasts(he_H, ihc_H, he_G, ihc_G)
    resample = min(50, int(100000 / patch_he.size[0]))

    he_H = run_historeg_preproc(he_H, resample=resample)
    ihc_H = run_historeg_preproc(ihc_H, resample=resample)

    print(he_H.GetOrigin(), ihc_H.GetOrigin())
    itk.imwrite(he_H, he_H_path)
    itk.imwrite(ihc_H, ihc_H_path)

    imsave(he_path.with_suffix("").with_suffix(".png"), he)
    he = itk.image_view_from_array(he, is_vector=True)
    he.SetDirection(np.array([[-1, 0], [0, -1]]))
    itk.imwrite(he, he_path)
    ihc = itk.image_view_from_array(ihc, is_vector=True)
    ihc.SetDirection(np.array([[-1, 0], [0, -1]]))
    itk.imwrite(ihc, ihc_path)

    print(f"[{pid}] Starting registration...")

    container = register(
        base_path,
        he_H_path,
        ihc_H_path,
        he_path,
        ihc_path,
        reg_path,
        patch_he.size,
        resample=resample,
        iterations=iterations,
        threads=threads,
    )

    print(f"[{pid}] Registration done...")

    return container
