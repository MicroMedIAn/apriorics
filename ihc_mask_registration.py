import numpy as np
from pathlib import Path
import numpy as np

from apriorics.registration import full_registration_ihc, get_coord_transform
from apriorics.masks import get_mask_function, get_tissue_mask
from PIL import Image

from multiprocessing import current_process

from pathaia.util.paths import get_files
from pathaia.util.types import Coord, Patch, Slide
import geopandas as gpd
import pandas as pd
from masking import get_annotations
from config_kartezio import BASE_PATH, ANNOTATION_PATH, SLIDE_LIST, SLIDE_TO_PATH, SLIDE_FOLDER, IHC_FOLDER


# For convenience
class arguments():

    ihc_type = "PHH3"
    patch_csv_folder=Path("/data/anapath/AprioricsSlides/train/PHH3/256_0/patch_csvs/")
    slidefolder=Path("/home/elliot/AprioricsSlides/slides/PHH3/HE/")
    ihcfolder=Path("/home/elliot/AprioricsSlides/slides/PHH3/IHC/")
    maskfolder=Path("/data/anapath/AprioricsSlides/masks/PHH3/HE/" )
    split_csv=Path("/data/anapath/AprioricsSlides/train/splits.csv")
    logfolder=Path("/data/anapath/AprioricsSlides/train/logs")
    gpu=0
    batch_size=32
    lr=2e-4
    wd=0.01
    epochs=10
    patch_size=224
    num_workers=32
    scheduler="one-cycle"
    horovod=False
    stain_matrices_folder=Path("")
    augment_stain=False

def format_annotation(geo_df):
    mitosis_area_dfs = []
    temp_df = None
    for index, row in geo_df.iterrows():
        if row['classification']['name'] == 'mitosis_area':
            if temp_df is not None:
                mitosis_area_dfs.append(temp_df)
                temp_df = None
            temp_df = pd.DataFrame([row])
        elif temp_df is not None:
            temp_df = pd.concat([temp_df, pd.DataFrame([row])], ignore_index=True)
            
    if temp_df is not None:
        mitosis_area_dfs.append(temp_df)
        
    return mitosis_area_dfs
        
def get_slides(slide_folder, ihc_folder, slide_number):
    slide_paths = get_files(
        slide_folder, extensions=".svs", recurse=False).sorted(key=lambda x: x.stem)
    ihc_paths = slide_paths.map(
        lambda x: ihc_folder / x.with_suffix(".svs").name
    )

    slide_backend: str = "cucim"
    slide_path = slide_paths[slide_number]
    print("slide_path = ", slide_path)
    ihc_path = ihc_paths[slide_number]
    slide_he = Slide(slide_path)
    slide_ihc = Slide(ihc_path)
    return slide_he, slide_ihc

def get_patches(slide_ihc, slide_he, dic_annotation, crop_size_x, crop_size_y):
    patch_ihc = Patch(
        id="#0",
        slidename="",
        position=dic_annotation['new_position'],
        level=0,
        size=Coord(dic_annotation['size'][0] + 2 * crop_size_x, dic_annotation['size'][1] + 2 * crop_size_y),
        size_0=Coord(dic_annotation['size'][0] + 2 * crop_size_x, dic_annotation['size'][1] + 2 * crop_size_y),
        parent=None
    )
    coord_tfm = get_coord_transform(slide_ihc, slide_he)
    patch_he = Patch(
        id=patch_ihc.id,
        slidename="",
        position=coord_tfm(*patch_ihc.position),
        size=patch_ihc.size,
        level=patch_ihc.level,
        size_0=patch_ihc.size_0,
    )
    return patch_he, patch_ihc

def register_image(slide_he, slide_ihc, patch_he, patch_ihc, base_path, box, idx):
    pid = str(current_process().pid)

    base_path_str = base_path / str(idx)
    
    if not base_path.exists():
        base_path.mkdir()
    if not base_path_str.exists():
        base_path_str.mkdir()

    restart = True
    iterations = 5000
    count = 0
    maxiter = 4

    while restart and count < maxiter:
        container = full_registration_ihc(
            slide_ihc,
            slide_he,
            patch_ihc,
            patch_he,
            base_path_str,
            dab_thr=0.03, #args.dab_thr
            object_min_size= 5, #args.object_min_size,
            iterations=iterations,
            threads=1,
        )
        restart = container is False
        if restart:
            break
        else:
            try:
                ihc = Image.open(base_path_str / "he_warped.png").convert("RGB").crop(box)
            except FileNotFoundError:
                print(f"[{pid}] ERROR: HE={patch_he}/IHC={patch_ihc}. Skipping...")
                container.stop()
                container.remove()
                break
            tissue_mask = get_tissue_mask(np.asarray(ihc.convert("L")), whitetol=256)
            if tissue_mask.sum() < 0.999 * tissue_mask.size:
                container.stop()
                container.remove()
                restart = True
                iterations *= 2
                count += 1

def main():

    for slide_number in SLIDE_LIST:
        
        geo_df = gpd.read_file(ANNOTATION_PATH / f"{SLIDE_TO_PATH[slide_number]}.geojson")

        # These shouldn't differ
        mitosis_area_dfs = format_annotation(geo_df)
        slide_he, slide_ihc = get_slides(SLIDE_FOLDER, IHC_FOLDER, slide_number)
        for idx, df in enumerate(mitosis_area_dfs):

            dic_annotation = get_annotations(df)
            crop_size_x = int(dic_annotation['size'][0] *0.10)
            crop_size_y = int(dic_annotation['size'][1] *0.10)
            dic_annotation['new_position'] = (dic_annotation['position'][0] - crop_size_x, dic_annotation['position'][1] - crop_size_y)
            
            patch_he, patch_ihc = get_patches(
                slide_ihc=slide_ihc, 
                slide_he=slide_he, 
                dic_annotation=dic_annotation, 
                crop_size_x=crop_size_x,
                crop_size_y=crop_size_y)
            
            box = (crop_size_x, crop_size_y, patch_ihc.size.x - crop_size_x, patch_ihc.size.y - crop_size_y)
            
            register_image(
                slide_he=slide_he, 
                slide_ihc=slide_ihc, 
                patch_he=patch_he, 
                patch_ihc=patch_ihc, 
                base_path=BASE_PATH / SLIDE_TO_PATH[slide_number], 
                box=box, 
                idx=idx)

            he_crop = Image.open(BASE_PATH / SLIDE_TO_PATH[slide_number] / str(idx) / "he.png").convert("RGB").crop(box)
            he_crop.save(BASE_PATH / SLIDE_TO_PATH[slide_number] / str(idx) / "he_crop.png")
            
            ihc_crop = Image.open(BASE_PATH / SLIDE_TO_PATH[slide_number] / str(idx) / "ihc.png").convert("RGB").crop(box)
            ihc_crop.save(BASE_PATH / SLIDE_TO_PATH[slide_number] / str(idx) / "ihc_crop.png")



if __name__ == "__main__":
    
    main()