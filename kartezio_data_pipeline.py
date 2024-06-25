import os
import cv2

from PIL import Image
import numpy as np
from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd


from masking import get_annotations
from ihc_mask_registration import format_annotation, get_slides
from pathlib import Path
from config_kartezio import BASE_PATH, ANNOTATION_PATH, SLIDE_LIST, SLIDE_TO_PATH, SLIDE_FOLDER, IHC_FOLDER


def construct_slide_dic():
    dic_slide = {}

    for slide_number in SLIDE_LIST:
        geo_df = gpd.read_file(ANNOTATION_PATH / f"{SLIDE_TO_PATH[slide_number]}.geojson")

        mitosis_area_dfs = format_annotation(geo_df)

        dic_out_annot = {}
        for idx, df in enumerate(mitosis_area_dfs):
            
            dic_annotation = get_annotations(df)
            slide_he, slide_ihc = get_slides(SLIDE_FOLDER, IHC_FOLDER, slide_number)
            crop_size_x = int(dic_annotation['size'][0] * 0.10)
            crop_size_y = int(dic_annotation['size'][1] * 0.10)
            # Adapt the position of the image, so after the cropping we still should have the full annotation

            dic_annotation['new_position'] = (dic_annotation['position'][0] - crop_size_x, dic_annotation['position'][1] - crop_size_y)
            dic_out_annot[idx] = dic_annotation
        
        dic_slide[slide_number] = dic_out_annot
    return dic_slide


def csv_format_annotations(dic_slide, base_path):
    # Sert à output les annotations sous une forme assez facilement exploitable

    for idx_sld, (idx_slide, dic_out_annot) in enumerate(dic_slide.items()):
        for idx, value in dic_out_annot.items():
            dic_df = {}
            for j, liste in enumerate(value['annotations']):
                dic_temp = {}
                dic_temp['x1'] = liste['coord'][0][0]
                dic_temp['y1'] = liste['coord'][0][1]
                dic_temp['y2'] = liste['coord'][1][1]
                dic_temp['x2'] = liste['coord'][2][0]
                dic_temp['label'] = liste['annotation_class']
                dic_df[j] = dic_temp
            df = pd.DataFrame.from_dict(dic_df, orient='index')
            print("writing to", base_path / SLIDE_TO_PATH[idx_slide] / str(idx) / "annotations.csv")
            df.to_csv(base_path / SLIDE_TO_PATH[idx_slide] / str(idx) / "annotations.csv")
        
def extract_annotated_thumbnails(base_path):
    for idx, slide in enumerate(SLIDE_LIST):
        path_ = base_path / SLIDE_TO_PATH[slide]
        list_folders = (next(os.walk(path_))[1])
        list_folders.sort()
        for element in list_folders:
            path_folder_ = path_ / element
            
            # Define the output directories
            output_dir_yes_ihc = path_folder_ / 'yes_ihc'
            output_dir_no_ihc = path_folder_ / 'no_ihc'
            
            # Define the output directories
            output_dir_yes_he = path_folder_ / 'yes_he'
            output_dir_no_he = path_folder_ / 'no_he'

            # Create output directories if they don't exist
            if not os.path.exists(output_dir_yes_ihc):
                os.makedirs(output_dir_yes_ihc)
            if not os.path.exists(output_dir_no_ihc):
                os.makedirs(output_dir_no_ihc)

            # Create output directories if they don't exist
            if not os.path.exists(output_dir_yes_he):
                os.makedirs(output_dir_yes_he)
            if not os.path.exists(output_dir_no_he):
                os.makedirs(output_dir_no_he)
            he = Image.open(path_folder_ / "he_crop.png").convert("RGB")
            he_array = np.asarray(he)

            ihc = Image.open(path_folder_ / "ihc_crop.png").convert("RGB")
            ihc_array = np.asarray(ihc)

            annotations = pd.read_csv(path_folder_ / 'annotations.csv', index_col=None) 
            
            # Add Padding here
            # Process each row in the annotations dataframe
            for index, row in annotations.iterrows():
                x1, y1, x2, y2, label = row['x1'], row['y1'], row['x2'], row['y2'], row['label']
                print(x1, y1, x2, y2, label)
                # Crop both images using the annotation coordinates
                he_crop = he_array[y1:y2, x1:x2]
                ihc_crop = ihc_array[y1:y2, x1:x2]

                # Determine the output directory based on the label
                if 'mitosis_yes' in label:
                    output_dir_ihc = output_dir_yes_ihc
                    output_dir_he = output_dir_yes_he

                else:
                    output_dir_ihc = output_dir_no_ihc
                    output_dir_he = output_dir_no_he

                # Save the cropped images to the output directory
                he_crop_filename = f'he_{index}.png'
                ihc_crop_filename = f'ihc_{index}.png'
                he_crop_path = os.path.join(output_dir_he, he_crop_filename)
                ihc_crop_path = os.path.join(output_dir_ihc, ihc_crop_filename)
                try :
                    cv2.imwrite(he_crop_path, he_crop)
                    cv2.imwrite(ihc_crop_path, ihc_crop)
                except:
                    continue
             
def kartezio_dataset_formating(base_path):   
    list_roi = []
    list_img_yes = []
    list_img_no = []
    list_path_ihc_imgs = []
    for slide in SLIDE_LIST:
        # Loop for all the slides
        for root, dirs, files in ((os.walk(base_path/SLIDE_TO_PATH[slide]))):
            # For all the patches in each slide
            print("root =", root, "dirs = ", dirs)
            for dir in dirs:
                list_path_ihc_imgs.append(f"{SLIDE_TO_PATH[slide]}/{dir}/ihc_crop.png")
                path_to_write_yes = f"{SLIDE_TO_PATH[slide]}/{dir}/yes_ihc"
                path_to_write_no = f"{SLIDE_TO_PATH[slide]}/{dir}/no_ihc"
                # Looping for all the images in "yes_annotation"
                for root2, dirs2, files2 in (os.walk((Path(root) / dir / "yes_ihc"))):
                    for file in files2:
                        name, ext = file.split('.')
                        if '_' in name:
                            _, name = name.split('_')
                        if ext == 'roi':
                            path_roi = f"{SLIDE_TO_PATH[slide]}/{dir}/yes_ihc/{file}"
                            list_roi.append(path_roi)
                        elif ext == 'png':
                            path_img = f"{SLIDE_TO_PATH[slide]}/{dir}/yes_ihc/{file}"
                            list_img_yes.append(path_img)
                
                for root3, dirs3, files3 in (os.walk((Path(root) / dir / "no_ihc"))):
                    for file in files3:
                        if ext=='png':
                            path_img = f"{SLIDE_TO_PATH[slide]}/{dir}/no_ihc/{file}"
                            list_img_no.append(path_img)
            # this break is needed, otherwise it goes to all subdirectories as well
            break

    list_img_yes.sort()
    list_roi.sort()
    dic = {}
    for idx, element in enumerate(list_img_yes):
        dic[idx] = (element, list_roi[idx], 'training')
    for idx2, element in enumerate(list_img_no):
        dic[idx + idx2] = (element, '', 'training')
    for idx3, element in enumerate(list_path_ihc_imgs):
        dic[idx + idx2 + idx3] = (element, '', 'testing')
        
    import pandas as pd
    df = pd.DataFrame.from_dict(dic, orient='index', columns=['input', 'label', 'set'])

    df.to_csv(base_path / 'dataset.csv', index=False)
    
def main():

    dic_slide = construct_slide_dic()
    csv_format_annotations(dic_slide, BASE_PATH)
    extract_annotated_thumbnails(BASE_PATH)
    kartezio_dataset_formating(BASE_PATH)
    
    
if __name__ == "__main__":
    main()