import cv2
import argparse
from kartezio.dataset import RoiPolygonReader, DataItem, Dataset
from pathlib import Path
import numpy as np
import pandas as pd
from numena.image.basics import image_new
from skimage.color import rgb2hsv
from kartezio.fitness import FitnessIOU, FitnessAP
from kartezio.training import train_model
from kartezio.apps.instance_segmentation import create_instance_segmentation_model

from config_kartezio import BASE_PATH as base_path


def format_dataset(df, selected_color_spaces):
    
    train_set = Dataset.SubSet(None)
    test_set = Dataset.SubSet(None)
    roipolygonreader = RoiPolygonReader(base_path)

    for raw in df.iterrows():

        str_ihc = (raw[1]['input'])
        str_he = str_ihc.replace("ihc", "he")
        datalist = []
        
        if raw[1]['set'] == "testing":
            img_ihc = cv2.imread(str(base_path / str_ihc))
            img_ihc = cv2.cvtColor(img_ihc, cv2.COLOR_BGR2RGB)
            
            img_he = cv2.imread(str(base_path / str_he))
            img_he = cv2.cvtColor(img_he, cv2.COLOR_BGR2RGB)
            
            if 'RGB' in selected_color_spaces:
                img_ihc_rgb = cv2.cvtColor(img_ihc, cv2.COLOR_BGR2RGB)
                img_he_rgb = cv2.cvtColor(img_he, cv2.COLOR_BGR2RGB)
                
                datalist += list(cv2.split(img_ihc_rgb))
                datalist += list(cv2.split(img_he_rgb))

            if 'HSV' in selected_color_spaces:
                img_ihc_hsv = cv2.cvtColor(img_ihc, cv2.COLOR_RGB2HSV)
                img_he_hsv = cv2.cvtColor(img_he, cv2.COLOR_RGB2HSV)
                
                datalist += list(cv2.split(img_ihc_hsv))
                datalist += list(cv2.split(img_he_hsv))

            if 'HED' in selected_color_spaces:
                img_ihc_hed = rgb2hsv(img_ihc).astype(np.uint8)
                img_he_hed = rgb2hsv(img_he).astype(np.uint8)
                
                datalist += list(cv2.split(img_ihc_hed))
                datalist += list(cv2.split(img_he_hed))
            
            X = DataItem(datalist=datalist, # Ici rajouter des conversions HSV etc
                        shape=img_ihc.shape[:2],
                        count=1,
                        visual=img_ihc)
            
            label = (raw[1]['label'])
            
            if isinstance(label, float):
                label_mask = image_new(img_ihc.shape[:2])
                label = ""
                Y = [label_mask]
            else:
                Y = roipolygonreader.read(Path(label), shape=img_ihc.shape[:2])
                Y = Y.datalist
                
            test_set.add_item(X.datalist, Y)
            test_set.add_visual(img_ihc_rgb)
            
        
        if raw[1]['set'] == "training":
            
            img_ihc = cv2.imread(str(base_path / str_ihc))
            img_ihc = cv2.cvtColor(img_ihc, cv2.COLOR_BGR2RGB)
            
            img_he = cv2.imread(str(base_path / str_he))
            img_he = cv2.cvtColor(img_he, cv2.COLOR_BGR2RGB)
            
            if 'RGB' in selected_color_spaces:
                img_ihc_rgb = cv2.cvtColor(img_ihc, cv2.COLOR_BGR2RGB)
                img_he_rgb = cv2.cvtColor(img_he, cv2.COLOR_BGR2RGB)
                
                datalist += list(cv2.split(img_ihc_rgb))
                datalist += list(cv2.split(img_he_rgb))

            if 'HSV' in selected_color_spaces:
                img_ihc_hsv = cv2.cvtColor(img_ihc, cv2.COLOR_RGB2HSV)
                img_he_hsv = cv2.cvtColor(img_he, cv2.COLOR_RGB2HSV)
                
                datalist += list(cv2.split(img_ihc_hsv))
                datalist += list(cv2.split(img_he_hsv))

            if 'HED' in selected_color_spaces:
                img_ihc_hed = rgb2hsv(img_ihc).astype(np.uint8)
                img_he_hed = rgb2hsv(img_he).astype(np.uint8)
                
                datalist += list(cv2.split(img_ihc_hed))
                datalist += list(cv2.split(img_he_hed))
            
            label = (raw[1]['label'])
            
            if isinstance(label, float):
                label_mask = image_new(img_ihc.shape[:2])
                label = ""
                Y = [label_mask]
            else:
                Y = roipolygonreader.read(Path(label), shape=img_ihc.shape[:2])
                Y = Y.datalist
                
            X = DataItem(datalist=datalist, # Ici rajouter des conversions HSV etc
                    shape=img_ihc.shape[:2],
                    count=1,
                    visual=img_he)
            if img_ihc.shape[0] > 300 or img_ihc.shape[1] > 300: #Pour gagner du temps et virer les grosses images
                continue
                train_set.add_item(X.datalist, Y)
                train_set.add_visual(img_ihc_rgb)
        

    dataset = Dataset(train_set=train_set,
                    test_set=test_set,
                    name='Train_set',
                    label_name='Mitoses',
                    inputs=len(selected_color_spaces) * 6)
    
    return dataset
    

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Choose one or multiple color space options")

    # Add an argument for the color space options
    parser.add_argument(
        '--color-spaces',
        choices=['HED', 'HSV', 'RGB'],
        nargs='+',  # Allows one or more arguments
        help='Choose one or more color spaces from HED, HSV, and RGB',
        required=False,  # Makes this argument required
        default=['RGB']
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check the selected color spaces
    selected_color_spaces = args.color_spaces
    print(f"Selected color spaces: {', '.join(selected_color_spaces)}")
            
    df = pd.read_csv(base_path / "dataset.csv")

    dataset = format_dataset(df, selected_color_spaces)
    ITERATIONS = 100
    LAMBDA = 5
    fitness=FitnessAP(thresholds=0.5)
    #preprocessing = TransformToHED()
    preprocessing = None

    model = create_instance_segmentation_model(
            ITERATIONS,
            LAMBDA,
            inputs=18,
            outputs=2, 
            fitness=fitness
        )

    elite, _ = train_model(
            model, dataset, './', preprocessing=preprocessing, callback_frequency=10
        )

if __name__ == "__main__":
    main()
