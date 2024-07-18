from pathlib import Path

# Where the geojson of the annotations are located
ANNOTATION_PATH = Path("/home/elliot/AprioricsSlides/mitoses_annotation_geojson")
assert ANNOTATION_PATH.exists()

# Where most of the data will be stored
BASE_PATH = Path("/data/elliot/test_pipeline")
assert BASE_PATH.exists()

# Slide number = slide number in files -4 (slides starts at 4)
# 996 and 997 refers to 1108 and 1204 because we added them manually after the 999 inital one

SLIDE_LIST = [105, 182, 724, 956, 997]

SLIDE_TO_PATH = {
    105: '21I000109-1-13-21_095813',
    182: '21I000186-1-15-21_153809',
    724: '21I000728-1-10-21_141128',
    956: '21I000960-1-2-21_133557',
    997: '21I001108-1-3-21_094208',
    999: '21I001108-1-3-21_094208'
}

SLIDE_TO_PID_HOVERNET = {
    105: '21I000109',
    182: '21I000186',
    724: '21I000728',
    956: '21I000960',
    997: '21I001108',
}

SLIDE_FOLDER = Path("/home/elliot/AprioricsSlides/slides/PHH3/HE/")
IHC_FOLDER = Path("/home/elliot/AprioricsSlides/slides/PHH3/IHC/")
HOVERNET_FOLDER = Path("/data/anapath/AprioricsSlides/hovernet_geojsons/PHH3")

