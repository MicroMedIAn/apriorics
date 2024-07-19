# APRIORICS

[![Documentation Status](https://readthedocs.org/projects/apriorics/badge/?version=latest)](https://apriorics.readthedocs.io/en/latest/?badge=latest)

## Install

First, we need to install my fork of [HistoReg](https://github.com/CBICA/HistoReg) that
contains a ready-to-use Dockerfile.

```bash
cd
git clone https://github.com/schwobr/HistoReg.git
docker build -t historeg HistoReg
```

We then need to create a conda environment with pytorch.

```bash
conda create -n apriorics python=3.9
conda activate apriorics
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install openslide -c conda-forge
```

NB: To check required `cudatoolkit` version, type `nvcc --version` in a shell. Cuda is always compatible with older versions of the same major release (for instance if your nvcc version is 11.5, you can install `cudatoolkit=11.3` here). Specific versions of `pytorch` are only available with few `cudatoolkit` versions, you can check it on [PyTorch official website](https://pytorch.org/get-started/locally/).

Make sure that you have blas and lapack installed:

```bash
sudo apt install libblas-dev libblapack-dev # debian-based systems
sudo yum install blas-devel lapack-devel # CentOS 7
sudo dnf install blas-devel lapack-devel # CentOS 8
```

We can then clone this repository and install necessary pip packages.

```bash
cd
git clone https://github.com/schwobr/apriorics.git
cd apriorics
pip install -r requirements.txt
```

You can also install this library as an editable pip package, which will install all dependencies automatically.

```bash
pip install -e .
```

---
# Specific documentation

## Transformers & Coco

### Data

In the context of the [Transformers](https://huggingface.co/docs/transformers/index) library, it was necessary to transform the data in the [COCO format](https://cocodataset.org/#format-data). First, please install the hugginface's transformer library following the official documentation. Then, to transform our dataset (for object detection) in this format:

```
python ./scripts/utils/tococodataset.py
```

### Training

Then, we can use the data created to train the transformers models using `/scripts/train/train_transformers.py`). Three models are supported : `detr`, `deformdabledetr` and `yolos`.

An example of the usage of this script :

```
python train_transformers.py -m detr
```

Transformer models are underperforming on this dataset, compared to Yolo.

## Yolo

### Data

We used [Yolov5](https://github.com/ultralytics/yolov5) and [Yolov8](https://github.com/ultralytics/ultralytics) is also usable. Please don't forget to clone the repository and install the dependencies as showed on the documentation of Yolo. Then, the first step is to transform the data in the Yolo format. To do so:

```
python ./scripts/utils/toyolo.py
```

You must also create a `yaml` file at the root of the yolo folder, following the format provided [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml). In our case, we only have 1 class : `Mitosis`.

### Training

Once we obtained the data, the training script can be used. An example of is the following :

```
python train.py --img 256 --batch 16 --epochs 3 --data yolo_format.yaml --weights yolov5s.pt
```

For more details about the parameters, please refer to the [official documentation](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

### Inference

If you want to evaluate the model and obtain evaluation measures, use the following script. Different models trained on the data are available at `/data/elliot/runs/train/`. The model with the best result is in `exp12_62 epochs` : 

```
python val.py --weights ./runs/train/exp12_62epochs/weights/best.pt --data yolo_dataset.yaml --img 256
```

If you want to do the inference on unlabeled data, use :

```
python detect.py --weights ./runs/train/exp12_62epochs/weights/best.pt --source /path/to/data
```

---

# Kartezio

## Data preparation


First step is to adapt `config_kartezio.py`:

```
# Where the geojson of the annotations are located
ANNOTATION_PATH = Path("/path/to/annotations") # Where the geojson files for the annotated slides are 
assert ANNOTATION_PATH.exists()

# Where most of the data will be stored
BASE_PATH = Path("/path/to/storage") # Where all the data will be stored
assert BASE_PATH.exists()

# Slide number = slide number in files -4 (slides starts at 4)
# 996 and 997 refers to 1108 and 1204 because we added them manually after the 999 inital one

SLIDE_LIST = [105, 182, 724, 956, 997] # Number of the slide IN THE FOLDER

SLIDE_TO_PATH = {
    105: '21I000109-1-13-21_095813',
    182: '21I000186-1-15-21_153809',
    724: '21I000728-1-10-21_141128',
    956: '21I000960-1-2-21_133557',
    997: '21I001108-1-3-21_094208',
    999: '21I001108-1-3-21_094208'
} # Correspondance between the slide number in the folder and the file name

SLIDE_FOLDER = Path("/path/to/AprioricsSlides/slides/PHH3/HE/") # Slide folder
IHC_FOLDER = Path("path/to/AprioricsSlides/slides/PHH3/IHC/") # IHC Folder
```

Once this file is configured, run the following scripts :

1 - Slide registration

```
python ihc_mask_registration.py
```

It takes as inputs the slides provided as well as their annotations (annotated patches). It extract the annotated patches and make the correspondance between HE and IHC patches.

2 - Data pipeline

```
python kartezio_data_pipeline.py
```

First, put the annotated data in a usable format.
Then extract the thumbnails of each annotated element, to prepare the manual annotation for kartezio. Once this is done, a manual annotation phase is neeeded. The general idea is to produce a `roi` file containing a drawing around the mitosis to detect (either on HE or IHC).
Finally, split the dataset into training/testing and output `dataset.csv` needed for kartezio.

3 - Training

First, follow the installation steps provided in the [kartezio github repository](https://github.com/KevinCortacero/Kartezio)

In the folder of the dataset and in the created environnement, run the following command :

`kartezio-dataset name=mitose label_name=mitose`

It creates a `meta.json` file. Please modify this file, in particular replace `label : type` from `csv` to `roi` and `label : format` from `ellipse` to `polygon`.

Then, run the following script :

```
python kartezio_training.py
```

By default, the color space used is RGB, but you can also use HED or HSV, or combine them :

```
python kartezio_training.py {RGB,HED,HSV}
```

This will output a trained model, stored in a folder named after a hash which should look like this `364699-f92d8cbd-0a27-480f-b10d-2cfa73b6debf`.

Please copy this hash to the notebook `kartezio_results.ipynb` and use the scripts to analyze the results.


4 - Inference and visualization

Two notebooks are provided in order to run the inference of the newly trained model (`kartezio_inference.ipynb`) and visualize the results (`kartezio_visualization.ipynb`). The first one must be run using `kartezio` environment while the second must be run using `apriorics` environment.