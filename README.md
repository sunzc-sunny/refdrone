
<p align="center">
  <h2/> RefDrone: A Challenging Benchmark for Drone Scene Referring Expression Comprehension</h2>
</p>

<img width="50%" src="./fig/intro.jpg" />

## TODO list
- ✅ Release RefDrone test dataset
- ✅ Release RefDrone train/val dataset
- ✅ Release RDAnnotator
- ✅ Release NGDINO


## Dataset
Please download RefDrone dataset from [Huggingface](https://huggingface.co/datasets/sunzc-sunny/RefDrone).

## Installation
The recommended configuration is 4 A100 GPUs, with CUDA version 12.1. The other configurations in MMDetection should also work.

Please follow the guide to install and set up of the mmdetection. 
```
conda create --name openmmlab python=3.10.6 -y
conda activate openmmlab

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install -U openmim
mim install mmengine
mim install "mmcv==2.2.0"
```

```
git@github.com:sunzc-sunny/refdrone.git
cd refdrone
pip install -v -e .
```

## Preparation
After downloading and unzipping the images and annotations, place the dataset (or create a symbolic link to it) inside the datasets/ directory. For convenience, it's recommended to store all images together in an all_images/ subdirectory. Your directory structure should look like this:
```
refdrone
├── configs
├── datasets
│   ├── VisDrone2019
│   │   ├── RefDrone_train_mdetr.json
│   │   ├── RefDrone_test_mdetr.json
│   │   ├── RefDrone_val_mdetr.json
│   │   ├── all_image
│   │   │   ├── xxx.jpg
│   │   │   ├── ...
```

Then use [coco2odvg.py](../../tools/dataset_converters/coco2odvg.py) to convert RefDrone_train_mdetr.json into the ODVG format required for training:
python tools/dataset_converters/refcoco2odvg.py datasets/VisDrone2019

## Usage

### Train
```bash
# single gpu
python tools/train.py configs/NGDINO/ngdino_swin-t_refdrone.py 

# multi gpu
bash tools/dist_train.sh configs/NGDINO/ngdino_swin-t_refdrone.py   NUM_GPUs
```

### Inference

```bash
# single gpu
python tools/test.py configs/NGDINO/ngdino_swin-t_refdrone.py  CHECKPOINT

# multi gpu
bash tools/dist_test.sh configs/NGDINO/ngdino_swin-t_refdrone.py  CHECKPOINT NUM_GPUs
```



