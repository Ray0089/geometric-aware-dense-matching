
# Geometric-aware dense matching network for 6D pose estimation of objects from RGB-D images

source code for our paper in Pattern Recognition.

## Installation

### Create conda environments and activate it.

```
conda create --name gdm6d python==3.7
conda activate gdm6d
```

### Install pytorch 1.10

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### Install pytorch-geometric for SplineCNN

```
pip install --no-index torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch_geometric==2.0.0
```

### Install mmcv

```
pip install -U openmim
mim install mmcv
```

### Install detectron2

```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```
### Other dependencies

```
pip install -r requirements.txt
```

### Compile RandLA according to [Readme](/models/RandLA/README.md)

## Data preparation

### 1. Download YCB-V and Linemod from [BOP6D](https://bop.felk.cvut.cz/datasets/)
### 2. Unpack and link them to datasets/lm/linemod and datasets/ycbv/ycbv, respectively
```
ln -s /your/path/to/lm dataset/ datasets/lm/linemod
ln -s /your/path/to/ycbv dataset/ datasets/ycbv/ycbv
```
### 3. Download the simplified model which contain 8192 vertices of each object from [here](https://pan.baidu.com/s/1fXdX-V3Vl82qKLz_fn-wag?pwd=nmh0)

### 4. Place the models to the datasets
``` 
mkdir -p datasets/lm/linemod/kps
cp /your/path/to/downloaded model/* datasets/lm/linemod/kps
mkdir -p datasets/ycbv/ycbv/kps
cp /your/path/to/downloaded model/* datasets/ycbv/ycbv/kps
```

## Train the model 

### run the corresponding .sh scripts to train the model. 
```
./train_lm.sh
./train_ycb.sh
```
## Test the model

### 1. Download the generated bbox from Mask-RCNN [here](https://pan.baidu.com/s/1fXdX-V3Vl82qKLz_fn-wag?pwd=nmh0)

### 2. Copy the real_det.json file to datasets/lm/linemod/test/ or datasets/ycbv/ycbv/test/ folder.
### 3. run the corresponding .sh scripts to test the model. 
```
./test_lmo.sh
./test_ycb.sh
```

## Declaration

Part of the source code are from [FFB6D](https://github.com/ethnhe/FFB6D),[CDPN](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi),[DPOD](https://github.com/yashs97/DPOD). We express our sincere gratitude for their work.

## Citation

If you find this code useful for you, please cite our paper
```
@article{wu2023geometric,
  title={Geometric-aware Dense Matching Network for 6D Pose Estimation of Objects from RGB-D Images},
  author={Wu, Chenrui and Chen, Long and Wang, Shenglong and Yang, Han and Jiang, Junjie},
  journal={Pattern Recognition},
  pages={109293},
  year={2023},
  publisher={Elsevier}
}
```
>>>>>>> master
