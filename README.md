## Trash or Treasure? An Interactive Dual-Stream Strategy for Single Image Reflection Separation (NeurIPS 2021)
by Qiming Hu, Xiaojie Guo.

### Dependencies
* Python3
* PyTorch>=1.0
* OpenCV-Python, TensorboardX, Visdom
* NVIDIA GPU+CUDA

## :rocket: 1. Single Image Reflection Separation
### Data Preparation

#### Training dataset
* 7,643 images from the
  [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/), center-cropped as 224 x 224 slices to synthesize training pairs.
* 90 real-world training pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal)

#### Tesing dataset
* 45 real-world testing images from [CEILNet dataset](https://github.com/fqnchina/CEILNet).
* 20 real testing pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal)
* 454 real testing pairs from [SIR^2 dataset](https://sir2data.github.io/), containing three subsets (i.e., Objects (200), Postcard (199), Wild (55)). 

### Usage

#### Training 
* For stage 1: ```python train_sirs.py --inet ytmt_ucs --model ytmt_model_sirs --name ytmt_ucs_sirs --hyper```
* For stage 2: ```python train_twostage_sirs.py --inet ytmt_ucs --model twostage_ytmt_model --name ytmt_uct_sirs --hyper --resume --resume_epoch xx --checkpoints_dir xxx```

#### Testing 
```python test_sirs.py --inet ytmt_ucs --model twostage_ytmt_model --name ytmt_uct_sirs_test --hyper --resume --icnn_path ./checkpoints/ytmt_uct_sirs/twostage_unet_68_077_00595364.pt```

## :rocket: 2. Single Image Denoising


## :rocket: 3. Single Image Demoireing
### Data Preparation

#### Training dataset
[AIM 2019 Demoireing Challenge](https://competitions.codalab.org/competitions/20165)

#### Tesing dataset
100 [moireing](https://data.vision.ee.ethz.ch/timofter/AIM19demoire/ValidationMoire.zip) and [clean](https://data.vision.ee.ethz.ch/timofter/AIM19demoire/ValidationClear.zip) pairs from AIM 2019 Demoireing Challenge. 

### Usage

#### Training 
```python train_moire.py --inet ytmt_ucs --model ytmt_model_demoire --name ytmt_ucs_demoire --hyper```

#### Testing 
```python test_moire.py --inet ytmt_ucs --model ytmt_model_demoire --name ytmt_ucs_demoire_test --hyper --resume --icnn_path ./checkpoints/ytmt_ucs_demoire/ytmt_ucs_opt_086_00860000.pt```

## :rocket: 4. Intrinsic Image Decomposition
