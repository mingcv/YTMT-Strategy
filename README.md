## [Trash or Treasure? An Interactive Dual-Stream Strategy for Single Image Reflection Separation (NeurIPS 2021)](https://arxiv.org/abs/2110.10546)
by Qiming Hu, Xiaojie Guo.

### Dependencies
* Python3
* PyTorch>=1.0
* OpenCV-Python, TensorboardX, Visdom
* NVIDIA GPU+CUDA

### Network Architecture
![figure_arch](https://github.com/mingcv/YTMT-Strategy/blob/main/figures/figure_ytmt_networks_final.png)

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
* For stage 1: ```python train_sirs.py --inet ytmt_ucs --model ytmt_model_sirs --name ytmt_ucs_sirs --hyper --if_align```
* For stage 2: ```python train_twostage_sirs.py --inet ytmt_ucs --model twostage_ytmt_model --name ytmt_uct_sirs --hyper --if_align --resume --resume_epoch xx --checkpoints_dir xxx```

#### Testing 
```python test_sirs.py --inet ytmt_ucs_old --model twostage_ytmt_model --name ytmt_uct_sirs_test --hyper --if_align --resume --icnn_path ./checkpoints/ytmt_uct_sirs/ytmt_uct_sirs_68_077_00595364.pt```

*Note: "ytmt_ucs_old" is only for our provided checkpoint, and please change it as "ytmt_ucs" when you train our model by yourself, since it is a refactorized verison for a better view.*

#### Trained weights
[Google Drive](https://drive.google.com/file/d/1yOKFzhhFUdbKzU3eafYKFLN7AdHqW4_7/view?usp=sharing)

#### Visual comparison on real20 and SIR^2
![figure_eval](https://github.com/mingcv/YTMT-Strategy/blob/main/figures/figure_eval_comparation.png)

#### Visual comparison on real45
![figure_test](https://github.com/mingcv/YTMT-Strategy/blob/main/figures/figure_test_comparation.png)

## :rocket: 2. Single Image Denoising

### Data Preparation

#### Training datasets
400 images from the [Berkeley segmentation dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/mftm-iccv01.pdf), following [DnCNN](https://arxiv.org/abs/1608.03981).

#### Tesing datasets
[BSD68 dataset and Set12](https://github.com/SaoYan/DnCNN-PyTorch/tree/master/data). 

### Usage

#### Training 
```python train_denoising.py --inet ytmt_pas --name ytmt_pas_denoising --preprocess True --num_of_layers 9 --mode B --preprocess True```

#### Testing 
```python test_denoising.py --inet ytmt_pas --name ytmt_pas_denoising_blindtest_25 --test_noiseL 25 --num_of_layers 9 --test_data Set68 --icnn_path ./checkpoints/ytmt_pas_denoising_49_157500.pt```

#### Trained weights
[Google Drive](https://drive.google.com/file/d/1FmmUHbWbvTfFlic-gR334cSlesiLZ-e2/view?usp=sharing)

#### Visual comparison on a sample from BSD68
![figure_eval_denoising](https://github.com/mingcv/YTMT-Strategy/blob/main/figures/figure_eval_denoising.png)

## :rocket: 3. Single Image Demoireing
### Data Preparation

#### Training dataset
[AIM 2019 Demoireing Challenge](https://competitions.codalab.org/competitions/20165)

#### Tesing dataset
100 [moireing](https://data.vision.ee.ethz.ch/timofter/AIM19demoire/ValidationMoire.zip) and [clean](https://data.vision.ee.ethz.ch/timofter/AIM19demoire/ValidationClear.zip) pairs from AIM 2019 Demoireing Challenge. 


### Usage

#### Training 
```python train_demoire.py --inet ytmt_ucs --model ytmt_model_demoire --name ytmt_uas_demoire --hyper --if_align```

#### Testing 
```python test_demoire.py --inet ytmt_ucs --model ytmt_model_demoire --name ytmt_uas_demoire_test --hyper --if_align --resume --icnn_path ./checkpoints/ytmt_ucs_demoire/ytmt_ucs_opt_086_00860000.pt```

#### Trained weights
[Google Drive](https://drive.google.com/file/d/16331tan6_1pTli8MNTnC2uoKGcO2cWcv/view?usp=sharing)

#### Visual comparison on the validation set of LCDMoire
![figure_eval_demoire](https://github.com/mingcv/YTMT-Strategy/blob/main/figures/figure_eval_demoire.png)

## :rocket: 4. Intrinsic Image Decomposition
### Data Preparation

[MIT-intrinsic dataset](https://github.com/davidstutz/grosse2009-intrinsic-images), pre-processed following [Direct Intrinsics](https://github.com/tnarihi/direct-intrinsics/tree/master/data/mit)

### Usage

#### Trained weights
[Google Drive](https://drive.google.com/file/d/1sor46AYKgp8rQ7fGScsm1cZzLgRjAWuk/view?usp=sharing)

#### Visual comparison on the validation split of MIT-Intrinsic

![figure_eval_intrinsic](https://github.com/mingcv/YTMT-Strategy/blob/main/figures/figure_eval_intrinsic_decomp.png)

#### Training 
```python train_intrinsic.py --inet ytmt_ucs --model ytmt_model_intrinsic_decomp --name ytmt_ucs_intrinsic```

#### Testing
```python test_intrinsic.py --inet ytmt_ucs --model ytmt_model_intrinsic_decomp --name ytmt_ucs_intrinsic --resume --icnn_path [Path to your weight]```
