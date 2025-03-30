# [AAAI 2025] Official Implementation of 3D$^2$-Actor: Learning Pose-Conditioned 3D-Aware Denoiser for Realistic Gaussian Avatar Modeling

## Installation
1. Clone this repo.
```
git clone https://github.com/silence-tang/GaussianActor.git
```
2. Install environments.
```
conda create -n gaussianactor python=3.10
conda activate gaussianactor
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
(Install pytorch3d) Please download a proper version of pytorch3d from pytorch.org and install locally.
pip install pytorch_lightning==2.0.2
pip install lightning==2.0.2
pip install opencv-python scikit-image matplotlib einops packaging spconv pytorch-fid==0.3.0 lpips==0.1.4 tensorboard==2.12.0
(NOTE) If a module is missing, please install it manually based on the error message.
```
3. Install 3DGS module
```
git clone AnimatableGaussians and place directory gaussians under GaussianActor/.
cd GaussianActor
cd gaussians/diff_gaussian_rasterization_depth_alpha
python setup.py install
cd ../..
```

## ZJU-MoCap Dataset

- Please download the preprocessed ZJU-MoCap dataset from the [huggingface space](https://huggingface.co/datasets/PolarisT/zjumocap/tree/main), place `CoreView_313.zip`, `CoreView_315.zip`, `CoreView_377.zip`, `CoreView_386.zip` under `GaussianActor/data/zju_mocap/` and finally unzip them.

## SMPL/SMPL-X Data

- Please download the preprocessed SMPL/SMPL-X data from the [huggingface space](https://huggingface.co/datasets/PolarisT/zjumocap/tree/main), place `body_models.zip` under `GaussianActor/` and finally unzip it.

## Train the 2D denoiser
```
cd full
CUDA_VISIBLE_DEVICES=0 python train_2d_v1_ok_zju.py
```

## Train the 3D rectifier
```
CUDA_VISIBLE_DEVICES=0 python train_v14_5_1_zju_lr5e-5.py
```

## Test
```
(NOTE) You need to place the pretrained 2D/3D modeles (ckpt files) under `GaussianActor/ckpt`.
CUDA_VISIBLE_DEVICES=0 python test_v16_zju_np.py
```

## Notes
- You may try to enable DDP or mixed-precision training (mix-16) to accelerate the training process.


