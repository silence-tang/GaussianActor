# [AAAI 2025] Official Implementation of 3D$^2$-Actor: Learning Pose-Conditioned 3D-Aware Denoiser for Realistic Gaussian Avatar Modeling

## Environment

- 3DGS related
  - Clone AnimatableGaussians and place `gaussians` under `GaussianActor/`.
  - `cd gaussians/diff_gaussian_rasterization_depth_alpha`
  - `python setup.py install`
  - `cd ../..`

## ZJU-MoCap Dataset

- Please download the preprocessed ZJU-MoCap dataset from the [huggingface space](https://huggingface.co/datasets/PolarisT/zjumocap/tree/main), place `CoreView_313.zip`, `CoreView_315.zip`, `CoreView_377.zip`, `CoreView_386.zip` under `GaussianActor/data/zju_mocap/` and finally unzip them.

## SMPL/SMPL-X Data

- Please download the preprocessed SMPL/SMPL-X data from the [huggingface space](https://huggingface.co/datasets/PolarisT/zjumocap/tree/main), `place body_models.zip` under `GaussianActor/` and finally unzip it.
