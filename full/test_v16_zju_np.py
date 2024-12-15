import argparse
import os

import pytorch_lightning as pl
import torch
import random
import numpy as np

from data_module_v3 import ZJUMOCAPDataModule
from models_v16_new import Lightning
from exp_new.ckpts import ZJUMOCAPCkpt


# precision
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Test_ZJU-MoCap_NP')

parser.add_argument('--id', type=str, default='313', help='ZJU-MoCap subject id.')
parser.add_argument('--data_path', type=str, default='/home/xxx/GaussianActor/data/zju_mocap', help='ZJU_MoCap dataset absolute path.')
parser.add_argument('--data_path_aux', type=str, default=None, help='ZJU_MoCap auxiliary undistorted pre-compute data.')
parser.add_argument('--log_path', type=str, default='/home/xxx/GaussianActor/test_out/', help='log dir.')
parser.add_argument('--ckpt3d', type=str, default=None, help='Specific checkpoint path to load from.')
parser.add_argument('--ckpt2d', type=str, default=None, help='Specific checkpoint path to load from.')
parser.add_argument('--run_name', type=str, default=None, help='Run name')

parser.add_argument('--st', type=int, default=20, help='sampling timestep.')
parser.add_argument('--split', type=int, default=300, help='3D 2D split timestep.')
parser.add_argument('--insert', type=int, default=2, help='3D gau insert times.')

if __name__ == "__main__":
    
    args = parser.parse_args()

    # configs
    num_workers = 4
    image_size = 512
    view_image_size = 512
    check_val_every_n_epoch = 5

    sampling_timesteps = args.st
    test_recur_timestep = -1
    train_lr = 4e-4

    test_mode = 'np'
    
    run_name = f'np_{args.id}_st{args.st}_split{args.split}_insert{args.insert}' if args.run_name is None else args.run_name
    
    # set data module

    data_module = ZJUMOCAPDataModule(
        data_dir=args.data_path,
        subject='CoreView_' + args.id,
        dataset_folder_aux_undistorted=args.data_path_aux,
        image_size=image_size,
        view_image_size=view_image_size,
        num_workers=num_workers,
        test_mode = test_mode,
        sample_rays = False,
        erode_mask = True,
    )

    root_dir = args.log_path

    lightning = Lightning(
        smpl_type = 'SMPL',
        # smpl_type = 'SMPLX',
        train_lr = train_lr,
        adam_betas = (0.9, 0.99),
        image_size = image_size,
        timesteps = 1000,                        # 1000
        sampling_timesteps=sampling_timesteps,   # 20
        test_recur_timestep=test_recur_timestep,
        test_mode=test_mode,
        log_dir=os.path.join(root_dir, run_name),
        checkpoint_3d_path=args.ckpt3d if args.ckpt3d is not None else ZJUMOCAPCkpt['CoreView_' + args.id]['3d'],
        checkpoint_2d_path=args.ckpt2d if args.ckpt2d is not None else ZJUMOCAPCkpt['CoreView_' + args.id]['2d'],
        stage_split_timestep = args.split,
        gau_insert_times = args.insert,
    
    )

    # set trainer
    trainer = pl.Trainer(
        # accumulate_grad_batches=4,
        num_sanity_val_steps=0,
        max_epochs=100000,
        strategy='ddp_find_unused_parameters_true',
        default_root_dir=root_dir,
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=False,
        log_every_n_steps=1,
        inference_mode=False,
    )

    # test
    trainer.test(
        model=lightning,
        datamodule=data_module,
    )
