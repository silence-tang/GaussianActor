import argparse
import os

import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from data_module_v3 import ZJUMOCAPDataModule
from models_v14_5_1_wo_diffusion import Lightning

# precision
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Train_ZJU-MoCap')

parser.add_argument('--id', type=str, default='313', help='ZJU-MoCap subject id.')
parser.add_argument('--data_path', type=str, default='/home/xxx/GaussianActor/data/zju_mocap', help='ZJU_MoCap dataset absolute path.')
parser.add_argument('--data_path_aux', type=str, default='/home/xxx/GaussianActor/data/zju_mocap_pre2', help='ZJU_MoCap auxiliary undistorted pre-compute data.')
parser.add_argument('--log_path', type=str, default='/home/xxx/GaussianActor/out/v14_gau/', help='log dir.')
parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint.')
parser.add_argument('--ckpt', type=str, default=None, help='Specific checkpoint path to load from. Only works when --resume.')
parser.add_argument('--no_log_img', action='store_true', help='Do not log validation images during training.')
parser.add_argument('--test', action='store_true', help='Test mode.')

if __name__ == "__main__":
    
    args = parser.parse_args()

    # configs
    num_workers = 4
    image_size = 512
    view_image_size = 512
    train_sample_views = 8
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1
    train_dataset_repeat = 1
    check_val_every_n_epoch=5

    sampling_timesteps = 5
    
    run_name = args.id + '3d_zju_v1_lr5e-5'
    
    train_lr = 5e-5
    
    # set data module

    data_module = ZJUMOCAPDataModule(
        data_dir=args.data_path,
        subject='CoreView_' + args.id,
        dataset_folder_aux_undistorted=args.data_path_aux,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        image_size=image_size,
        view_image_size=view_image_size,
        num_workers=num_workers,
        train_dataset_repeat = train_dataset_repeat,
        train_sample_views = train_sample_views,
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
        timesteps = 1000,
        sampling_timesteps = sampling_timesteps,
        log_val_img = not args.no_log_img,
        log_dir=os.path.join(root_dir, 'lightning_logs', run_name),
    )

    tensorboard = pl_loggers.TensorBoardLogger(
        save_dir=root_dir,
        version=run_name,
    )

    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=3, monitor="psnr", mode='max')

    # set trainer
    trainer = pl.Trainer(
        # accumulate_grad_batches=4,
        # num_sanity_val_steps=0,
        # precision='16-mixed',
        accelerator='gpu',
        devices=0,
        max_epochs=100000,
        # strategy='ddp_find_unused_parameters_true',
        default_root_dir=root_dir,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=tensorboard if not args.test else False,
        log_every_n_steps=1,
    )

    ckpt_path = os.path.join(root_dir, 'lightning_logs', run_name, 'checkpoints', 'last.ckpt') if args.ckpt is None else args.ckpt

    # train
    trainer.fit(
        model=lightning,
        datamodule=data_module,
        ckpt_path=ckpt_path if args.resume else None,
    )
