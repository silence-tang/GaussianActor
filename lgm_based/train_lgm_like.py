import argparse
import os

import lightning.pytorch as pl
import torch
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from lgm_based.data_module_v3 import ZJUMOCAPDataModule
from lgm_based.model_lgm_like import Lightning

# precision
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Train_ZJU-MoCap')

parser.add_argument('--id', type=str, default='315',
                    help='ZJU-MoCap subject id.')
parser.add_argument('--data_path', type=str, default='./data/zju_mocap',
                    help='ZJU_MoCap dataset path.')
parser.add_argument('--data_path_aux', type=str, default='./data/zju_mocap_pre2',  # todo
                    help='ZJU_MoCap auxiliary undistorted pre-compute data.')
parser.add_argument('--log_path', type=str, default='./out/lgm_based/lgm_like',
                    help='Root dir.')
parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint.')
parser.add_argument('--ckpt', type=str, default=None,
                    help='Specific checkpoint path to load from. Only works when --resume.')
parser.add_argument('--test', action='store_true', help='Test mode.')

if __name__ == "__main__":
    
    args = parser.parse_args()

    # configs
    num_workers = 4
    image_size = 256
    view_image_size = 512
    train_sample_views = 8
    check_val_every_n_epoch=5
    
    run_name = args.id + '_v1'
    
    train_lr = 5e-5

    # set data module

    data_module = ZJUMOCAPDataModule(
        data_dir=args.data_path,
        subject='CoreView_' + args.id,
        dataset_folder_aux_undistorted=args.data_path_aux,
        image_size=image_size,
        view_image_size=view_image_size,
        num_workers=num_workers,
        train_sample_views = train_sample_views,
        erode_mask = True,
    )

    root_dir = args.log_path

    lightning = Lightning(
        train_lr = train_lr,
        adam_betas = (0.9, 0.99),
        image_size = image_size,
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
        max_epochs=100000,
        strategy='ddp_find_unused_parameters_true',
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