import json
import math
import os
import sys
from collections import OrderedDict, namedtuple
from functools import partial, wraps
from random import random
from typing import Any, Optional, Tuple, Union

import cv2
import kiui
import lightning.pytorch as pl
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.gs import GaussianRenderer
from core.options import Options
from core.unet import UNet
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from kiui.lpips import LPIPS
from packaging import version
from pytorch3d.ops import knn_points
from pytorch3d.renderer import (HardPhongShader, MeshRasterizer, MeshRenderer,
                                PointLights, RasterizationSettings)
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from skimage.metrics import structural_similarity as compute_ssim
from torch import einsum, nn
from torchvision import utils
from tqdm.auto import tqdm

from gaussians.gaussian_renderer import render3
from utils.general_utils import (build_scaling_rotation, inverse_sigmoid,
                                 strip_symmetric)

# metrics

def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def ssim_metric(img_pred, img_gt, mask_at_box):
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]

    # compute the ssim
    ssim = compute_ssim(img_pred, img_gt, multichannel=True, channel_axis=2, data_range=1.0)
    return ssim

def lpips_metric(img_pred, img_gt, mask_at_box, loss_fn_vgg, device):
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]

    # compute the lpips
    img_pred = torch.tensor(img_pred, dtype=torch.float32, device=device).reshape(1, h, w, 3).permute(0, 3, 1, 2)
    img_gt = torch.tensor(img_gt, dtype=torch.float32, device=device).reshape(1, h, w, 3).permute(0, 3, 1, 2)

    score = loss_fn_vgg(img_pred, img_gt, normalize=True)
    return score.item()

def render_gaussians(gaussian_vals, R, T, K, image_size, bg_color=(0., 0., 0.)):
    """
    Args:
        gaussian_vals: dict
        R: [v, 3, 3]
        T: [v, 3]
        K: [v, 3, 3]
        image_size: [v, 2]
        bg_color: (0., 0., 0.)
    Returns:
        render_out: dict
    """
    device = K.device
    bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(device)
    v_num = K.shape[0]
    ret = []
    for v in range(v_num):
        extr = torch.eye(4, dtype=R.dtype, device=device)
        extr[:3, :3] = R[v]
        extr[:3, 3] = T[v]
        intr = K[v]
        img_w = image_size[v][1]
        img_h = image_size[v][0]

        render_ret = render3(
            gaussian_vals,
            bg_color,
            extr,
            intr,
            img_w,
            img_h,
        )

        ret.append(render_ret)
    
    render_out = {}
    
    for key in render_ret.keys():
        vals = torch.stack([i[key] for i in ret], dim=0)
        render_out[key] = vals

    return render_out

class LGM(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        # unet
        self.unet = UNet(
            9, 14, 
            # down_channels=(64, 128, 256, 512, 1024, 1024),
            down_channels=(32, 64, 64, 128, 128, 256),
            down_attention=(False, False, False, True, True, True),
            mid_attention=True,
            # up_channels=(1024, 1024, 512, 256, 128),
            up_channels=(256, 128, 128, 64, 64),
            up_attention=(True, True, True, False, False),
        )
        self.splat_size = 128

        # activations
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)
    
    def forward_gaussians(self, images):
        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images) # [B*4, 14, h, w]
        x = x.reshape(B, 4, 14, self.splat_size, self.splat_size)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians
    
    def forward(self, data):
        results = {}

        images = data['input'] # [B, 4, 9, h, W], input features
        gaussians = self.forward_gaussians(images) # [B, N, 14]
        results['gaussians'] = gaussians

        return results
    
    def render_gaussians(self, gaussians, render, bg_color=(0., 0., 0.)):
        assert gaussians.shape[0] == 1
        gaussian_vals = {
            'positions': gaussians[0, :, :3],
            'opacity': gaussians[0, :, 3:4],
            'scales': gaussians[0, :, 4:7],
            'rotations': gaussians[0, :, 7:11],
            'colors': gaussians[0, :, 11:],
            'max_sh_degree': 0,
        }

        R = render.get('R')  # [v, 3, 3]
        T = render.get('T')  # [v, 3]
        K = render.get('K')  # [v, 3, 3]
        image_size = render.get('image_size')  # [v, 2]

        ret = render_gaussians(
            gaussian_vals=gaussian_vals,
            R=R,
            T=T,
            K=K,
            image_size=image_size,
            bg_color=bg_color,
        )

        return ret






class Lightning(pl.LightningModule):
    def __init__(
        self,
        train_lr = 1e-4,
        adam_betas = (0.9, 0.99),
        image_size = 256,
        log_dir = None,
        rgb_loss_weight = 1.,
        mask_loss_weight = 1.,
    ):
        super().__init__()
        self.save_hyperparameters()

        # todo: model
        self.model = LGM()

        self.image_size = image_size
        self.train_lr = train_lr
        self.adam_betas = adam_betas

        self.rgb_loss_weight = rgb_loss_weight
        self.mask_loss_weight = mask_loss_weight

        self.loss_fn_vgg = lpips.LPIPS(net='vgg')

        self.validation_step_outputs = {
            'generated': [],
            'gt': [],
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }
    
    def training_step(self, batch, batch_idx):
        input_masked_images = self.get_masked_input_image(batch, data='current_data_list')
        input_ray_images = self.get_ray_images(batch, data='current_data_list')
        model_input_images = torch.cat([input_masked_images, input_ray_images], dim=2)  # [1, v, 9, image_size, image_size]

        model_out = self.model({
            'input': model_input_images
        })

        render_info = self.get_render_info(batch)
        render_out = self.model.render_gaussians(model_out['gaussians'], render_info)

        loss = self.calc_loss(batch, render_out)
        return loss
    
    def calc_loss(self, batch, model_out):
        render_image = model_out['render']  # [v, 3, h, w]
        render_mask = model_out['mask']  # [v, 1, h, w]

        gt_image = self.get_masked_input_image(batch, data='view_data_list').squeeze(0)  # [v, 3, h, w]
        gt_mask_erode = torch.stack([i['image_mask_erode'] for i in batch['view_data_list']], dim=0)  # [v, 1, h, w], {0, 1, 100}
        mask_valid = (gt_mask_erode==0) | (gt_mask_erode==1)

        # RGB loss
        rgb_loss = F.mse_loss(render_image, gt_image)
        self.log('rgb_loss', rgb_loss, prog_bar=True)

        # mask loss
        mask_loss = F.mse_loss(render_mask[mask_valid], (gt_mask_erode != 0).float()[mask_valid])
        self.log('mask_loss', mask_loss, prog_bar=True)

        # all
        loss = rgb_loss * self.rgb_loss_weight +\
               mask_loss * self.mask_loss_weight
        
        self.log('loss', loss, prog_bar=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        input_masked_images = self.get_masked_input_image(batch, data='current_data_list')
        input_ray_images = self.get_ray_images(batch, data='current_data_list')
        model_input_images = torch.cat([input_masked_images, input_ray_images], dim=2)  # [1, v, 9, image_size, image_size]

        model_out = self.model({
            'input': model_input_images
        })

        render_info = self.get_render_info(batch)
        render_out = self.model.render_gaussians(model_out['gaussians'], render_info)

        val_rays = self.compose_rays_inputs(batch['view_data_list'], packed=True)

        result_dict = self.validation_step_outputs
        
        val_out = render_out['render'].unsqueeze(0)  # [1, v, 3, image_size, image_size]
        result_dict['generated'].append(val_out)

        images = self.get_masked_input_image(batch, data='view_data_list')
        result_dict['gt'].append(images)

        pred_img = val_out.permute(0, 1, 3, 4, 2).squeeze().detach().cpu().numpy()
        gt_img = images.permute(0, 1, 3, 4, 2).squeeze().detach().cpu().numpy()
        bbox_mask = val_rays['sample_image_mask'].detach().cpu().numpy()
        for v_idx in range(images.shape[1]):
            _pred_img = pred_img[v_idx]
            _gt_img = gt_img[v_idx]
            _bbox_mask = bbox_mask[v_idx]
            _psnr = psnr_metric(_pred_img[_bbox_mask], _gt_img[_bbox_mask])
            _ssim = ssim_metric(_pred_img, _gt_img, _bbox_mask)
            _lpips = lpips_metric(_pred_img, _gt_img, _bbox_mask, self.loss_fn_vgg, self.device)
            result_dict['psnr'].append(_psnr)
            result_dict['ssim'].append(_ssim)
            result_dict['lpips'].append(_lpips)

    def on_validation_epoch_end(self):
        result_dict = self.validation_step_outputs
        
        all_preds = torch.cat(result_dict['generated'], dim=0)
        validation_outputs = self.all_gather(all_preds)
        validation_outputs = validation_outputs.reshape(-1, *validation_outputs.shape[-3:])
        grid = utils.make_grid(validation_outputs)
        try:
            self.logger.experiment.add_image(f"generated_images", grid, self.global_step)
        except:
            pass
        result_dict['generated'].clear()

        all_gt = torch.cat(result_dict['gt'], dim=0)
        gts = self.all_gather(all_gt)
        gts = gts.reshape(-1, *gts.shape[-3:])
        grid = utils.make_grid(gts)
        try:
            self.logger.experiment.add_image(f"gt_images", grid, self.global_step)
        except:
            pass
        result_dict['gt'].clear()

        all_psnr = torch.Tensor(result_dict['psnr']).mean()
        all_ssim = torch.Tensor(result_dict['ssim']).mean()
        all_lpips = torch.Tensor(result_dict['lpips']).mean()

        psnr = self.all_gather(all_psnr).mean()
        ssim = self.all_gather(all_ssim).mean()
        lpips = self.all_gather(all_lpips).mean()
        
        self.log('psnr', psnr, rank_zero_only=True, sync_dist=True)
        self.log('ssim', ssim, rank_zero_only=True, sync_dist=True)
        self.log('lpips', lpips, rank_zero_only=True, sync_dist=True)

        self.validation_step_outputs = {
            'generated': [],
            'gt': [],
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.train_lr, betas = self.adam_betas)
        return optimizer
    
    def get_masked_input_image(self, batch, data='current_data_list'):
        image = torch.cat([i['image'] for i in batch[data]], dim=0).permute([0, 3, 1, 2]).clone()  # [v, 3, image_size, image_size]
        image_mask = torch.cat([i['image_mask'] for i in batch[data]], dim=0)  # [v, image_size, image_size]
        image[image_mask.unsqueeze(1).repeat(1, 3, 1, 1) == 0] = 0
        return image.unsqueeze(0)  # [1, v, 3, image_size, image_size]
    
    def get_ray_images(self, batch, data='current_data_list'):
        rays_o = torch.cat([i['cam_loc'] for i in batch[data]], dim=0)  # [v, 3]
        rays_d = torch.cat([i['sample_ray_dirs_all'] for i in batch[data]], dim=0)  # [v, image_size, image_size, 3]
        rays_o = rays_o[:, None, None, :].expand(rays_d.shape)
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [v, h, w, 6]
        rays_plucker = rays_plucker.permute(0, 3, 1, 2).unsqueeze(0)  # [1, v, 6, image_size, image_size]
        return rays_plucker
    
    def get_render_info(self, batch):
        R = []
        T = []
        K = []
        image_size = []

        for view_item in batch['view_data_list']:
            _R = view_item['R']  # [1, 3, 3]
            _T = view_item['T']  # [1, 3]
            _K = view_item['K']  # [1, 3, 3]
            _image_size = torch.cat([view_item['img_height'], view_item['img_width']], dim=-1)
            R.append(_R)
            T.append(_T)
            K.append(_K)
            image_size.append(_image_size)

        R = torch.cat(R, dim=0)  # [v, 3, 3]
        T = torch.cat(T, dim=0)  # [v, 3]
        K = torch.cat(K, dim=0)  # [v, 3, 3]
        image_size = torch.stack(image_size, dim=0)  # [v, 2]
        inputs = {
            'R': R,  # [v, 3, 3]
            'T': T,  # [v, 3]
            'K': K,  # [v, 3, 3]
            'image_size': image_size,  # [v, 2]
        }
        return inputs

    def compose_rays_inputs(self, data_list, packed=False, get_alpha=False):
        cam_loc = []
        sample_ray_dirs = []
        sample_body_bounds_intersections = []
        sample_pixels = []
        sample_image_mask = []
        sample_alpha = []

        for view_item in data_list:
            _cam_loc = view_item['cam_loc']  # [1, 3]
            _sample_ray_dirs = view_item['sample_ray_dirs']  # [1, p, 3]
            _sample_body_bounds_intersections = view_item['sample_body_bounds_intersections']  # [1, p, 2]
            _cam_loc = _cam_loc.unsqueeze(1).repeat(1, _sample_ray_dirs.shape[1], 1)  # [1, p, 3]
            _sample_pixels = view_item['sample_pixels']  # [1, p, 3]
            _sample_image_mask = view_item.get('sample_image_mask')  # [1, image_size, image_size]
            cam_loc.append(_cam_loc)
            sample_ray_dirs.append(_sample_ray_dirs)
            sample_body_bounds_intersections.append(_sample_body_bounds_intersections)
            sample_pixels.append(_sample_pixels)
            sample_image_mask.append(_sample_image_mask)

            if get_alpha:
                _sample_alpha = view_item['sample_alpha']  # [1, p, 1]
                sample_alpha.append(_sample_alpha)

        cat_dim = 1 if packed else 0
        cam_loc = torch.cat(cam_loc, dim=cat_dim)  # [v, p, 3] or [1, v*p, 3]
        sample_ray_dirs = torch.cat(sample_ray_dirs, dim=cat_dim)  # [v, p, 3] or [1, v*p, 3]
        sample_body_bounds_intersections = torch.cat(sample_body_bounds_intersections, dim=cat_dim)  # [v, p, 2] or [1, v*p, 2]
        sample_pixels = torch.cat(sample_pixels, dim=cat_dim)  # [v, p, 3] or [1, v*p, 3]
        if sample_image_mask[0] is not None:
            sample_image_mask = torch.cat(sample_image_mask, dim=0)
        inputs = {
            'ray_o': cam_loc,  # [v, p, 3] or [1, v*p, 3]
            'ray_d': sample_ray_dirs,  # [v, p, 3] or [1, v*p, 3]
            'near': sample_body_bounds_intersections[..., :1],  # [v, p, 1] or [1, v*p, 1]
            'far': sample_body_bounds_intersections[..., 1:],  # [v, p, 1] or [1, v*p, 1]
            'sample_pixels': sample_pixels,  # [v, p, 3] or [1, v*p, 3]
            'sample_image_mask': sample_image_mask,  # [v*(None)] or [v, image_size, image_size]
        }

        if get_alpha:
            sample_alpha = torch.cat(sample_alpha, dim=cat_dim)  # [v, p, 1] or [1, v*p, 1]
            inputs.update({
                'sample_alpha': sample_alpha,  # [v, p, 1] or [1, v*p, 1]
            })

        return inputs