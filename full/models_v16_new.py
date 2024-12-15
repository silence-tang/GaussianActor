import json
import math
import os
import sys
from collections import OrderedDict, namedtuple
from functools import partial, wraps
from random import random
# from typing import Any, Optional, Tuple, Union

import cv2
import pytorch_lightning as pl
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch
import torch.nn.functional as F

from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.metrics import structural_similarity as compute_ssim
from torch import einsum, nn
from torchvision import utils
from tqdm.auto import tqdm

from models_2d_v1 import Denoiser as Denoiser_2d
from models_v14_5_1_wo_diffusion import GauModel

# helper functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def safe_normalize(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        l = np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        l = torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))
    return x / l

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

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# gaussian diffusion

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    Linear schedule, proposed in the original DDPM paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule
    As proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    Sigmoid schedule
    Proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    Better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'add'])

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        gau_model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_x0',
        beta_schedule='sigmoid',
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.,
        auto_normalize=True,
        min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5
    ):
        super().__init__()
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.gau_model = gau_model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'Objective must be either pred_noise (predict noise), pred_x0 (predict image start), or pred_v (predict v [v-parameterization as defined in appendix D of the progressive distillation paper, used successfully in imagen-video])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'Unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # Sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps)  # Default number of sampling timesteps to number of timesteps during training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps  # True if using DDIM sampling
        self.ddim_sampling_eta = ddim_sampling_eta  # Typically 0

        # Helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # Above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Derive loss weight
        # SNR - Signal-to-Noise Ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # Auto-normalization of data [0, 1] -> [-1, 1] - can be turned off by setting to False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, data, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t, data, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        additional_info = None

        if self.objective == 'pred_noise':
            raise NotImplementedError(f'Objective {self.objective} not implemented yet!')
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            b, v_num, c_num, h, w, _ = *x.shape, 0
            x_start = torch.zeros_like(x).reshape(b, v_num, h, w, c_num)  # [1, v, image_size, image_size, 3]
            x_start = model_output['out'].unsqueeze(0)
            x_start = self.normalize(x_start)

            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

            additional_info = {
                't': t,
            }

        elif self.objective == 'pred_v':
            raise NotImplementedError(f'Objective {self.objective} not implemented yet!')
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start, additional_info)

    def p_mean_variance(self, x, t, data, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, data, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, data, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, data=data, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.  # No noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, data, return_all_timesteps=False):
        raise NotImplementedError
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, data, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, data, return_all_timesteps=False, partial_start_time=-1, x0=None):

        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if partial_start_time > -1:
            if partial_start_time < times[-2]:
                partial_start_time = times[-2]
            assert x0 is not None
            max_time = next(v for _, v in enumerate(times) if v <= partial_start_time)
            max_time = torch.full((batch,), max_time, dtype=torch.long, device=device)
            x_start = self.normalize(x0)
            noise = torch.randn_like(x_start)
            img = self.q_sample(x_start=x_start, t=max_time, noise=noise)
        else:
            img = torch.randn(shape, device=device)
            x_start = None

        imgs = [img]        

        # for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
        for time, time_next in time_pairs:
            if time > partial_start_time > -1:
                continue
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, additional_info, *_ = self.model_predictions(
                img, time_cond, data, self_cond, clip_x_start=True, rederive_pred_noise=True
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)

        if partial_start_time > -1:
            additional_info['recur_noised'] = self.unnormalize(imgs[0])
        return ret, additional_info

    @torch.no_grad()
    def sample(self, data, batch_size=1, views=4, **kwargs):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # sample_fn = self.ddim_3d_aware_sample
        return sample_fn((batch_size, views, channels, image_size, image_size), data, **kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, data, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, data, self_cond)

        return img
    
    @torch.no_grad()
    def reconstruction(self, img, t, data, noise=None):
        b, v, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        t = torch.full((b,), t, dtype=torch.long, device=device)
        img = self.normalize(img)
        noise = default(noise, lambda: torch.randn_like(img))
        x = self.q_sample(x_start=img, t=t, noise=noise)
        self_cond = img if self.self_condition else None
        pred_noise, x_start, additional_info, *_ = self.model_predictions(
            img, t, data, self_cond, clip_x_start=True, rederive_pred_noise=True
        )
        rec = self.unnormalize(x_start)
        return rec, additional_info


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, data, current_data=None, noise=None):
        b, v, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Noise sampling

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # If doing self-conditioning, 50% of the time, predict x_start from the current set of times
        # and condition with the U-Net with that
        # This technique will slow down training by 25%, but seems to significantly lower FID

        x_self_cond = None
        if self.self_condition and random() < 0.5 and current_data is not None:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, current_data).pred_x_start
                x_self_cond.detach_()

        # Predict and take gradient step

        model_out = self.model(x, t, data, x_self_cond)

        if self.objective == 'pred_noise':
            raise NotImplementedError(f'Objective {self.objective} not implemented yet!')
            target = noise
        elif self.objective == 'pred_x0':
            # prediction = model_out['rgb_map']
            # target = data['rays']['sample_pixels']
            ret = model_out
        elif self.objective == 'pred_v':
            raise NotImplementedError(f'Objective {self.objective} not implemented yet!')
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'Unknown objective {self.objective}')

        # loss = self.loss_fn(prediction, target, reduction='none')
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')

        # loss_raw = loss

        # loss = loss * extract(self.loss_weight, t, loss.shape)
        # return loss.mean(), loss_raw.mean()

        loss_weight = self.loss_weight.gather(-1, t)
        return ret, loss_weight

    def forward(self, img, *args, **kwargs):
        b, v, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'Height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
    
    @torch.no_grad()
    def ddim_3d_aware_sample(
        self, cur_cond, view_info, partial_start_time=-1, x0=None, 
        return_all=False, back_to_3d_add_info=False, 
        stage_split_timestep=200, gau_insert_times=4
    ):
        batch = 1

        # cur_cond: anchor view information
        # view_info: view information required for rendering the final output image
        # partial_start_time: Only set to 250 when generating the second frame of video, otherwise -1
        # x0: Only set to reposed_img when generating the second frame of video, otherwise None
        # return_all = False, unchanged
        # back_to_3d_add_info: True only when generating video, otherwise False
        # stage_split_timestep = 200, unchanged
        # gau_insert_times = 4, unchanged

        # total_timesteps: 1000
        # sampling_timesteps: 20
        # eta: 0
        device, total_timesteps, sampling_timesteps, eta = (
            self.betas.device, 
            self.num_timesteps, 
            self.sampling_timesteps, 
            self.ddim_sampling_eta
        )

        # cur_view_num represents the number of anchor views, view_view_num represents the number of views required for rendering
        cur_view_num, view_view_num = cur_cond['R'].shape[0], view_info['R'].shape[0]

        times = torch.linspace(-1, total_timesteps-1, steps=sampling_timesteps+1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # 20 pairs, [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # Only set to 250 when generating the second frame of video, otherwise -1
        if partial_start_time > -1:
            # todo: partial_start_time
            # partial_start_time -= 1
            if partial_start_time < times[-2]:
                partial_start_time = times[-2]
            
            assert x0 is not None
            
            # Find a max_time in valid times that <= partial_start_time to add a small amount of noise
            max_time = next(v for _, v in enumerate(times) if v <= partial_start_time)
            max_time = torch.full((batch,), max_time, dtype=torch.long, device=device)
            # NOTE: modified
            x_start = self.normalize(x0)
            noise = torch.randn_like(x_start)
            img = self.q_sample(x_start=x_start, t=max_time, noise=noise)
        else:
            # Start denoising from pure noise image
            img = torch.randn(
                (batch, cur_view_num, self.channels, self.image_size, self.image_size), 
                device=device
            )
            x_start = None

        # pass
        assert gau_insert_times >= 2

        # Only set to true when generating the second frame of video, otherwise False
        if partial_start_time > -1:
            # Ensure partial denoising timestep >= split timestep, otherwise cannot insert 3D rectification step
            assert partial_start_time >= stage_split_timestep

        # Split time_pairs into 3D (before split timestep) and pure 2D (after split timestep)
        time_pairs_stage_3d = []
        time_pairs_stage_2d = []

        for time_pair in time_pairs:
            if time_pair[0] >= stage_split_timestep:
                time_pairs_stage_3d.append(time_pair)
            else:
                time_pairs_stage_2d.append(time_pair)
        
        # stage 1: 3D-aware denoising
        
        # (len(time_pairs_stage_3d) + gau_insert_times - 3) // (gau_insert_times - 1)
        # 1 @ 2 3 4 @ 5 6 7 @ 8 9 10 ^@
        # 1 @ 2 3 4 5 @ 6 7 8 9 @ 10 11 12 13 ^@
        # 1 @ 2 3 4 5 @ 6 7 8 9 @ 10 11 12 ^@
        # 1 @ 2 3 4 5 6 @ 7 8 9 10 11 @ 12 13 14 ^@
        insert_between = (len(time_pairs_stage_3d) + gau_insert_times - 3) // (gau_insert_times - 1)
        inserted_time_pairs_stage_3d = []

        for ind in range(len(time_pairs_stage_3d)):
            time, time_next = time_pairs_stage_3d[ind]
            # When generating the second frame of video, need to step to partial_start_time first
            if time > partial_start_time > -1:
                continue
            # Construct inserted_time_pairs_stage_3d
            if (ind == 0 or ind % insert_between == 0) and time_next != time_pairs_stage_3d[-1][1]:
                inserted_time_pairs_stage_3d.append((time, "@"))
                inserted_time_pairs_stage_3d.append(("@", time_next))
            else:
                inserted_time_pairs_stage_3d.append((time, time_next))
        
        if inserted_time_pairs_stage_3d[-1] == "@":
            inserted_time_pairs_stage_3d.pop(-1)

        stage_3d_denoiser_cond = {
            'cond': cur_cond,
        }

        # prepare return all
        # pass
        if return_all:
            imgs = [
                (partial_start_time + 1 if partial_start_time > -1 else self.num_timesteps, img.cpu())
            ]
            x_starts = []
            if partial_start_time > -1:
                x_starts.append((partial_start_time + 1, x_start.cpu()))
            x_starts_gau_refine = []
            x_starts_gau_refine_mid = []

        for time, time_next in inserted_time_pairs_stage_3d:
            
            # 3D rectification step
            if time == "@":
                last_x_start = x_start
                # "Pure image"
                x_start_unnorm = unnormalize_to_zero_to_one(x_start)  # [1, vc, 3, image_size, image_size], [0, 1]
                # 3D rectification step: Start from the "pure" images of 4 anchor views to obtain 3dgs corresponding to the pose
                # Then render the anchor views to obtain more 3D-consistent results
                # Calculate inference time
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                gau_out = self.gau_model(
                    x_start_unnorm,
                    data=cur_cond,
                    render=cur_cond,  # 3dgs also only render anchor views here
                )
                end_event.record()
                torch.cuda.synchronize()
                print(f"3D module took: {start_event.elapsed_time(end_event):.4f} milliseconds")
                # More 3D-consistent results
                x_start_refined_unnorm = gau_out['render'].unsqueeze(0)  # [1, vc, 3, image_size, image_size], [0, 1]
                # NOTE: Here, x_start is equivalent to the pure image output by the 3D denoising step
                x_start = normalize_to_neg_one_to_one(x_start_refined_unnorm)
                x_start = torch.clamp(x_start, min=-1., max=1.)
                # Must map to [-1,1] before encoding
                # Weighted combination of pred_noise and x_start to obtain the denoised result of the current 3D step, i.e., "x_{t-1}" to be used for the next 2D step
                # Here, img is the denoised result of the previous 2D denoising step
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)
                # pass
                if return_all:
                    x_starts_gau_refine.append((time_next + 1, last_x_start.cpu(), x_start.cpu()))
            
            # 2D refine step
            else:
                # do 2d denoising
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                self_cond = x_start if self.self_condition else None
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                pred_noise, x_start, *_ = self.model_predictions(
                    img, time_cond, stage_3d_denoiser_cond, self_cond, 
                    clip_x_start=True, rederive_pred_noise=True
                )
                end_event.record()
                torch.cuda.synchronize()  # Wait for all GPU operations to complete
                print(f"2D module took: {start_event.elapsed_time(end_event):.4f} milliseconds")
                # x_start: [1, vc, 3, image_size, image_size], [-1, 1]
                last_time = time
            
            if time_next != "@":
                if time_next < 0:
                    img = x_start
                    if return_all:
                        imgs.append((0, img.cpu()))
                        x_starts.append((0, x_start.cpu()))
                    continue

                alpha = self.alphas_cumprod[last_time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(img)

                # Use x_start and pred_noise to compute the denoised image at the current step (x_{t-1}), which will be used for the next 2D step
                img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

                # pass
                if return_all:
                    imgs.append((time_next + 1, img.cpu()))
                    x_starts.append((time_next + 1, x_start.cpu()))
            
        # ------------------------------------------------------------------------------------------------
        # stage 2: generate multi-view images
        
        # Concatenate current camera to view camera
        # back_to_3d_add_info is True only when generating video, otherwise False
        if back_to_3d_add_info:
            view_info_new = {}
            for key in ['R', 'T', 'K', 'image_size']:
                cur_val = cur_cond[key]
                view_val = view_info[key]
                view_info_new[key] = torch.cat([view_val, cur_val], dim=0)
        else:
            view_info_new = view_info

        last_x_start = x_start
        x_start_unnorm = unnormalize_to_zero_to_one(x_start)  # [1, vc, 3, image_size, image_size], [0, 1]
        
        # Last 3D rectification step
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        gau_out = self.gau_model(
            x_start_unnorm,
            data=cur_cond,
            render=view_info_new,
        )
        end_event.record()
        torch.cuda.synchronize()  # Wait for all GPU operations to complete
        print(f"3D module took: {start_event.elapsed_time(end_event):.4f} milliseconds")
        x_start_refined_unnorm = gau_out['render'].unsqueeze(0)  # [1, vv, 3, image_size, image_size], [0, 1]
        x_start = normalize_to_neg_one_to_one(x_start_refined_unnorm)
        x_start = torch.clamp(x_start, min=-1., max=1.)
        # At this point, x_start represents the multi-view pure images output by the last second Gaussian denoising step

        if return_all:
            x_starts_gau_refine_mid.append((time_next + 1, last_x_start.cpu(), x_start.cpu()))

        # stage 3: 2D denoising
        # Finally refine a few steps with 2D
        if time_next > -1:
            noise = torch.randn_like(x_start)
            time_next_tensor = torch.full((batch,), time_next, dtype=torch.long, device=device)
            # Inputs to the 2D stage are always x_start with added noise
            img = self.q_sample(x_start=x_start, t=time_next_tensor, noise=noise)
            pred_noise = self.predict_noise_from_start(img, time_next_tensor, x_start)
            stage_2d_denoiser_cond = {'cond': {}}
            stage_2d_denoiser_cond['cond'].update(cur_cond)
            stage_2d_denoiser_cond['cond'].update(view_info_new)
        else:
            img = x_start
            if return_all:
                imgs.append((0, img.cpu()))

        for time, time_next in time_pairs_stage_2d:
            # do 2d denoising
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            # At this point, we are close to the end of the entire 2D3D denoising process, and img (the output of the previous Gaussian step) is considered to be nearly pure ground truth
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, stage_2d_denoiser_cond, self_cond, 
                clip_x_start=True, rederive_pred_noise=True
            )
            end_event.record()
            torch.cuda.synchronize()  # Wait for all GPU operations to complete
            print(f"2D module took: {start_event.elapsed_time(end_event):.4f} milliseconds")
            # x_start: [1, vc, 3, image_size, image_size], [-1, 1]
            
            if time_next < 0:
                img = x_start
                if return_all:
                    imgs.append((0, img.cpu()))
                    x_starts.append((0, x_start.cpu()))
                # If time_next < 0, directly end the loop
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            if return_all:
                imgs.append((time_next + 1, img.cpu()))
                x_starts.append((time_next + 1, x_start.cpu()))

        # Finally return the image [0, 1]
        ret = self.unnormalize(img)

        # stage 4: back to 3D
        # back_to_3d_add_info is True only when generating video, otherwise False
        if back_to_3d_add_info:
            cur_final = ret[:, -cur_view_num:, ...]  # Last time, use 4 anchor view images
            ret = ret[:, :view_view_num, ...]
            gau_out = self.gau_model(
                cur_final,
                data=cur_cond,
                render=None,
            )

        add_info = {
            'gaussian_vals': gau_out['gaussian_vals'],
            'local_coords': gau_out['local_coords'],
        }

        if return_all:
            add_info.update({
                'imgs': [(time, self.unnormalize(image)) for time, image in imgs],
                'x_starts': [(time, self.unnormalize(image)) for time, image in x_starts],
                'x_starts_gau_refine': [(time, image_ori, image_refine) for time, image_ori, image_refine in x_starts_gau_refine],
                'x_starts_gau_refine_mid': [(time, image_ori, image_refine) for time, image_ori, image_refine in x_starts_gau_refine_mid],
            })
        
        return ret, add_info


# Lightning module
class Lightning(pl.LightningModule):
    def __init__(
        self,
        smpl_type='SMPL',
        train_lr=1e-4,
        adam_betas=(0.9, 0.99),
        image_size=512,
        timesteps=1000,
        sampling_timesteps=250,
        test_mode=None,
        log_dir=None,
        test_recur_timestep=200,   # -1 for np.py, 250 for video.py
        checkpoint_3d_path=None,
        checkpoint_2d_path=None,

        stage_split_timestep=200, 
        gau_insert_times=4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.gau_model = GauModel(smpl_type=smpl_type)

        self.denoiser_2d = Denoiser_2d(smpl_type=smpl_type)

        self.diffusion = GaussianDiffusion(
            model=self.denoiser_2d,
            gau_model=self.gau_model,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
        )

        # Sampling and training hyperparameters
        self.image_size = image_size
        self.train_lr = train_lr
        self.adam_betas = adam_betas

        self.test_step_outputs = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }

        # LPIPS
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        self.test_mode = test_mode  # None, 'nv', 'np', 'cam'
        self.log_dir = log_dir
        self.test_recur_timestep = test_recur_timestep
        self.stage_split_timestep = stage_split_timestep
        self.gau_insert_times = gau_insert_times

        self.test_last_add_info = None

        if checkpoint_3d_path is not None:
            ckpt = torch.load(checkpoint_3d_path)
            old_state_dict = ckpt['state_dict']
            state_dict = OrderedDict()
            for k, v in old_state_dict.items():
                if k.startswith('model'):
                    state_dict[k[6:]] = v
            load_ret = self.gau_model.load_state_dict(state_dict, strict=False)
            print(f'{load_ret}')
        
        if checkpoint_2d_path is not None:
            ckpt = torch.load(checkpoint_2d_path)
            old_state_dict = ckpt['state_dict']
            state_dict = OrderedDict()
            for k, v in old_state_dict.items():
                if k.startswith('denoiser'):
                    state_dict[k[9:]] = v
            load_ret = self.denoiser_2d.load_state_dict(state_dict, strict=False)
            print(f'{load_ret}')

    ############################################################################################################
    # Execute test_step for data points after 2000 frames
    def test_step(self, batch, batch_idx):
        # vc = view_cur, vv = view_view
        cur_masked_images = self.get_masked_input_image(batch, data='current_data_list')  # [1, vc, 3, image_size, image_size]
        # Represents a new pose of a frame
        cur_cond = self.get_diffusion_conds(batch, data='current_data_list')
        cur_view_num = cur_masked_images.shape[1]

        if self.test_mode == 'cam':  # Single frame rotating around in 3D+2D denoising process
            # method 1
            # todo: rot axis
            self.cam_config = 1
            assert self.cam_config == 1
            assert self.test_last_add_info is None or self.test_recur_timestep <= -1

            # generate view cams
            view_info_ori = self.get_render_info(batch)
            device = view_info_ori['R'].device

            center_loc = np.array([0.1, 0.3, 1])
            radius = 3
            elevation = -10
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            # Convert to radians
            elevation = np.deg2rad(elevation)
            azimuth = np.deg2rad(azimuth)

            T_all = []

            for azi in azimuth:

                x = radius * np.cos(elevation) * np.sin(azi)
                y = -radius * np.sin(elevation)
                z = radius * np.cos(elevation) * np.cos(azi)

                campos = np.array([z, x, y]) + center_loc  # [3]
            
                T = np.eye(4, dtype=np.float32)

                # forward is camera --> target
                forward_vector = safe_normalize(center_loc - campos)
                up_vector = np.array([0, 0, -1], dtype=np.float32)
                right_vector = -safe_normalize(np.cross(forward_vector, up_vector))
                up_vector = -safe_normalize(np.cross(right_vector, forward_vector))
                R = np.stack([right_vector, up_vector, forward_vector], axis=1)

                T[:3, :3] = R
                T[:3, 3] = campos

                T = torch.from_numpy(T).unsqueeze(0).to(device)
                T = torch.inverse(T)
                T_all.append(T)
            
            T_all = torch.cat(T_all, dim=0)  # [v?, 4, 4]

            # for R, T, K, image_size in zip(
            #     torch.split(T_all[:, :3, :3], split_size, dim=0),
            #     torch.split(T_all[:, :3, 3], split_size, dim=0),
            #     torch.split(view_info_ori['K'][:1].repeat(T_all.shape[0], 1, 1), split_size, dim=0),
            #     torch.split(view_info_ori['image_size'][:1].repeat(T_all.shape[0], 1), split_size, dim=0),
            # ):
            view_info = {
                'R': T_all[:, :3, :3],
                'T': T_all[:, :3, 3],
                'K': view_info_ori['K'][:1].repeat(T_all.shape[0], 1, 1),
                'image_size': view_info_ori['image_size'][:1].repeat(T_all.shape[0], 1),
                'split': 20,
            }

            # from pure noise
            diffusion_out, _ = self.diffusion.ddim_3d_aware_sample(
                cur_cond=cur_cond,   # Represents anchor view information
                view_info=view_info, # Represents view information required for rendering the final output image
                # x_0 = None, represents starting from noise
                back_to_3d_add_info=self.test_recur_timestep > -1, # False
                stage_split_timestep=self.stage_split_timestep,
                gau_insert_times=self.gau_insert_times,
            )  # [1, v??, 3, image_size, image_size]
            
            # save images and return
            save_dir = os.path.join(self.log_dir, self.test_mode)
            rgb_img_dir = os.path.join(save_dir, 'rgb')
            for _dir in [save_dir, rgb_img_dir]:
                if not os.path.exists(_dir):
                    os.makedirs(_dir)

            frame_idx = int(batch['view_data_list'][0]['frame_idx'])
            for _i, _v in enumerate(range(diffusion_out.shape[1])):
                rgb_image = diffusion_out[0, _v]
                utils.save_image(rgb_image, os.path.join(rgb_img_dir, f'rgb_raw_f{frame_idx:06d}_cam{_i:03d}.png'))
            return

            # method 2: ???

            #     cam_poses.append(torch.from_numpy(orbit_camera(elevation, azi, radius=radius.detach().cpu().numpy(), target=center_loc.detach().cpu().numpy(), opengl=True)).unsqueeze(0).to(cam_locs.device))
            # cam_poses = torch.cat(cam_poses, dim=0)

            # for rad in torch.deg2rad(torch.arange(0, 360, 10).to(device)):
            #     # todo: rot axis
            #     self.axis = 3
            #     assert(self.axis == 3)
            #     if self.axis == 3:
            #         rot = radius * torch.FloatTensor([
            #             [  torch.cos(rad), torch.sin(rad),   0, center_loc[0].cpu()],
            #             [- torch.sin(rad), torch.cos(rad),   0, center_loc[1].cpu()],
            #             [               0,              0,   1, center_loc[2].cpu()],
            #             [               0,              0,   0,                   1],
            #         ]).to(device)



            # method 3: interpolate NO!!!!!!!!
            # interpolate_size = 20
            # view_info_ori = self.get_render_info(batch)
            # view_info = {
            #     'R': F.interpolate(view_info_ori['R'].permute(1, 2, 0), size=interpolate_size, mode='linear', align_corners=True).permute(2, 0, 1),
            #     'T': F.interpolate(view_info_ori['T'].permute(1, 0).unsqueeze(0), size=interpolate_size, mode='linear', align_corners=True).squeeze(0).permute(1, 0),
            #     'K': view_info_ori['K'][:1].repeat(interpolate_size, 1, 1),
            #     'image_size': view_info_ori['image_size'][:1].repeat(interpolate_size, 1),
            # }

        # Both np and video execute the following code, but their test_recur_timestep are different, -1 -> 250
        # When generating video, the batch here is a video dataset, batch_size=1 means each test_step generates a frame in the sequence
        view_masked_images = self.get_masked_input_image(batch, data='view_data_list')  # [1, vv, 3, image_size, image_size]
        view_info = self.get_render_info(batch)
        view_view_num = view_masked_images.shape[1]

        # Video generation: first time do this, self.test_recur_timestep = 250, self.test_last_add_info is initially None and then not None
        # np generation: do this, self.test_recur_timestep = -1, self.test_last_add_info is not used
        # Only when generating video, back_to_3d_add_info is true

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        if self.test_last_add_info is None or self.test_recur_timestep <= -1:
            # from pure noise, single frame generation [1, vv, 3, image_size, image_size]
            # NOTE: modified
            
            start_event.record()

            diffusion_out, add_info = self.diffusion.ddim_3d_aware_sample(
                cur_cond=cur_cond,
                view_info=view_info,
                back_to_3d_add_info=self.test_recur_timestep > -1, # np generation: False, video generation: True
                stage_split_timestep=self.stage_split_timestep,
                gau_insert_times=self.gau_insert_times,
            )

            end_event.record()
            torch.cuda.synchronize()  # Wait for all GPU operations to complete
            print(f"Single frame generation took: {start_event.elapsed_time(end_event):.4f} milliseconds")

            # After first generation, assign self.test_last_add_info
            self.test_last_add_info = add_info
            repose_imgs = None

        # Video generation: after running test_step for the first time, do this
        else:
            # from last frame
            # step 1: render current pose with last gaussians
            gau_model_input_data = {}
            gau_model_input_data.update(cur_cond)
            gau_model_input_data.update(self.test_last_add_info)
            gau_model_ret = self.gau_model.forward(
                noised_image=None,
                data=gau_model_input_data,
                render=gau_model_input_data,
            )

            repose_imgs = gau_model_ret['render'].unsqueeze(0)  # [1, vc, 3, image_size, image_size]

            # step 2: video generation partial denoising
            # When generating video, to ensure certain speed, adopt a partial denoising strategy starting from 250 steps, i.e., partial denoising
            # NOTE: modified
            diffusion_out, add_info = self.diffusion.ddim_3d_aware_sample(
                cur_cond=cur_cond,
                view_info=view_info,
                partial_start_time=self.test_recur_timestep,        # 250
                x0=repose_imgs,                                     # Only when video generation starts from the second frame, x0 is not None
                back_to_3d_add_info=self.test_recur_timestep > -1,  # True
                stage_split_timestep=self.stage_split_timestep,
                gau_insert_times=self.gau_insert_times,
            )  # [1, vv, 3, image_size, image_size]
            # step 3: record gaussians
            self.test_last_add_info = add_info

        ####################################### Compute Metrics #######################################
        # test_rays = self.compose_rays_inputs(batch['view_data_list'], packed=True)
        # gt_img = view_masked_images.permute(0, 1, 3, 4, 2).squeeze().detach().cpu().numpy()
        # bbox_mask = test_rays['sample_image_mask'].detach().cpu().numpy()

        # # metric 2d
        # result_dict = self.test_step_outputs

        # pred_img = diffusion_out[0].permute(0, 2, 3, 1).detach().cpu().numpy()
        
        # for v_idx in range(view_masked_images.shape[1]):
        #     _pred_img = pred_img[v_idx]
        #     _gt_img = gt_img[v_idx]
        #     _bbox_mask = bbox_mask[v_idx]
        #     _psnr = psnr_metric(_pred_img[_bbox_mask], _gt_img[_bbox_mask])
        #     _ssim = ssim_metric(_pred_img, _gt_img, _bbox_mask)
        #     _lpips = lpips_metric(_pred_img, _gt_img, _bbox_mask, self.loss_fn_vgg, self.device)
        #     result_dict['psnr'].append(_psnr)
        #     result_dict['ssim'].append(_ssim)
        #     result_dict['lpips'].append(_lpips)

        # save images
        save_dir = os.path.join(self.log_dir, self.test_mode)
        gt_img_dir = os.path.join(save_dir, 'gt')
        rgb_img_dir = os.path.join(save_dir, 'rgb')
        diffusion_img_dir = os.path.join(save_dir, 'diffusion')
        repose_img_dir = os.path.join(save_dir, 'repose')
        for _dir in [save_dir, gt_img_dir, rgb_img_dir, diffusion_img_dir, repose_img_dir]:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        
        frame_idx = int(batch['view_data_list'][0]['frame_idx'])
        
        for _i, _v in enumerate(range(view_masked_images.shape[1])):
            gt_image = view_masked_images[0, _v]
            rgb_image = diffusion_out[0, _v]
            utils.save_image(gt_image, os.path.join(gt_img_dir, f'gt_cam{_i:02d}_f{frame_idx:06d}.png'))
            utils.save_image(rgb_image, os.path.join(rgb_img_dir, f'rgb_raw_cam{_i:02d}_f{frame_idx:06d}.png'))
        
        if repose_imgs is not None:
            for _i, _v in enumerate(range(cur_masked_images.shape[1])):
                repose_image = repose_imgs[0, _v]
                utils.save_image(repose_image, os.path.join(repose_img_dir, f'gt_cam{_i:02d}_f{frame_idx:06d}.png'))
        

    def on_test_epoch_end(self):
        if self.test_mode == 'cam':
            return
        
        save_dir = os.path.join(self.log_dir, self.test_mode)
        gt_img_dir = os.path.join(save_dir, 'gt')
        rgb_img_dir = os.path.join(save_dir, 'rgb')

        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0

        fid = calculate_fid_given_paths(
            paths=[rgb_img_dir, gt_img_dir],
            batch_size=50,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dims=2048,
            num_workers=num_workers,
        )

        all_psnr = torch.Tensor(self.test_step_outputs['psnr']).mean()
        all_ssim = torch.Tensor(self.test_step_outputs['ssim']).mean()
        all_lpips = torch.Tensor(self.test_step_outputs['lpips']).mean()

        psnr = self.all_gather(all_psnr).mean()
        ssim = self.all_gather(all_ssim).mean()
        lpips = self.all_gather(all_lpips).mean()

        out_str = '\n'
        out_str += ('=============================\n')
        out_str += (f' Mode: {self.test_mode}\n')
        out_str += (f' PSNR: {psnr.detach().cpu().numpy()}\n')
        out_str += (f' SSIM: {ssim.detach().cpu().numpy()}\n')
        out_str += (f'LPIPS: {lpips.detach().cpu().numpy()}\n')
        out_str += (f'  FID: {fid}\n')
        out_str += ('=============================\n')
        out_str += ('\n')

        print(out_str)
        
        with open(os.path.join(save_dir, 'results.txt'), 'a') as f:
            f.write(out_str)

        self.test_step_outputs = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=self.train_lr, betas=self.adam_betas)
        return optimizer
    
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
    
    def get_masked_input_image(self, batch, data='current_data_list'):
        image = torch.cat([i['image'] for i in batch[data]], dim=0).permute([0, 3, 1, 2]).clone()  # [v, 3, image_size, image_size]
        image_mask = torch.cat([i['image_mask'] for i in batch[data]], dim=0)  # [v, image_size, image_size]
        image[image_mask.unsqueeze(1).repeat(1, 3, 1, 1) == 0] = 0
        return image.unsqueeze(0)  # [1, v, 3, image_size, image_size]
    
    def get_diffusion_conds(self, batch, data='current_data_list'):
        R = []
        T = []
        K = []
        image_size = []

        for view_item in batch[data]:
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
            'verts': batch['smpl_vertices'].clone().detach(),  # [1, SMPL_NODE_NUM, 3]
            'verts_cano': batch['minimal_shape'].clone().detach(),  # [1, SMPL_NODE_NUM, 3]
            'verts_T_inv': batch['T_inv'],  # [1, SMPL_NODE_NUM, 4, 4]
            'R': R,  # [v, 3, 3]
            'T': T,  # [v, 3]
            'K': K,  # [v, 3, 3]
            'image_size': image_size,  # [v, 2]
        }
        return inputs
    
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


def save_image(path, image):
    return cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
