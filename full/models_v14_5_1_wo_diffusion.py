import json
import math
import os
import sys
from collections import namedtuple
from functools import partial, wraps
from random import random
from typing import Tuple, Union
from datetime import datetime

import cv2
import pytorch_lightning as pl
import lpips
# import matplotlib.pyplot as plt
import numpy as np
# import pytorch3d
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from packaging import version
from pytorch3d.ops import knn_points
from pytorch3d.renderer import (MeshRasterizer, RasterizationSettings)
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from skimage.metrics import structural_similarity as compute_ssim
from torch import einsum, nn
from torchvision import utils

sys.path.append("..")
from gaussians.gaussian_renderer import render3

# Helper functions
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

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

# Metrics

def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def ssim_metric(img_pred, img_gt, mask_at_box):
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]

    # Compute the SSIM
    ssim = compute_ssim(img_pred, img_gt, multichannel=True, channel_axis=2, data_range=1.0)
    return ssim

def lpips_metric(img_pred, img_gt, mask_at_box, loss_fn_vgg, device):
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]

    # Compute the LPIPS
    img_pred = torch.tensor(img_pred, dtype=torch.float32, device=device).reshape(1, h, w, 3).permute(0, 3, 1, 2)
    img_gt = torch.tensor(img_gt, dtype=torch.float32, device=device).reshape(1, h, w, 3).permute(0, 3, 1, 2)

    score = loss_fn_vgg(img_pred, img_gt, normalize=True)
    return score.item()

# Normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

# Small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class Interpolate(nn.Module):
    def __init__(self, scale_factor=0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    Weight Standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# Sinusoidal positional embeddings

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ Following @crowsonkb's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# Building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # Determine efficient attention configs for CUDA and CPU

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        # if device_properties.major == 8 and device_properties.minor == 0:
        #     print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
        #     self.cuda_config = AttentionConfig(True, False, False)
        # else:
        #     print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
        #     self.cuda_config = AttentionConfig(False, True, True)

        self.cuda_config = AttentionConfig(True, True, True)

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        if mask is not None:
            mask = mask.contiguous()

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # PyTorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.
            )

        return out.contiguous()

    def forward(self, q, k, v, mask=None):
        """
        Einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v, mask)
        
        assert mask is None

        scale = q.shape[-1] ** -0.5

        # Similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # Attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # Aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = True
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# UNet model

class SFTLayer(nn.Module):
    '''https://github.com/xinntao/SFTGAN.git'''
    def __init__(
            self,
            feature_dim,
            cond_dim,
            mid_dim,
    ):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(cond_dim, mid_dim, 1)
        self.SFT_scale_conv1 = nn.Conv2d(mid_dim, feature_dim, 1)
        self.SFT_shift_conv0 = nn.Conv2d(cond_dim, mid_dim, 1)
        self.SFT_shift_conv1 = nn.Conv2d(mid_dim, feature_dim, 1)

    def forward(self, x, cond):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        return x * (scale + 1) + shift

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        pix_cond_dim = None,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        enable_attn=[False, False, False, True],
        enable_mid_attn=True,
    ):
        super().__init__()

        # Determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # Layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = None),
                SFTLayer(dim_in, pix_cond_dim, dim_in),
                block_klass(dim_in, dim_in, time_emb_dim = None),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if enable_attn[ind] else nn.Identity(),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1),
                # Downsample(pix_cond_dim, pix_cond_dim) if not is_last else nn.Identity(),
                Interpolate(scale_factor=0.5) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = None)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim))) if enable_mid_attn else nn.Identity()
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = None)
        enable_attn.reverse()

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = None),
                SFTLayer(dim_out, pix_cond_dim, dim_out),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = None),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if enable_attn[ind] else nn.Identity(),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = None)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, pix_cond):

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, mod1, block2, attn, downsample, downsample_pix_cond in self.downs:
            x = block1(x)
            x = mod1(x, pix_cond)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)

            h.append(pix_cond)

            x = downsample(x)
            pix_cond = downsample_pix_cond(pix_cond)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, mod1, block2, attn, upsample in self.ups:
            pix_cond = h.pop()
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x)
            x = mod1(x, pix_cond)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)
        return self.final_conv(x)
    
# 3D Gaussian Splatting (3DGS)
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_raw(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_raw(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        return self._features
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_raw(self):
        return self._opacity
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_pcd(self, points, colors, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(np.asarray(points))
        if not isinstance(colors, torch.Tensor):
            points = torch.tensor(np.asarray(colors))
        fused_point_cloud = points.float()
        features = colors.float()

        # print("Number of points at initialization: ", fused_point_cloud.shape[0])
        # dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        dist2 = torch.clamp_min(knn_points(fused_point_cloud[None], fused_point_cloud[None], K = 4)[0][0, :, 1:].mean(-1), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features = nn.Parameter(features.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]))


def project_xy(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = torch.matmul(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = torch.matmul(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


# Render SMPL images from different views
def rasterize_smpl(mesh_bar, cam_rot, cam_trans, K, image_size, device):
    """
    Args:
        mesh_bar: Meshes object
        cam_rot: [v, 3, 3]
        cam_trans: [v, 3]
        K: [v, 3, 3]
        image_size: int
        device: torch.device
    Returns:
        rendered_list: list of rendered outputs
    """

    image_size_tensor = torch.tensor([[image_size, image_size]], dtype=torch.float32, device=device)
    raster_settings = RasterizationSettings(
        image_size=(image_size, image_size),
        bin_size=0,
    )
    rendered_list = []
    for v in range(cam_trans.shape[0]):
        cameras = cameras_from_opencv_projection(
            cam_rot[v: v + 1, ...],
            cam_trans[v: v + 1, ...],
            K[v: v + 1, ...],
            image_size=image_size_tensor,
        ).to(device)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        rendered = rasterizer(mesh_bar)
        rendered_list.append(rendered)
    return rendered_list

def gen_seg(face_feat, rendered_list):
    """
    Args:
        face_feat: [SMPL_FACE_NUM, dim]
        rendered_list: [rendered]
    Returns:
        seg: [v, img_size, img_size, dim]
    """
    seg = torch.zeros([len(rendered_list), *rendered_list[0].pix_to_face.shape[1:3], face_feat.shape[1]], dtype=torch.float32, device=face_feat.device)
    
    face_feat_pad = torch.cat([face_feat, torch.zeros_like(face_feat[0:1])], dim=0)

    for i, rendered in enumerate(rendered_list):
        pix2face = rendered.pix_to_face
        seg[i] = face_feat_pad[pix2face.squeeze(), :]

    return seg

class Projector(nn.Module):
    # https://github.com/lucidrains/vit-pytorch
    def __init__(self):
        super().__init__()
    
    def forward_avg_vis_proj(self, verts, faces, cam_rot, cam_trans, K, image_size, feature, rendered_list):
        """
        Args:
            verts: [NODE_NUM, 3]
            faces: [SMPL_FACE_NUM]
            cam_rot: [v, 3, 3]
            cam_trans: [v, 3]
            K: [v, 3, 3]
            image_size: int
            feature: [v, c, image_size, image_size]
            rendered_list: [rendered]
        Returns:
            projected_vert_features: [SMPL_NODE_NUM, c]
        """
        device = verts.device
        verts_mask = []
        sampled_feature = []
        for v in range(feature.shape[0]):
            rendered = rendered_list[v]
            fg_mask = rendered.pix_to_face >= 0
            fg_faces = rendered.pix_to_face[fg_mask]
            fg_v = torch.isin(faces, fg_faces)
            _verts_mask = torch.zeros([verts.shape[0]], device=device, dtype=torch.bool)
            _verts_mask[fg_v] = True  # [NODE_NUM]
            verts_mask.append(_verts_mask.unsqueeze(0))

            v_2d_loc = project_xy(verts, K[v], torch.cat([cam_rot[v], cam_trans[v].unsqueeze(-1)], dim=-1))
            v_2d_uv = (v_2d_loc / (image_size - 1) * 2 - 1)  # [NODE_NUM, 2]
            # NOTE: use grid_sample to fetch features
            _sampled_feature = torch.nn.functional.grid_sample(feature[v: v+1, ...], v_2d_uv[None, None, ...], mode='bilinear', align_corners=True).permute([0, 2, 3, 1]).squeeze()  # [NODE_NUM, c]
            sampled_feature.append(_sampled_feature.unsqueeze(0))

        sampled_feature = torch.cat(sampled_feature, dim=0)  # [v, NODE_NUM, c]
        verts_mask = torch.cat(verts_mask, dim=0)  # [v, NODE_NUM]
        verts_mask = verts_mask.type_as(feature).unsqueeze(-1)
        
        # Average features from all views for invisible points
        verts_mask[(verts_mask.sum(dim=0, keepdim=True) == 0).repeat(verts_mask.shape[0], 1, 1)] = 1.
        projected_vert_features = (sampled_feature * verts_mask).sum(dim=0) / verts_mask.sum(dim=0)  # [NODE_NUM, PFD]

        return projected_vert_features
    
    def forward_cat_vis_proj(self, verts, faces, cam_rot, cam_trans, K, image_size, feature, rendered_list):
        """
        Args:
            verts: [NODE_NUM, 3]
            faces: [SMPL_FACE_NUM]
            cam_rot: [v, 3, 3]
            cam_trans: [v, 3]
            K: [v, 3, 3]
            image_size: int
            feature: [v, c, image_size, image_size]
            rendered_list: [rendered]
        Returns:
            projected_vert_features: [SMPL_NODE_NUM, v * c]
        """
        device = verts.device
        verts_mask = []
        sampled_feature = []
        for v in range(feature.shape[0]):
            rendered = rendered_list[v]
            fg_mask = rendered.pix_to_face >= 0
            fg_faces = rendered.pix_to_face[fg_mask]
            fg_v = torch.isin(faces, fg_faces)
            _verts_mask = torch.zeros([verts.shape[0]], device=device, dtype=torch.bool)
            _verts_mask[fg_v] = True  # [NODE_NUM]
            verts_mask.append(_verts_mask.unsqueeze(0))

            v_2d_loc = project_xy(verts, K[v], torch.cat([cam_rot[v], cam_trans[v].unsqueeze(-1)], dim=-1))
            v_2d_uv = (v_2d_loc / (image_size - 1) * 2 - 1)  # [NODE_NUM, 2]
            
            _sampled_feature = torch.nn.functional.grid_sample(feature[v: v+1, ...], v_2d_uv[None, None, ...], mode='bilinear', align_corners=True).permute([0, 2, 3, 1]).squeeze()  # [NODE_NUM, c]
            sampled_feature.append(_sampled_feature.unsqueeze(0))

        sampled_feature = torch.cat(sampled_feature, dim=0)  # [v, NODE_NUM, c]
        verts_mask = torch.cat(verts_mask, dim=0)  # [v, NODE_NUM]
        sampled_feature[~verts_mask] = 0

        projected_vert_features = rearrange(sampled_feature, 'v n c -> n (v c)') # [NODE_NUM, v * c]

        return projected_vert_features
    
    def forward(self, mode='avg_vis', *args, **kwargs):
        if mode == 'avg_vis':
            return self.forward_avg_vis_proj(*args, **kwargs)
        if mode == 'cat_vis':
            return self.forward_cat_vis_proj(*args, **kwargs)
        else:
            raise NotImplementedError

# PyTorch3D: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/sample_points_from_meshes.html
def sample_points_from_meshes(
    meshes,
    smpl_sample_dict,
    return_normals: bool = False,
    return_textures: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    faces = meshes.faces_packed()
    num_meshes = len(meshes)

    sample_face_idxs = smpl_sample_dict['sample_face_idxs']
    num_samples = sample_face_idxs.shape[-1]

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Barycentric coords. These are precomputed and directly used?
    w0, w1, w2 = smpl_sample_dict['w0'], smpl_sample_dict['w1'], smpl_sample_dict['w2']

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]

    # Calculate the barycentric coordinates for each triangle as the initial observation space coordinates of all Gaussian points in the current frame
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    # If returning normals
    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[meshes.valid] = vert_normals

    # If returning textures
    if return_textures:
        # Fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
        pix_to_face = sample_face_idxs.view(len(meshes), num_samples, 1, 1)  # NxSx1x1
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # NxSx1x1x3
        # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
        dummy = torch.zeros(
            (len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32
        )  # NxSx1x1
        fragments = MeshFragments(
            pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy
        )
        textures = meshes.sample_textures(fragments)  # NxSx1x1xC
        textures = textures[:, :, 0, 0, :]  # NxSxC

    # Return based on requested outputs
    if return_normals and return_textures:
        return samples, normals, textures
    if return_normals:  # return_textures is False
        return samples, normals
    if return_textures:  # return_normals is False
        return samples, textures
    return samples

def local_coord_to_positions(
    meshes,
    smpl_sample_dict,
    local_coords,
):
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = meshes.faces_packed()
    num_meshes = len(meshes)

    sample_face_idxs = smpl_sample_dict['sample_face_idxs']
    num_samples = sample_face_idxs.shape[-1] # Number of Gaussian points sampled on each mesh?

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]

    # Calculate normals
    normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
    vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
    vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
        min=sys.float_info.epsilon
    )
    vert_normals = vert_normals[sample_face_idxs]
    normals[meshes.valid] = vert_normals

    positions = torch.einsum('b n l, b n d l -> b n d', local_coords, torch.cat([a[..., None], b[..., None], c[..., None], normals[..., None]], dim=-1))

    return positions

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
    # bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float16).to(device)
    v_num = K.shape[0]
    ret = []
    for v in range(v_num):
        extr = torch.eye(4, dtype=R.dtype, device=device)
        extr[:3, :3] = R[v]
        extr[:3, 3] = T[v]
        intr = K[v]
        img_w = image_size[v][1]
        img_h = image_size[v][0]

        # with torch.cuda.amp.autocast(enabled=False):
        # gaussian_vals['positions'] = gaussian_vals['positions'].to(torch.float32)
        render_ret = render3(
            gaussian_vals,
            bg_color,
            extr,
            intr,
            img_w,
            img_h,
        )

        # render_ret['render'] = render_ret['render'].to(torch.float16)
        # render_ret['mask'] = render_ret['mask'].to(torch.float16)

        # render_ret
        ret.append(render_ret)
    
    render_out = {}
    
    for key in render_ret.keys():
        vals = torch.stack([i[key] for i in ret], dim=0)
        render_out[key] = vals

    return render_out

# 3D Denoiser
class GauModel(nn.Module):
    def __init__(self, channels=3, smpl_type='SMPL', multi_view=4, multires=5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.self_condition = False  # Self-conditioning not supported yet.
        
        if smpl_type == 'SMPL':
            smpl_node_num = 6890
            faces = np.load('/home/xxx/GaussianActor/body_models/misc/faces.npz')['faces']  # [13776, 3]
            self.register_buffer('smpl_faces_seg', torch.load('/home/xxx/GaussianActor/body_models/misc/faces_seg.pt'))
            smpl_part_num = 24
            # 373056 points
            self.register_buffer('cano_samples', torch.load('/home/xxx/GaussianActor/body_models/misc/cano_samples.pt').squeeze(0))  # [373056, 3]
            # What is the sample_dict used for here?
            smpl_sample_dict = torch.load('/home/xxx/GaussianActor/body_models/misc/smpl_sample_dict.pt')
            self.smpl_sample_dict_keys = smpl_sample_dict.keys()
            for key, val in smpl_sample_dict.items():
                self.register_buffer(f'smpl_sample_dict_{key}', val)
            

        # TODO: check name
        elif smpl_type == 'SMPLX':
            # raise NotImplementedError
            smpl_node_num = 10475
            faces = np.load('/home/xxx/GaussianActor/body_models/smplx/SMPLX_NEUTRAL.npz')['f']  # [20908, 3]
            self.register_buffer('smpl_faces_seg', torch.load('/home/xxx/GaussianActor/body_models/misc/faces_seg_x.pt'))
            smpl_part_num = 27
            # 373056 points
            self.register_buffer('cano_samples', torch.load('/home/xxx/GaussianActor/body_models/misc/cano_samples_smplx.pt').squeeze(0))  # [373056, 3]
            smpl_sample_dict = torch.load('/home/xxx/GaussianActor/body_models/misc/smplx_sample_dict.pt')
            self.smpl_sample_dict_keys = smpl_sample_dict.keys()
            for key, val in smpl_sample_dict.items():
                self.register_buffer(f'smpl_sample_dict_{key}', val)

        self.register_buffer('smpl_faces', torch.from_numpy(faces.astype(np.int64)))

        self.pixel_feature_dim = 32

        self.extract_net = Unet(
            dim=16,
            init_dim=None,
            out_dim=32,  # DIM_1
            dim_mults=(1, 2, 4, 8),
            channels=self.channels,
            pix_cond_dim=smpl_part_num,
            self_condition=self.self_condition,
            enable_attn=[False, False, False, False],
            enable_mid_attn=True,
        )

        self.projector = Projector()

        self.max_sh_degree = 0
        self.cano_gaussian_model = GaussianModel(sh_degree = self.max_sh_degree)

        # Initialize all Gaussian point attributes here, ignoring position attributes; positions will be re-initialized from barycentric coordinates later, and other attributes initialized to their respective values
        self.cano_gaussian_model.create_from_pcd(self.cano_samples, torch.rand_like(self.cano_samples), spatial_lr_scale = 2.5)

        self.embed_fn = None
        input_ch = 3
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn

        self.offset_net = nn.Sequential(
            nn.Linear(input_ch + 16 * multi_view, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self.extra_net = nn.Sequential(
            nn.Linear(input_ch + 16, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1 + 3 + 4 + 3),
        )

    
    def forward(
            self,
            noised_image,
            data,
            render,
            bg_color=(0., 0., 0.),
    ):
        """
        Args:
            noised_image: [1, v, 3, image_size, image_size]
            data: dict
            render: dict
        Returns:
            ret
        """
        if noised_image is not None and noised_image.shape[0] != 1:
            raise NotImplementedError("Batch size > 1 is not supported yet.")
        
        verts = data['verts']
        # verts_cano = data.get('verts_cano')  # [1, SMPL_NODE_NUM, 3]
        # verts_T_inv = data.get('verts_T_inv')  # [1, SMPL_NODE_NUM, 4, 4]
        
        device = verts.device

        ret = {}

        gaussian_vals = data.get('gaussian_vals')
        local_coords = data.get('local_coords')
        
        # During video generation, gaussian_vals is not None, so pass
        if gaussian_vals is None:

            R = data.get('R')  # [v, 3, 3]
            T = data.get('T')  # [v, 3]
            K = data.get('K')  # [v, 3, 3]

            smpl_sample_dict = {key: getattr(self, f'smpl_sample_dict_{key}') for key in self.smpl_sample_dict_keys}

            mesh_posed = Meshes(verts=verts, faces=self.smpl_faces.unsqueeze(0))
            
            # Sample the initial observation space coordinates of all Gaussian points from the current frame's SMPL model (using precomputed barycentric coordinates)
            posed = sample_points_from_meshes(mesh_posed, smpl_sample_dict).squeeze(0)  # [GAU_NUM, 3]

            extract_input_image = noised_image[0]  # [v, 3, image_size, image_size]

            # TODO: check
            # The SMPLX of a certain frame is definitely posed
            # with torch.cuda.amp.autocast(enabled=False):
            rendered = rasterize_smpl(
                mesh_bar=mesh_posed,
                cam_rot=R, 
                cam_trans=T, 
                K=K, 
                image_size=noised_image.shape[-1],
                device=device,
            )

            seg_cond = gen_seg(self.smpl_faces_seg, rendered)  # [v, image_size, image_size, seg_cond_dim]

            # seg_cond = seg_cond.to(torch.float16)
            
            # Pass through UNet to get feature maps
            pixel_features = self.extract_net(extract_input_image, seg_cond.permute(0, 3, 1, 2).contiguous())

            # Query features for each Gaussian point
            projected_offset_features = self.projector(
                mode='cat_vis',
                verts=posed,
                faces=smpl_sample_dict['sample_face_idxs'].squeeze(0), 
                cam_rot=R, 
                cam_trans=T, 
                K=K, 
                image_size=noised_image.shape[-1],
                feature=pixel_features[:, :16, ...], 
                rendered_list=rendered,
            )  # [GAU_NUM, v * PFD]

            cano_samples = self.cano_samples.detach()
            
            if self.embed_fn is not None:
                cano_samples = self.embed_fn(cano_samples)

            # Pass through a lightweight MLP to compute local coordinates (including offset)
            offset_raw = self.offset_net(torch.cat([cano_samples, projected_offset_features], dim=-1))
            
            # All Gaussian local coordinates [1, num_gs, 4], each local coordinate [w0, w1, w2, m]
            local_coords = torch.zeros([1, offset_raw.shape[0], 4], dtype=offset_raw.dtype, device=offset_raw.device)
            local_coords[0, :, :2] = offset_raw[:, :2]
            local_coords[0, :, 2] = 1. - local_coords[0, :, 0] - local_coords[0, :, 1]
            local_coords[0, :, 3] = offset_raw[:, 2]
            
            # Transform local coordinates to observation space coordinates and compute the final position of Gaussians in the current frame
            positions = local_coord_to_positions(mesh_posed, smpl_sample_dict, local_coords).squeeze(0)

            # Additional outputs
            offset = positions - posed

            # Project the offset Gaussians to feature maps to get offset features for other Gaussian attributes
            projected_extra_features = self.projector(
                mode='avg_vis',
                verts=positions, 
                faces=smpl_sample_dict['sample_face_idxs'].squeeze(0), 
                cam_rot=R, 
                cam_trans=T, 
                K=K, 
                image_size=noised_image.shape[-1], 
                feature=pixel_features[:, 16:, ...], 
                rendered_list=rendered,
            )  # [GAU_NUM, PFD]
            
            # Pass through a lightweight MLP to compute attribute offsets for each Gaussian (including offset)
            extra_info = self.extra_net(torch.cat([cano_samples, projected_extra_features], dim=-1))

            # Add the offsets to the initialized values to get the final attribute values
            opacity, scales, rotations, colors = torch.split(extra_info, [1, 3, 4, 3], 1)
            opacity = self.cano_gaussian_model.opacity_activation(opacity + self.cano_gaussian_model.get_opacity_raw)
            scales = self.cano_gaussian_model.scaling_activation(scales + self.cano_gaussian_model.get_scaling_raw)
            rotations = self.cano_gaussian_model.rotation_activation(rotations + self.cano_gaussian_model.get_rotation_raw)
            colors = colors + self.cano_gaussian_model.get_features

            gaussian_vals = {
                'positions': positions,
                'opacity': opacity,
                'scales': scales,
                'rotations': rotations,
                'colors': colors,
                'max_sh_degree': self.max_sh_degree
            }

            ret['gaussian_vals'] = gaussian_vals
            ret['offset'] = offset
            ret['local_coords'] = local_coords
        
        # During video generation, do this
        elif local_coords is not None:
            # Repose
            # last_positions = gaussian_vals['positions']
            smpl_sample_dict = {key: getattr(self, f'smpl_sample_dict_{key}') for key in self.smpl_sample_dict_keys}
            mesh_posed = Meshes(verts=verts, faces=self.smpl_faces.unsqueeze(0))
            new_positions = local_coord_to_positions(mesh_posed, smpl_sample_dict, local_coords)
            gaussian_vals['positions'] = new_positions[0]

        # Both do this
        if render is not None:
            split = render.get('split', -1)
            R = render.get('R')  # [v, 3, 3]
            T = render.get('T')  # [v, 3]
            K = render.get('K')  # [v, 3, 3]
            image_size = render.get('image_size')  # [v, 2]
            if split == -1:

                # Render Gaussian points to generate multi-view images for the current frame based on the provided render parameters
                ret.update(render_gaussians(
                    gaussian_vals=gaussian_vals,
                    R=R,
                    T=T,
                    K=K,
                    image_size=image_size,
                    bg_color=bg_color,
                ))
            else:
                gau_rets = []
                for _R, _T, _K, _image_size in zip(
                    torch.split(R, split, dim=0),
                    torch.split(T, split, dim=0),
                    torch.split(K, split, dim=0),
                    torch.split(image_size, split, dim=0),
                ):
                    gau_ret = render_gaussians(
                        gaussian_vals=gaussian_vals,
                        R=_R,
                        T=_T,
                        K=_K,
                        image_size=_image_size,
                        bg_color=bg_color,
                    )
                    gau_rets.append(gau_ret)
                
                ret_gau = {}

                for k in gau_ret.keys():
                    ret_gau[k] = torch.cat([gau[k] for gau in gau_rets], dim=0)
                ret.update(ret_gau)

        
        return ret

# Lightning Module
class Lightning(pl.LightningModule):
    def __init__(
        self,
        smpl_type = 'SMPL',
        train_lr = 1e-4,
        adam_betas = (0.9, 0.99),
        image_size = 256,
        timesteps = 1000,
        sampling_timesteps = 250,
        test_mode = None,
        log_dir = None,
        reconstruction_timestep = 0,
        test_recur_timestep = 200,
        rgb_loss_weight = 1.,
        mask_loss_weight = 1.,
        log_val_img = True,
        use_ema = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = GauModel(smpl_type=smpl_type)

        # Sampling and training hyperparameters
        self.image_size = image_size
        self.train_lr = train_lr
        self.adam_betas = adam_betas

        self.validation_step_outputs = {
            'generated': [],
            'gt': [],
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'psnr_all_true': [],
            'ssim_all_true': [],
            'lpips_all_true': [],
        }

        self.test_step_outputs = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'psnr_all_true': [],
            'ssim_all_true': [],
            'lpips_all_true': [],
        }

        # LPIPS
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')

        self.test_mode = test_mode  # None, 'nv', 'np'
        self.log_dir = log_dir
        self.reconstruction_timestep = reconstruction_timestep
        self.test_recur_timestep = test_recur_timestep

        self.rgb_loss_weight = rgb_loss_weight
        self.mask_loss_weight = mask_loss_weight

        self.log_val_img = log_val_img

        self.test_last_add_info = None

    ########################################################################################
    def training_step(self, batch, batch_idx):
        image = self.get_masked_input_image(batch)  # current_data_list, [1, n_view, 3, image_size, image_size]
        cond = self.get_diffusion_conds(batch)      # cond are the camera parameters for training views
        render_info = self.get_render_info(batch)   # render_info contains the camera parameters needed for rendering
        # The 3D denoiser is not actually a denoiser; it simply reconstructs the Gaussians for the current frame using several anchor clean frames
        model_out = self.model(image, data=cond, render=render_info)
        # Calculate loss
        loss = self.calc_loss(batch, model_out)
        torch.cuda.empty_cache()
        return loss
    

    def calc_loss(self, batch, model_out):
        render_image = model_out['render']  # [12, 3, h, w]
        render_mask = model_out['mask']  # [12, 1, h, w]

        # view_data_list is the list of views needed to compute the loss
        gt_image = self.get_masked_input_image(batch, data='view_data_list').squeeze(0)  # [v, 3, h, w]
        gt_mask_erode = torch.stack([i['image_mask_erode'] for i in batch['view_data_list']], dim=0)  # [v, 1, h, w], {0, 1, 100}
       
        # NOTE: Background is False, 0
        # gt_mask_erode = gt_mask_erode.float()
        mask_valid = (gt_mask_erode==0.) | (gt_mask_erode==1.)

        # RGB loss
        rgb_loss = F.mse_loss(render_image, gt_image)
        self.log('rgb_loss', rgb_loss, prog_bar=True)

        # Mask loss
        mask_loss = F.mse_loss(render_mask[mask_valid], (gt_mask_erode != 0).float()[mask_valid])
        self.log('mask_loss', mask_loss, prog_bar=True)

        # Total loss
        loss = rgb_loss * self.rgb_loss_weight + mask_loss * self.mask_loss_weight
        
        self.log('loss', loss, prog_bar=False)

        return loss


    def validation_step(self, batch, batch_idx):
        # Get input and conditions
        input_masked_images = self.get_masked_input_image(batch)  # [1, n_view, 3, image_size, image_size]
        images = self.get_masked_input_image(batch, data='view_data_list')
        cond = self.get_diffusion_conds(batch)
        render_info = self.get_render_info(batch)
        # Do not use rays_o/rays_d, only use mask
        val_rays = self.compose_rays_inputs(batch['view_data_list'], packed=True)

        # Call GauModel to generate results
        render_out = self.model(input_masked_images, data=cond, render=render_info)
        val_out = render_out['render'].unsqueeze(0)  # [1, v, 3, image_size, image_size]
        
        # Store in validation outputs
        self.validation_step_outputs['generated'].append(val_out)
        self.validation_step_outputs['gt'].append(images)

        # result_dict['generated'].append(val_out)
        # result_dict['gt'].append(images)
        # result_dict = self.validation_step_outputs

        pred_img = val_out.permute(0, 1, 3, 4, 2).squeeze().detach().cpu().numpy() # [v, 512, 512, 3]
        gt_img = images.permute(0, 1, 3, 4, 2).squeeze().detach().cpu().numpy()    # [v, 512, 512, 3]
        bbox_mask = val_rays['sample_image_mask'].detach().cpu().numpy()           # [v, 512, 512]
        bbox_mask_all_true = np.ones(bbox_mask.shape, dtype=bool)                  # [v, 512, 512]
        
        # TODO: Consider removing bbox or using another computation method
        for v_idx in range(images.shape[1]):
            _pred_img = pred_img[v_idx]
            _gt_img = gt_img[v_idx]
            _bbox_mask = bbox_mask[v_idx]
            _psnr = psnr_metric(_pred_img[_bbox_mask], _gt_img[_bbox_mask])
            _ssim = ssim_metric(_pred_img, _gt_img, _bbox_mask)
            _lpips = lpips_metric(_pred_img, _gt_img, _bbox_mask, self.loss_fn_vgg, self.device)
            # Compute with all-true mask
            _bbox_mask_all_true = bbox_mask_all_true[v_idx]
            _psnr_all_true = psnr_metric(_pred_img[_bbox_mask_all_true], _gt_img[_bbox_mask_all_true])
            _ssim_all_true = ssim_metric(_pred_img, _gt_img, _bbox_mask_all_true)
            _lpips_all_true = lpips_metric(_pred_img, _gt_img, _bbox_mask_all_true, self.loss_fn_vgg, self.device)
            # result_dict['psnr'].append(_psnr)
            # result_dict['ssim'].append(_ssim)
            # result_dict['lpips'].append(_lpips)
            self.validation_step_outputs['psnr'].append(_psnr)
            self.validation_step_outputs['ssim'].append(_ssim)
            self.validation_step_outputs['lpips'].append(_lpips)
            self.validation_step_outputs['psnr_all_true'].append(_psnr_all_true)
            self.validation_step_outputs['ssim_all_true'].append(_ssim_all_true)
            self.validation_step_outputs['lpips_all_true'].append(_lpips_all_true)


    def on_validation_epoch_end(self):
        # result_dict = self.validation_step_outputs
        if self.log_val_img:
            # Log predicted images
            all_preds = torch.cat(self.validation_step_outputs['generated'], dim=0)
            validation_outputs = self.all_gather(all_preds)
            validation_outputs = validation_outputs.reshape(-1, *validation_outputs.shape[-3:])
            grid = utils.make_grid(validation_outputs)
            self.logger.experiment.add_image(f"generated_images", grid, self.global_step)
            self.validation_step_outputs['generated'].clear()

            # Log ground truth images
            all_gt = torch.cat(self.validation_step_outputs['gt'], dim=0)
            gts = self.all_gather(all_gt)
            gts = gts.reshape(-1, *gts.shape[-3:])
            grid = utils.make_grid(gts)
            self.logger.experiment.add_image(f"gt_images", grid, self.global_step)
            self.validation_step_outputs['gt'].clear()

        all_psnr = torch.Tensor(self.validation_step_outputs['psnr']).mean()
        all_ssim = torch.Tensor(self.validation_step_outputs['ssim']).mean()
        all_lpips = torch.Tensor(self.validation_step_outputs['lpips']).mean()
        all_psnr_all_true = torch.Tensor(self.validation_step_outputs['psnr_all_true']).mean()
        all_ssim_all_true = torch.Tensor(self.validation_step_outputs['ssim_all_true']).mean()
        all_lpips_all_true = torch.Tensor(self.validation_step_outputs['lpips_all_true']).mean()

        psnr = self.all_gather(all_psnr).mean()
        ssim = self.all_gather(all_ssim).mean()
        lpips = self.all_gather(all_lpips).mean()
        psnr_all_true = self.all_gather(all_psnr_all_true).mean()
        ssim_all_true = self.all_gather(all_ssim_all_true).mean()
        lpips_all_true = self.all_gather(all_lpips_all_true).mean()
        
        self.log('psnr', psnr, rank_zero_only=True, sync_dist=True)
        self.log('ssim', ssim, rank_zero_only=True, sync_dist=True)
        self.log('lpips', lpips, rank_zero_only=True, sync_dist=True)
        self.log('psnr_all_true', psnr_all_true, rank_zero_only=True, sync_dist=True)
        self.log('ssim_all_true', ssim_all_true, rank_zero_only=True, sync_dist=True)
        self.log('lpips_all_true', lpips_all_true, rank_zero_only=True, sync_dist=True)

        self.validation_step_outputs = {
            'generated': [],
            'gt': [],
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'psnr_all_true': [],
            'ssim_all_true': [],
            'lpips_all_true': [],
        }


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.train_lr, betas = self.adam_betas)
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
    
    
    # NOTE: Modified
    def get_masked_input_image(self, batch, data='current_data_list'):
        image = torch.cat([i['image'] for i in batch[data]], dim=0).permute([0, 3, 1, 2]).clone()  # [v, 3, image_size, image_size]
        image_mask = torch.cat([i['image_mask'] for i in batch[data]], dim=0)  # [v, image_size, image_size]
        image[image_mask.unsqueeze(1).repeat(1, 3, 1, 1) == 0] = 0
        # image[~image_mask.unsqueeze(1).repeat(1, 3, 1, 1)] = 0
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
            # SMPL-related parameters are always batch_size=1 regardless of the number of views used
            'verts': batch['smpl_vertices'].clone().detach(),  # [1, SMPL_NODE_NUM, 3] SMPL model vertices, used here
            # 'verts_cano': batch['minimal_shape'].clone().detach(),  # [1, SMPL_NODE_NUM, 3] Not used
            # 'verts_T_inv': batch['T_inv'],  # [1, SMPL_NODE_NUM, 4, 4] Not used
            # Each view has its own set of RTK parameters, so batch_size=v
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
