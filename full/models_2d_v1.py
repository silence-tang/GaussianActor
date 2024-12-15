import json
import math
import os
from collections import namedtuple
from functools import partial, wraps
from random import random

import cv2
import pytorch_lightning as pl
import lpips
import matplotlib.pyplot as plt
import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from packaging import version
from pytorch3d.renderer import (HardPhongShader, MeshRasterizer, MeshRenderer, PointLights, RasterizationSettings)
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from skimage.metrics import structural_similarity as compute_ssim
from torch import einsum, nn
from torchvision import utils

# helpers functions

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

# small helper modules

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
    weight standardization purportedly works synergistically with group normalization
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

# sinusoidal positional embeds

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
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
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

# building block modules

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
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

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

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.
            )

        return out.contiguous()

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
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

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

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

# sft model
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

# unet model
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
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        enable_attn=[False, False, False, True],
        enable_mid_attn=True,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1) # 3
        init_dim = default(init_dim, dim) # init_dim=None -> init_dim=8

        # in layers
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        # time embedder
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # inti layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        #  down blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                SFTLayer(dim_in, pix_cond_dim, dim_in),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if enable_attn[ind] else nn.Identity(),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1),
                # Downsample(pix_cond_dim, pix_cond_dim) if not is_last else nn.Identity(),
                Interpolate(scale_factor=0.5) if not is_last else nn.Identity(),
            ]))

        # mid blocks
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim))) if enable_mid_attn else nn.Identity()
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        enable_attn.reverse()

        # up blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                SFTLayer(dim_out, pix_cond_dim, dim_out),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if enable_attn[ind] else nn.Identity(),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # out layers
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, pix_cond, x_self_cond = None):
        # pass
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, mod1, block2, attn, downsample, downsample_pix_cond in self.downs:
            x = block1(x, t)
            x = mod1(x, pix_cond)  # seg map参与sft layer的计算
            h.append(x)

            x = block2(x, t)
            x = attn(x)  # attn or identity
            h.append(x)

            h.append(pix_cond)

            x = downsample(x)
            pix_cond = downsample_pix_cond(pix_cond)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) # attn or identity
        x = self.mid_block2(x, t)

        for block1, mod1, block2, attn, upsample in self.ups:
            pix_cond = h.pop()
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = mod1(x, pix_cond)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# render
# https://github.com/zju3dv/neuralbody.git

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists,
         torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)],
        -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

class Renderer:
    def __init__(self, net):
        self.net = net

        self.perturb = 1
        self.N_samples = 64
        self.raw_noise_std = 0
        self.white_bkgd = False

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if self.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        # used for feature interpolation
        sp_input['bounds'] = batch['bounds']
        sp_input['verts_input_feature'] = batch['cond']['verts_input_feature']  # [SMPL_NODE_NUM, dim]

        return sp_input

    def get_density_color(self, wpts, viewdir, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        raw = raw_decoder(wpts, viewdir)
        return raw

    def get_pixel_value(self, ray_o, ray_d, near, far, feature_volume,
                        sp_input, batch):
        # sampling points along camera rays
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)

        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        raw_decoder = lambda x_point, viewdir_val: self.net.calculate_density_color(
            x_point, viewdir_val, feature_volume, sp_input)

        # compute the color and density
        wpts_raw = self.get_density_color(wpts, viewdir, raw_decoder)

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        raw = wpts_raw.reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, self.raw_noise_std, self.white_bkgd)

        ret = {
            'rgb_map': rgb_map.view(n_batch, n_pixel, -1),
            'disp_map': disp_map.view(n_batch, n_pixel),
            'acc_map': acc_map.view(n_batch, n_pixel),
            'weights': weights.view(n_batch, n_pixel, -1),
            'depth_map': depth_map.view(n_batch, n_pixel)
        }

        return ret

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        # encode neural body
        sp_input = self.prepare_sp_input(batch)
        feature_volume = self.net.encode_sparse_voxels(sp_input)

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               feature_volume, sp_input, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret

# query

class SparseConvNet(nn.Module):
    def __init__(self, feature_dim):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(feature_dim, feature_dim, 'subm0')
        self.down0 = stride_conv(feature_dim, 64, 'down0')

        self.conv1 = double_conv(64, 64, 'subm1')
        self.down1 = stride_conv(64, 128, 'down1')

        self.conv2 = triple_conv(128, 128, 'subm2')
        self.down2 = stride_conv(128, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes

def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())


# utils functions

def render_smpl_mesh(verts, faces, cam_rot, cam_trans, K, image_size, smpl_type='SMPL'):
    """
    Args:
        verts: [1, SMPL_NODE_NUM, 3]
        faces: [SMPL_FACE_NUM, 3]
        cam_rot: [v, 3, 3]
        cam_trans: [v, 3]
        K: [v, 3, 3]

    Returns:
        images: [v, 3, img_size, img_size]
    """

    if smpl_type == 'SMPL':
        seg_json_path = 'body_models/misc/smpl_vert_segmentation.json'

    elif smpl_type == 'SMPLX':
        seg_json_path = 'body_models/misc/smplx_vert_segmentation.json'
    device = verts.device

    with open(seg_json_path) as f:
        seg_json = json.load(f)

    parts_num = len(seg_json)  # 24
    cmap = plt.colormaps['Spectral']
    parts_color = cmap(np.linspace(0., 1., parts_num))[..., :3]  # [parts_num, 3]
    parts_color = torch.tensor(parts_color, dtype=verts.dtype, device=device)

    verts_rgb = torch.zeros_like(verts)  # [1, SMPL_NODE_NUM, 3]
    
    for i, v_list in enumerate(list(seg_json.values())):
        verts_rgb[0, v_list, :] = parts_color[i]
    
    textures = Textures(verts_rgb=verts_rgb.to(device))
    mesh_bar = Meshes(verts=verts, faces=faces.unsqueeze(0), textures=textures)
    image_size_tensor = torch.tensor([[image_size, image_size]], dtype=torch.float32, device=device)
    raster_settings = RasterizationSettings(image_size=(image_size, image_size))

    images = []
    for v in range(cam_trans.shape[0]):
        cameras = cameras_from_opencv_projection(
            cam_rot[v: v + 1, ...],
            cam_trans[v: v + 1, ...],
            K[v: v + 1, ...],
            image_size=image_size_tensor,
        ).to(device)

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(
                device=device, 
                cameras=cameras,
                lights=PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            )
        )
        image = renderer(mesh_bar)
        images.append(image[..., :3])

    images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
    return images


def rasterize_smpl(verts, faces, cam_rot, cam_trans, K, image_size):
    """
    Args:
        verts: [1, SMPL_NODE_NUM, 3]
        faces: [SMPL_FACE_NUM, 3]
        cam_rot: [v, 3, 3]
        cam_trans: [v, 3]
        K: [v, 3, 3]
    Returns:
        [rendered]
    """

    # verts = verts.to(torch.float32)
    # faces = faces.to(torch.float32)
    # cam_rot = cam_rot.to(torch.float32)
    # cam_trans = cam_trans.to(torch.float32)
    # K = K.to(torch.float32)
    # image_size = image_size

    device = verts.device
    mesh_bar = Meshes(verts=verts, faces=faces.unsqueeze(0))
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
        # MeshRasterizer from pytorch3d
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
        seg: [v, img_size, img_size, dim], [v, H, W, C]
    """
    seg = torch.zeros([len(rendered_list), *rendered_list[0].pix_to_face.shape[1:3], face_feat.shape[1]], dtype=torch.float32, device=face_feat.device)
    
    face_feat_pad = torch.cat([face_feat, torch.zeros_like(face_feat[0:1])], dim=0)

    for i, rendered in enumerate(rendered_list):
        pix2face = rendered.pix_to_face
        seg[i] = face_feat_pad[pix2face.squeeze(), :]

    return seg


# 2d denoiser
class Denoiser(nn.Module):
    def __init__(self, channels=3, smpl_type='SMPL', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.self_condition = False  # self condition not supported yet.
        
        if smpl_type == 'SMPL':
            smpl_node_num = 6890
            faces = np.load('/home/xxx/GaussianActor/body_models/misc/faces.npz')['faces']  # [13776, 3]
            self.register_buffer('smpl_faces_seg', torch.load('/home/xxx/GaussianActor/body_models/misc/faces_seg.pt'))
            smpl_part_num = 24
        
        elif smpl_type == 'SMPLX':
            smpl_node_num = 10475
            faces = np.load('/home/xxx/GaussianActor/body_models/smplx/SMPLX_NEUTRAL.npz')['f']  # [20908, 3]
            self.register_buffer('smpl_faces_seg', torch.load('/home/xxx/GaussianActor/body_models/misc/faces_seg_x.pt'))
            smpl_part_num = 27
        
        self.register_buffer('smpl_faces', torch.from_numpy(faces.astype(np.int64)))

        self.extract_net = Unet(
            dim=8,
            init_dim=None,
            out_dim=3,
            dim_mults=(1, 2, 4, 8),
            channels=self.channels,
            pix_cond_dim=smpl_part_num,
            self_condition=self.self_condition,
            enable_attn=[False, False, False, False],
            enable_mid_attn=True,
        )

        self.random_or_learned_sinusoidal_cond = self.extract_net.random_or_learned_sinusoidal_cond
    
    def forward(
            self,
            noised_image,  # [1, 3, image_size, image_size]
            time,  # [1,]
            data,
            x_self_cond=None
    ):
        """
        Args:
            noised_image: [1, v, 3, image_size, image_size]
            time: [1,]
            data: dict
            x_self_cond: [1, v, 3, image_size, image_size]
        Returns:
            render_out: dict
                rgb_map: [num_views, num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
                disp_map: [num_views, num_rays]. Disparity map. 1 / depth.
                acc_map: [num_views, num_rays]. Accumulated opacity along each ray. Comes from fine model.
                weights
                depth_map
                verts_input_feature
        """
        if noised_image is not None and noised_image.shape[0] != 1:
            raise NotImplementedError("Batch size > 1 is not supported yet.")
        
        # unpack data
        data_cond = data.get('cond')

        verts = data_cond.get('verts')  # [1, SMPL_NODE_NUM, 3]
        # verts_cano = data_cond.get('verts_cano')  # [1, SMPL_NODE_NUM, 3]
        # verts_T_inv = data_cond.get('verts_T_inv')  # [1, SMPL_NODE_NUM, 4, 4]
        
        R = data_cond.get('R')  # [v, 3, 3]
        T = data_cond.get('T')  # [v, 3]
        K = data_cond.get('K')  # [v, 3, 3]

        split = data_cond.get('split', -1)

        extract_input_image = noised_image[0]  # [v, 3, image_size, image_size]
        extract_input_time = time.repeat(extract_input_image.shape[0])  # [v,]

        if split == -1:

            # with torch.cuda.amp.autocast(enabled=False):
            rendered = rasterize_smpl(
                verts=verts,
                faces=self.smpl_faces,
                cam_rot=R,
                cam_trans=T,
                K=K,
                image_size=noised_image.shape[-1],
            )

            seg_cond = gen_seg(self.smpl_faces_seg, rendered)  # ok, [v, image_size, image_size, seg_cond_dim]
            
            # seg_cond = seg_cond.to(torch.float16)
            # NOTE: real unet, predict x_0
            # input: [v, 3, image_size, image_size]
            pixel_features = self.extract_net(extract_input_image, extract_input_time, seg_cond.permute(0, 3, 1, 2).contiguous())

        # pass
        else:
            pixel_features = []
            
            for _R, _T, _K, _image, _time in zip(
                torch.split(R, split, dim=0),
                torch.split(T, split, dim=0),
                torch.split(K, split, dim=0),
                torch.split(extract_input_image, split, dim=0),
                torch.split(extract_input_time, split, dim=0),
            ):
                rendered = rasterize_smpl(
                    verts=verts, 
                    faces=self.smpl_faces, 
                    cam_rot=_R, 
                    cam_trans=_T, 
                    K=_K, 
                    image_size=_image.shape[-1], 
                )
                seg_cond = gen_seg(self.smpl_faces_seg, rendered)  # [v, image_size, image_size, seg_cond_dim]
                _features = self.extract_net(_image, _time, seg_cond.permute(0, 3, 1, 2).contiguous())
                pixel_features.append(_features)

            pixel_features = torch.cat(pixel_features, dim=0)
        
        render_out = {}
        render_out['out'] = pixel_features

        return render_out
    
# gaussian diffusion

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
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
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_x0',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        # register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float16))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)

        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)

        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    # def predict_start_from_noise(self, x_t, t, noise):
    #     return (
    #         extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
    #         extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    #     )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # def predict_v(self, x_start, t, noise):
    #     return (
    #         extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
    #         extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
    #     )

    # def predict_start_from_v(self, x_t, t, v):
    #     return (
    #         extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
    #         extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
    #     )

    # def q_posterior(self, x_start, x_t, t):
    #     posterior_mean = (
    #         extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
    #         extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    #     )
    #     posterior_variance = extract(self.posterior_variance, t, x_t.shape)
    #     posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
    #     return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, data, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, data, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
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
            x_start = self.normalize(x_start)  # [-1, 1]

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

    # def p_mean_variance(self, x, t, data, x_self_cond = None, clip_denoised = True):
    #     preds = self.model_predictions(x, t, data, x_self_cond)
    #     x_start = preds.pred_x_start

    #     if clip_denoised:
    #         x_start.clamp_(-1., 1.)

    #     model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
    #     return model_mean, posterior_variance, posterior_log_variance, x_start

    # @torch.no_grad()
    # def p_sample(self, x, t: int, data, x_self_cond = None):
    #     b, *_, device = *x.shape, x.device
    #     batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
    #     model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, data = data, x_self_cond = x_self_cond, clip_denoised = True)
    #     noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
    #     pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    #     return pred_img, x_start

    # @torch.no_grad()
    # def p_sample_loop(self, shape, data, return_all_timesteps = False):
    #     raise NotImplementedError
    #     batch, device = shape[0], self.betas.device

    #     img = torch.randn(shape, device = device)
    #     imgs = [img]

    #     x_start = None

    #     for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
    #         self_cond = x_start if self.self_condition else None
    #         img, x_start = self.p_sample(img, t, data, self_cond)
    #         imgs.append(img)

    #     ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    #     ret = self.unnormalize(ret)
    #     return ret

    @torch.no_grad()
    def ddim_sample(self, shape, data, return_all_timesteps = False, partial_start_time = -1, x0 = None):
        # total_timesteps=1000, sampling_timesteps=250
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # pass
        if partial_start_time > -1:
            if partial_start_time < times[-2]:
                partial_start_time = times[-2]
            assert x0 is not None
            max_time = next(v for _, v in enumerate(times) if v <= partial_start_time)
            max_time = torch.full((batch,), max_time, dtype=torch.long, device=device)
            x_start = self.normalize(x0)
            noise = torch.randn_like(x_start)
            img = self.q_sample(x_start = x_start, t = max_time, noise = noise)
        else:
            img = torch.randn(shape, device = device)
            x_start = None

        imgs = [img]        

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:
            # pass
            if time > partial_start_time > -1:
                continue

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None # None
            pred_noise, x_start, additional_info, *_ = self.model_predictions(img, time_cond, data, self_cond, clip_x_start = True, rederive_pred_noise = True)

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


        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1) # return_all_timesteps=False

        ret = self.unnormalize(ret)

        if partial_start_time > -1:
            additional_info['recur_noised'] = self.unnormalize(imgs[0])

        return ret, additional_info

    @torch.no_grad()
    def sample(self, data, batch_size = 1, views = 4, return_all_timesteps = False, partial_start_time = -1, x0 = None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, views, channels, image_size, image_size), data, return_all_timesteps = return_all_timesteps, partial_start_time = partial_start_time, x0 = x0)

    # @torch.no_grad()
    # def interpolate(self, x1, x2, data, t = None, lam = 0.5):
    #     b, *_, device = *x1.shape, x1.device
    #     t = default(t, self.num_timesteps - 1)

    #     assert x1.shape == x2.shape

    #     t_batched = torch.full((b,), t, device = device)
    #     xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

    #     img = (1 - lam) * xt1 + lam * xt2

    #     x_start = None

    #     for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
    #         self_cond = x_start if self.self_condition else None
    #         img, x_start = self.p_sample(img, i, data, self_cond)

    #     return img
    
    # @torch.no_grad()
    # def reconstruction(self, img, t, data, noise=None):
    #     b, v, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
    #     t = torch.full((b,), t, dtype=torch.long, device=device)
    #     img = self.normalize(img)
    #     noise = default(noise, lambda: torch.randn_like(img))
    #     x = self.q_sample(x_start = img, t = t, noise = noise)
    #     self_cond = img if self.self_condition else None
    #     pred_noise, x_start, additional_info, *_ = self.model_predictions(img, t, data, self_cond, clip_x_start = True, rederive_pred_noise = True)
    #     rec = self.unnormalize(x_start)
    #     return rec, additional_info

    # 加噪
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
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, data, current_data = None, noise = None):
        b, v, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None

        # pass
        if self.self_condition and random() < 0.5 and current_data is not None:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, current_data).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
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
            raise ValueError(f'unknown objective {self.objective}')

        # loss = self.loss_fn(prediction, target, reduction = 'none')
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')

        # loss_raw = loss

        # loss = loss * extract(self.loss_weight, t, loss.shape)
        # return loss.mean(), loss_raw.mean()

        loss_weight = self.loss_weight.gather(-1, t)
        return ret, loss_weight

    def forward(self, img, *args, **kwargs):
        b, v, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # [0,1] -> [-1,1]
        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# lightning
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
        log_val_img = True,
        pre_trained_unet_path = None,
        pre_trained_query_path = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.denoiser = Denoiser(smpl_type=smpl_type)
        
        self.diffusion = GaussianDiffusion(
            model=self.denoiser,
            image_size=image_size,
            timesteps = timesteps,
            sampling_timesteps = sampling_timesteps,
        )

        # sampling and training hyperparameters
        self.image_size = image_size
        self.train_lr = train_lr
        self.adam_betas = adam_betas

        self.validation_step_outputs = {
            'generated': [],
            'gt': [],
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }

        self.test_step_outputs = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }

        # lpips
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')

        self.test_mode = test_mode  # None, 'nv', 'np'
        self.log_dir = log_dir
        self.reconstruction_timestep = reconstruction_timestep
        self.test_recur_timestep = test_recur_timestep

        self.rgb_loss_weight = rgb_loss_weight

        self.log_val_img = log_val_img

        self.test_last_add_info = None

    ########################################################################################
    def training_step(self, batch, batch_idx):
        image = self.get_masked_input_image(batch)  # [1, n_view, 3, image_size, image_size]
        cond = self.get_diffusion_conds(batch)
        data = {
            'cond': cond,
        }

        model_out, loss_weight = self.diffusion(image, data=data, current_data=None)

        loss = self.calc_loss(image, model_out, loss_weight)

        return loss
    

    def calc_loss(self, image, model_out, loss_weight):
        loss_raw = 0
        
        prediction_rgb = model_out['out'].unsqueeze(0)
        target_rgb = image
        rgb_loss_raw = F.mse_loss(prediction_rgb, target_rgb)
        self.log('rgb_loss_raw', rgb_loss_raw, prog_bar=True)
        loss_raw += rgb_loss_raw * self.rgb_loss_weight

        loss = loss_raw * loss_weight
        self.log('loss', loss, prog_bar=False)

        return loss


    def validation_step(self, batch, batch_idx):
        image = self.get_masked_input_image(batch)  # [1, n_view, 3, image_size, image_size]
        cond = self.get_diffusion_conds(batch)
        data = {
            'cond': cond,
        }
        batch_size, views = image.shape[0], image.shape[1]

        val_out, _ = self.diffusion.sample(data=data, batch_size=batch_size, views=views)
        self.validation_step_outputs['generated'].append(val_out)
        self.validation_step_outputs['gt'].append(image)

        pred_img = val_out.permute(0, 1, 3, 4, 2).reshape(-1, 3).detach().cpu().numpy()
        gt_img = image.permute(0, 1, 3, 4, 2).reshape(-1, 3).detach().cpu().numpy()
        
        psnr = psnr_metric(pred_img, gt_img)
        self.validation_step_outputs['psnr'].append(psnr)
        
        pred_img = val_out.permute(0, 1, 3, 4, 2).squeeze().detach().cpu().numpy()
        gt_img = image.permute(0, 1, 3, 4, 2).squeeze().detach().cpu().numpy()
        
        bbox_mask = self.get_sample_image_mask(batch).detach().cpu().numpy()

        for v_idx in range(image.shape[1]):
            _pred_img = pred_img[v_idx]
            _gt_img = gt_img[v_idx]
            _bbox_mask = bbox_mask[v_idx]
            _ssim = ssim_metric(_pred_img, _gt_img, _bbox_mask)
            _lpips = lpips_metric(_pred_img, _gt_img, _bbox_mask, self.loss_fn_vgg, self.device)
            self.validation_step_outputs['ssim'].append(_ssim)
            self.validation_step_outputs['lpips'].append(_lpips)


    def on_validation_epoch_end(self):
        if self.log_val_img:
            all_preds = torch.cat(self.validation_step_outputs['generated'], dim=0)
            validation_outputs = self.all_gather(all_preds)
            validation_outputs = validation_outputs.reshape(-1, *validation_outputs.shape[-3:])
            grid = utils.make_grid(validation_outputs)
            self.logger.experiment.add_image(f"generated_images", grid, self.global_step)
            self.validation_step_outputs['generated'].clear()

            all_gt = torch.cat(self.validation_step_outputs['gt'], dim=0)
            gts = self.all_gather(all_gt)
            gts = gts.reshape(-1, *gts.shape[-3:])
            grid = utils.make_grid(gts)
            self.logger.experiment.add_image(f"gt_images", grid, self.global_step)
            self.validation_step_outputs['gt'].clear()

        all_psnr = torch.Tensor(self.validation_step_outputs['psnr']).mean()
        all_ssim = torch.Tensor(self.validation_step_outputs['ssim']).mean()
        all_lpips = torch.Tensor(self.validation_step_outputs['lpips']).mean()

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


    def test_step(self, batch, batch_idx):
        image = self.get_masked_input_image(batch)  # [1, n_view, 3, image_size, image_size]
        cond = self.get_diffusion_conds(batch)
        data = {
            'cond': cond,
        }
        batch_size, views = image.shape[0], image.shape[1]
        test_out, _ = self.diffusion.sample(data=data, batch_size=batch_size, views=views)

        pred_img = test_out.permute(0, 1, 3, 4, 2).squeeze().detach().cpu().numpy()
        gt_img = image.permute(0, 1, 3, 4, 2).squeeze().detach().cpu().numpy()
        
        bbox_mask = self.get_sample_image_mask(batch).detach().cpu().numpy()
        for v_idx in range(image.shape[1]):
            _pred_img = pred_img[v_idx]
            _gt_img = gt_img[v_idx]
            _bbox_mask = bbox_mask[v_idx]
            _psnr = psnr_metric(_pred_img[_bbox_mask], _gt_img[_bbox_mask])
            _ssim = ssim_metric(_pred_img, _gt_img, _bbox_mask)
            _lpips = lpips_metric(_pred_img, _gt_img, _bbox_mask, self.loss_fn_vgg, self.device)
            self.test_step_outputs['psnr'].append(_psnr)
            self.test_step_outputs['ssim'].append(_ssim)
            self.test_step_outputs['lpips'].append(_lpips)

        save_dir = os.path.join(self.log_dir, self.test_mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        frame_idx = [int(batch['current_data_list'][i]['frame_idx']) for i in range(views)]
        cam_idx = [int(batch['current_data_list'][i]['cam_idx']) for i in range(views)]

        for view in range(views):
            gt_image = image[0, view]
            rgb_image = test_out[0, view]
            utils.save_image(gt_image, os.path.join(save_dir, 'gt_f{:06d}_cam{:02d}.png'.format(frame_idx[view], cam_idx[view])))
            utils.save_image(rgb_image, os.path.join(save_dir, 'rgb_f{:06d}_cam{:02d}.png'.format(frame_idx[view], cam_idx[view])))


    def on_test_epoch_end(self):
        all_psnr = torch.Tensor(self.test_step_outputs['psnr']).mean()
        all_ssim = torch.Tensor(self.test_step_outputs['ssim']).mean()
        all_lpips = torch.Tensor(self.test_step_outputs['lpips']).mean()

        psnr = self.all_gather(all_psnr).mean()
        ssim = self.all_gather(all_ssim).mean()
        lpips = self.all_gather(all_lpips).mean()

        # self.log('psnr', psnr, rank_zero_only=True)
        # self.log('ssim', ssim, rank_zero_only=True)
        # self.log('lpips', lpips, rank_zero_only=True)

        with open(os.path.join(self.log_dir, self.test_mode, 'results.txt'), 'a') as f:
            f.write('=============================\n')
            f.write(f' Mode: {self.test_mode}\n')
            f.write(f' PSNR: {psnr.detach().cpu().numpy()}\n')
            f.write(f' SSIM: {ssim.detach().cpu().numpy()}\n')
            f.write(f'LPIPS: {lpips.detach().cpu().numpy()}\n')
            f.write('=============================\n')
            f.write('\n')

        self.validation_step_outputs = {
            'generated': [],
            'gt': [],
            'psnr': [],
            'ssim': [],
            'lpips': [],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.diffusion.parameters(), lr = self.train_lr, betas = self.adam_betas)
        return optimizer
    

    def get_masked_input_image(self, batch):
        image = torch.cat([i['image'] for i in batch['current_data_list']], dim=0).permute([0, 3, 1, 2]).clone()  # [v, 3, image_size, image_size]
        image_mask = torch.cat([i['image_mask'] for i in batch['current_data_list']], dim=0)  # [v, image_size, image_size]
        image[image_mask.unsqueeze(1).repeat(1, 3, 1, 1) == 0] = 0
        # image[~image_mask.unsqueeze(1).repeat(1, 3, 1, 1)] = 0
        return image.unsqueeze(0)  # [1, v, 3, image_size, image_size]
    

    def get_sample_image_mask(self, batch):
        sample_image_mask = torch.cat([i['sample_image_mask'] for i in batch['current_data_list']], dim=0).clone()  # [v, image_size, image_size]
        return sample_image_mask
    

    # v是训练使用的视角的个数
    def get_diffusion_conds(self, batch):
        R = []
        T = []
        K = []

        for view_item in batch['current_data_list']:
            _R = view_item['R']  # [1, 3, 3]
            _T = view_item['T']  # [1, 3]
            _K = view_item['K']  # [1, 3, 3]
            R.append(_R)
            T.append(_T)
            K.append(_K)

        R = torch.cat(R, dim=0)  # [v, 3, 3]
        T = torch.cat(T, dim=0)  # [v, 3]
        K = torch.cat(K, dim=0)  # [v, 3, 3]

        inputs = {
            'verts': batch['smpl_vertices'].clone().detach(),         # [1, SMPL_NODE_NUM, 3]
            # 'verts_cano': batch['minimal_shape'].clone().detach(),    # [1, SMPL_NODE_NUM, 3]
            # 'verts_T_inv': batch['T_inv'],                          # [1, SMPL_NODE_NUM, 4, 4]
            'R': R,  # [v, 3, 3]
            'T': T,  # [v, 3]
            'K': K,  # [v, 3, 3]
        }
        return inputs


def save_image(path, image):
    return cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
