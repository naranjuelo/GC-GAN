# Copyright (C) 2022 ByteDance Inc.inv
# All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# The software is made available under Creative Commons BY-NC-SA 4.0 license
# by ByteDance Inc. You can use, redistribute, and adapt it
# for non-commercial purposes, as long as you (a) give appropriate credit
# by citing our paper, (b) indicate any changes that you've made,
# and (c) distribute any derivative works under the same license.

# THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# summary of changes (GC-GAN):
# 25/06/2024: new generate functions for gaze-conditioned models

import numpy as np
import torch
from scipy.interpolate import CubicSpline

color_map = {
    0: [0, 0, 0],
    1: [239, 234, 90],
    2: [44, 105, 154],
    3: [4, 139, 168],
    4: [13, 179, 158],
    5: [131, 227, 119],
    6: [185, 231, 105],
    7: [107, 137, 198],
    8: [241, 196, 83],
    9: [242, 158, 76],
    10: [234, 114, 71],
    11: [215, 95, 155],
    12: [207, 113, 192],
    13: [159, 89, 165],
    14: [142, 82, 172],
    15: [158, 115, 200], 
    16: [116, 95, 159],
}

def generate_img(model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images = []
    for head in range(0, styles.size(0), batch_size):
        images_, _ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_)
    images = torch.cat(images,0)
    return tensor2image(images)

def check_unbroken_mask(mask):
    unbroken = []
    thr = 10000
    for sample in mask:
        for i, mask in enumerate(sample):            
            if i == 5:
                if (mask > 0).sum() > thr:
                    unbroken.append(False)
                else:
                    unbroken.append(True)

def generate(model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs = [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_.detach().cpu())        
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images,0), torch.cat(segs,0)    
    return tensor2image(images), tensor2seg(segs)

def generate_D(model, model_D, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs, preds = [], [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        pred = model_D(images_, segs_)
        print(pred)
        preds.append(pred.detach().cpu().numpy())
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images,0), torch.cat(segs,0)
    return tensor2image(images), tensor2seg(segs), preds

def generate_infeats(model, styles, feats, depths, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs = [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_ = model([styles[head:head+batch_size]], feats,depths, input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images,0), torch.cat(segs,0)
    return tensor2image(images), tensor2seg(segs)

def generate_outfeats(model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs, feats, depths = [], [], [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_, feats_, depths_ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
        feats.append(feats_)
        depths.append(depths_)
    images, segs = torch.cat(images,0), torch.cat(segs,0)
    return tensor2image(images), tensor2seg(segs), feats, depths

def generate_gaze(model, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs, gazes = [], [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_, gz_ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
        gazes.append(gz_.detach().cpu())
    images, segs = torch.cat(images,0), torch.cat(segs,0)
    return tensor2image(images), tensor2seg(segs), gazes

def generate_gaze_estimD(model, discriminator, styles, mean_latent=None, truncation=1.0, batch_size=16, *args, **kwargs):
    images, segs, gazes, estim_gazes = [], [], [], []
    for head in range(0, styles.size(0), batch_size):
        images_, segs_ = model([styles[head:head+batch_size]], input_is_latent=True,
                                    truncation=truncation, truncation_latent=mean_latent, *args, **kwargs)        
        fake_pred, fake_g_estim = discriminator(images_, segs_)
        
        estim_gazes.append(fake_g_estim.detach().cpu())
        images.append(images_.detach().cpu())
        segs.append(segs_.detach().cpu())
    images, segs = torch.cat(images,0), torch.cat(segs,0)
    return tensor2image(images), tensor2seg(segs), gazes, estim_gazes


def discrim_estimD(images, segs, discriminator):
    estim_fake, estim_gazes = [], []
         
    fake_pred, fake_g_estim = discriminator(images, segs)
            
    estim_gazes.append(fake_g_estim.detach().cpu())
    estim_fake.append(fake_pred.detach().cpu())
    return estim_gazes, estim_fake

def tensor2image(tensor):
    images = tensor.cpu().clamp(-1,1).permute(0,2,3,1).detach().numpy()
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images
    
def tensor2seg(sample_seg):
    seg_dim = sample_seg.size(1)
    sample_seg = torch.argmax(sample_seg, dim=1).detach().cpu().numpy()
    sample_mask = np.zeros((sample_seg.shape[0], sample_seg.shape[1], sample_seg.shape[2], 3), dtype=np.uint8)
    for key in range(seg_dim):
        sample_mask[sample_seg==key] = color_map[key]
    return sample_mask

def cubic_spline_interpolate(styles, step):
    device = styles.device
    styles = styles.detach().cpu().numpy()
    N, K, D = styles.shape
    x = np.linspace(0.0, 1.0, N)
    y = styles.reshape(N,K*D)
    spl = CubicSpline(x, y)
    x_out = np.linspace(0.0, 1.0, step)
    results = spl(x_out) # Step x KD
    results = results.reshape(step,K,D)
    return torch.tensor(results, device=device).float()
