# Copyright (C) 2022 ByteDance Inc.
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
# 25/06/2024: gaze-conditioned model generation, random gaze selection, conditioned image generation, interpolation between different gaze directions

import os
import argparse
import shutil
import numpy as np
import imageio
import torch
import cv2
#import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import make_model_gaze
from generate.utils import generate, cubic_spline_interpolate

latent_dict_celeba = {
    2: "bcg_1",
    3: "bcg_2",
    4: "face_shape",
    5: "face_texture",
    6: "eye_shape",
    7: "eye_texture",
    8: "eyebrow_shape",
    9: "eyebrow_texture",
    10: "nose_shape",
    11: "nose_texture",
    0: "coarse_1",
    1: "coarse_2",
}


def pick_random_gaze(batch, db_labels):
    random_gazes = [db_labels[np.random.randint(len(db_labels))] for _ in range(batch)]
    print(random_gazes[0])
    r = [float(random_gazes[0].split('_')[0]), float(random_gazes[0].split('_')[1])]
    for i, g in enumerate(random_gazes):
        random_gazes[i] = r
    return random_gazes

def generate_progressive_gazes():
    random_gazes = []
    min_x = -0.5 #-0-1
    max_x = 0.5
    min_y = -0.5 #-0.1
    max_y = 0.5
    step = (max_x - min_x) / 8
    for x in np.arange(min_x, max_x, step):
        for y in np.arange(min_y, max_y, step):
            random_gazes.append([x, y])
    return random_gazes


def draw_gaze(image_in, pitchyaw, thickness=4, color=(255, 0,0)):
    """Draw gaze angle on given image with a given eye positions."""
    image_in = image_in[0:256, 0:256]
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 3 #2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out


# def draw_gaze_distrib(gazes):
#     px = []
#     py = []
#     for i, gaze in enumerate(gazes):
#         r = [float(gazes[i].split('_')[0]), float(gazes[i].split('_')[1])]
#         px.append(r[0])
#         py.append(r[1])

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     hist, xedges, yedges = np.histogram2d(px, py, bins=16, range=[[-2, 2], [-2, 2]])

#     # Construct arrays for the anchor positions of the 16 bars.
#     xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
#     xpos = xpos.ravel()
#     ypos = ypos.ravel()
#     zpos = 0

#     # Construct arrays with the dimensions for the 16 bars.
#     dx = dy = 0.5 * np.ones_like(zpos)
#     dz = hist.ravel()

#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
#     plt.savefig('distrib_gazes.jpg')

#     fig = plt.figure()
#     plt.hist(px, bins=20)
#     plt.savefig('distrib_gazes_x.jpg')
#     fig = plt.figure()
#     plt.hist(py, bins=20)
#     plt.savefig('distrib_gazes_y.jpg')

#     return fig


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('ckpt', type=str, help="path to the model checkpoint")
    parser.add_argument('--latent', type=str, default=None,
                        help="path to the latent numpy")
    parser.add_argument('--outdir', type=str, default='./results/interpolation/',
                        help="path to the output directory")
    parser.add_argument('--batch', type=int, default=8, help="batch size for inference")
    parser.add_argument("--sample", type=int, default=8,
                        help="number of latent samples to be interpolated")
    parser.add_argument("--steps", type=int, default=160,
                        help="number of latent steps for interpolation")
    parser.add_argument("--truncation", type=float, default=0.7, help="truncation ratio")
    parser.add_argument("--truncation_mean", type=int, default=10000,
                        help="number of vectors to calculate mean for the truncation")
    parser.add_argument("--dataset_name", type=str, default="celeba",
                        help="used for finding mapping between latent indices and names")
    parser.add_argument('--device', type=str, default="cuda",
                        help="running device for inference")
    parser.add_argument('--gaze_lbs', type=str, help="path to the dataset file with gaze labels")
    args = parser.parse_args()

    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    model = make_model_gaze(ckpt['args'])
    model.to(args.device)
    model.eval()
    model.load_state_dict(ckpt['g_ema'])
    mean_latent = model.style(torch.randn(args.truncation_mean, model.style_dim, device=args.device)).mean(0)

    print("Generating original image ...")
    db_gazes = np.load(args.gaze_lbs, allow_pickle=True).item()
    #d = draw_gaze_distrib(db_gazes)
    random_gazes_z = pick_random_gaze(1, db_gazes)
    random_gazes_z = np.asarray(random_gazes_z)
    random_gazes_z = torch.from_numpy(random_gazes_z)
    random_gazes_z = random_gazes_z.to(args.device)
    with torch.no_grad():
        if args.latent is None:
            styles = model.style(torch.randn(1, model.style_dim, device=args.device))
            styles = args.truncation * styles + (1 - args.truncation) * mean_latent.unsqueeze(0)
        else:
            styles = torch.tensor(np.load(args.latent), device=args.device)
        if styles.ndim == 2:
            assert styles.size(1) == model.style_dim
            styles = styles.unsqueeze(1).repeat(1, model.n_latent, 1)
        images, segs = generate(model, styles, gaze=random_gazes_z, mean_latent=mean_latent, randomize_noise=False,
                                batch_size=args.sample)
        imageio.imwrite(f'{args.outdir}/image.jpeg', images[0])
        imageio.imwrite(f'{args.outdir}/seg.jpeg', segs[0])
    original_im = images[0]

    print("Generating videos ...")
    if args.dataset_name == "celeba":
        latent_dict = latent_dict_celeba
    else:
        raise ValueError("Unknown dataset name: f{args.dataset_name}")

    random_gazes_z = pick_random_gaze(args.sample, db_gazes)
    random_gazes_z = np.asarray(random_gazes_z)
    random_gazes_z = torch.from_numpy(random_gazes_z)
    random_gazes_z = random_gazes_z.to(args.device)

    with torch.no_grad():
        for latent_index, latent_name in latent_dict.items():
            #continue
            styles_new = styles.repeat(args.sample, 1, 1)
            mix_styles = model.style(torch.randn(args.sample, 512, device=args.device))
            mix_styles[-1] = mix_styles[0]
            mix_styles = args.truncation * mix_styles + (1 - args.truncation) * mean_latent.unsqueeze(0)
            mix_styles = mix_styles.unsqueeze(1).repeat(1, model.n_latent, 1)
            styles_new[:, latent_index] = mix_styles[:, latent_index]
            styles_new = cubic_spline_interpolate(styles_new, step=args.steps)
            images, segs = generate(model, styles_new, gaze=random_gazes_z, mean_latent=mean_latent,
                                    randomize_noise=False, batch_size=args.sample)
            frames = [np.concatenate((img, seg), 1) for (img, seg) in zip(images, segs)]
            #imageio.mimwrite(f'{args.outdir}/{latent_index:02d}_{latent_name}.mp4', frames, fps=20)
            imageio.mimsave(f'{args.outdir}/{latent_index:02d}_{latent_name}.gif', frames)
            print(f"{args.outdir}/{latent_index:02d}_{latent_name}.gif")
            for kk, im in enumerate(frames):
                if kk > 3:
                    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                 #   cv2.imwrite(args.outdir + '/' +latent_name + '_' + str(latent_index) + '_' + str(latent_index) + str(kk) + '.jpg', im_rgb)

    # MODIFY GAZE:
    prog_gazes = generate_progressive_gazes()
    prog_gazes = np.asarray(prog_gazes)
    prog_gazes = prog_gazes[0:args.sample]
    print(len(prog_gazes))
    prog_gazes = torch.from_numpy(prog_gazes)
    prog_gazes = prog_gazes.to(args.device)
    
    with torch.no_grad():
        latent_index = 0
        styles_new = styles.repeat(args.sample, 1, 1)
        mix_styles = model.style(torch.randn(args.sample, 512, device=args.device))
        mix_styles[-1] = mix_styles[0]
        mix_styles = args.truncation * mix_styles + (1 - args.truncation) * mean_latent.unsqueeze(0)
        mix_styles = mix_styles.unsqueeze(1).repeat(1, model.n_latent, 1)
        styles_new[:, latent_index] = mix_styles[:, latent_index]
        styles_new = cubic_spline_interpolate(styles_new, step=args.steps)
        images, segs = generate(model, styles_new, gaze=prog_gazes, mean_latent=mean_latent, randomize_noise=False, batch_size=args.sample)

        frames = [np.concatenate((img, seg), 1) for (img, seg) in zip(images, segs)]
        imageio.mimwrite(f'{args.outdir}/{latent_index:02d}_gaze.mp4', frames, fps=2)
        
    dframes = []
    dif = 0
    for i, g in enumerate(prog_gazes):
        newim = draw_gaze(frames[i], g.cpu().numpy())
        im_rgb = cv2.cvtColor(newim, cv2.COLOR_BGR2RGB)
        dframes.append(newim)
        dif += cv2.subtract(newim, original_im)
        #cv2.imwrite(args.outdir + '/' + str(i) + '.jpg', im_rgb)
    imageio.mimwrite(f'{args.outdir}/{latent_index:02d}_gaze_draw.gif', dframes)

    # MODIFY STYLE OF THE SAME GAZE:
    # prog_gazes = generate_progressive_gazes()
    # prog_gazes = np.asarray(prog_gazes)
    # prog_gazes = prog_gazes[0:args.sample]
    # prog_gazes = torch.from_numpy(prog_gazes)
    # prog_gazes = prog_gazes.to(args.device)

    # with torch.no_grad():
    #     latent_index = 0
    #     styles_new = styles.repeat(args.sample, 1, 1)
    #     mix_styles = model.style(torch.randn(args.sample, 512, device=args.device))
    #     mix_styles[-1] = mix_styles[0]
    #     mix_styles = args.truncation * mix_styles + (1 - args.truncation) * mean_latent.unsqueeze(0)
    #     mix_styles = mix_styles.unsqueeze(1).repeat(1, model.n_latent, 1)
    #     images, segs = generate(model, styles_new, gaze=prog_gazes, mean_latent=mean_latent,
    #                             randomize_noise=False, batch_size=args.sample)

    #     frames = [np.concatenate((img, seg), 1) for (img, seg) in zip(images, segs)]
    #     imageio.mimwrite(f'{args.outdir}/{latent_index:02d}_gaze.mp4', frames, fps=20)
    #     imageio.mimsave(f'{args.outdir}/{latent_index:02d}_gaze.gif', frames)
    #     for ff, f in enumerate(frames):
    #         cv2.imwrite(f'{args.outdir}/{latent_index:02d}_gaze_' + str(ff) + '_v2.jpg', f)
