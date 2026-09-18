import logging
import math
import os
import sys
import warnings
import cv2
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim, FusedSSIMMap
from lpips import LPIPS
from pytorch_msssim import MS_SSIM
from torchvision.transforms.functional import gaussian_blur

from gsplat import (
    project_gaussians_2d_scale_rot,
    rasterize_gaussians_no_tiles,
    rasterize_gaussians_sum,
)
from utils.flip import LDRFLIPLoss
from utils.image_utils import (
    compute_image_gradients,
    get_grid,
    load_texture,
    save_as_rgb,
    save_as_rgba,
    save_as_greyscale,
    save_error_maps,

    # separate_image_channels,
    visualize_added_gaussians,
    visualize_gaussian_position,
)
from utils.misc_utils import clean_dir, get_latest_ckpt_step, save_cfg, set_random_seed
from utils.quantization_utils import ste_quantize
from utils.saliency_utils import get_smap

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips")


class GaussianSplatting2D(nn.Module):
    def __init__(self, args):
        super(GaussianSplatting2D, self).__init__()
        set_random_seed(seed=args.seed)
        self.evaluate = args.eval
        self.device = args.device
        self.dtype = torch.float32
        self.gamma = args.gamma
        self.disable_lr_schedule = args.disable_lr_schedule
        self.quantize = args.quantize
        self.pos_bits = args.pos_bits
        self.scale_bits = args.scale_bits
        self.rot_bits = args.rot_bits
        self.feat_bits = args.feat_bits
        self.disable_prog_optim = args.disable_prog_optim
        self.initial_ratio = args.initial_ratio
        self.add_times = args.add_times
        self.add_steps = args.add_steps
        self.ckpt_file = args.ckpt_file
        self.topk = args.topk
        self.eps = 1e-7 if args.disable_tiles else 1e-4
        self.init_scale = args.init_scale
        self.disable_topk_norm = args.disable_topk_norm
        self.disable_inverse_scale = args.disable_inverse_scale
        self.disable_color_init = args.disable_color_init
        self.block_h, self.block_w = 16, 16
        self.l1_loss = None
        self.l2_loss = None
        self.ssim_loss = None
        self.l1_loss_ratio = args.l1_loss_ratio
        self.l2_loss_ratio = args.l2_loss_ratio
        self.ssim_loss_ratio = args.ssim_loss_ratio
        self.disable_tiles = args.disable_tiles
        self.decay_ratio = args.decay_ratio
        self.check_decay_steps = args.check_decay_steps
        self.max_decay_times = args.max_decay_times
        self.decay_threshold = args.decay_threshold
        self.start_step = 1
        self.max_steps = args.max_steps
        self.pos_lr = args.pos_lr
        self.scale_lr = args.scale_lr
        self.rot_lr = args.rot_lr
        self.feat_lr = args.feat_lr
        self.init_mode = args.init_mode
        self.init_random_ratio = args.init_random_ratio
        self.smap_filter_size = args.smap_filter_size

        self._init_logging(args)
        self._init_target(args)
        self._init_gaussians(args)

        self.optimizer = torch.optim.Adam([
          {'params': self.xy, 'lr': self.pos_lr},
          {'params': self.scale, 'lr': self.scale_lr},
          {'params': self.rot, 'lr': self.rot_lr},
          {'params': self.feat, 'lr': self.feat_lr}])

        if self.evaluate:
             self._load_model()
        else:
             self._init_pos_scale_feat()

    # =======================================
    # Initialization
    # =======================================

    def _init_logging(self, args):
        self.log_dir = args.log_dir
        self.log_level = args.log_level
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.train_dir = os.path.join(self.log_dir, "train")
        self.eval_dir = os.path.join(self.log_dir, "eval")
        self.save_image_format = args.save_image_format
        self.save_plot_format = args.save_plot_format
        self.vis_gaussians = args.vis_gaussians
        self.save_image_steps = args.save_image_steps
        self.save_ckpt_steps = args.save_ckpt_steps
        self.eval_steps = args.eval_steps
        if not self.evaluate:
            clean_dir(path=self.log_dir)
            os.makedirs(self.log_dir, exist_ok=False)
            os.makedirs(self.ckpt_dir, exist_ok=False)
            os.makedirs(self.train_dir, exist_ok=False)
        else:
            os.makedirs(self.eval_dir, exist_ok=True)
        self._gen_logger(args)
        if not self.evaluate:
            save_cfg(path=f"{self.log_dir}/cfg_train.yaml", args=args)

    def _gen_logger(self, args):
        log_fname = "log_train"
        if self.evaluate:
            log_fname = "log_eval"
        log_level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(level=log_level)
        self.worklog = logging.getLogger("Image-GS Logger")
        self.worklog.propagate = False
        datefmt = "%Y/%m/%d %H:%M:%S"
        fileHandler = logging.FileHandler(f"{self.log_dir}/{log_fname}.txt", mode="a", encoding="utf8")
        fileHandler.setFormatter(logging.Formatter(fmt="[{asctime}] {message}", datefmt=datefmt, style="{"))
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter(fmt="\x1b[32m[{asctime}] \x1b[0m{message}", datefmt=datefmt, style="{"))
        self.worklog.handlers = [fileHandler, consoleHandler]
        action = "rendering" if self.evaluate else "optimizing"
        self.worklog.info(f"Start {action} {args.num_gaussians:d} Gaussians for '{args.input_path}'")
        self.worklog.info("***********************************************")

    def _init_target(self, args):
        path = os.path.join(args.data_root, args.input_path)
        rgb, alpha, bit_depth = load_texture(path) # load RGBA texture
        self.bit_depth = bit_depth # 8 or 16
        self.input_channels = 3 # 3 channels: RGB
        self.feat_dim = 3

        alpha_epsilon = 0.1
        hole_mask_np = alpha <= alpha_epsilon  # (H, W) bool, True = invalid/hole pixel
        valid_mask_np = ~hole_mask_np
        # Conserva l'RGB originale per la valutazione finale
        rgb_original = rgb.copy()
        # inpainting solo per riempire i buchi, senza toccare i validi
        if hole_mask_np.any():
            rgb_hwc = rgb.transpose(1, 2, 0)
            rgb_uint8 = np.clip(rgb_hwc * 255.0, 0, 255).astype(np.uint8)
            inpaint_mask = hole_mask_np.astype(np.uint8) * 255

            rgb_filled_uint8 = cv2.inpaint(
                rgb_uint8,
                inpaint_mask,
                inpaintRadius=5,
                flags=cv2.INPAINT_TELEA,
            )
            rgb_filled = rgb_filled_uint8.astype(np.float32) / 255.0
            # I pixel validi rimangono ESATTAMENTE quelli originali
            rgb_filled[valid_mask_np] = rgb_hwc[valid_mask_np]
            rgb = rgb_filled.transpose(2, 0, 1)

        self.gt_image_original = torch.from_numpy(rgb_original).to(dtype=self.dtype, device=self.device)
        self.gt_image = torch.from_numpy(rgb).to(dtype=self.dtype, device=self.device)
        self.alpha = torch.from_numpy(alpha).to(dtype=self.dtype, device=self.device)

        self.valid_mask = self.alpha > alpha_epsilon # valid pixels are those with alpha > 0.1
        self.hole_mask = ~self.valid_mask # hole pixels are those with alpha <= 0.1

        self.num_valid_pixels = self.valid_mask.sum().item() # calculate the number of valid pixels
        if self.num_valid_pixels == 0:
            raise ValueError("Texture contains no valid pixels according to the alpha mask.")

        self.img_h, self.img_w = self.gt_image.shape[1:]
        self.num_pixels = self.img_h * self.img_w
        self.tile_bounds = (
            (self.img_w + self.block_w - 1) // self.block_w,
            (self.img_h + self.block_h - 1) // self.block_h,
            1,
        )

        self.pixel_xy = get_grid(h=self.img_h, w=self.img_w).to(dtype=self.dtype,device=self.device).reshape(-1, 2)
        self.valid_pixel_indices = torch.where(self.valid_mask.reshape(-1))[0]

        if not self.evaluate:
          path = f"{self.log_dir}/gt_res-{self.img_h:d}x{self.img_w:d}.png"
          save_as_rgba(
              image_rgb=self.gt_image,
              alpha=self.alpha,
              save_path=path,
              bit_depth=self.bit_depth,
          )

    def _init_gaussians(self, args):
        # The number of gaussians cannot exceed the number of valid pixels
        self.total_num_gaussians = min(args.num_gaussians, self.num_valid_pixels)
        if self.total_num_gaussians < args.num_gaussians:
            self.worklog.info(
                f"Requested {args.num_gaussians:d} gaussians but only {self.num_valid_pixels:d} "
                f"valid pixels available. Capping total_num_gaussians to {self.total_num_gaussians:d}."
            )

        if not self.disable_prog_optim and not self.evaluate:
            self.num_gaussians = math.ceil(self.initial_ratio * self.total_num_gaussians)
            self.num_gaussians = min(self.num_gaussians, self.num_valid_pixels)  # difesa ridondante ma innocua
            self.max_add_num = math.ceil(float(self.total_num_gaussians - self.num_gaussians) / self.add_times)
            min_steps = self.add_steps * self.add_times + args.post_min_steps
            if args.max_steps < min_steps:
                self.worklog.info(f"Max steps ({args.max_steps:d}) is too small for progressive optimization. Resetting to {min_steps:d}")
                args.max_steps = min_steps
        else:
            self.num_gaussians = self.total_num_gaussians

        # Campionamento unificato degli indici validi (garantisce coerenza tra xy e feat)
        perm = torch.randperm(self.num_valid_pixels, device=self.device)[:self.num_gaussians]
        flat_idx = self.valid_pixel_indices[perm]
        # Inizializzazione XY (con jitter, stessa logica di _sample_valid_xy)
        row = (flat_idx // self.img_w).to(self.dtype)
        col = (flat_idx % self.img_w).to(self.dtype)
        jitter_row = (torch.rand(self.num_gaussians, device=self.device, dtype=self.dtype) - 0.5)
        jitter_col = (torch.rand(self.num_gaussians, device=self.device, dtype=self.dtype) - 0.5)
        y = (row + 0.5 + jitter_row) / self.img_h
        x = (col + 0.5 + jitter_col) / self.img_w
        self.xy = nn.Parameter(torch.stack([x, y], dim=-1), requires_grad=True)
        # Inizializzazione Feature (campionate dall'RGB originale nei soli pixel validi)
        # gt_image_original ha forma (C, H, W). Lo appiattiamo a (H*W, C) per l'indicizzazione avanzata
        gt_flat = self.gt_image_original.reshape(self.feat_dim, -1).T
        sampled_feat = gt_flat[flat_idx]  # Forma: (num_gaussians, feat_dim)
        self.feat = nn.Parameter(sampled_feat, requires_grad=True)

        self.scale = nn.Parameter(torch.ones(self.num_gaussians, 2, dtype=self.dtype, device=self.device), requires_grad=True)
        self.rot = nn.Parameter(torch.zeros(self.num_gaussians, 1, dtype=self.dtype, device=self.device), requires_grad=True)
        self.vis_feat = nn.Parameter(torch.rand_like(self.feat), requires_grad=False)
        self._log_compression_rate()

    def _log_compression_rate(self):
        bytes_uncompressed = float(self.num_valid_pixels) * self.feat_dim * (self.bit_depth / 8.0)
        bpp_uncompressed = float(self.feat_dim) * self.bit_depth
        bppc_uncompressed = bpp_uncompressed / self.feat_dim
        self.worklog.info(f"Uncompressed: {bytes_uncompressed/1e3:.2f} KB | {bpp_uncompressed:.3f} bpp | {bppc_uncompressed:.3f} bppc")

        bits_compressed = (2*self.pos_bits + 2*self.scale_bits + self.rot_bits + self.feat_dim*self.feat_bits) * self.total_num_gaussians
        bytes_compressed = bits_compressed / 8.0
        bpp_compressed = float(bits_compressed) / self.num_valid_pixels
        bppc_compressed = bpp_compressed / self.feat_dim
        self.num_bytes = bytes_compressed
        self.worklog.info(f"Compressed: {bytes_compressed/1e3:.2f} KB | {bpp_compressed:.3f} bpp | {bppc_compressed:.3f} bppc")
        self.worklog.info(f"Compression rate: {bpp_uncompressed/bpp_compressed:.2f}x | {100.0*bpp_compressed/bpp_uncompressed:.2f}%")
        self.worklog.info("***********************************************")

    def _sample_valid_xy(self, n):
        perm = torch.randperm(self.num_valid_pixels, device=self.device)[:n]
        flat_idx = self.valid_pixel_indices[perm]
        row = (flat_idx // self.img_w).to(self.dtype)
        col = (flat_idx % self.img_w).to(self.dtype)
        jitter_row = (torch.rand(n, device=self.device, dtype=self.dtype) - 0.5)
        jitter_col = (torch.rand(n, device=self.device, dtype=self.dtype) - 0.5)
        y = (row + 0.5 + jitter_row) / self.img_h
        x = (col + 0.5 + jitter_col) / self.img_w
        return torch.stack([x, y], dim=-1)

    def _init_pos_scale_feat(self):
        valid_indices = self.valid_pixel_indices.cpu().numpy()
        with torch.no_grad():
            # Position
            if self.init_mode == 'gradient':
                self._compute_gmap()
                self.xy.copy_(self._sample_pos(prob=self.image_gradients, valid_indices=valid_indices))
            elif self.init_mode == 'saliency':
                self._compute_smap(path="models")
                self.xy.copy_(self._sample_pos(prob=self.saliency, valid_indices=valid_indices))
            else:
                replace = self.num_gaussians > valid_indices.size
                selected = np.random.choice(valid_indices, self.num_gaussians, replace=replace, p=None)
                self.xy.copy_(self.pixel_xy.detach().clone()[selected])
            # Scale
            self.scale.fill_(self.init_scale if self.disable_inverse_scale else 1.0/self.init_scale)
            # Feature
            if not self.disable_color_init:
                self.feat.copy_(self._get_target_features(positions=self.xy).detach().clone())

    def _sample_pos(self, prob, valid_indices):
        # Restrict the sampling probability distribution to valid pixels, renormalize
        prob = prob[valid_indices]
        prob = prob / prob.sum()

        num_random = round(self.init_random_ratio*self.num_gaussians)
        num_other = self.num_gaussians - num_random

        replace_random = num_random > valid_indices.size
        selected_random = np.random.choice(valid_indices, num_random, replace=replace_random, p=None)

        nonzero_count = int((prob > 0).sum())
        replace_other = num_other > nonzero_count
        selected_other = np.random.choice(valid_indices, num_other, replace=replace_other, p=prob)

        return torch.cat([self.pixel_xy.detach().clone()[selected_random], self.pixel_xy.detach().clone()[selected_other]], dim=0)

    def _compute_gmap(self):
        gy, gx = compute_image_gradients(self.gt_image.detach().cpu().clone().numpy())
        g_norm = np.hypot(gy, gx).astype(np.float32)

        hole_mask_np = self.hole_mask.detach().cpu().numpy()
        g_norm[hole_mask_np] = 0.0  # ignora i bordi/gradienti spuri nei buchi

        max_grad = g_norm.max()
        if max_grad > 0:
            g_norm = g_norm / max_grad
        else:
            g_norm.fill(0.0)

        path = f"{self.log_dir}/gmap_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}"
        save_as_greyscale(g_norm, path, bit_depth=self.bit_depth)

        g_norm = np.power(g_norm.reshape(-1), 2.0)
        grad_sum = g_norm.sum()
        if grad_sum > 0:
            self.image_gradients = g_norm / grad_sum
        else:
            self.image_gradients = np.zeros_like(g_norm)
        self.worklog.info("Image gradient map successfully saved")
        self.worklog.info("***********************************************")

    def _compute_smap(self, path):
        smap = get_smap(self.gt_image.detach().clone(), path, self.smap_filter_size)
        smap = smap.masked_fill(self.hole_mask, 0.0)

        path = f"{self.log_dir}/smap_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}"
        save_as_greyscale(smap, path, bit_depth=self.bit_depth)

        smap = smap.reshape(-1)
        smap_sum = smap.sum()
        if smap_sum > 0:
            self.saliency = (smap / smap_sum).detach().cpu().numpy()
        else:
            self.saliency = np.zeros(smap.numel(), dtype=np.float32)
        self.worklog.info("Saliency map successfully saved")
        self.worklog.info("***********************************************")

    def _get_target_features(self, positions):
        with torch.no_grad():
            # gt_images [1, C, H, W]; positions [1, 1, P, 2]; top-left [-1, -1]; bottom-right [1, 1]
            target_features = F.grid_sample(self.gt_image.unsqueeze(0), positions[None, None, ...] * 2.0 - 1.0, align_corners=False)
            target_features = target_features[0, :, 0, :].permute(1, 0)  # [P, C]
        return target_features

    def _load_model(self):
        if self.ckpt_file != "":
            ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_file)
        else:
            latest_step = get_latest_ckpt_step(self.ckpt_dir)
            if latest_step == -1:
                raise FileNotFoundError(f"No checkpoint found in '{self.ckpt_dir}'")
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_step-{latest_step:d}.pt")
        checkpoint = torch.load(ckpt_path, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.start_step = checkpoint["step"]+1
        self.worklog.info(f"Checkpoint '{ckpt_path}' successfully loaded")
        self.worklog.info("***********************************************")

    # =======================================
    # Optimization
    # =======================================

    def optimize(self):
        self.psnr_curr, self.ssim_curr = 0.0, 0.0
        self.best_psnr, self.best_ssim = 0.0, 0.0
        self.decay_times, self.no_improvement_steps = 0, 0
        self.render_time_accum, self.total_time_accum = 0.0, 0.0
        self.lpips_final, self.flip_final, self.msssim_final = 1.0, 1.0, 0.0

        self.step = 0
        with torch.no_grad():
            self._log_images(log_final=False, plot_gaussians=self.vis_gaussians)
        for step in range(self.start_step, self.max_steps+1):
            self.step = step
            self.optimizer.zero_grad()
            # Rendering
            images, render_time = self.forward(self.img_h, self.img_w, self.tile_bounds)
            self.render_time_accum += render_time
            # Optimization
            begin = perf_counter()
            self._get_total_loss(images)
            self.total_loss.backward()
            self.optimizer.step()
            self.total_time_accum += (perf_counter() - begin + render_time)
            # Logging
            terminate = False
            with torch.no_grad():
                if self.step % self.eval_steps == 0:
                    self._evaluate(log=True)
                    if not self.disable_lr_schedule and self.num_gaussians == self.total_num_gaussians:
                        terminate = self._lr_schedule()
                if self.step % self.save_image_steps == 0:
                    self._log_images(log_final=False, plot_gaussians=self.vis_gaussians)
                if self.step % self.save_ckpt_steps == 0 and self.num_gaussians == self.total_num_gaussians:
                    self._save_model()
                if not self.disable_prog_optim and self.step % self.add_steps == 0 and self.num_gaussians < self.total_num_gaussians:
                    self._add_gaussians(self.max_add_num, plot_gaussians=self.vis_gaussians)
                if terminate:
                    break
        with torch.no_grad():
            self._log_images(log_final=True, plot_gaussians=self.vis_gaussians)
            self._save_model()
        self.worklog.info("Optimization completed")
        self.worklog.info("***********************************************")
        self.worklog.info(f"Mean scale: {self._get_scale().mean().item():.4f} (pixel) | {self.scale.mean().item():.4f} (raw)")
        self.worklog.info("***********************************************")
        return self.psnr_curr, self.ssim_curr

    def _get_total_loss(self, images):
        self.total_loss = 0
        images_valid = images[:, self.valid_mask]
        gt_valid = self.gt_image_original[:, self.valid_mask]

        if self.l1_loss_ratio > 1e-7:
            self.l1_loss = self.l1_loss_ratio * F.l1_loss(images_valid, gt_valid)
            self.total_loss += self.l1_loss
        else:
            self.l1_loss = None

        if self.l2_loss_ratio > 1e-7:
            self.l2_loss = self.l2_loss_ratio * F.mse_loss(images_valid, gt_valid)
            self.total_loss += self.l2_loss
        else:
            self.l2_loss = None

        if self.ssim_loss_ratio > 1e-7:
            valid_mask_3d = self.valid_mask.unsqueeze(0).float()
            hole_mask_3d = self.hole_mask.unsqueeze(0).float()
            images_hybrid = images * valid_mask_3d + self.gt_image * hole_mask_3d

            C1, C2 = 0.01 ** 2, 0.03 ** 2
            ssim_map = FusedSSIMMap.apply(
                C1, C2, images_hybrid.unsqueeze(0), self.gt_image.unsqueeze(0), "same", True, 2
            )  # [1, C, H, W], ancora differenziabile rispetto a images_hybrid

            # Verifica che la mappa abbia la stessa risoluzione dell'input
            if ssim_map.dim() == 4:      # (N, C, H, W)
                valid_mask_map = self.valid_mask.view(1, 1, *self.valid_mask.shape).expand_as(ssim_map)
            elif ssim_map.dim() == 3:    # (N, H, W)
                valid_mask_map = self.valid_mask.view(1, *self.valid_mask.shape).expand_as(ssim_map)
            else:
                raise ValueError(f"Shape inattesa per ssim_map: {ssim_map.shape}")

            ssim_value = ssim_map[valid_mask_map].mean()   # media SOLO sui pixel validi

            self.ssim_loss = self.ssim_loss_ratio * (1 - ssim_value)
            self.total_loss += self.ssim_loss
        else:
            self.ssim_loss = None

    def _evaluate(self, log=True):
        # 1. Rendering e correzione gamma
        images = torch.pow(torch.clamp(self._render_images(), 0.0, 1.0), 1.0/self.gamma)

        # Prepariamo i target nello spazio gamma (come l'immagine renderizzata)
        gt_original_gamma = torch.pow(self.gt_image_original, 1.0/self.gamma)
        gt_inpainted_gamma = torch.pow(self.gt_image, 1.0/self.gamma)

        # 2. Calcolo PSNR (Esclusivamente su pixel validi vs RGB Originale)
        images_valid = images[:, self.valid_mask]
        gt_valid = gt_original_gamma[:, self.valid_mask]

        mse = F.mse_loss(images_valid, gt_valid)
        # Calcolo PSNR standard (assumendo range [0,1] dopo gamma correction)
        # Usiamo clamp per evitare log(0) nel caso (improbabile) di errore nullo
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10)).item()

        # 3. Calcolo SSIM (Trucco immagine ibrida + Mappa mascherata)
        valid_mask_3d = self.valid_mask.unsqueeze(0).float()
        hole_mask_3d = self.hole_mask.unsqueeze(0).float()
        # Costruiamo l'immagine ibrida: renderizzata nei validi, inpainted nei buchi
        images_hybrid = images * valid_mask_3d + gt_inpainted_gamma * hole_mask_3d

        # Calcoliamo la mappa SSIM usando FusedSSIMMap
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = FusedSSIMMap.apply(
            C1, C2, images_hybrid.unsqueeze(0), gt_inpainted_gamma.unsqueeze(0), "same", True, 2
        )

        # Adattiamo la maschera alla dimensionalità della mappa SSIM
        if ssim_map.dim() == 4:
            valid_mask_map = self.valid_mask.view(1, 1, *self.valid_mask.shape).expand_as(ssim_map)
        elif ssim_map.dim() == 3:
            valid_mask_map = self.valid_mask.view(1, *self.valid_mask.shape).expand_as(ssim_map)
        else:
            raise ValueError(f"Shape inattesa per ssim_map: {ssim_map.shape}")

        # Media della SSIM calcolata SOLO sui pixel validi
        ssim = ssim_map[valid_mask_map].mean().item()

        if log:
            self.psnr_curr, self.ssim_curr = psnr, ssim
            loss_results = f"Loss: {self.total_loss.item():.4f}"
            loss_results += f", L1: {self.l1_loss.item():.4f}" if self.l1_loss is not None else ""
            loss_results += f", L2: {self.l2_loss.item():.4f}" if self.l2_loss is not None else ""
            loss_results += f", SSIM: {self.ssim_loss.item():.4f}" if self.ssim_loss is not None else ""
            time_results = f"Total: {self.total_time_accum:.2f} s | Render: {self.render_time_accum:.2f} s"
            self.worklog.info(f"Step: {self.step:d} | {time_results} | {loss_results} | PSNR: {self.psnr_curr:.2f} | SSIM: {self.ssim_curr:.4f}")

        return psnr, ssim

    def _evaluate_extra(self):
        # Ottieni il render grezzo e applica il clamp/gamma
        images_raw = torch.clamp(self._render_images(), 0.0, 1.0)
        images_hybrid = images_raw.clone()
        # Sovrascrivi i buchi con l'inpainted (così sono identici al target)
        images_hybrid[:, self.hole_mask] = self.gt_image[:, self.hole_mask]
        # Applica gamma
        images = torch.pow(images_hybrid, 1.0/self.gamma)[None, ...]
        # Prendi l'inpainted e applica il gamma
        gt_images = torch.pow(self.gt_image, 1.0/self.gamma)[None, ...]

        msssim_metric = MS_SSIM(data_range=1.0, size_average=True, channel=self.feat_dim).to(device=self.device).eval()
        self.msssim_final = msssim_metric(images, gt_images).item()
        lpips_metric = LPIPS(net='alex').to(device=self.device).eval()
        flip_metric = LDRFLIPLoss().to(device=self.device).eval()
        num_channels = 1 if self.feat_dim < 3 else 3
        self.lpips_final = lpips_metric(images[:, :num_channels], gt_images[:, :num_channels]).item()
        if self.feat_dim >= 3:
            self.flip_final = flip_metric(images[:, :3], gt_images[:, :3]).item()

    def _add_gaussians(self, add_num, plot_gaussians=False):
        add_num = min(add_num, self.max_add_num, self.total_num_gaussians - self.num_gaussians)
        if add_num <= 0:
            return

        raw_images = self._render_images()
        images = torch.pow(torch.clamp(raw_images, 0.0, 1.0), 1.0 / self.gamma)
        # Usiamo gt_image (inpainted) per evitare problemi ai bordi, ma SENZA blur
        gt_images = torch.pow(self.gt_image, 1.0 / self.gamma)

        # Calcolo diretto della mappa di errore (L2 sui canali mediati)
        diff_map = (gt_images - images).detach()
        error_map = torch.pow(torch.abs(diff_map).mean(dim=0).reshape(-1), 2.0)

        # Azzera i buchi nella mappa di errore (fondamentale)
        error_map[self.hole_mask.reshape(-1)] = 0.0
        error_sum = error_map.sum()

        # Calcolo delle probabilità
        valid_mask_flat = self.valid_mask.reshape(-1).cpu().numpy().astype(np.float64)

        if error_sum > 1e-8:
            sample_prob = (error_map / error_sum).cpu().numpy().astype(np.float64)
            sample_prob += 1e-12 * valid_mask_flat  # Probabilità di base per evitare crash
        else:
            sample_prob = valid_mask_flat

        sample_prob = sample_prob / sample_prob.sum()

        # Sicurezza finale sul numero di campioni
        num_available = np.count_nonzero(sample_prob)
        safe_add_num = min(add_num, num_available)

        if safe_add_num <= 0:
            return

        selected = np.random.choice(self.num_pixels, safe_add_num, replace=False, p=sample_prob)

        # ... il resto del codice rimane invariato

        # New Gaussians
        new_xy = self.pixel_xy.detach().clone()[selected]
        new_scale = torch.ones(add_num, 2, dtype=self.dtype, device=self.device)
        init_scale = self.init_scale
        new_scale.fill_(init_scale if self.disable_inverse_scale else 1.0/init_scale)
        new_rot = torch.zeros(add_num, 1, dtype=self.dtype, device=self.device)
        new_feat = diff_map.permute(1, 2, 0).reshape(-1, self.feat_dim)[selected]
        new_vis_feat = torch.rand_like(new_feat)
        # Old Gaussians
        old_xy = self.xy.detach().clone()
        old_scale = self.scale.detach().clone()
        old_rot = self.rot.detach().clone()
        old_feat = self.feat.detach().clone()
        old_vis_feat = self.vis_feat.detach().clone()
        # Update trainable parameters
        self.num_gaussians += add_num
        all_xy = torch.cat([old_xy, new_xy], dim=0)
        all_scale = torch.cat([old_scale, new_scale], dim=0)
        all_rot = torch.cat([old_rot, new_rot], dim=0)
        all_feat = torch.cat([old_feat, new_feat], dim=0)
        all_vis_feat = torch.cat([old_vis_feat, new_vis_feat], dim=0)
        self.xy = nn.Parameter(all_xy, requires_grad=True)
        self.scale = nn.Parameter(all_scale, requires_grad=True)
        self.rot = nn.Parameter(all_rot, requires_grad=True)
        self.feat = nn.Parameter(all_feat, requires_grad=True)
        self.vis_feat = nn.Parameter(all_vis_feat, requires_grad=False)
        # Plot Gaussians
        if plot_gaussians:
            path = f"{self.train_dir}/add-gaussians_step-{self.step:d}_num-{self.num_gaussians:d}_res-{self.img_h:d}x{self.img_w:d}"
            every_n = max(1, self.total_num_gaussians // 2000)
            size = (self.img_h * self.img_w) / 1e4
            visualize_added_gaussians(path, raw_images, old_xy, new_xy, self.input_channels, size=size, every_n=every_n,
                                      alpha=0.8, bit_depth=self.bit_depth, save_image_format=self.save_plot_format)
        # Update optimizer
        self.optimizer = torch.optim.Adam([
          {'params': self.xy, 'lr': self.pos_lr}, {'params': self.scale, 'lr': self.scale_lr},
          {'params': self.rot, 'lr': self.rot_lr}, {'params': self.feat, 'lr': self.feat_lr}])
        self.worklog.info(f"Step: {self.step:d} | Adding {add_num:d} Gaussians ({self.num_gaussians-add_num:d} -> {self.num_gaussians:d})")
        self.worklog.info("***********************************************")

    def _save_model(self):
        if self.quantize:
            self._quantize()
        psnr, ssim = self._evaluate(log=False)
        self._evaluate_extra()
        ckpt_data = {"step": self.step,
                     "psnr": psnr,
                     "ssim": ssim,
                     "lpips": self.lpips_final,
                     "flip": self.flip_final,
                     "msssim": self.msssim_final,
                     "bytes": self.num_bytes,
                     "time": self.total_time_accum,
                     "state_dict": self.state_dict(),
                     "optim_state_dict": self.optimizer.state_dict()}
        save_path = f"{self.ckpt_dir}/ckpt_step-{self.step:d}.pt"
        torch.save(ckpt_data, save_path)
        self.worklog.info(f"Checkpoint 'ckpt_step-{self.step:d}.pt' successfully saved")
        self.worklog.info(
            f"PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | LPIPS: {self.lpips_final:.4f} | FLIP: {self.flip_final:.4f} | MS-SSIM: {self.msssim_final:.4f}")
        self.worklog.info("***********************************************")

    def _lr_schedule(self):
        if (self.psnr_curr <= self.best_psnr + 100*self.decay_threshold or self.ssim_curr <= self.best_ssim + self.decay_threshold):
            self.no_improvement_steps += self.eval_steps
            if self.no_improvement_steps >= self.check_decay_steps:
                self.no_improvement_steps = 0
                self.decay_times += 1
                if self.decay_times > self.max_decay_times:
                    return True
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= self.decay_ratio
                self.worklog.info(f"Learning rate decayed by {self.decay_ratio:.1f}")
                self.worklog.info("***********************************************")
            return False
        else:
            self.best_psnr = self.psnr_curr
            self.best_ssim = self.ssim_curr
            self.no_improvement_steps = 0
            return False

    def _quantize(self):
        with torch.no_grad():
            self.xy.copy_(ste_quantize(self.xy, self.pos_bits))
            self.scale.copy_(ste_quantize(self.scale, self.scale_bits))
            self.rot.copy_(ste_quantize(self.rot, self.rot_bits))
            self.feat.copy_(ste_quantize(self.feat, self.feat_bits))


    # =======================================
    # Rendering
    # =======================================

    def render(self):
        img_h, img_w = self.img_h, self.img_w
        tile_bounds = ((img_w + self.block_w - 1) // self.block_w, (img_h + self.block_h - 1) // self.block_h, 1)
        with torch.no_grad():
            num_prep_runs = 2
            for _ in range(num_prep_runs):
                self.forward(img_h, img_w, tile_bounds, benchmark=True)
            images, render_time = self.forward(img_h, img_w, tile_bounds)
            path = f"{self.eval_dir}/render_res-{img_h:d}x{img_w:d}.{self.save_image_format}"
            save_as_rgb(images, path, bit_depth=self.bit_depth)
        self.worklog.info(f"Step: {self.start_step-1:d} | Time: {render_time:.6f} s")
        self.worklog.info(f"Rendering at resolution ({img_h:d}, {img_w:d}) completed")
        self.worklog.info("***********************************************")

    def forward(self, img_h, img_w, tile_bounds, benchmark=False):
        scale = self._get_scale()
        xy, rot, feat = self.xy, self.rot, self.feat
        if self.quantize:
            xy, scale, rot, feat = ste_quantize(xy, self.pos_bits), ste_quantize(
                scale, self.scale_bits), ste_quantize(rot, self.rot_bits), ste_quantize(feat, self.feat_bits)
        begin = perf_counter()
        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp
        if not self.disable_tiles:
            enable_topk_norm = not self.disable_topk_norm
            tmp = xy, radii, conics, num_tiles_hit, feat, img_h, img_w, self.block_h, self.block_w, enable_topk_norm
            out_image = rasterize_gaussians_sum(*tmp)
        else:
            tmp = xy, conics, feat, img_h, img_w
            out_image = rasterize_gaussians_no_tiles(*tmp)
        render_time = perf_counter() - begin
        if benchmark:
            return render_time
        out_image = out_image.view(-1, img_h, img_w, self.feat_dim).permute(0, 3, 1, 2).contiguous()
        return out_image.squeeze(dim=0), render_time

    def _render_images(self):
        images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds)
        return images

    def _get_scale(self):
        scale = self.scale
        if not self.disable_inverse_scale:
            scale = 1.0 / scale
        return scale

    def _visualize_gaussian_id(self, img_h, img_w, tile_bounds):
        scale = self._get_scale()
        xy, rot, feat = self.xy, self.rot, self.feat
        if self.quantize:
            xy, scale, rot, feat = ste_quantize(xy, self.pos_bits), ste_quantize(
                scale, self.scale_bits), ste_quantize(rot, self.rot_bits), ste_quantize(feat, self.feat_bits)
        feat = self.vis_feat * feat.norm(dim=-1, keepdim=True)
        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp
        if not self.disable_tiles:
            enable_topk_norm = not self.disable_topk_norm
            tmp = xy, radii, conics, num_tiles_hit, feat, img_h, img_w, self.block_h, self.block_w, enable_topk_norm
            out_image = rasterize_gaussians_sum(*tmp)
        else:
            tmp = xy, conics, feat, img_h, img_w
            out_image = rasterize_gaussians_no_tiles(*tmp)
        out_image = out_image.view(-1, img_h, img_w, self.feat_dim).permute(0, 3, 1, 2).contiguous()
        return out_image.squeeze(dim=0)

    def _log_images(self, log_final=False, plot_gaussians=False):
        images = self._render_images()
        if log_final:
          path = f"{self.log_dir}/render_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}"
          save_as_rgb(images, path, bit_depth=self.bit_depth, gamma=self.gamma)
        psnr, ssim = self._evaluate(log=False)

        path = (
            f"{self.train_dir}/render_step-{self.step:d}"
            f"_psnr-{psnr:.2f}_ssim-{ssim:.4f}"
            f"_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}"
        )
        save_as_rgb(images, path, bit_depth=self.bit_depth, gamma=self.gamma)

        if plot_gaussians:
            path = f"{self.train_dir}/flip-error_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            save_error_maps(
              path=path,
              images=images,
              gt_images=self.gt_image_original,
              gamma=self.gamma,
              valid_mask=self.valid_mask,
              save_image_format="jpg",
            )

            path = f"{self.train_dir}/gaussian-position_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            every_n = max(1, self.total_num_gaussians // 1000)
            size = 1.5 * (self.img_h * self.img_w) / 1e4
            visualize_gaussian_position(
                filepath=path,
                images=images,
                xy=self.xy,
                input_channels=self.input_channels,
                bit_depth=self.bit_depth,
                alpha=0.9,
                save_image_format=self.save_plot_format,
                every_n=every_n,
                color="#c0b1fc",
                size=size,
            )

            path = (
                f"{self.train_dir}/gaussian-id_step-{self.step:d}"
                f"_psnr-{psnr:.2f}_ssim-{ssim:.4f}"
                f"_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}"
            )
            images_gid = self._visualize_gaussian_id(self.img_h, self.img_w, self.tile_bounds)
            save_as_rgb(images_gid, path, bit_depth=self.bit_depth, gamma=None)
