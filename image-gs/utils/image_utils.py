import os

import cv2
import flip_evaluator
import imageio.v3 as iio
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from numpy.linalg import norm
from PIL import Image
from scipy.ndimage import sobel

FONT_PATH = "assets/fonts/linux_libertine/LinLibertine_R.ttf"
font_manager.fontManager.addfont(FONT_PATH)
FONT_PROP = font_manager.FontProperties(fname=FONT_PATH).get_name()

plt.rcParams['font.family'] = FONT_PROP
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['legend.title_fontsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

PLOT_DPI = 72.0
GAUSSIAN_ZOOM = 5
GAUSSIAN_COLOR = "#80ed99"

def load_texture(load_path):
    """
    Load a single RGBA PNG texture.

    Returns:
        rgb: np.ndarray, shape (3, H, W), float32 in [0, 1]
        alpha: np.ndarray, shape (H, W), float32 in [0, 1]
        bit_depth: int, 8 or 16
    """
    if not (os.path.isfile(load_path) and os.path.splitext(load_path)[1].lower() == ".png"):
        raise FileNotFoundError(f"No PNG texture found at '{load_path}'")

    image = cv2.imread(load_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{load_path}'")
    if image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError(f"Expected RGBA image with 4 channels, got shape {image.shape}")

    # cv2 loads as BGRA -> reorder to RGBA
    image = image[..., [2, 1, 0, 3]]

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
        bit_depth = 8
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
        bit_depth = 16
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")

    rgb = image[..., :3].transpose(2, 0, 1)  # (3, H, W)
    alpha = image[..., 3]                    # (H, W)
    return rgb, alpha, bit_depth

def save_texture(rgb, alpha, save_path, bit_depth=8):
    """
    Save an RGB image + alpha channel as a single RGBA PNG.

    Args:
        rgb: torch.Tensor or np.ndarray, shape (3, H, W), values in [0, 1]
        alpha: torch.Tensor or np.ndarray, shape (H, W), values in [0, 1]
        save_path: destination path, must end in .png
        bit_depth: int, 8 or 16 (default 8)
    """
    if os.path.splitext(save_path)[1].lower() != ".png":
        raise ValueError(f"Invalid image format (expected .png): {save_path}")

    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().clone().numpy()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().cpu().clone().numpy()

    if rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor with shape (3, H, W), got {rgb.shape}")
    if alpha.shape != rgb.shape[1:]:
        raise ValueError(f"Alpha shape {alpha.shape} does not match RGB spatial shape {rgb.shape[1:]}")

    rgb = rgb.transpose(1, 2, 0).astype(np.float32)   # (H, W, 3)
    alpha = alpha.astype(np.float32)                   # (H, W)

    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)  # (H, W, 4)
    rgba = np.clip(rgba, 0.0, 1.0)
    if bit_depth == 8:
        rgba = (255.0 * rgba).astype(np.uint8)
    elif bit_depth == 16:
        rgba = (65535.0 * rgba).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported bit_depth: {bit_depth}")

    # RGBA -> BGRA for cv2
    rgba = rgba[..., [2, 1, 0, 3]]
    cv2.imwrite(save_path, rgba)

def save_grayscale(image, save_path):
    """Save a single-channel float32 map (e.g. gradient/saliency) as an 8-bit grayscale PNG for visualization only."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().clone().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (255.0 * image).astype(np.uint8)
    cv2.imwrite(save_path, image)

def get_grid(h, w, x_lim=np.asarray([0, 1]), y_lim=np.asarray([0, 1])):
    x = torch.linspace(x_lim[0], x_lim[1], steps=w + 1)[:-1] + 0.5 / w
    y = torch.linspace(y_lim[0], y_lim[1], steps=h + 1)[:-1] + 0.5 / h
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid

def compute_image_gradients(image):
    gy, gx = [], []
    for image_channel in image:
        gy.append(sobel(image_channel, 0))
        gx.append(sobel(image_channel, 1))
    gy = norm(np.stack(gy, axis=0), ord=2, axis=0).astype(np.float32)
    gx = norm(np.stack(gx, axis=0), ord=2, axis=0).astype(np.float32)
    return gy, gx



def save_error_maps(path, images, gt_images, mask, save_image_format="png"):
    images = torch.clamp(images, 0.0, 1.0)
    gt_image = gt_images.detach().cpu().clone().numpy().transpose(1, 2, 0)
    image = images.detach().cpu().clone().numpy().transpose(1, 2, 0)

    flip_error_map, _, _ = flip_evaluator.evaluate(reference=gt_image, test=image, dynamicRangeString="LDR", inputsRGB=True, applyMagma=True)

    # Zero out the error visualization in hole regions, so they don't visually
    # suggest error where none was actually measured/optimized
    mask_np = mask.detach().cpu().numpy()  # (H, W), bool
    flip_error_map[~mask_np] = 0.0

    save_rgb_image(flip_error_map, f"{path}.{save_image_format}")

def save_rgb_image(image, save_path):
    """Save an RGB array/tensor of shape (H, W, 3) as an 8-bit image for visualization only."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().clone().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (255.0 * image).astype(np.uint8)
    image = image[..., ::-1]  # RGB -> BGR for cv2
    cv2.imwrite(save_path, image)

def get_psnr_masked(image1, image2, mask, num_valid_pixels, feat_dim, max_value=1.0):
    diff2 = (image1 - image2) ** 2 * mask
    mse = diff2.sum() / (num_valid_pixels * feat_dim)
    if mse.item() <= 1e-7:
        return float('inf')
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr

def visualize_gaussian_position(filepath, image, xy, color="#7bf1a8", size=700, every_n=10, alpha=0.8, save_image_format="png"):
    """
    Visualize the position of Gaussians using dots.
    """
    image_height, image_width = image.shape[1:]
    xy = xy.detach().cpu().clone().numpy()[::every_n]
    x, y = xy[:, 0] * image_width, xy[:, 1] * image_height

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().clone().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (255.0 * image).astype(np.uint8)
    image = image.transpose(1, 2, 0)  # (C, H, W) -> (H, W, 3)

    fig = plt.figure()
    fig.set_dpi(PLOT_DPI)
    fig.set_size_inches(w=image_width/PLOT_DPI, h=image_height/PLOT_DPI, forward=False)
    plt.imshow(Image.fromarray(image))
    plt.scatter(x, y, s=size, c=color, marker='o', alpha=alpha)
    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{filepath}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
    plt.close()

def visualize_gaussian_footprint(filepath, xy, scale, rot, feat, img_h, img_w, input_channels, alpha=0.8, gamma=None, save_image_format="jpg"):
    """
    Visualize the footprint of Gaussians using colored elliptical disks.
    """
    if feat.shape[1] != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {feat.shape[1]:d} vs {sum(input_channels):d}")
    xy = xy.detach().cpu().clone().numpy()
    y, x = xy[:, 1] * img_h, xy[:, 0] * img_w
    scale = GAUSSIAN_ZOOM * scale.detach().cpu().clone().numpy()
    rot = rot.detach().cpu().clone().numpy()
    if gamma is not None:
        feat = torch.pow(feat, 1.0/gamma)
    feat = np.clip(feat.detach().cpu().clone().numpy(), 0.0, 1.0)

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        curr_feat = feat[:, curr_channel:curr_channel+num_channels]
        if curr_feat.shape[1] == 1:
            curr_feat = np.repeat(curr_feat, 3, axis=1)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=img_w/PLOT_DPI, h=img_h/PLOT_DPI, forward=False)
        ax = plt.gca()
        for gid in range(len(xy)):
            ellipse = Ellipse(xy=(x[gid], y[gid]), width=scale[gid, 0], height=scale[gid, 1],
                              angle=rot[gid, 0]*180/np.pi, alpha=alpha, ec=None, fc=curr_feat[gid], lw=None)
            ax.add_patch(ellipse)
        plt.xlim(0, img_w)
        plt.ylim(img_h, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels

def visualize_added_gaussians(filepath, images, old_xy, new_xy, input_channels, size=500, every_n=5, alpha=0.8, gamma=None, save_image_format="jpg"):
    """
    Visualize the positions of added Gaussians during error-guided progressive optimization.
    """
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_height, image_width = images.shape[1:]
    old_xy = old_xy.detach().cpu().clone().numpy()[::every_n]
    new_xy = new_xy.detach().cpu().clone().numpy()[::every_n]
    old_x, old_y = old_xy[:, 0] * image_width, old_xy[:, 1] * image_height
    new_x, new_y = new_xy[:, 0] * image_width, new_xy[:, 1] * image_height

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        image = images[curr_channel:curr_channel+num_channels]
        image = to_output_format(image, f".{save_image_format}", gamma)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=image_width/PLOT_DPI, h=image_height/PLOT_DPI, forward=False)
        plt.imshow(Image.fromarray(image), cmap='gray', vmin=0, vmax=255)
        plt.scatter(old_x, old_y, s=size, c="#ef476f", marker='o', alpha=alpha)  # red
        plt.scatter(new_x, new_y, s=size, c="#06d6a0", marker='o', alpha=alpha)  # green
        plt.xlim(0, image_width)
        plt.ylim(image_height, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels
