import os
import cv2
import flip_evaluator
import matplotlib
#import matplotlib.font_manager as font_manager
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
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


def to_cv2_image(tensor, bit_depth, to_bgr=False, gamma=None):
    """Converte un tensore (C, H, W) in [0, 1] in un array numpy per cv2.imwrite."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if tensor.dtype == np.float16:
        tensor = tensor.astype(np.float32)
    if tensor.dtype == np.bool_:
        tensor = tensor.astype(np.float32)
    if tensor.ndim == 3:
        tensor = np.transpose(tensor, (1, 2, 0))

    # gamma PRIMA dello scaling
    if gamma is not None and tensor.dtype in [np.float32, np.float64]:
        tensor = np.power(np.clip(tensor, 0.0, 1.0), 1.0 / gamma)
    if tensor.dtype in [np.float32, np.float64]:
        max_val = 255.0 if bit_depth == 8 else 65535.0
        tensor = np.clip(tensor * max_val, 0, max_val)
    if to_bgr and tensor.ndim == 3 and tensor.shape[-1] == 3:
        tensor = tensor[..., ::-1]

    # Cast al tipo intero corretto
    if bit_depth == 8:
        return tensor.astype(np.uint8)
    else:
        return tensor.astype(np.uint16)

def save_as_greyscale(image, save_path, bit_depth=8):
    """Salva un'immagine a singolo canale (1, H, W) o (H, W)."""
    img = to_cv2_image(image, bit_depth, to_bgr=False, gamma=None)
    assert img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1)
    # OpenCV preferisce array 2D per le immagini in scala di grigi
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    cv2.imwrite(save_path, img)

def save_as_rgb(image, save_path, bit_depth=8, gamma=None):
  """Salva un'immagine RGB (3, H, W)."""
  img = to_cv2_image(image, bit_depth, to_bgr=True, gamma=gamma)
  assert img.ndim == 3 and img.shape[-1] == 3, "save_as_rgb expects 3 channels"
  cv2.imwrite(save_path, img)

def save_as_rgba(image_rgb, alpha, save_path, bit_depth=8, gamma=None):
    assert image_rgb.shape[0] == 3, "save_as_rgba expects RGB, not RGBA"
    assert alpha.ndim == 2 or (alpha.ndim == 3 and alpha.shape[0] == 1)

    # 1. Converti RGB in BGR. to_cv2_image lo porta da (3, H, W) a (H, W, 3)
    bgr = to_cv2_image(image_rgb, bit_depth, to_bgr=True, gamma=gamma)

    # 2. Converti Alpha.
    # Se era (1, H, W), to_cv2_image lo porta automaticamente a (H, W, 1).
    # Se era (H, W), resta (H, W).
    a = to_cv2_image(alpha, bit_depth, to_bgr=False, gamma=None)

    # 3. Assicurati che 'a' sia (H, W, 1) per poter fare il concatenate sull'ultimo asse
    if a.ndim == 2:
        a = a[..., np.newaxis]

    # 4. Concatena e salva
    bgra = np.concatenate([bgr, a], axis=-1)
    cv2.imwrite(save_path, bgra)

def save_error_maps(path, images, gt_images, gamma, valid_mask=None, save_image_format="jpg"):
    """
    Calcola e salva la mappa di errore FLIP tra render e ground truth.

    Args:
        path: percorso base del file (senza estensione).
        images: render (C,H,W) float [0,1].
        gt_images: ground truth (C,H,W) float [0,1]. Usare gt_image_original.
        gamma: valore di gamma per la correzione.
        valid_mask: maschera booleana (H,W) opzionale. Se fornita, i buchi vengono azzerati nella mappa.
        save_image_format: estensione del file (es. "jpg", "png").
    """
    # Correzione gamma
    images = torch.pow(torch.clamp(images, 0.0, 1.0), 1.0 / gamma)
    gt_images = torch.pow(torch.clamp(gt_images, 0.0, 1.0), 1.0 / gamma)

    # Converti in numpy (H,W,C)
    images_np = images.detach().cpu().numpy().transpose(1, 2, 0)
    gt_images_np = gt_images.detach().cpu().numpy().transpose(1, 2, 0)

    # Calcola mappa FLIP (RGB, magma)
    flip_error_map, _, _ = flip_evaluator.evaluate(
        reference=gt_images_np,
        test=images_np,
        dynamicRangeString="LDR",
        inputsRGB=True,
        applyMagma=True
    )

    # Normalizza se necessario (FLIP può restituire uint8 o float > 1)
    if flip_error_map.dtype == np.uint8:
        flip_error_map = flip_error_map.astype(np.float32) / 255.0
    elif flip_error_map.max() > 1.0:
        flip_error_map = flip_error_map / 255.0

    # Applica maschera: azzera i buchi
    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.detach().cpu().numpy()
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[0]  # (H,W)
        flip_error_map = flip_error_map * valid_mask[..., np.newaxis]

    # Trasponi in (C,H,W) per save_as_rgb
    flip_error_map = flip_error_map.transpose(2, 0, 1)

    # Salva come RGB (8-bit)
    save_as_rgb(flip_error_map, f"{path}.{save_image_format}", bit_depth=8, gamma=None)

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




def visualize_gaussian_position(
    filepath, images, xy, input_channels, bit_depth=8, color="#7bf1a8", size=700, every_n=10, alpha=0.8, gamma=None, save_image_format="jpg"
):
    """
    Visualize the position of Gaussians using dots.
    """
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_height, image_width = images.shape[1:]
    xy = xy.detach().cpu().clone().numpy()[::every_n]
    x, y = xy[:, 0] * image_width, xy[:, 1] * image_height

    curr_channel = 0
    vmax = 255 if bit_depth == 8 else 65535
    for image_id, num_channels in enumerate(input_channels, 1):
        image = images[curr_channel:curr_channel+num_channels]
        image = to_cv2_image(image, bit_depth, to_bgr=False, gamma=gamma)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=image_width/PLOT_DPI, h=image_height/PLOT_DPI, forward=False)
        plt.imshow(Image.fromarray(image), cmap='gray', vmin=0, vmax=vmax)
        plt.scatter(x, y, s=size, c=color, marker='o', alpha=alpha)
        plt.xlim(0, image_width)
        plt.ylim(image_height, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels

def visualize_added_gaussians(
  filepath, images, old_xy, new_xy, input_channels,
  bit_depth=8, size=500, every_n=5, alpha=0.8, save_image_format="jpg"
):
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
    vmax = 255 if bit_depth == 8 else 65535
    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        image = images[curr_channel:curr_channel + num_channels]
        image = to_cv2_image(image, bit_depth, to_bgr=False, gamma=None)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=image_width/PLOT_DPI, h=image_height/PLOT_DPI, forward=False)
        plt.imshow(Image.fromarray(image), cmap='gray', vmin=0, vmax=vmax)
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


# def visualize_gaussian_footprint(filepath, xy, scale, rot, feat, img_h, img_w, input_channels, alpha=0.8, gamma=None, save_image_format="jpg"):
#     """
#     Visualize the footprint of Gaussians using colored elliptical disks.
#     """
#     if feat.shape[1] != sum(input_channels):
#         raise ValueError(f"Incompatible number of channels: {feat.shape[1]:d} vs {sum(input_channels):d}")
#     xy = xy.detach().cpu().clone().numpy()
#     y, x = xy[:, 1] * img_h, xy[:, 0] * img_w
#     scale = GAUSSIAN_ZOOM * scale.detach().cpu().clone().numpy()
#     rot = rot.detach().cpu().clone().numpy()
#     if gamma is not None:
#         feat = torch.pow(feat, 1.0/gamma)
#     feat = np.clip(feat.detach().cpu().clone().numpy(), 0.0, 1.0)

#     curr_channel = 0
#     for image_id, num_channels in enumerate(input_channels, 1):
#         curr_feat = feat[:, curr_channel:curr_channel+num_channels]
#         if curr_feat.shape[1] == 1:
#             curr_feat = np.repeat(curr_feat, 3, axis=1)
#         fig = plt.figure()
#         fig.set_dpi(PLOT_DPI)
#         fig.set_size_inches(w=img_w/PLOT_DPI, h=img_h/PLOT_DPI, forward=False)
#         ax = plt.gca()
#         for gid in range(len(xy)):
#             ellipse = Ellipse(xy=(x[gid], y[gid]), width=scale[gid, 0], height=scale[gid, 1],
#                               angle=rot[gid, 0]*180/np.pi, alpha=alpha, ec=None, fc=curr_feat[gid], lw=None)
#             ax.add_patch(ellipse)
#         plt.xlim(0, img_w)
#         plt.ylim(img_h, 0)
#         plt.axis('off')
#         plt.tight_layout()
#         suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
#         plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
#         plt.close()
#         curr_channel += num_channels
