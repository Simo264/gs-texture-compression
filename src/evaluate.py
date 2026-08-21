import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

if __name__ == "__main__":
  original_path = "anime7.png"
  compressed_path = "render_res-2048x2048.jpg"

  img_orig = cv2.imread(original_path)
  img_comp = cv2.imread(compressed_path)
  if img_orig.shape != img_comp.shape:
    print("Warning: images have different dimensions. Resizing...")
    img_comp = cv2.resize(img_comp, (img_orig.shape[1], img_orig.shape[0]))

  # Evaluate PSNR
  psnr_value = psnr(img_orig, img_comp, data_range=255)
  print(f"PSNR: {psnr_value:.2f} dB")
  
  # Evaluate SSIM: if the image is color, calculate the mean over the three channels
  if len(img_orig.shape) == 3:
    ssim_value = ssim(img_orig, img_comp, channel_axis=2, data_range=255)
  else:
    ssim_value = ssim(img_orig, img_comp, data_range=255)

  print(f"SSIM: {ssim_value:.4f}")
