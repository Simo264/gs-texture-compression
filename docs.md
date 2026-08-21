> Configurazione di Collab

Image-GS si basa su CUDA, siccome io ho una GPU AMD non riesco ad eseguire in locale. Per questo ci viene in aiuto la piattaforma Google Colab che ci mette a disposizione una GPU NVIDIA T4.

In Collab eseguiamo i seguenti script:
```bash
!git clone --recurse-submodules https://github.com/Simo264/gs-texture-compression.git


# Installa le dipendenze:
!pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

!pip install flip-evaluator lpips==0.1.4 matplotlib==3.9.2 numpy==2.0.2 opencv-python==4.12.0.88 scikit-image==0.24.0 scipy==1.13.1 torchmetrics==1.5.2 pytorch-msssim==1.0.0

!pip install -e "/content/gs-texture-compression/image-gs/gsplat"

!mkdir -p /content/gs-texture-compression/image-gs/media/images
```

Image-GS ci dà la possibilità di comprimere semplice immagini JPG, PNG ma anche texture color, normal, roughness, ecc. 
Nel mio caso specifico quello che mi interessa è Image Compression perché quando scarico texture da scansioni io ho solamente una immagine color png.

Dentro la cartella `image-gs` eseguiamo i seguenti script python:
```bash
# Addestra il modello di Gaussian Splatting con 10'000 gaussiane
%cd /content/gs-texture-compression/image-gs

!python main.py \
--input_path="images/anime7_2k.png" \
--exp_name="test/anime7_2k" \
--num_gaussians=10000 \
--quantize

# Carica il modello già addestrato e renderizza l'immagine
!python main.py \
--input_path="images/anime7_2k.png" \
--exp_name="test/anime7_2k" \
--num_gaussians=10000 \
--quantize \
--eval
```

Valutiamo la qualità dell'immagine compressa. **PSNR (Peak Signal-to-Noise Ratio)** calcola il rapporto segnale/rumore di picco (in decibel) tra due immagini. Questo rapporto viene utilizzato come misura della qualità tra l'immagine originale e quella compressa. Quanto più alto è il PSNR, tanto migliore è la qualità dell'immagine compressa o ricostruita:
- \>40: eccellente (Lossless compression)
- 30-40: buono (High-quality compression)
- 20-30: decente (Moderate compression)
- <20: scarsa (Heavy compression)

**Structural Similarity Index (SSIM)** è una metrica percettiva che quantifica il degrado della qualità dell'immagine causato da elaborazioni quali la compressione dei dati o da perdite nella trasmissione dei dati:

L'immagine originale pesa 6'186'720 B (6Mib).

L'immagine compressa rappresentata con 10'000 gaussiane pesa 9Kib (un risparmio dell'85%) ma con una precisione mediocre:
```
Filename: render_res-2048x2048.jpg
Filesize: 917941B
PSNR: 25.54 | SSIM: 0.8006
```

Con 50'000 gaussiane, invece, pesa un pò di più ma il risultato della compressione è buono/ottimo:
```
Filesize: 1.14894MiB
PSNR: 32.54 | SSIM: 0.9250
```