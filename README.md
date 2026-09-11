# Gaussian Splatting for Texture Compression

## Descrizione

Questo progetto studia l'uso di Gaussian Splatting 2D per la compressione di texture, con attenzione al caso delle texture derivate da scansioni 3D di oggetti reali.

L'approccio consiste nel fittare un insieme di gaussiane 2D a una texture, così da ottenere una rappresentazione compressa dell'immagine. La qualità della ricostruzione viene valutata confrontando l'immagine renderizzata dalle gaussiane con quella originale, tramite le metriche PSNR e SSIM.

A differenza delle texture standard, le texture derivate da scansioni 3D presentano spesso aree prive di dati validi (dei buchi), generate automaticamente durante il processo di UV-mapping/baking della scansione. Questi pixel non hanno un valore di colore significativo e non devono influenzare né l'inizializzazione delle gaussiane, né il training, né la valutazione finale.

Il progetto si basa sull'implementazione originale di [ImageGS](https://github.com/NYU-ICL/image-gs) (NYU-ICL), estesa per gestire correttamente questi casi.

## Config Google Colab

Image-GS si basa su CUDA, per questo motivo devo utilizzare la piattaforma Google Colab che mette a disposizione una GPU NVIDIA T4.

Da terminale:

```bash
git clone https://github.com/Simo264/gs-texture-compression.git

# creare un ambiente con Python 3.11
pip install uv
uv venv --seed --python 3.11 /content/venv311
source venv311/bin/activate

# Installa le dipendenze:
pip install pyyaml flip-evaluator lpips matplotlib numpy opencv-python pytorch-msssim scikit-image scipy torchmetrics torch torchvision torchaudio

pip install --index-url https://download.pytorch.org/whl/cu124

pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

cd /content/gs-texture-compression/image-gs/gsplat
pip install . --no-build-isolation
```

## How to use

Per verificare che l'installazione sia avvenuta correttamente, proviamo a effettuare una compressione di un'immagine di test con ImageGS:

```bash
python main.py \
--input_path="images/robot.png" \
--exp_name="test/robot" \
--num_gaussians=10000 \
--quantize

python main.py \
--input_path="images/robot.png" \
--exp_name="test/robot" \
--num_gaussians=10000 \
--quantize \
--eval
```

Per compressione di texture da scansioni 3D con ImageGS possiamo aggiungere l'opzione `--is_texture_scan`:

```bash
python main.py \
  --is_texture_scan \
  --input_path=textures/texture-scan-3d.png \
  --exp_name=test/texture-scan-3d \
  --num_gaussians=10000 \
  --quantize

python main.py \
  --is_texture_scan \
  --input_path=textures/texture-scan-3d.png \
  --exp_name=test/texture-scan-3d \
  --num_gaussians=10000 \
  --quantize \
  --eval \
```

> Le immagini si trovano dentro alla directory image-gs/media
