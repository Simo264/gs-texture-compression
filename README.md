# Gaussian Splatting for Texture Compression

## Descrizione

Questo progetto studia l'uso di Gaussian Splatting 2D per la compressione di texture, con attenzione al caso delle texture derivate da scansioni 3D di oggetti reali.

L'approccio consiste nel fittare un insieme di gaussiane 2D a una texture, così da ottenere una rappresentazione compressa dell'immagine. La qualità della ricostruzione viene valutata confrontando l'immagine renderizzata dalle gaussiane con quella originale, tramite le metriche PSNR e SSIM.

A differenza delle texture standard, le texture derivate da scansioni 3D presentano spesso aree prive di dati validi (dei buchi), generate automaticamente durante il processo di UV-mapping/baking della scansione. Questi pixel non hanno un valore di colore significativo e non devono influenzare né l'inizializzazione delle gaussiane, né il training, né la valutazione finale.

Il progetto si basa sull'implementazione originale di [ImageGS](https://github.com/NYU-ICL/image-gs) (NYU-ICL), modificato per gestire correttamente questo caso.


## How to use

Chiamare il programma `main.py` con i seguenti argomenti:

```bash
python main.py \
  --input_path=textures/texture-scan-3d.png \
  --exp_name=test/texture-scan-3d \
  --num_gaussians=10000 \
  --quantize

python main.py \
  --input_path=textures/texture-scan-3d.png \
  --exp_name=test/texture-scan-3d \
  --num_gaussians=10000 \
  --quantize \
  --eval \
```