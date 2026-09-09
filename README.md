# Gaussian Splatting for Texture Compression

## Descrizione

Questo progetto studia l'uso di Gaussian Splatting 2D per la compressione di texture, con attenzione al caso delle texture derivate da scansioni 3D di oggetti reali.

L'approccio consiste nel fittare un insieme di gaussiane 2D a una texture, così da ottenere una rappresentazione compressa dell'immagine. La qualità della ricostruzione viene valutata confrontando l'immagine renderizzata dalle gaussiane con quella originale, tramite le metriche PSNR e SSIM.

A differenza delle texture standard, le texture derivate da scansioni 3D presentano spesso aree prive di dati validi (dei buchi), generate automaticamente durante il processo di UV-mapping/baking della scansione. Questi pixel non hanno un valore di colore significativo e non devono influenzare né l'inizializzazione delle gaussiane, né il training, né la valutazione finale.

Il progetto si basa sull'implementazione originale di [ImageGS](https://github.com/NYU-ICL/image-gs) (NYU-ICL), estesa per gestire correttamente questi casi.

## Struttura del repository

Il flag `is_texture_scan` (in cfgs/default.yaml, oppure passabile da CLI come --is_texture_scan) determina il comportamento:
- `is_texture_scan`: False è il comportamento di default di ImageGS. Nessuna maschera viene applicata.
- `is_texture_scan`: True abilita la maschera di buchi, per texture derivate da scansioni 3D.

## How to use

Per compressione di immagini standard con ImageGS:

```bash
python main.py \
  --input_path=images/anime-1_2k.png \
  --exp_name=test/anime-1_2k \
  --num_gaussians=10000 \
  --quantize

python main.py \
  --input_path=images/anime-1_2k.png \
  --exp_name=test/anime-1_2k \
  --num_gaussians=10000 \
  --quantize \
  --eval \
```

Per compressione di texture da scansioni 3D con ImageGS:

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

Le immagini si trovano dentro alla directory image-gs/media

## Dataset utilizzati

- Texture standard: [Poly Haven](https://polyhaven.com)
- Texture da scansioni 3D: [texturedmesh.isti.cnr.it](https://texturedmesh.isti.cnr.it)
