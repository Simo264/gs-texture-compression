# Gaussian Splatting for Texture Compression

Siamo interessati a studiare come usare GS per comprimere textures derivate da scansioni 3D di oggetti.
L'idea è di fittare gaussiane 2D alle textures derivate dalle scansioni, e valutare la qualità della compressione in termini di accuratezza della texture ricostruita rispetto all'originale. Textures derivate da scansioni 3D spesso contengono dettagli fini e rumore, quindi sarà interessante vedere come Gaussian Splatting riesce a catturare questi aspetti. Inoltre, queste textures sono spesso piene di "buchi" (aree senza dati), dovuti ai metodi automatici di generazione delle textures dalle scansioni 3D. In questi "buchi", non ci interessano i valori delle immagini, dato che non verranno usati nel rendering finale.

## Algoritmo 

Usare l'implementazione di Gaussian Splatting per immagini 2D. Applicare Gaussian Splatting alle textures derivate da scansioni 3D, fittando un numero variabile di gaussiane.
Durante il fitting, ignorare le aree "vuote" della texture (ad esempio, aree completamente nere o bianche, a seconda del formato della texture). Modificare il codice in modo che non inserisca nuove gaussiane in queste aree durante il fitting, nè le inserisca all'inizio. Modificare il calcolo della loss in modo che non consideri i pixel in queste aree.
Possibilmente, pravare ad aggiungere la base di Beta Splatting per migliorare la qualità del fitting.

## Valutazione

Valutare la qualità della compressione confrontando l'immagine ricostruita con l'immagine originale, usando metriche come PSNR o SSIM.
Utilizzare il codice di ImageGS (https://github.com/NYU-ICL/image-gs).
Utilizzare textures derivate da scansioni 3D ad esempio da https://texturedmesh.isti.cnr.it. Confrontare con texture standard usate da https://polyhaven.com.

## Suggerimenti

Comincia a scaricare le textures da scansioni 3D reali e textures standard da usare come baseline di confronto.

Il progetto si appoggia esplicitamente a ImageGS. Prima di iniziare a scrivere codice, clona il repo e fallo girare così com'è su un'immagine qualsiasi, per capire cosa produce in output e come è strutturato il codice. Solo dopo inizierei a modificare il codice:
1. dove vengono inizializzate le gaussiane?
2. dove viene calcolata la loss?
3. dove avviene la densification/adaptive addition?
4. dove avviene il ciclo di training?

Il task chiede tre modifiche concrete:
1. non inizializzare gaussiane nelle aree vuote; serve una maschera binaria
2. non aggiungere nuove gaussiane lì durante la densification
3. escludere quei pixel vuoti dalla loss

Ti conviene scrivere prima uno script separato che, data una texture, produce la maschera binaria (soglia su nero/bianco/alpha).

PSNR e SSIM vanno calcolati solo sui pixel validi, altrimenti i buchi falsano il punteggio.

## Google Colab

Image-GS si basa su CUDA, siccome io ho una GPU AMD non riesco ad eseguire in locale. Per questo ci viene in aiuto la piattaforma Google Colab che ci mette a disposizione una GPU NVIDIA T4.

Per usare Image-GS su Google Colab bisogna clonare il repository e installare tutte le dipendenze all'interno del notebook.

Crea le cartelle di lavoro:
```
!mkdir -p /content/image-gs
!mkdir -p /content/data/scans
!mkdir -p /content/data/standard
!mkdir -p /content/experiments
```

Clonare e installare Image-GS:
```
%cd /content/
!git clone https://github.com/NYU-ICL/image-gs.git
%cd image-gs
```

Installa le dipendenze con pip:
```
!pip install torch torchvision torchaudio numpy opencv-python scipy matplotlib Pillow tensorboard lpips pytorch_msssim flip_evaluator

!pip install --index-url https://download.pytorch.org/whl/cu118

!pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

%cd gsplat
!pip install -e ".[dev]"
%cd ..
```


Image-GS ci dà la possibilità di comprimere semplice immagini JPG, PNG ma anche texture color, normal, roughness, ecc. Nel mio caso specifico quello che mi interessa è Image Compression perché quando scarico texture da scansioni io ho solamente una immagine color png.

Crea la directory media/images:
```
!mkdir -p /content/image-gs/media/images/
```

In Collab possiamo eseguire image-gs con immagine di prova
```
# Optimize an Image-GS representation for an input image
# using 10000 Gaussians with half-precision parameters

!python main.py \
--input_path="images/anime-7_2k.png" \
--exp_name="test/anime-7_2k" \
--num_gaussians=10000 \
--quantize

# Render the corresponding optimized Image-GS representation at a
# new resolution with height 4000

!python main.py \
--input_path="images/anime-7_2k.png" \
--exp_name="test/anime-7_2k" \
--num_gaussians=10000 \
--quantize \
--eval \
--render_height=4000
```

Nella directory results/test è stata generate l'immagine compressa. Quello che è successo è che abbiamo preso l'immagine originale di 6Mib: 
```
Image:
  Filename: anime-7_2k.png
  Format: PNG
  Filesize: 6'186'720 B
```

in una immagine compressa di 9Kib (risparmiando oltre l'85% dello spazio) rappresentata da 10.000 gaussiane:
```
Image:
  Filename: render_res-2048x2048.jpg
  Format: JPEG 
  Filesize: 917'941 B
```
