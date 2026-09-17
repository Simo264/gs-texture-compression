# Contesto

Sto adattando il repository ImageGS (NYU-ICL) per la compressione di texture RGB derivate da scansioni 3D di oggetti. La texture di input è una singola immagine PNG RGBA.

**Vincoli**:
- Nessuna operazione di multi-formato, upsampling o downsampling.
- La risoluzione della texture deve rimanere invariata durante tutto il processo.

**Problema**: Le texture da scansioni 3D contengono dettagli fini, rumore e pixel senza dati reali (buchi), identificati dal canale alpha:
```python
hole_mask  = (alpha <= alpha_epsilon)   # buchi
valid_mask = (alpha >  alpha_epsilon)   # validi
```

I buchi non devono essere ricostruiti, né usati per loss, gradienti, densificazione o metriche.

Per facilitare l'ottimizzazione ai bordi si applica `cv2.inpaint(..., cv2.INPAINT_TELEA)` solo sui buchi. I pixel validi restano intatti. Si mantengono quindi due immagini:
1. `gt_image_original`: RGB originale, unico ground truth per le metriche.
2. `gt_image`: RGB con buchi inpainted, solo ausilio (bordi, densificazione, SSIM ibrida).

**Obiettivo**: Adattare ImageGS affinché Gaussiane, loss, metriche (PSNR/SSIM/LPIPS), densificazione e rendering ignorino completamente i buchi e lavorino esclusivamente sui pixel validi, confrontandosi sempre con l'RGB originale.

# Fase 1 (Inizializzazione) [COMPLETATA]

## Gestione Target e Inpainting  (`_init_target`)
Qua viene calcolata la maschera dei buchi; viene applicato inpainting sui buchi per evitare discontinuità di gradiente ai bordi che attirerebbero Gaussiane spurie durante la densificazione. I pixel validi nell'immagine inpainted vengono sovrascritti con i valori originali.

##  Inizializzazione Gaussiane (`_init_gaussians` e `_init_pos_scale_feat`)

Il numero massimo di Gaussiane è limitato a `num_valid_pixels`.
Le posizioni iniziali (`xy`) sono campionate esclusivamente da `valid_pixel_indices` con un jitter sub-pixel $\plusmn 0.5$. Nessuna Gaussiana nasce in un buco.
Le mappe di probabilità per le modalità gradient e saliency vengono azzerate nei buchi.
Le feature di colore iniziali (self.feat) sono campionate direttamente da `gt_image_original` usando gli indici dei pixel validi.

## Attributi Chiave
| Attributo                        | Tipo              | Ruolo                                                                          |
| -------------------------------- | ----------------- | ------------------------------------------------------------------------------ |
| self.gt_image_original           | Tensor (C,H,W)    | Ground Truth Assoluto. Usato per Loss L1/L2, PSNR, e campionamento feature.    |
| self.gt_image                    | Tensor (C,H,W)    | RGB inpainted. Usato per SSIM (hybrid trick) e densificazione.                 |
| self.valid_mask / self.hole_mask | Tensor (bool)     | Maschere booleane (H,W)                                                        |
| self.valid_pixel_indices         | Tensor (1D, long) | Indici flattened dei soli pixel validi. Fondamentale per campionamenti e loss. |
| self.num_valid_pixels            | int               | Numero di pixel validi. Cap per il numero di Gaussiane.                        |
| self.xy                          | Tensor (G, 2)     | Posizioni Gaussiane. Garanzia: campionate solo da valid_pixel_indices.         |
| self.feat                        | Tensor (G, 3)     | Colori Gaussiane. Garanzia: campionati da gt_image_original su pixel validi.   |


# 2. Ottimizzazione

##### 2.1 `optimize()`
Il ciclo principale di addestramento è gestito da `optimize()`, che per ogni step esegue rendering, calcolo della loss, backpropagation, aggiornamento dei pesi e, periodicamente, valutazione delle metriche e densificazione. La struttura del ciclo è corretta e non richiede modifiche: tutti gli interventi per escludere i buchi sono stati isolati nelle funzioni ausiliarie delegate.

##### 2.2 `_get_total_loss()`
Calcola L1, L2 e SSIM. Il problema è che usare l'RGB originale genererebbe un errore enorme nei buchi, mentre usare l'inpainted produrrebbe gradienti indesiderati. Per L1 e L2, la loss viene calcolata esclusivamente sui pixel validi (`images[:, valid_mask]`) confrontando con `gt_image_original`, garantendo gradienti nulli sui buchi. Per la SSIM, che opera su finestre locali e non può essere semplicemente mascherata senza corrompere i bordi, si costruisce un'immagine ibrida (`images_hybrid`) che contiene il render sui pixel validi e l'inpainted sui buchi. Poiché nei buchi l'ibrida è identica al target, l'errore locale è nullo e la chain rule azzera automaticamente i gradienti durante la backpropagation. La mappa SSIM pixel-per-pixel viene ottenuta con `FusedSSIMMap.apply` e la media scalare è calcolata solo sui pixel validi per evitare la diluizione della loss. castate al dtype del tensore immagine per prevenire RuntimeError in CUDA.

##### 2.3 `_evaluate`
Calcola le metriche di qualità. Il problema è che i buchi falsano pesantemente PSNR e SSIM, specialmente nelle finestre locali vicino ai bordi. Il PSNR viene calcolato come MSE esclusivamente sui pixel validi confrontando con `gt_image_original` nello spazio gamma corretto. La SSIM utilizza la stessa strategia dell'immagine ibrida implementata nella loss, confrontando il render con `gt_image` (inpainted) in spazio gamma e mediando la mappa risultante solo sui pixel validi.

##### 2.4 `_add_gaussians`
Gestisce la densificazione progressiva campionando nuove posizioni da una mappa di errore. Il problema è che se l'errore include i buchi, il modello spreca Gaussiane cercando di riempire aree mancanti. La `diff_map` viene calcolata confrontando il render con `gt_image` (inpainted) in spazio gamma, fornendo un target continuo ai bordi senza bisogno di applicare gaussian_blur: poiché l'obiettivo è la compressione di texture, preservare i dettagli ad alta frequenza è prioritario e l'immagine inpainted garantisce già la continuità necessaria. 
I valori della `error_map` corrispondenti ai buchi vengono poi azzerati esplicitamente. Questo garantisce che le nuove Gaussiane vengano create esclusivamente dove c'è vero dettaglio da ricostruire.

# 3. Rendering

##### 3.1 Metodo `render()`



# Strategia di lavoro richiesta all'agente

- Procedere in modo incrementale.
- Se trovi problemi, proponi la modifica necessaria, spiegando perché garantisce il requisito e (eventualmente) quali effetti collaterali può introdurre.
- Se non trovi problemi gravi, dichiaralo e fermati.
- Verificare la correttezza della soluzione
- Non riscrivere intere funzioni se non necessario, ma solo le posizioni specifiche su dove intervenire.