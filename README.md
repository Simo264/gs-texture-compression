# Gaussian Splatting for Texture Compression

Siamo interessati a studiare come usare GS per comprimere textures derivate da scansioni 3D di oggetti.
L'idea è di fittare gaussiane 2D alle textures derivate dalle scansioni, e valutare la qualità della compressione in termini di accuratezza della texture ricostruita rispetto all'originale. Textures derivate da scansioni 3D spesso contengono dettagli fini e rumore, quindi sarà interessante vedere come Gaussian Splatting riesce a catturare questi aspetti. Inoltre, queste textures sono spesso piene di "buchi" (aree senza dati), dovuti ai metodi automatici di generazione delle textures dalle scansioni 3D. In questi "buchi", non ci interessano i valori delle immagini, dato che non verranno usati nel rendering finale.

## Algoritmo 

Usare l'implementazione di Gaussian Splatting per immagini 2D. Applicare Gaussian Splatting alle textures derivate da scansioni 3D, fittando un numero variabile di gaussiane.
Durante il fitting, ignorare le aree "vuote" della texture (ad esempio, aree completamente nere o bianche, a seconda del formato della texture). Modificare il codice in modo che non inserisca nuove gaussiane in queste aree durante il fitting, nè le inserisca all'inizio. Modificare il calcolo della loss in modo che non consideri i pixel in queste aree.
Possibilmente, pravare ad aggiungere la base di Beta Splatting per migliorare la qualità del fitting.

## Valutazione

Valutare la qualità della compressione confrontando l'immagine ricostruita con l'immagine originale, usando metriche come PSNR o SSIM.

Attenzione: calcolare PSNR e SSIM su tutta l'immagine includendo i buchi falserebbe i
risultati. Bisogna calcolare le metriche solo sui pixel validi.

## Suggerimenti

Scaricare textures derivate da scansioni 3D da https://texturedmesh.isti.cnr.it.
Scaricare texture standard da https://polyhaven.com.
Utilizzare il codice di ImageGS (https://github.com/NYU-ICL/image-gs) come base per il fitting.

Il progetto si appoggia esplicitamente a ImageGS, più precisamente ci interessa
il file `model.py`.

> 1. Dove vengono inizializzate le gaussiane? 

L'inizializzazione avviene nei metodi `_init_gaussians` e `_init_pos_scale_feat`.

Non inizializzare gaussiane nelle aree vuote. Occorre modificare la fase di campionamento iniziale. L'idea è di creare una maschera binaria (dove 1 = pixel valido,
0 = buco) e usarla per filtrare i pixel da cui campionare.
Una volta creata la maschera, modificare la funzione `_sample_pos` in model.py per usare
la maschera.

> 2. Dove viene calcolata la loss?

Avviene nel metodo `_get_total_loss`. Per escludere i buchi, devi modificare il calcolo
di ogni loss in modo che consideri solo i pixel validi, usando la maschera.
Per ogni loss (L1, L2, SSIM), devi applicare la maschera. In particolare, per L1 e L2
puoi moltiplicare le immagini per la maschera prima di calcolare la loss, oppure usare
un parametro reduction='none' e poi fare la media solo sui pixel validi.
La SSIM è più complessa perché considera patch di pixel. 

> 3. Dove avviene la densification/adaptive addition?

L'aggiunta progressiva di gaussiane avviene nel metodo `_add_gaussians`.
Questo metodo viene chiamato dal training loop `optimize()`, finché non si raggiunge 
il numero totale di gaussiane desiderato.
All'interno di `_add_gaussians`, le nuove gaussiane vengono campionate in base a una
error_map; le aree con errore più alto ricevono più nuove gaussiane.

Qua noi non dobbiamo aggiungere nuove gaussiane nelle aree vuote. Dobbiamo modificare
il calcolo della `sample_prob` in `_add_gaussians`.

> 4. Dove avviene il ciclo di training?

Il ciclo di training principale è nel metodo `optimize()`.
Qua non è necessario apportare modifiche.

> 5. Dove avviene la valutazione?

Infine per la valutazione, nel metodo `_evaluate()` vengono calcolati PSNR e SSIM su
tutta l'immagine

Dovremo modificare `_evaluate()` per calcolare le metriche solo sui pixel validi.