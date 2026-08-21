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