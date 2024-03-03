

-----------------------------------------------------------------------------
- Whatsapp
-----------------------------------------------------------------------------

[Gabriele]
ciao, abbiamo due aspetti da sviluppare Bayesian Optimization e Data Distillation. Dell'aspetto Ottimizzazione
Bayesiana abbiamo già detto (ho condiviso anche a Gianni la cartella /Gabriele_corrado); riguardo alla Distillation,
io intendevo focalizzarmi sul sotto-problema della "coreset selection" (mentre la distillation comprende anche la
creazione di esempi rappresentativi artefatti, la coreset selection si limita, nella sua accezione restrittiva a
selezionare solo esempi già esistenti nel dataset originario). Ci sono un sacco di articoli in giro con vari obiettivi
e differenze nelle definizioni (più o meno ampie), ma credo di avere trovato il nostro bandolo della matassa: si tratta
dell'articolo "DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning" di Guo et al., che hanno
raccolto e implementato anche con PyTorch tutti i metodi principali e forniscono anche un sacco di benchmark, così da
consentirci di avere già un ampio spettro di termini di confronto. Altro articolo che fornisce dei benchmark dataset
(ma non codice) è "DC-BENCH: Dataset Condensation Benchmark" di Cui ed al. Li salvo entrambi nella cartella condivisa.


iniziando con deep core si ha una visione del problema della coreset selection, in parallelo o separatamente uno può
guardare per la bayesian optimization la molto figurativa introduzione fornita dal libro di Nguyen e l'a chiarissima
e formale presentazioni di Candelieri (entrambe queste ultime salvate adesso nella cartella di corrado)


ho aggiunto anche il libro del mio senior collega Archetti, il cui capitolo 6 presenta un bel po' di strumenti software
per la Bayesian optimization

[Gabriele]
Ho salvato nella cartella un altro papiro (che avevo già mostrato lunedì): "Efficient Black-Box Combinatorial
Optimization" del 2020. Due cose vanno dette: 1) mi ha fregato l'idea di usare lo shapley/banzaf value per selezionare
il subset (come avevamo fatto nella tesi di Corrado), infatti il papiro usa la base di Fourier invece della base di
Moebius, quindi semplicemente codifica presenza assenza con +1 e -1 anziché con 1 e 0, ma strutturalmente è la stessa
idea 2) la buona notizia è che non l'ha notato quasi nessuno, cioè il papiro non è citato da altri che ci costruiscano
sopra, ma solo nei related work di lavori sulla black-box optimization. Quindi c'è spazio per costruirci sopra,
riversando anche la nostra expertise in "Shapleyologia comparata" 😉


L'idea di fondo è questa: noi vogliamo ottimizzare una set function (che è black box). Ad ogni valutazione (dispendiosa)
aggiorniamo il modello probabilistico (il surrogate model che adesso descrivo) e poi scegliamo il prossimo insieme
usando un'acquisition function opportuna (ad esempio la Expected Improvement). Il modello probabilistico in Bayesian
Optimization è una distribuzione di probabilità sulle possibili funzioni pseudo-booleane (con n punti candidati a far
parte dell'insieme ottimale abbiamo funzioni 2^n-->R). Questo modello a rigore abiterebbe in uno spazio 2^n
dimensionale, ma noi lo approssimiamo con lo Shapley value dei singoli punti, quindi lavoriamo in uno spazio n
dimensionale, e se vogliamo anche usare l'interaction value aggiungiamo n(n-1)/2 dimensioni. Il resto è procedura
standard (che estrinseco quando ci vediamo).


Va detto che noi non effettuiamo una ricerca sull'intero reticolo booleano, ma abbiamo come obiettivo quello di trovare
un insieme di cardinalità k<<n, appunto il coreset; quindi possiamo usare il banzaf ridotto, come nella teso di Corrado
(ecco un'altra novità rispetto al papiro salvato), ma con un numero di candidati n grande il calcolo è comunque
proibitivo).


Adesso penso a come sfruttare un fatto nuovo rispetto alla tesi di Corrado, una "mossa magica": qui possiamo fare il
training sull'intero campione di un modello di Machine Learning trasparente, come un Decision Tree, così da ottenere
informazioni sulle frontiere tra le regioni e poi cercare i punti candidati proprio lì vicino, perché è lì che ci
saranno i punti più informativi. Questa è la mi idea ad oggi, e si differenzia dal papiro salvato perché quello parla
di ottimizzazione compibatoria generica, mentre qui noi abbiamo a che fare con un caso molto particolare, grazie alla
"mossa magica".


La mossa magica può sembrare una mossa truffaldina, ma non lo è. Infatti l'obiettivo dell'ottimizzazione è ottenere un
insieme (di k punti estratto degli n, ad esempio con con n=10k) sfruttando un certo budget di calcolo. Ora è noto che
un round di training implica un effort direttamente proporzionale al numero di esempi utilizzato: indico il budget con
b x n, dove b è il budget consumato per singolo esempio. Se mi danno un budget b x (2 n), io posso usare il budget b x n
per la singola valutazione dell'intero training dataset, e 10 volte b k per valutare dieci insiemi candidati, ciascuno
di k punti: quindi dopo la mossa magica ho 10 shot grazie ai quali valutare la funzione obiettivo (10 iterazioni dellla
Bayesian Optimization). Questi numeri sono solo esemplificativi. Il concetto è che io posso spezzare il budget in modo
ottimale tra mossa magica e numero di iterazioni consentite.


Questo da luogo ad un problema interessante, e cioè come si spezza in modo ottimale il budget.


Sembrerebbe un problema di ottimizzazione ad un terzo livello (forse semplice se la funzione associata al budget è
convessa): il livello più esterno è ottimizzare il budget utilizzando campioni di dimensione ad esempio decrescente,
poi usare l'ottimizzazione bayesiana per scegliere il campione di un certo livello k, poi all'interno di ogni singola
iterazione dell'ottimizzazione bayesiana ottimizzare la funzione d'acquisizione (anche questo sembra un problema
semplice).


Tutto ciò senza assumere alcuna struttura nello spazio dei dati di input. Ma se ciascun punto x = (x_1,...,x_D)
appartiene ad uno spazio D-dimensionale (ad esempio Euclideo) si può cercare un embedding in uno spazio d-dimensionale
con d<D (il che aiuta sicuramente la ricerca) anche senza ricorrere minimamente alla dispendiosa valutazione della
funzione f(S).


Siccome il dataset da condensare contiene le coppie (x,y) (cioè le coppie (input, label)), l'embedding è particolarmente
informativo e si sovrappone alla "mossa magica", cioè al primo round di classificazione che utilizza tutto il campione.
Come sfruttare questo fatto, ad esempio con delle SVM, è da chiarire.


Ho inoltre un'osservazione riguardo a quanto avevo scritto sopra: mi è stata chiara fin da domenica, mentre viaggiavo
sul flixbus per Lione, senza connessione internet, quindi la segnalo adesso che sono tornato e ho pututo rimettere
la testa sulle cose di ricerca.


Mi scuso perché nella foga ho commesso una svista, per troppo pessimismo, quindi rimuovendola si guadagna.


Avevo detto che sostituivamo la funzione pseudo-booleana 2^n dimensionale con la sua approssimazione tramite
Shapley/Banzaf (cioè l’approssimazione di grado 1) per cercare l’insieme che da il massimo di quest’ultima: ma ciò
viene gratis, perché essendo quell’approssimazione additiva negli atomi, quando hai stimato lo shapley value di quelli,
li metti in ordine e tieni i top k per costruire il “dream-team” dei migliori k elementi.
Quindi questa operazione si riduce al calcolo dello shapley value e non richiede ottimizzazione bayesiana.
Ma per l’approssimazione di grado 2 non si può usare questa scorciatoia.
Sappiamo che il problema si può formulare in termini di grafi (il second’ordine rappresenta l’interazione, diciamo
grossomodo lo shapley interaction value) e che la sua soluzione è equivalente a un problema di min-cut o max-cut.
Questo è un problema combinatorio degno dell’ottimizzazione Bayesiana.
Forse troviamo già qualcosa sul connubbio dei due. Però il nostro caso è speciale, perché noi non ci muoviamo alla
cieca come abbiamo fatto nella feature selection.
Come dicevo possiamo contare su: 1) un embedding dello spazio X in cui i punti sono immersi, e anche dello spazio (X,Y),
poi 2) sotto opportune ipotesi possiamo permetterci una mossa esplorativa in cui facciamo il training di un modello
trasparente su tutti i punti.

sto elaborando un un documento word la procedura senza tutti gli ammennicoli che ho aggiunto qui sopra: ne verrà fuori
un one-pager che condividerò

-----

Scusate ho avuto un'illuminazione (che purtroppo rende abbastanza superfluo quel che ho scritto fin qui, per fortuna 
non completamente). Mi spiego.

Nella distillation dobbiamo creare k punti con k<<n in grado di ottimizzare la funzione obiettivo (ad esempio
l’accuratezza di un classificatore.
Se assumiamo che ogni punto sperimentale (x,y) abbia un x definito su R^d, allora un insieme di tali punti può essere
rappresentato in n x d dimensioni, in R^{n x d}, quindi si può ricorrere all’ottimizzazione Bayesiana standard, anche
se alto-dimensionale.
E il risultato della distilation può essere usato come punto di partenza per la coreset selection: come coreset
candidato si prendono i punti esistenti più vicini ai prototipi prodotti dalla distillation. Poi si può raffinare con
altri metodi.

Quindi abbiamo un approcccio semplice e possiamo cominciare a sperimentare. L'articolo sarà "A bayesian optimization
approach to dataset distillation". Ho controllato in letteratura e non ho trovato niente che applichi la BO al
problema, quindi possiamo procedere con fiducia.

-----

Un punto importante: le immagini vivono in R^{NxD} dove N è il numero degli esempi e D il numero delle feature. Ho
letto il papiro fondantivo della distillation (Wang, Zhu, Torralba, Efron, Dataset Distillation) e in effetti loro
osservano che se le feature stanno in R^D e quindi le soluzioni vivono in R^{NxD}, si può stimare il gradiente della
loss rispetto allo spazio in questione e quindi si possono applicare metodi standard di gradient based optimization.
Loro non dedicano nemmeno una riga in più al problema e si concentrano sul ridurre il costo dell'inner loop di training.

Quindi con le immagini la Bayesian Optimization è probabilimente come ammazzare la mosca col cannone. Però tanto che
ci siamo in confronto con un metodo di libreria (es. Wang et al.) lo possiamo fare: i metodi gradient based effettuano
prevalentemente expoitation, invece la BO introduce l'esploration e questo può avere qualche vantaggio.

Però il passo notevole sarebbe applicarla a problemi con un spazio di feature categoriche. Come capita per la model
selection and hyperparameter optimization in AutoML. Lì si che avremmo ragione ad invocare questo approccio!

----

Distanza categorica: 