software decomposition (~colorazione grafo)
    rileggere l'articolo/revisione testo/ultime sezioni
    controllare il body dell'articolo
    dalla 3 sezione in poi

    IEEE Transactions on Services Computing (TSC) 
    bio/biography


-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------

Special Issue: articolo nuovo
    elaborazione dati
    tesi corrado, ridurre la parte di ricerca, sottometterla
    multi view learning/co-training


preparazione del dato
    data distillitaion (classificazione!/regressione?)
        data pruning    tenendosi solo la frontiera
        data creation   per creare i dati mancanti

-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------

GUO Dataset pruning

approccio tipo active learning
bayesian optimization/gausian processes

generazione dati variazionale

-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------

statistica non parametrica
    All of Statistics - 2004
    All of Nonparametric Statistics - 2006

regressione non parametrica/kernel density

target learning: framework machine learning per fare delle stime
    influence functions

-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------
A Tutorial on Bayesian Optimization
https://arxiv.org/pdf/1807.02811.pdf

Gaussian Processes for Machine Learning
https://gaussianprocess.org/gpml/chapters/RW.pdf

Bayesian Optimization 
https://bayesoptbook.com/book/bayesoptbook.pdf


-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------
Uso di embedding per lavorare su un ogetto discreto
L'emedding converte lo spazio discreto in uno continuo
Ottimizzaizone boolean sul continuo
Inverso sul discreto


dataset
decision tree
ridurre il dataset
dataset -> array di 128 variabili booleane


datas pruning (data distillation e' un supercaso)


-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------
discreto -> continuo
    proiezioni casuali (modelli lineari)
    linear embedding
    random projection embedding
    bayesian optimization random embedding


ottimizazione bayesiana su reticoli combinatori
    ottimo del livello 10 su 100

metodo COMBO



-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------
core set selection (survey??)
    dipende dal task

POI distillation (survey??)

-----------------------------------------------------------------------------
- Discussione articolo
-----------------------------------------------------------------------------

Ricavare il coreset
Distillation: generare artefatti che aiutano l'algoritmo

1) Dimesnionality Reduction

2) mossa magica: Decision Tree per classificazione

    Riprodure: tasselazione dell'albero

3) scegliere dei punti che ricostruisce l'albero originale

    Mondrian: suddivisione 2D -> nD

    
KNN: punti prototipi (centroidi)

Discriminant Analisys

Come aiutare la BayesOpt usando euristiche che indicano quali zone sono piu' interessanti


-----------------------------------------------------------------------------

dataset R[n,d], B -> R[n*d] x B[n]    e' un punto singolo in R[n*d] con n etichette


generazione dataset:
    decidere il numero di dimensioni
    per ogni dimensione decidere il numero di suddivisioni (random [1,R])
    generare suddivisioni random in range [0,1] (o [-1, 1])
    assegnare etichette random ad ogni intersezione, oppure in modo che 
        "smart" in modo da non avere 2 celle adiacenti con lo stesso colore
        numero passi di suddivisioni DISPARI!
    generare dataset con categorie bilanciate

distillazione (NO coreset) -> coreset dai punti piu' vicini dopo la distillazione


modello pangloss (alternativa al federated learning -> centralized learning)


servono 2 classificatori (logistico):
    uno sul GT
    uno sul dataset distillato


budget numero di tentativi
accuracy non migliora piu' di tot


ottimizzazione 1 dimensionale


svm, mistura di gaussiane, polinomio
    modelli SEMPLICI (O(n))


acquisition function: come implementarla?
    modello: class dei polinomi


Expected improvement: che cosa ottimizzare con BayesOpt
Botorch: Bayesian Optimization Torch
--------------------------------

ottimizzazione bayesiana:
       min/max f(z)
         z in Z



dataset: 100x2
dataset:  10x2
classificazione: 2 categorie
    classificatore logistico
        metrica: accuracy


------------------------------------------------
aumento dimensione dataset e dimensioni
    100_000/10_000/1000
    100
    ..

dataset a dimensione piu' bassa rispetto

-----------------------------------------------------------------------------
- Whatsapp
-----------------------------------------------------------------------------

Una proiezione lineare da fare (da piccolo a grande) e' semplice: basta generare una matrice riempita dei numeri random.

MA (domandona): esiste un modo SEMPLICE per fare qualche proiezione NON lineare (anche se la non linearita' e di tipo "semplice")?

L'idea che mi era venuta e' quella di generare dei "segmenti di curva" (nello spazio grande), una per ogni asse dello spazio piccolo, e poi usare un po' di interpolazione per traformare le coordinate dei punti dallo spazio piccolo a quello grande.
Ma mi pare arzigogolato.
E poi SOLO se ne vale la pena.
Penso che una trasformazione lineare, per iniziare, dovrebbe essere gia' abbastanza



UNO
Supponiamo di avere tre classi R,G e B, giusto per fare un po’ di colore.
Le associo a tre segmenti che coprono l’intervallo [0,1] dell’asse t. Esiste una funzione semplice che mappa t in f(t) (la classe/colore). Adesso aggiungo una dimensione non informativa z, così che f(t,z)=f(t).

Dal punto di vista di un algoritmo come Isomap, t-sne, UMAP, questa roba può essere riproiettata su dimensionalità 1D facilmente. [questo è il ritornello 1D]

DUE
Adesso riparto da zero e adotto una variante, con trasformazione di coordinate e produco uno swiss-roll. Parto da una funzione semplice che mappa t in f(t) (la classe/colore).
Prendo la dimensione lineare t (su cui generare i punti), e passo a coordinate radiali con r=r(t) e theta=theta(t); esempio semplice r=t e theta=t, così da avere una curva parametrica nel piano descritta da (t cos t, t sin t). Poi aggiungo la dimensione non informativa z, così da avere (t cos t, t sin t, z) e infine, magari ruoto di due angoli a caso tutta la baracca.
Le dimensioni dello spazio sono 3, quelle della varietà sono 2, ma quella informativa è una sola.
[ritornello 1D]

TRE
Adesso prendo, scrivendo un po’ a caso, (t cos t, t sin t, t^2 cos 3t, t^3 sin 2t, z1, z2, z3), con z1, z2 e z3 non informative. E’ chiaro che la dimensione informativa è una sola. [Ritornello 1D]

QUATTRO
Aggiungo altri due parametri u e v che si combinano tra di loro e con t, in modo che esista f(t,u,v) con tutte e tre le dimensioni informative (esempio f(t,u,v)=parity(ceiling(10 t) ceiling(10 u) ceiling(10 v)))
Adesso combino le cose un po’ a caso
(v sin t cos u, t sin v sin u, v u cos 3 t, t^3 sin 2t, z1, z2, z3), con le z non informative
E’ chiaro che le dimensioni effettive sono 3 [ritornello 3D]

INSOMMA
Basta decidere le dimensionalità k della manifold effettivamente informativa (poi decidere la f() di k variabili), combinare le k variabili con delle funzioni continue e finite ad esempio trigonometriche, e aggiungere delle dimensioni non informative. Poi se uno vuole può anche ruotare nello spazio alto dimensionale. Quel che salta fuori è una roba sufficientemente complessa. E può anche peggiorare facilmente complicando f().


