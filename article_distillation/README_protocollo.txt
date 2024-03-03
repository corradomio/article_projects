-----------------------------------------------------------------------------
- Protocollo v2
-----------------------------------------------------------------------------

Protocollo da applicare a dataset contenenti valori categorici.
Sono usati 2 dataset:

    mushroom:       SOLO categorici
    census_income:  categorici E numerici


--
Poiche' l'obbiettivo e' generare un dataset di valori sintetici,
NON SERVE suddividere i dataset originale in train/test!

Il protocollo e' il seguente:

    1) usando TUTTO il dataset originale, fare il training di un classificatore
       che chiameremo GTC (Ground Truth Classifier)

    2) generare D punti le cui coordinate sono scelte in questo modo

        - coordinata categorica: valore categorico scelto random con
          distrib uniforme
        - cordinata numerica:    valore random con distrib uniforme
          in range [min, max]

    3) usare il GTC per assegnare la label ai D punti

    4) addestrare un classificatore con i D punti, che chiameremo DC
       (Distilled Classifier)

    5) usando DC, predire le label del dataset originale

    6) calcolare l'accuracy della predizione.
       Questo e' il valore da ottimizzare:

            MASSIMIZZARE l'accuracy


I parametri dell'ottimizzatore solo le coordinate dei D punti.
Se ci sono M coordinate, il numero di parametri dell'ottimizzatore sono

    D*M

L'ottimizzatore deve poter ottenere un certo numero di parametri (diciamo K)
generati in modo random, poiche' sono i punti dello spazio D*M da cui partire.
Per fare questo gli servono le informazioni su quale possa essere il range di
valori da cui ottenere i valori iniziali. La libreria supporta al minimo:

    1) list[str|int]: valore categorico
    2) (min, max)   : valore continuo





-----------------------------------------------------------------------------
- Protocollo v2
-----------------------------------------------------------------------------

WARNING: vecchio protocollo usato con il dataset

    Cubo di Rubik in N dimensioni

-----------------------------------------------------------------------------

Protocollo da applicare a dataset contenente SOLO valori numerici in range
predefinito. In questo caso [0,1].


Supponiamo un problema di classificazione binaria (con valori {0,1})

Sia X una matrice NxM con valori reali in [0,1]. 
    Per dare dei numeri, supponiamo: N=100, M=2
    Uso M=2 per essere sicuro di ragionare in termini di "matrici" e non di vettori.
    In questo modo, se uso M=1 o M=1000, il ragionamento non cambia.

Sia y un vettore di N interi in {0,1} (le due categorie)

1) addestro GTC, il classificatore Ground Truth con (X, y)

Ora, supponiamo di voler trovare K punti, diciamo K=10, da usare per addestrare
un classificatore DC, dello stesso tipo di GTC. Gli "iperparametri" di DC sono
questi K punti, ma poiche' ogni punto e' composto da M dimensioni, il numero
totale di iperparametri di DC e' K*M. In questo caso 10*2=20.

Quindi, posso considerare, come spazio degli iperparametri di DC lo spazio [0,1]^K*M

Il protocollo da seguire e' il seguente:

1) genero un punto in [0,1]^KxM: 'Pd'

2) converto 'Pd' in una matrice KxM, cioe' in K punti in [0,1]^M, che chiamo 'Xd'

3) uso GTC per assegnare le etichette a questi K punti: 'yd'
   A questo punto ho un "dataset distillato": (Xd, yd)

4) creo un nuovo DC e lo addestro con il "dataset distillato" (Xd, yd)

5) applico DC a 'X' per ottenere 'yp', le predizioni sulle etichette fornite da DC
   sul dataset originario 'X'

6) calcolo l'accuracy: 'Ad=accuracy(y, yp)'.
   A questo punto abbiamo il 'punto' da usare con l'ottimizzatore bayesiano: (Xd, Ad)
   (piu' precisamente:  (Pd, Ad))

7) aggiorno l'ottimizzatore con il nuovo punto (Xd, Ad)

8) chiedo all'ottimizzatore di propormi un'altro punto, 'Pd_prossimo'

9) se ho raggiunto la condizione di termine elaborazione, STOP,
   altrimenti uso 'Pd_prossimo' come 'Pd' e goto 2)


