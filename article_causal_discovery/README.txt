Protocollo
----------

    1) creare i DAG

        'gen_digraphs.py'

    2) generare i datasets

        'gen_datasets.py'

    3) processare i dataset

        'process_datasets.py'


Grafi generati
--------------

    Vengono generati TUTTI i DAG di ordine 2,3,4,5
    Vengono generati 1000 istanze per DAG con

        - ordine 10, 15, 20, 25
        - densita' 10%, 15%, 20%

    Elenco di coppie (n nody, n archi)

        2     (2,  1),         1      1
        3     (3,  2),         3      4
              (3,  3),         1
        4     (4,  3),        16     38
              (4,  4),        15
              (4,  5),         6
              (4,  6),         1
        5     (5,  4),       125    728
              (5,  5),       222
              (5,  6),       205
              (5,  7),       120
              (5,  8),        45
              (5,  9),        10
              (5, 10),         1
       10    (10, 10),      1000
             (10, 15),      1000
             (10, 20),      1000
       15    (15, 23),      1000
             (15, 34),      1000
             (15, 45),      1000
       20    (20, 40),      1000
             (20, 60),      1000
             (20, 80),      1000
       25    (25, 63),      1000    (excluded)
             (25, 94),      1000
             (25, 125)      1000


Configurazione algoritmi
------------------------

    Gli algoritmi da utilizzare sono specificati nel file JSON

        data/graphs-algorithms.json

    La definizione di un algoritmo puo' avere le seguenti forme:

        "algo_name": {
            "class": "fully-qualified-class-name",
            "param_name": <value>,
            ...
        }

    oppure

        "algo_name": {
            "class": "fully-qualified-class-name",
            "graph-order": {
                "param_name": <value>,
            }
            ...
        }

    se la configurazione dipende dall'ordine del grafo (numero di vertici/nodi)


Errori nelle predizioni
-----------------------

    Puo' caputare che un algoritmo fallisca nella fase di training e quindi non sia
    in grado di generare il grafo causale. In questo caso la matrice di adiacenza
    sara' quella di un grafo senza archi. In questo modo la matrice puo' ancora
    essere usata per le statistiche, visto che e' rappresenta un possibile grafo
    quando un algo, funzionante correttamente, non e' in grado di identificare
    le dipendenze



Python da usare
---------------

    E' stato usato Python 3.10 e le librerie
    specificate in "requirements.txt" installate
    usando 'pip'

        pip install -r requirements.txt


    Nota: conviene create un environment dedicato
    onde evitare conflitti con librerie gia' installate

        conda create -n <env_name> python=3.10
        conda activate <env_name>
        pip install -r requirements.txt
        conda deactivate


Parallelismo
------------

    Al momento l'implementazione richiede

        500MB + n*300MB

    con n il parallelismo


NN usata
--------

    Per evitare conflitti ed incompatibilita', conviene usare
    Pytorch ultima versione.

    In teoria si potrebbe usare 'mindspore' ma sembra sia disponibile
    SOLO per Python 3.7 e 3.8.
    La compatibilita' con Python 3.10 non e' conosciuta.
    Nota: 'mindspore' e' Cinese!

