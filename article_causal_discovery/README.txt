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


NN usata
--------

Per evitare conflitti ed incompatibilita', conviene usare
Pytorch ultima versione.

In teoria si potrebbe usare 'mindspore' ma sembra sia disponibile
SOLO per Python 3.7 e 3.8. 
La compatibilita' con Python 3.10 non e' conosciuta.
Nota: 'mindspore' e' Cinese!

