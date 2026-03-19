Grafi
-----

    I DAG, usati per generare i dataset, sono salvati in file JSON aveti la seguente struttura:
    
    {
        "date": <data di creazione del file>,
        "graphs": {
            <graph order>: {
                <weisfeiler_lehman_graph_hash>: {
                    "n": <graph order/n of nodes>,
                    "m": <graph size/n of edges>,
                    "adjacency_matrix": <adiacency matrix 0/1>
                },
                ...
            }
        }
    }
    
    C'e' un livello di nidificazione inutile, ma presente per motivi storici.


Dataset
-------
    Il nome del file ha la seguente struttura:
    
        dataset-<graph order>-<dataset id>.hdf5
        
    <dataset id> e' presente perche' in origine c'erano piu' dataset per lo stesso ordine dei grafi.
    

    I dataset sono memorizzati in file HDF5, aventi una struttura simile al formato JSON:
        
        <graph order>/<weisfeiler_lehman_graph_hash>
            attributi:
                "n": <graph order/n of nodes>,
                "m": <graph size/n of edges>,
                "wl_hash": <weisfeiler_lehman_graph_hash>
                "adjacency_matrix": <adiacency matrix 0/1>
                "fun": <informazioni sulle funzioni booleane usate, in formato JSON, serializzato come stringa>
            datasets:
                "datase": (<n of instances>, <n of records>, <n of nodes>)


    Ogni "dataset" contiene 30 dataset generati a partire dallo stesso DAG (indicato da "wl_hash").
    Ogni istanza e' generata da una DIVERSA selezione di funzioni booleane.
    Il numero di record per ogni istanza e' 4000 per i grafi di ordine 5-8, 8000 per quelli di ordine 9, 16000 per quelli di ordine 10

    Le funzioni booleane usate sono memorizzate in "fun": e' una lista di 30 'oggetti' aventi la seguente struttura:
        
        {
            "instance": 0,
            "n": 4,
            "m": 4,
            "wl_hash": "09bc3a1379797b3871e3b04b468f0300",
            "nodes": {
                "0": {
                    "n": 0,
                    "f": "x0",
                    "params": [],
                    "noise_prob": 0.5
                },
                "1": {
                    "n": 1,
                    "f": "x1",
                    "params": [],
                    "noise_prob": 0.5
                },
                "2": {
                    "n": 2,
                    "f": "~x0 & ~x1",
                    "params": [0, 1],
                    "noisep": 0.20506474025638027,
                    "fx": "1"
                },
                "3": {
                    "n": 3,
                    "f": "x2 & ~x0",
                    "params": [0, 2],
                    "noisep": 0.2916061937581988,
                    "fx": "2"
                }
            }
        }
        
    dove:
        "instance": <intero in range [0,29]>
        "n": <graph order/n of nodes>,
        "m": <graph size/n of edges>,
        "wl_hash": <weisfeiler_lehman_graph_hash>
        "nodes": {
            <node id>: {
                "n": <node id>,
                "f": <boolean function used>,
                "params": <parent nodes>
                "noisep": <binomial distribution parameter>
                "fx": <hex representation of the function>
            }
        }

    "f" contiene la versione "simbolica" della funzione usata, dove "xi" si riferisce al nodo i-mo
    "params" contiene la lista dei nodi "parent" della funzione, cioe' i suoi parametri. 
        Se la la funzione non ha parametri, "params" e' la lista vuota.
        "fx", quando presente, rappresenta la funzione booleana codificata in esadecimale, con il bit
        meno significativo a sinistra.

    Nota: "noise_prob" e' la stessa cosa di "noisep". Dimenticanza storico.
    Nota: anche le duplicazioni sono "dimenticanze storiche".


Results
-------

    Il nome del file ha la seguente struttura:
    
        <algorithm>-<graph order>-<dataset id>-<thread id>.hdf5
        
    <thread id> e' l'ID del thread che ha processato il corrispondente dataset
    
    La struttura interna e' simile a quella dei dataset:
    
        <graph order>/<weisfeiler_lehman_graph_hash>
            attributi:
                "n": <graph order/n of nodes>,
                "m": <graph size/n of edges>,
                "wl_hash": <weisfeiler_lehman_graph_hash>
            datasets:
                "causal_matrices": (<1 + n of instances>, <n of nodes>, <n of nodes>)

    La PRIMA istanza e' la matrice di adiacenza GROUND TRUTH.
    Le altre, sono le matrici di adiacenza del dedotte dall'algoritmo sulle varie istanze del dataset.
    
    SE c'e' stato un errore di qualche tipo, la matrice di adiacenza e' composta solo da ZERO.
    
    Nota: BISOGNA controllare quante matrici nulle ci sono!
