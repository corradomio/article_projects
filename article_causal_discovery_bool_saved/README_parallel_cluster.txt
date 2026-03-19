Purtroppo ogni cluster ha la sua modalita' di configurazione.

Le simulazioni possono essere partizionate a livello di algoritmo e dataset.
Ogni simulazione puo' essere configurata per processare i grafi all'interno 
dello stesso dataset in parallelo in base al numero di thread disponibili 
sulla macchina/nodo.

Al momento, il parallelismo e' pre configurato nel file "2_causal_discovery.py"
variabile globale "N_JOBS".

Il valore di default e' "auto", vuol dire che vengono usati tutti i thread visibili
da Python.


