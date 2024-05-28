Sto cercando di preparare i dataset per l'articolo sulla causal inference.
Siore e Siori, venghino, venghino 😉
Sto cercando di preparare i dataset per l'articolo sulla Causal Inference.
In un DAG possiamo suddividere i nodi in 3 gruppi:
1) nodi con "indegree = 0 & outdegree > 0": corrispondenti alle "cause" INIZIALI
2) nodi con "indegree > 0 & outdegree > 0": sono i nodi interni del DAG
3) nodi con "indegree > 0 & outdegree = 0": che corrispondono ai "sintomi"/"effetti" TERMINALI
Ora si pone il seguente problemino:
un DAG ha almeno UN nodo di tipo 3), MA ne puo' avere PIU' di 1.
C'e' da decidere la "filosofia" da seguire:

1) ci interessano SOLO DAG con un SINGOLO sintomo/effetto, nel qual caso bisogna modificare il DAG
   per ottenre un'UNICO nodo di tipo 3). Ci sono diverse strategie possibili
   1.1) SCEGLIAMO uno dei nodi di tipo 3) e aggiungiamo archi per collegare gli altri nodi di tipo 3) a questo nodo
   1.2) AGGIUNGIAMO UN nodo ed aggiungiamo un arco dai nodi di tipo 3) al nodo appena aggiunto
   1.3) ci sarebbero altre soluzioni, ma mi sembrano inferiori alla 1.1) e 1.2).

2) manteniamo il DAG cosi' come e' con la semantica che ogni nodo di tipo 3) rappresenta un possibile
   "sintomo"/"effetto"

3) manca un'informazione' che sto cercando di ricuperare: ma gli algo di causal discoveri SONO IN GRADO di
   gestire "effetti multipli" (nodi di tipo 3) multipli)?

--

Altro problema: come generare il DAG: Gabriele proponeva grafo "scale free"

https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model

Comunque abbiamo troppi pochi nodi (da 10 a 25 con passi di 5) per poter differenziare un DAG scale-free da un DAG generato 
in modo casuale.

--

La libreria Python "NetworkX" mette a disposizione un po' di generatori per grafi diretti:

https://networkx.org/documentation/stable/reference/generators.html

Ma sono "grafi diretti" NON DAG.

--

Altra nota: affinche' un DAG sia connesso, serve almeno il 10% di archi.
Risultato "spannometrico" ottenuto nel seguente modo:
scelto il numero di nodi e di archi, genera il numero di archi richiesto e poi, se il DAG non e' ancora connesso, aggiungi 
tutti gli archi che mancano (SEMPRE in modo casuale). Da cui il risultato.
Quindi, come densita'/n di archi, direi: 10%, 15%, 20%.

--

Altro "trick" che potrebbe essere utile: poiche' ci servono DAG con diverse densita', invece di create DAG nuovi ogni volta, 
iniziamo con un DAG  a densita' piu' bassa, quindi aggiungiamo archi per ottenere le versioni a densita' piu' alte.
"Filosoficamente" questo vuol dire aggiungere "conoscenza" relativa alle "interazioni" tra i vari nodi/cause.

-- GG

Un numero di effetti-foglia maggiore di 1 non è un problema per gli algoritmi di discovery. Potremo considerare questo problema 
quando ci occuperemo di una discovery "targeted": in tal caso ci assicureremo che il target sia 1 solo. Per il momento lasciamo 
così.

ero. Non sono particolarmente affezionato agli scale-free, era una scelta arbitraria. Si potrebbe tentare una qualcune altra 
strategia difendibile in sede di articolo: ad esempio un dataset esaustivo fino al grado 7 (non si può andare molto oltre 
come mostro qua sotto), un altro dataset in cui si procede con un campionamento con construzione casuale da definire.
Ecco il numero di DAG in funzione del numero n dei nodi:

n = 2: 3
n = 3: 25
n = 4: 543
n = 5: 29281
n = 6: 3781503
n = 7: 113877926851
n = 8: 7510654605845232
n = 9: 1056630960571740802332
n = 10: 316593438211472706535203030
n = 11: 205377073056618514177463299145434
n = 12: 278236967409015780533635983997929348303
n = 13: 793538065213725634779008184537474996562540901
n = 14: 4701447390671871963178435210702994787992541324594307
n = 15: 57921768121822096678038306644501599301557082677644045823881
n = 16: 1451412402626730340113958100163365771162076061518725111368573224242
n = 17: 75841495710823473995128929984133354044688199761461394713993998422012964784
n = 18: 8039848291544052397725723581682972246865540884101354442402452727750926268687526515
n = 19: 1758820024846273398058086313660855157483092592724024374963241235193180381102498679312106189
n = 20: 7914255217356239256991949049805488362024532569630992245766425439218461616565029137303443137746412

direi che 7 è già una bella botta: ma probabilmente tutti gli algoritmi si comportano abbastanza bene per n bassi...

Concordo: le densità diverse vengono esaminate separatamente, questo trucco si può difendere.

--
n = 7: 113_877_926_851
114 MILIARDI di grafi mi sembrano un po' tantini 🤣
n = 6: 3_781_503
con grado 6 abbiamo circa 4 MILIONI di grafi 🤣

-- GG

in lire turche 114 miliardi non sono tanti: ci compri un caffé 😉

va bene il 6: per le numerosità di nodi maggiori occorre trovare una strategia difendibile, che non introduca troppo bias


--

Comunque bella domanda: come generare TUTTI i DAG di un certo ordine?
(Non e' una domanda per te a meno che tu non sappia gia' la risposta 😊)
A pensarci bene, dovrebbe bastare giocare con le permutazioni (piu' qualche constraint aggiuntivo)
Cerco un po', e magari mi invento qualcosa.

-- GG

Anche i 3 milioni, non bisogna provarli tutti: se si trova un metodo per la generazione sitematica si puo scegliere se studiarne 
uno lancianod una moneta, ad esempio con 1/1000 di probabilità di dire OK.

--

Un generatore random di dag non e' difficile da implementare. Gia implementato, ovviamente
Il problema potrebbe essere generare 2 DAG isomorfi, MA pescarne 2 isomorfi bisogna avere una gran sfortuna. A partire da 
n=6

-- GG

con n nodi c'è una matrice di adiacenza con  n(n-1) elementi (la diagonale la mettiamo a 0, così non ci sono self-loop), 
questi elementi possono essere accesi o spenti quindi ci sono 2^{n(n-1)} matrici di adiacenza, possiamo imporre facilmente 
il vincolo che non ci siano 2-loops (a connesso a b che è connesso ad a). Si può generare a caso una matrice di questo tipo, 
poi controllare che sia un DAG (c'è il trucchetto di esponenziare la matrice e vedere se gli elementi sulla diagonale sono 
tutti 0: se c'è un 1 c'è un loop). Che ne dici?

--

Infatti il generatore che ho implementato si basa esattamente su questo meccanismo!
Basta che la matrice sia triangolare (superiore o inferiore non fa differenza) e la diagonale sia tutta a ZERO

-- GG

Ottimo l'uso del meccanismo.

--

Ora se "linearizziamo" la parte triangolare e "mescoliamo" il vettore cosi' generato, dovremmo ottenere un nuovo DAG.
Il vincolo dovrebbe essere: non ci possono esser righe o colonne tutte a zero


-- GG

Non sono sicuro che le matrici triangolari coprano tutti i dag. Ma forse partendo da una triangolare si possono ricavare 
tutti i dag scambiando di posto gli elementi simmetrici rispetto alla diagonale... (ma ho una visibilità limitata sulla tua 
procedura, quindi ci avrai già pensato)

c'è comunque da esponenziare sempre per assicurarsi dellla dagness

meglio:

siccome stiamo parlando di 6 nodi o 7 nodi o giù di lì... se l'esponenziazione costa troppo si può semplicemente elevare 
la matrice alla 6, alla 7 o alla n

--

Partendo da un DAG si puo' sempre ottenere un ordine topologico dal che e' sempre possibile riordinare i nodi (ed assegnare 
loro un indice) in modo che ogni nodo sia connesso SOLO a nodi con "indice" piu' alto, da cui la matrice traingolare dovrebbe 
coprire tutti i dag

-- GG

mi sembra ottimo

Ho ottenuto una interessante risposta da ScholarGPT, della quale rporto solo la parte più semplice (le altre parti ricorrono 
a Markov Chain Monte Carlo

--

Ed io l'ho chiesto a "gemini" 😊
Esistono diversi algoritmi per generare tutti i grafi diretti aciclici (DAG) di un certo ordine. Ecco alcuni dei metodi più 
comuni:

1. Algoritmo basato sulla ricerca in ampiezza (BFS):

    Inizia con un grafo vuoto contenente solo i nodi.
    Per ogni nodo nell'ordine dato, esplora tutti i nodi non ancora visitati adiacenti ad esso e aggiungi un arco diretto 
dal nodo corrente a ciascun nodo non ancora visitato.
    Aggiorna l'ordine dei nodi in modo che i nodi già visitati siano sempre prima dei nodi non ancora visitati.
    Ripeti i passaggi 2 e 3 finché tutti i nodi non sono stati visitati.

2. Algoritmo basato sul backtracking:

    Inizia con un grafo vuoto contenente solo i nodi.
    Per ogni nodo nell'ordine dato, considera tutte le possibili combinazioni di archi diretti uscenti da esso.
    Per ogni combinazione di archi, crea un nuovo grafo che include i nodi e gli archi selezionati.
    Controlla se il nuovo grafo è aciclico. Se lo è, aggiungilo all'insieme dei grafi DAG generati.
    Esegui il backtracking per esplorare tutte le possibili combinazioni di archi per i nodi rimanenti.

3. Algoritmo basato sulla programmazione dinamica:

    Definisci una matrice di sottoproblemi in cui ogni cella rappresenta il numero di DAG con un sottoinsieme di nodi specificato.

    Riempi la matrice ricorsivamente usando i risultati dei sottoproblemi più piccoli.
    Traccia i percorsi nella matrice per ricostruire tutti i DAG possibili.

La scelta dell'algoritmo migliore dipende da diversi fattori, come il numero di nodi nel grafo e le risorse computazionali 
disponibili. In generale, l'algoritmo BFS è più semplice da implementare, mentre l'algoritmo basato sulla programmazione 
dinamica può essere più efficiente per grafi di grandi dimensioni.

Oltre agli algoritmi sopra menzionati, esistono diverse librerie e strumenti software che possono essere utilizzati per generare 
DAG. Questi strumenti possono offrire implementazioni ottimizzate degli algoritmi o funzionalità aggiuntive come la visualizzazione 
dei grafi o il calcolo di metriche specifiche.

Ecco alcune risorse che potrebbero esserti utili:

    https://pt.wikipedia.org/wiki/D%C3%ADgrafo: fornisce una definizione formale di DAG e le sue proprietà.
    https://it.wikipedia.org/wiki/Digrafo_aciclico: presenta un algoritmo basato sul DFS per l'ordinamento topologico di 
DAG.
    https://www.binance.com/en/square/post/447326: spiega l'utilizzo dei DAG nel contesto delle criptovalute.

Fammi sapere se hai altre domande o se desideri approfondire un aspetto specifico della generazione di DAG.

-- GG

Ottimo. Per completare la cosa (messa da parte) dello MCMC ecco un articolo:
https://link.springer.com/article/10.1007/s11222-013-9428-y

Tutto ciò è già un bel related work per la parte di generazione :) - io ti direi di scegliere l'approccio più semplice e 
di prendere nota degli altri per usarli successivamente come alternativa.

--

I numeri che hai indicato includono anche gli isomorfismi, cioe' grafi diversi SOLO per il riodino dei vertici.
Ad esempio: {1->2->3} e' isomorfo a {2->3->1},  {3->1->2}, {3->2->1}, {2->1->3}, {1->3->2}
Se escludiamo gli isomorfismi, il numero di dag sono nell'ordine di O(n^2).
Poiche il dataset lo generiamo noi, va da se che gli isomorfismi non ci interessano 😊.
Quindi mooooolti di meno, quindi si possono testare TUTTI i grafi diretti fino ad un ordine bello alto. 😊

-- GG

se si riesce a trovare un sistema per generare evitando gli isomorfi allora O(n^2) è bellissimo...

--
Certo che c'e':
https://en.wikipedia.org/wiki/Combinatorial_number_system
usato per la tesi (non si butta via niente 😊)

--

BugFix: il numero di DAG NON E' O(n^2) (troppa grazia Sant' Antonio 😊)
MA O(2^(n^2)).
In pratica tutti i modi di riempire la matrice triangolare superiore (o inferirore) di una matriice n*n, escludendo la diagonale, 
cioe' n*(n-1)/2 celle!

-- JL

Un altro garantito non specifico di pesce è il Libanese Flower. A metà tra Al Wahda Mall e Burjeel Hospital

-- GG

L'articolo che ho indicato ieri è veramente scritto bene. Chiarisce tra l'altro perché gli approcci basati sul sampling da 
matrici triangolari non siano uniformi sullo spazio dei DAG. Gli stessi autori, se si vanno a vedere le citazioni su scholar, 
hanno prodotto articoli successivi sullo stesso tema, e anche un package R per la scoperta di grafi bayesiani, che dovrebbe 
incorporare anche programmi per la generazione uniforme (basata su MCMC). Stamattina non ho tempo di guardarci dentro, ma 
il link è questo
https://cran.r-project.org/web/packages/BiDAG/index.html

resta il fatto che per il primo giro noi siamo interessati a generare dei grafi un po' come viene, tanto per mettere alla 
prova l'intera pipeline: successivamente useremo una generazione uniforme (se no i reviewer ci chiederanno comunque di farlo...):


--

Aggiornamento: Ho generato i seguenti grafi:
per gli ordini 2,3,4,5, TUTTI i possibili DAG, che sono rispettivamente 1, 4, 38, 728
per gli ordini 10, 15, 20, 25 ho generato 1000 DAG random per 3 diverse densita: 10%, 15%, 20%, quindi un totale di 3000 
grafi per ogno ordine.
Per essere sicuro di non generare grafi isomorfi, ho usato il "weisfeiler lehman test" il quale dice:
SE due grafi hanno il WL hash diverso, NON SONO sicuramente isomorfi, ma se hanno lo stesso hash, POTREBBERO essere isomorfi. 
Mi sono assicurato di avere WL hash TUTTI DIVERSI.
.
Passo succesivo: generare i dataset.
Direi di generare 1000 dataset diversi per ogni grafo e per ogni combinazione dei parametri messi a disposizione dalla classe 
IIDSimulation di "gcastle".
.
Ora serve un contenitore abbastanza flessibile per gestire tutti questi dataset: ad esempio HDF

-- GG

grazie, ottimo: non sapevo del "weisfeiler lehman test", c'è davvero un test per qualunque cosa!

tu suggerisci sia opportuno "generare 1000 dataset diversi per ogni grafo e combinazione di parametri",
- intendi "record"? se è così teniamo presente che la numerosità del campione è un parametro della sperimentazione, quindi 
si possono generare diverse numerosità (magari sottoinsiemi di un campione di dimensione massima);
- se davveero intendi "dataset" questo fattore mille è piuttosto impegnativo: forse è meglio spenderlo in modo diverso, magari 
campionando più grafi o più combinazioni di parametri.
che ne dici?

--

No, proprio DATASET. OGNI dataset con 1000 record, per ora.
Si puo' fare, 😊
Comunque al momento sto scrivendo il codice.
Per esempi invece di 1000 dataset magari ne bastano 10.
visto che abbiamo 10 modi diversi di generarli (usando IIDSimulator con tutte le accopiate "method/sym_type"). Quindi gia' 
cosi avremmo 100 dataset per ogni grafo.
Poi si potrebbero anche creare "dataset ibridi" combinando dataset generati in modo diverso

Buona idea!
INVECE di create 10 dataset da 1000 record, un UNICO dataset da 10000 record che poi si possono scegliere in un infinita 
di modi diversi.
Si, mi piace!  👍


-- GG

Credo di aver capito. Le numerosità sono una caratteristica su cui è importante intenderci. Possiamo convenire di parlare 
di grafo e di set-up separatamente. Dunque
- per una data numerosità ( N ) di nodi e archi (E) abbiamo G grafi;
- per ogni grafo g abbiamo S set-up diversi (poi ritorno su questo)
- per ogni set-up s abbiamo un certo numero D di dataset (tu suggerivi inzialmente D=10)
- per ogni dataset d abbiamo R record
quindi complessivamente per ogni numerosità di nodi ed archi (N,E) abbiamo GxSxD dataset
qui finisce la generazione, ma poi c'è la ricostruzione
Sentiti libero di modificare il testo qui sopra per segnalarmi dove eventualmente mi sono sbagliato
(scusa se cito Woitila: "Se mi sbalgio mi corrigerete!" ;-) )

Ritorno sul numero dei setup: per ogni nodo senza genitori c'è una distribuzione di probabilità, per ogni arco diretto c'è 
una dipendenza funzionale, per ogni collider c'è un aggregatore... direi che il combinatorio è enorme, anche per un grafo 
prefissato  di tre nodi a forma di collider...

--

"""
- per ogni set-up s abbiamo un certo numero D di dataset (tu suggerivi inzialmente D=10)
- per ogni dataset d abbiamo R record
"""
invece di D=10 dataset distinti, ne ho fatto SOLO UNO (D=1), invece di soli R=1000 record, di R=10_000 (praticamente come 
aver concatenato i 10 dataset di cui sopra). Non vale la pena generarne piu' di uno.
Per il resto, tutto giusto

-- GG

Perfetto! D=1 è il massimo per il green computing 😉

--

per grafi di ordine 2,3,4,5 ho generato TUTTI i DAG (escludendo gli isomorfismi).
per grafi di ordine 10, 15, 20, 25  ho usato 3 densita (10%, 15%, 20%) e per ogni coppia (rodine/densita) ho generato 1000 
DAG random.
Il WL test assicura di non aver generato DAG isomorfi
---
Sul WL test c'e' da dire questo: l'implementazione messa a disposizione da networkx funziona su grafi NON DIRETTI.
Ed infatti, per i DAG di ordine 2,3,4,5 ci sono casi in cui il WL test genera lo stesso valore di hash

-- GG

Non vedo grossi problemi sulla qualità del campionamento dei DAG, anzi!. Sicuramente la parte N=2-5 affrontata in modo esaustivo 
garantisce uniformità del sampling. Per l'altra, N>5, possiamo accontentarci di buone approssimazioni (e questa mi sembra 
già molto buona), per mettere in piedi la pipieline. Se presto o tardi decidiamo che non siamo contenti, possiamo ricorrere 
ad algoritmi già pronti basati su MCMC, così da essere completamente tranquilli su questo aspetto (e spazzare via i dubbi 
dei revisori).

-- JL

Una osservazione ovvia ma che almeno mi serve per double-check di aver compreso: noi non dovremo sostenere di fare inferenza 
sulla popolazione di tutti i DAG di ordine N, ma che fissato un certo DAG (o una classe di DAG isomorfi) facciamo inferenza 
usando un campione (di osservazioni multivariate) da quella rete bayesiana oppure inferenza da molti campioni. Ho capito 
bene?

-- GG

Si

-- GG

Un pensiero a posteriori circa la variabilità dei set-up: in quel caso non esiste una collezione di scelte "uniforme" in 
senso assoluto. La loro arbitrarietà inelimnabile: ad esempio se si ha un grafo che consiste solo in un collider con nodi 
A, B e C (quest'ultimo è il nodo collider) non è possibile scegliere "uniformemente" tra le coppie di distribuzioni di probabilità 
da assegnare ad A e B, ne tra le coppie di funzioni da assegnare ad A-C, e B->C, ne tra le funzioni di aggregazione sul nodo 
C.
Le nostre scelte per ognuna di queste caratteristiche sarà tra un set di distribuzioni e di funzioni arbitrario, ad esempio 
sceglieremo uniformemente dal set di funzioni che ci offre il package. Quindi i nostri risultati numerici saranno validi 
"in relazione al package".

Ci stiamo affannando a trovare una scelta uniforme sui grafi, ma l'arbitrarietà dei set-up rischia di pesare molto di più. 
Per protegger le nostre conclusioni da questa dipendenza dovremo tener traccia delle diverse scelte ( il mix arbitrario di 
parametri di set-up ) e "stratificare" le nostre conclusioni: se la dipendenza dallo "strato" delle conclusioni sarà bassa, 
allora potremo dichiarare che dipendono debolmente dal mix dei diversi set-up. Questo però lo faremo a posteriori, fase d'analisi.


So che ho scritto una cosa ovvia, ma volevo fissare la considerazione.

-- JL

Grazie per il chiarimento, ho capito. Anche a me sembra che l'uniformità nel sampling della struttura DAG sia poi cagionevole 
di configurazioni di distribuzione che deviano completamente da una uniformità o altra distribuzione ideale. Ma in letteratura 
di modelli grafici infatti, se non erro, ogni volta che si fanno delle scelte di distribuzione condizionata sui parents o 
di tipologia di variabili si dà un nome diverso al modello

I bayesiani penso studiano sempre casi molto particolari e precisi, mentre noi vogliamo mettere insieme tanti modelli in 
un pool