Giusto come ipotetico protocollo "iniziale":

0) probabilmente vale la pena normalizzare tutti i numeri con media 0 e varianza 1
1) si addestra un classificatore con i dati disponibili
2) si addestra un Variational Autoencoder con solo  i dati genetici (a quanto vedo sono solo numeri)
3) si usa la parte di decoder per generare nuovi valori dei dati genetici usando un generatore di numeri casuali di tipo uniforme da dare in ingress al decoder
4) si usa il classificatore addestrato in 1) per generare le ettichette.


----

Le GAN richiedono molti dati.

I VAE, anche no: dato un INPUT, lo comprimi, poi lo decomprimi e cerchi la configurazione di parametri che MEGLIO rigenera I dati in input. Quindi, basta che c'e' ne siano un po'!

Diciamo che per generare dei dati che tra di loro hanno qualche "correlazione" un VAE dovrebbe essere quello che serve.

Fondamentalmente cerca un sottospazio continuo in cui le dimensioni sono indipendenti, che poi usa per ricostruire i dati originali. E poiche' le dimensioni originali sono PIU' del sottospazio, in qualche modo deve tener traccia delle "dipendenze".

Inoltre usare media/sdev fa si che che I pesi della rete non vadano a spasso in giro per l'universo, ma stiano vicini alla loro media (che poi e' la ragione per il quale sono I VAE stati inventati)

"A me mi sembra" che sia esattamente quello che serve.

Poi cerco se c'e' qualcosa di meglio. 
Ma almeno abbiamo una prima "soluzione"

----

L'alternativa e': media e sdev per ogni Colonna, e considerare le colonne indipendenti.
.
Quindi si puo' fare un mix: 
distribuzione gaussiana con matrice di covarianza completa per il subset interessante, 
random (con distrib gaussiana/indipendente) per le altre.


Per la feature selection, vedo che cosa si puo' fare.
Con 13000 colonne, diventa obbligatoria.

