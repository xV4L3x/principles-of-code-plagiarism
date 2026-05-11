La rilevazione automatica del plagio nel codice sorgente pone una sfida complessa che va oltre il semplice confronto testuale: richiede la capacità di trasformare programmi informatici in rappresentazioni astratte che ne rendano confrontabile la logica, indipendentemente dallo stile di scrittura. La letteratura scientifica ha prodotto nel tempo una vasta gamma di algoritmi per rispondere a questa esigenza, i quali possono essere classificati efficacemente in base al livello di profondità con cui interpretano il codice sorgente.

== Analisi Basata sugli Attributi (Attribute-based)
L'analisi basata sugli attributi, definita anche come approccio metrico, non esamina direttamente la struttura logica o il testo del codice, ma si focalizza sull'estrazione di un insieme di caratteristiche quantitative (software metrics) che descrivono il programma. L'assunto di base è che ogni autore possieda uno "stile di programmazione" unico, che si riflette in parametri statistici costanti.

=== Metriche software principali
In questo approccio, il codice sorgente viene trasformato in un vettore di attributi. Le metriche più comunemente utilizzate includono:
- *Metriche di Halstead*: Si basano sul conteggio degli operatori e degli operandi per misurare la complessità algoritmica, il volume e lo sforzo implementativo @hamer1982halstead.
- *Complessità Ciclomatica di McCabe*: Misura il numero di percorsi linearmente indipendenti attraverso il grafo di flusso del controllo del programma, indicando quanto sia articolata la logica decisionale @mccabe1976complexity.
- *Indicatori di stile*: Conteggio delle righe di codice, numero di commenti, frequenza di determinate parole chiave del linguaggio, o la lunghezza media degli identificatori.

=== Meccanismo di confronto
Una volta estratte le metriche da due diversi file sorgente, la similarità viene calcolata determinando la "distanza" tra i due vettori risultanti (spesso utilizzando la distanza euclidea o la similarità del coseno) @cosmaLSA. Se la differenza tra i valori rientra in una soglia prestabilita, il sistema segnala un potenziale caso di plagio

=== Analisi Critica: Robustezza e Limiti
Sebbene l'analisi basata sugli attributi offra vantaggi significativi in termini di efficienza computazionale e scalabilità su dataset di vaste dimensioni @plagDetectionSurvey, la sua efficacia nella rilevazione di plagi sofisticati è severamente limitata. Il limite principale risiede nella natura stessa delle metriche software, le quali risultano estremamente sensibili a trasformazioni superficiali del codice @ref5.

L'inserimento di «codice morto» o di istruzioni irrilevanti altera i parametri quantitativi del programma, ingannando facilmente gli algoritmi di confronto basati sulle metriche. Poiché mancano di una reale comprensione della logica o della sintassi, questi approcci risultano vulnerabili a tecniche di offuscamento elementari. Di conseguenza, pur essendo utili per una scansione preliminare rapida, tali tecniche non sono considerate sufficientemente robuste per identificare casi di plagio strutturale o semantico.

== Analisi Basata sul Testo (String-based)
L'analisi basata sul testo tratta il codice sorgente come una semplice sequenza di caratteri o stringhe. In questo approccio, il software viene confrontato a livello testuale per identificare segmenti di codice identici o simili tra due o più file.

=== Algoritmi e funzionamento
Le tecniche string-based si affidano ad algoritmi classici di _string-matching_. Tra i più rilevanti in letteratura si distinguono:
- *Longest Common Subsequence (LCS)*: utilizzato per trovare la più lunga sottosequenza comune tra due stringhe, permettendo di identificare porzioni di codice che mantengono lo stesso ordine relativo @Hunt1977.
- *Greedy String Tiling (GST)*: un algoritmo che cerca di coprire un file con il maggior numero possibile di sottostringhe (o "tesseri") estratte da un altro file, risultando efficace anche se i blocchi di codice sono stati riordinati @wise1993string.

=== Efficacia e limitazioni
Questi metodi risultano particolarmente efficaci nel rilevamento del *plagio letterale (verbatim)*, ovvero la copia esatta di file in cui le uniche modifiche riguardano elementi non funzionali come spazi bianchi o commenti. Tuttavia, l'analisi testuale pura presenta vulnerabilità critiche come la sua sensibilità all'offuscamento lessicale. Gli approcci string-based sono messi in crisi da tecniche come lo _scrambling_ degli identificatori e l'inserimento di "codice morto", poichè queste modifiche alterano la sequenza dei caratteri senza cambiare la logica.

Un altro problema di questo tipo di approcci riguarda le trasformazioni strutturali. Poichè questi strumenti non hanno una reale comprensione della sintassi, trasformazioni strutturali come il passaggio da un ciclo `for` a un ciclo `while` rendono il codice irriconoscibile per un confronto testuale puro.

=== Esempio di Plagio Letterale
Il codice originale e quello plagiato sono identici come sequenza di caratteri.

_Codice originale:_
```C
int calculate_sum(int a, int b) {
  return a + b;
}
```
_Codice plagiato rilevabile da string-based:_
```C
int calculate_sum(int a, int b) {
  return a + b;
}
```
_Codice plagiato *non* rilevabile da string-based:_
```C
int sum(int x, int y) {
  return x+y;
}
```
Se il plagiatore cambiasse semplicemente i nomi delle variabili e/o delle funzioni di un software, un confornto basato puramente su stringhe potrebbe fallire o dare una similarità ridotta.

== Analisi Basata sui Token (Token-based)
L'analisi basata sui token rappresenta un'evoluzione significativa rispetto all'analisi testuale pura. In questo approccio, il codice sorgente non viene trattato come una sequenza di caratteri, ma viene trasformato in una sequennza di *simboli lessicali* (token) attraverso un processo di analisi lessicale (*lexing*).
=== Tokenizzazione e Normalizzazione
Il vantaggio principale di questo metodo risiede nella sua capacità di astrarre il codice dalla sua forma superficiale. Il processo si articola generalmente in due fasi:
+ *Lexing*: Il codice sorgente viene scansionato e convertito in una stringa ti token che rappresentano le unità logiche del linguaggio (es. `IF`, `WHILE`, `VAR`, `ASSIGN`).
+ *Normalizzazione*: Durante questa fase, vengono eliminate le informazioni irrilevanti dal punto di vista funzionale In particolare vengono rimossi commenti e spazi bianchi, e gli identificatori (come nomi di variabili, funzioni o classi) vengono sostituiti da un token generico, in modo che il nome di una variabile non influisca sull'analisi.

Grazie a questa astrazione, il sistema è in grado di riconoscere la similarità anche se il plagiatore ha modificato radicalmente i nomi delle variabili (scrambling) o lo stile di formattazione.
=== Algoritmi di confronto
Una volta ottenuta la sequenza di token, il sistema applica algoritmi di confronto per identificare sottosequenze comuni. L'algoritmo più noto in questo ambito è il *Running Karp-Rabin Greedy String Tiling (RK-GST)*, utilizzato da strumenti come JPlag @jplagFinding. Questo algoritmo è particolarmente efficace in quanto è in grado di rilevare blocchi di codice che sono stati riordinati o spostati e utilizza tecniche di hashing (Karp-Rabin) per velocizzare il confronto su grandi volumi di dati, rendendolo più scalabile rispetto alla ricerca esaustiva.

Un altro approccio che vale la pena citare riguarda l'utilizzo di algoritmi di *winnowing*. Strumenti come MOSS @plagDetectionSurvey fanno uso di questo tipo di tecniche che, invece di confrontare l'intera sequenza, suddividono il codice in _k-grammi_ (sottosequenze di lunghezza $k$). Di questi _k-grammi_ viene calcolato l'hash e solo un sottoinsieme selezionato viene archiviato utilizzando una "finestra scorrevole".
=== Robustezza e limitazioni
L'analisi token-based è ampiamente considerata lo standard "de facto" per la maggior parte degli strumenti accademici grazie al suo equilibrio tra precisione e complessità computazionale. Tuttavia, come ogni metodologia di analisi statica, presenta specifici punti di forza e vulnerabilità critiche che ne determinano l'efficacia.

Il merito principale è la sua resistenza all'offuscamento lessicale. Poichè il processo di normalizzazione elimina commenti, spazi bianchi e rinomina gli identificatori, gli algoritmi basati su token risultano immuni a modifiche superficiali come lo _scrambling_ delle variabili, o il cambiamento dello stile di formattazione. 

=== Esempio di offuscamento lessicale
L'analisi basata sui token converte il codice in una sequenza di simboli astratti, ignorando i nomi delle variabili.

_Codice A (originale)_:
```Java
int sumArray(int[] arr) {
  int total = 0;
  for (int i = 0; i < arr.length; i++) {
    total += arr[i];
  }
  return total;
}
```

_Codice B (plagiato rilevabile da token-based):_
```java
int calcola(int[] valori) {
  int x = 0;
  for (int j = 0; k < valori.length; k++) {
    x += valori[k];
  }
  return x;
}
```

_Codice C (plagiato *non* rilevabile da token-based):_
```java
int sumArray(int[] arr) {
  int total = 0;
  for (int i = 0; i < arr.length; i++) {
    int noise = 99;
    noise = noise * 2;
    total += arr[i];
  }
  return total;
}
```
Per comprendere il risultato del confronto, osserviamo come questi frammenti vengono "visti" dall'algoritmo dopo la fase di lexing e normalizzazione. In questa fase, commenti e spazi vengono rimossi e ogni identificatore (nome di variabile o funzione) viene sostituito da un token generico `ID`.

La sequenza generata per il *Codice A* ed il *Codice B* è identica poichè la struttura sintattica è la stessa, garantendo una rilevazione del 100%:

`FOR` `(` ... `)` `{` `ID` `+=` `ID` `[` `ID` `]` `;` `}`

Per il *Codice C* invece la sequenza viene disturbata dall'aggiunta di rumore all'interno del corpo del codice:

`FOR` `(` ... `)` `{` `INT` `ID` `=` `LIT` `;` `ID` `=` `ID` `*` `LIT` `;` `ID` `+=` `ID` `[` `ID` `]` `;` `}`

Come si evince dal confronto, mentre i codici A e B producono la stessa "impronta digitale", il codice C introduce nuovi statements, interrompendo la contiguità necessaria per la rilevazione del plagio.

== Analisi Basata sulla Struttura (Tree-based)
L'analisi basata sulla struttura rappresenta un approccio più sofisticato rispetto all'analisi lessicale (token-based), in quanto non si limita a convertire il codice in una sequenza lineare di simboli, ma ne astrae la struttura sintattica e logica completa. In questo metodo, il codice sorgente viene analizzato (parsed) per costruire una _Intermediate Representation_ (IR) tipicamente sotto la forma di un _Abstract Syntax Tree_ (AST) o, in alcuni casi, di un _Program Dependency Graph_ (PDG).

=== Costruzione dell'AST e Astrazione
Il processo inizia con il parsing del codice sorgente, che trasforma il testo in una struttura ad albero dove ogni nodo rappresenta un costrutto sintattico (es. cicli, assegnazioni, espressioni condizionali). A differenza dei metodi _string-based_ o _token_based_, l'approccio _tree-based_ elimina intrinsecamente le differenze superficiali: la struttura dell'albero rimane identica indipendentemente dai nomi delle variabili, dai commenti o dalla formattazione del codice @plagDetectionSurvey @wang2020detecting. Ad esempio, l'AST cattura la gerarchia delle operazioni e l'annidamento dei blocchi logici, offrendo una "impronta digitale" strutturale del programma.

=== Algoritmi di confronto
Una volta generati gli AST per i programmi da confrontare, la rilevazione del plagio si riduce ad un problema di _tree matching_ o di ricerca di sotto-alberi isomorfi
#footnote[Due sotto-alberi $T_1$ e $T_2$ si dicono isomorfi se esiste una corrispondenza biunivoca tra i loro nodi tale che la struttura di adiacenza sia preservata. In termini pratici, due alberi sono isomorfi se possono essere sovrapposti perfettamente tramite una rotazione o un riordinamento dei figli, mantenendo invariate le connessioni gerarchiche.].
Gli algoritmi cercano di identificare sotto-strutture simili o identiche tra due AST.

Si parla di *confronto esatto* quando la verifica richiede che i due alberi siano identici. È efficace solo per copie quasi esatte. Si parla invece di *confronto approssimato* quando vengono utilizzate metriche di similarità per quantificare quanto due alberi si sovrappongono. Questo è necessario poichè piccole modifiche al codice (come l'inserimento di un'istruzione vuota) possono alterare la struttura dell'albero.

=== Vantaggi e limitazioni
L'approccio Tree-based rileva il Plagio di Tipo 3 (piccole aggiunte/rimozioni di codice) meglio dei token, poiché un ramo aggiunto all'albero non invalida la struttura dei rami adiacenti. Il vantaggio principale di questo approccio è appunto la sua robustezza contro l'offuscamento lessicale e il riordinamento di funzioni o metodi, poichè la struttura gerarchica (ad esempio della classe) viene analizzata indipendentemente dall'ordine testuale. Tuttavia, l'analisi _tree-based_ presenta svantaggi in termini di complessità computazionale. Il confronto di alberi è un'operazione costosa (di complessità polinomiale o superiore), rendendo questi strumenti meno scalabili su grandi repository rispetto ai metodi _token_based_. Inoltre, modifiche strutturali significative, come la sostituzione di un ciclo `for` con un ciclo `while`, modificano la forma dell'albero, rendendo il plagio più difficile da rilevare se l'algoritmo di confronto è troppo rigido.

Per superare i limiti strutturali, ricerche recenti hanno proposto di arricchire la classica rappresentazione ad albero integrando esplicitamente archi supplementari che tracciano il flusso di controllo e dei dati. Questa rappresentazione ibrida prende il nome di Flow-Augmented AST @wang2020detecting e permette di mantenere la gerarchia sintattica dell'albero incorporando le dipendenze semantiche necessarie a rilevare manipolazioni complesse del codice.

=== Esempio di Plagio Strutturale
L'analisi basata sulla struttura (AST) astrae il codice in una gerarchia logica. Questo la rende resistente all'inserimento di blocchi di codice superflui che alterano la sequenza lineare dei token, ma rimane vulnerabile a cambiamenti radicali dell'algoritmo.

_Codice A (Originale):_
```java
int sumArray(int[] arr) {
  int total = 0;
  for (int i = 0; i < arr.length; i++) {
    total += arr[i];
  }
  return total;
}
```

_Codice B (Plagiato rilevabile da Tree-based):_
```java
int arraySum(int[] values) {
  {
    int dummy = 0;
    int result = 0;
    for (int k = 0; k < values.length; k++) {
      result += values[k]
    }
    return values;
  }
}
```

_Codice C (Plagiato non rilevabile da tree-based):_
```java
int sumArray(int[] arr) {
  int total = 0;
  int i = 0;
  while(i < arr.length) {
    total += arr[i];
    i++;
  }
  return total;
}
```

Per comprendere la differenza rispetto all'analisi token-based, osserviamo come questi frammenti vengono rappresentati gerarchicamente dall'albero sintattico.

Nel confronto tra *Codice A* e *Codice B*, l'algoritmo di tree matching è in grado di rilevare la similarità poichè cerca sotto-alberi isomorfi.
- *AST Codice A*: Il nodo radice del metodo contiene direttamente il nodo `ForStatement`.
- *AST Codice B*: Il nodo radice contiene un nodo `Block` intermedio, che a sua volta contiene il nodo `ForStatement`.

L'algoritmo ignora il "contenitore" extra e identifica che il sotto-albero che parte da `ForStatement` è strutturalmente identico (ha gli stessi figli: inizializzazione, condizione, incremento e corpo) in entrambi i casi.

Nel confronto con il *Codice C*, invece, la topologia dell'albero cambia, in quanto il nodo `ForStatement` viene sostituito dal nodo `WhileStatement` e l'istruzione di inizializzazione `int i = 0` che era prima incapsulata nel ciclo viene ora spostata fuori. 

Poiché i nodi non corrispondono più né per tipo né per posizione gerarchica, la distanza di edit tra gli alberi (Tree Edit Distance) risulta elevata, portando a una mancata rilevazione del plagio.

== Analisi Basata sul Grafo (Graph-based)
L'analisi basata sul grafo rappresenta un ulteriore passo verso la comprensione semantica del codice. Mentre l'AST analizza la gerarchia sintattica (com'é scritto il codice), l'approccio _graph-based_ modella il flusso di esecuzione e le dipendenze tra i dati (cosa fa il codice).

La rappresentazione più utilizzata in questo ambito è il *Program Dependency Graph* (PDG). In un PDG, i *nodi* rappresentano le istruzioni o le espressioni e gli *archi* rappresentano due tipi di relazioni:
+ *Dipendenza dai dati*: L'istruzione $B$ usa una variabile definita in $A$.
+ *Dipendenza dal controllo*: L'istruzione $B$ viene eseguita solo se la condizione in $A$ è vera

=== Superare i limiti della sintassi
Il vantaggio cruciale del PDG é che astrae completamente l'ordine testuale delle istruzioni che non sono dipendenti tra loro, e soprattutto, unifica costrutti sintattici diversi che hanno lo stesso comportamento. Se un ciclo `for` e un ciclo `while` generano AST diversi nell'analisi _tree-based_, poiché in realtà entrambi implementano la stessa logica di iterazione, generano PDG pressoché identici. Le dipendenze rimangono invariate, permettendo all'algoritmo di rilevare plagio anche su cloni di tipo 4/4#footnote[Il Tipo 4 (Semantic Clone) si verifica quando due frammenti di codice eseguono la stessa computazione (stesso comportamento funzionale) ma sono implementati attraverso varianti sintattiche differenti. Esempi tipici includono la sostituzione di un ciclo for con un while o l'uso della ricorsione al posto dell'iterazione @viertel2019detecting].

=== Algoritmi di confronto: Isomorfismo di Sottografi
Una volta costruiti i PDG dei programmi da analizzare, il rilevamento del plagio si riduce a un problema matematico noto come Isomorfismo di Sottografi (Subgraph Isomorphism). Formalmente, dato un grafo "pattern" $G_P$ (il codice sospetto o una sua parte) e un grafo "target" $G_T$ (il codice originale), l'obiettivo è determinare se esiste una mappatura biunivoca che permetta di sovrapporre $G_P$ a una porzione di $G_T$, preservando la struttura delle adiacenze e i tipi di dipendenza (dati o controllo).

Il limite principale di questo approccio è la sua intrattabilità teorica: il problema dell'isomorfismo di sottografi appartiene alla classe di complessità NP-Hard#footnote[Un problema è NP-Hard se ogni problema appartenente alla classe NP è ad esso riducibile in tempo polinomiale. Di conseguenza, non è noto alcun algoritmo che lo risolva in tempo polinomiale nel caso peggiore, rendendo la ricerca di soluzioni esatte computazionalmente intrattabile per istanze di grandi dimensioni.]. Ciò significa che, nel caso peggiore, il tempo necessario per il confronto cresce esponenzialmente con il numero di nodi del grafo. Applicare un algoritmo esatto di graph matching su repository contenenti migliaia di file sorgente risulterebbe computazionalmente proibitivo.

Per rendere questa tecnica applicabile in scenari reali, gli strumenti dello stato dell'arte come GPLAG @liu2006gplag adottano strategie di approssimazione e filtraggio. Ad esempio, prima di avviare il costoso confronto sui grafi, viene eseguito un pre-screening rapido basato su metriche semplici (come il numero di nodi o la distribuzione dei tipi di istruzioni). Se due grafi sono troppo diversi "macroscopicamente", vengono scartati immediatamente.

=== Esempio di Plagio semantico
Come detto precedentemente il PDG non si basa solamente sulla struttura sintattica del programma, ma riesce a creare un'astrazione delle dipendenze per poter analizzare il flusso di esecuzione senza dover necessariamente eseguire il programma. Il rilevatore fallisce nel momento in cui l'algoritmo viene effettivamente cambiato.

_Codice A (Originale):_
```java
int sumArray(int[] arr) {
  int total = 0;
  for (int i = 0; i < arr.length; i++) {
    total += arr[i]
  }
  return total;
}
```
_Codice B (Plagiato rilevabile da Graph-based):_
```java
int arraySum(int[] values) {
  int result = 0;
  int j = 0;
  while (j < values.length) {
    result += values[j];
    j++;
  }
  return result;
}
```

_Codice C (Plagiato non rilevabile da Graph-based):_
```java
int sumArray (int[] arr) {
  return Arrays.stream(arr).sum();
}
```
Osservando i frammenti proposti, emerge chiaramente la capacità del Grafo delle Dipendenze del Programma di superare le barriere sintattiche che limitavano gli approcci precedenti. Nel confronto tra il *Codice A* e il *Codice B*, sebbene la struttura testuale sia stata alterata sostituendo il ciclo `for` con un costrutto `while` e dislocando le istruzioni di inizializzazione e incremento, il grafo risultante è isomorfo. In entrambi i casi, infatti, l'algoritmo rileva le medesime relazioni di dipendenza: la variabile accumulatore dipende dal valore estratto dall'array, il quale a sua volta è vincolato all'indice che governa la condizione di uscita del ciclo. 

Al contrario, il Codice C introduce una variazione paradigmatica che rende il plagio invisibile a questa tecnica. L'utilizzo delle API funzionali (Stream) rimuove esplicitamente le variabili di controllo e i cicli iterativi, generando un grafo delle dipendenze lineare o monolitico che non presenta alcuna somiglianza strutturale con il grafo ciclico generato dall'algoritmo originale.

== Approcci basati su Machine Learning e Deep Learning (Learning-based)
Le tecniche analizzate nei paragrafi precedenti (String, Token, Tree, Graph) condividono una caratteristica comune: si basano su regole e algoritmi deterministici definiti a priori dall'uomo. Sebbene efficaci, questi metodi faticano a scalare verso la comprensione semantica profonda, specialmente quando due codici svolgono la stessa funzione usando paradigmi completamente diversi.

Negli ultimi anni, la ricerca si è spostata verso approcci _learning-based_, sfruttando il Machine Learning e, più recentemente, il Deep Learning per addestrare modelli capaci di classificare la similarità del codice automaticamente.

=== Rappresentazione Vettoriale
Il cuore di questa rivoluzione risiede nel superamento dell'ingegneria manuale delle feature. Invece di contare metriche o confrontare alberi, le moderne architetture di Deep Learning (come le Graph Neural Networks o i modelli basati su Transformer come CodeBERT @feng2020codebert) trasformano il codice sorgente in *Code Embeddings*. 

Un embedding è una rappresentazione del codice sotto forma di vettore numerico denso (un array di numeri reali) in uno spazio multidimensionale. Il modello viene addestrato affinché frammenti di codice semanticamente simili vengano mappati in punti vicini nello spazio vettoriale, mentre codici con funzionalità diverse vengano proiettati in punti distanti. La rilevazione del plagio diventa quindi un calcolo geometrico: si misura ad esempio la Similarità del Coseno tra i vettori dei due programmi. Se l'angolo tra i vettori è sufficientemente ridotto, i programmi sono considerati cloni, indipendentemente dalla loro struttura sintattica superficiale.

=== Interpretabilità dei risultati

Tuttavia, questa straordinaria capacità di astrazione comporta un costo significativo in termini di trasparenza, noto come il problema della *Black Box*. Mentre strumenti come JPlag (token-based) offrono una prova visiva immediata, evidenziando le righe di codice copiate e mostrando la corrispondenza esatta, un modello di Deep Learning restituisce unicamente un valore di probabilità (es. "Similarità: 98%").

Poiché il vettore è il risultato di milioni di operazioni matriciali non lineari all'interno della rete neurale, è estremamente difficile, se non impossibile, risalire a quali specifiche istruzioni abbiano determinato il verdetto di plagio. Questa mancanza di Explainability rappresenta un ostacolo critico in ambito accademico e legale, dove l'accusa di plagio deve essere supportata da prove tangibili e interpretabili, e non solo da un punteggio statistico.

Approcci avanzati basati su Graph Neural Networks (GNN) stanno però iniziando a mitigare il problema della Black Box @wang2020detecting. Applicando meccanismi di attenzione (attention mechanisms) alle rappresentazioni vettoriali del codice, è possibile evidenziare quali specifici nodi o sottostrutture del grafo abbiano contribuito maggiormente al calcolo della similarità. Questo non solo migliora l'accuratezza su plagi semantici, ma offre un primo livello di interpretabilità visiva, permettendo di identificare le porzioni di codice sospette anche in assenza di corrispondenze testuali esatte.

=== Esempio di Plagio con Variazione di Paradigma
Nonostante i limiti di spiegabilità, il Deep Learning è l'unica arma efficace contro i plagi semantici complessi. Riprendiamo l'esempio che aveva messo in crisi l'analisi Graph-based (PDG) nel capitolo precedente.

_Codice A (originale):_
```java
int sumArray(int[] arr) {
  int total = 0;
  for (int i = 0; i < arr.length; i++) {
    total += arr[i];
  }
  return total;
}
```
_Codice C (paradigma funzionale):_
```java
int sumArray(int[] arr) {
  return Arrays.stream(arr).sum();
}
```
Un modello di Deep Learning pre-addestrato su milioni di repository open-source ha "imparato" che il pattern `Arrays.stream(...).sum()` appare frequentemente negli stessi contesti del pattern `for (...) { ... += ... }`. Di conseguenza, il modello genererà due vettori numerici molto simili per il *Codice A* e il *Codice C*, rilevando il plagio semantico che era sfuggito a tutte le tecniche precedenti.

== Approcci LLM-Based
I Large Language Models (LLM) rappresentano una categoria qualitativamente nuova rispetto a tutti gli approcci discussi in precedenza. A differenza dei modelli learning-based classici, un LLM non viene addestrato su coppie di codice etichettate come "plagio/non plagio": la sua comprensione del codice emerge dalla pre-esposizione a miliardi di parametri addestrati su corpus enormi che includono codice sorgente, documentazione tecnica e linguaggio naturale. Il task di SCPD viene quindi affrontato in modalità zero-shot (senza nessun addestramento specifico sul problema), mediante la costruzione di un prompt in linguaggio naturale che descrive il problema.

=== Meccanismo di funzionamento
L'interfaccia con un LLM non avviene attraverso una pipeline deterministica come nei metodi precedenti, ma tramite un prompt: un testo in linguaggio naturale che descrive il task, fornisce i frammenti di codice da confrontare e specifica il formato atteso della risposta. La qualità del prompt è determinante per l'affidabilità del risultato — un prompt vago produce verdetti inconsistenti, mentre un prompt strutturato e dettagliato, che specifica esplicitamente la tassonomia dei tipi di clone e richiede una motivazione, tende a produrre analisi più precise e riproducibili.
La risposta del modello può essere ottenuta in linguaggio naturale — dove il modello argomenta liberamente le somiglianze rilevate — oppure in un formato strutturato come JSON, specificandolo nel prompt, in modo da rendere l'output comparabile con i punteggi numerici degli altri strumenti. Questa flessibilità è una caratteristica distintiva dell'approccio: lo stesso modello può adattarsi a contesti diversi semplicemente modificando il prompt, senza alcuna modifica al sistema sottostante.

=== Evidenze Empiriche
Studi recenti hanno iniziato a valutare sistematicamente le capacità degli LLM come strumenti di SCPD, confrontandoli con i tool tradizionali su dataset che coprono tutti e quattro i tipi di plagio. I risultati mostrano che GPT-4o raggiunge un'accuratezza complessiva del 78,70% con F1 score di 86,97%, mentre LLaMA 3 — modello open source e self-hostabile — ottiene un'accuratezza del 71,53% con F1 score di 82,75% @llm2024scpd. Entrambi i modelli dimostrano capacità superiori agli strumenti token-based nel rilevamento dei cloni di Tipo 4, ovvero le variazioni semantiche profonde che JPlag e MOSS non riescono a catturare.

=== Vantaggi e Limiti
Il vantaggio principale rispetto a tutti gli approcci precedenti è la spiegabilità nativa: invece di un punteggio numerico opaco, il modello produce un ragionamento in linguaggio naturale che identifica esplicitamente quali costrutti, logiche o pattern siano stati replicati. Questo supera il problema della Black Box discusso nella sezione 3.6.2 e rende i risultati più direttamente utilizzabili in un contesto accademico o legale.

Un secondo vantaggio è il supporto cross-language: un LLM pre-addestrato su più linguaggi simultaneamente può confrontare un frammento Python con uno Java, rilevando somiglianze semantiche che nessuno strumento monolingua potrebbe identificare. Questa capacità è particolarmente rilevante nel contesto delle nuove sfide discusse nella Sezione 6.

Tuttavia l'approccio presenta limiti strutturali che ne condizionano l'adozione su scala. Il primo è il costo computazionale: confrontare un corpus di centinaia di submission richiede un numero elevato di chiamate API o l'esecuzione locale di modelli da miliardi di parametri, rendendo l'approccio significativamente più lento rispetto a JPlag o MOSS. Il secondo è il non determinismo: lo stesso prompt eseguito due volte può produrre verdetti leggermente diversi, rendendo difficile la piena riproducibilità degli esperimenti. Il terzo è il rischio di falsi positivi: gli LLM tendono a segnalare come plagio frammenti che condividono pattern comuni di implementazione — come algoritmi di ordinamento standard — senza che vi sia un effettivo riutilizzo non autorizzato.

=== Esempio di Plagio con variazione di Paradigma
Per illustrare la capacità distintiva dell'approccio LLM-based, consideriamo una variante dell'esempio utilizzato nelle sezioni precedenti in cui non solo il paradigma ma anche il linguaggio di programmazione cambia tra i due frammenti.

_Codice A (Originale — Java):_
```java
int sumArray(int[] arr) {
    int total = 0;
    for (int i = 0; i < arr.length; i++) {
        total += arr[i];
    }
    return total;
}
```

_Codice B (Plagiato — Python, paradigma funzionale):_
```python
def sum_array(arr):
    return sum(arr)
```

Qualsiasi strumento deterministico — string-based, token-based, tree-based o graph-based — produrrebbe una similarità prossima allo zero: i due frammenti non condividono alcun token, nessuna struttura sintattica, nessuna dipendenza confrontabile. Persino un modello DL fine-tuned su Java faticherebbe a generalizzare su questa coppia, in quanto i due frammenti appartengono a spazi vettoriali addestrati su distribuzioni diverse.

Un LLM interrogato con un prompt adeguato riconosce invece che entrambi i frammenti implementano la stessa operazione — la somma degli elementi di una sequenza — ragionando sul comportamento osservabile piuttosto che sulla forma. La risposta del modello potrebbe argomentare che la funzione built-in `sum()` di Python è semanticamente equivalente al pattern di accumulazione con ciclo `for` in Java, identificando il plagio cross-language nonostante la completa diversità sintattica, strutturale e linguistica. Questa capacità, unica tra tutti gli approcci analizzati, è al contempo il punto di forza più rilevante e la fonte principale del rischio di falsi positivi discusso in 3.7.3: la somma di un array è un pattern talmente comune che due implementazioni indipendenti potrebbero essere erroneamente classificate come plagio.

== Sintesi Comparativa
A conclusione delle tecniche teoriche di rilevazione del plagio del codice sorgente, la seguente tabella riassume le capacità ed i limiti delle diverse metodologie discusse.

#figure(
  caption: [Confronto tra le principali metodologie di rilevazione del plagio del codice sorgente],
  block(width: 100%)[
    #set text(size: 9pt)
    #table(
      columns: (1fr, 1fr, 1fr, 1.2fr, 1fr),
      inset: 4pt,
      align: horizon,
      fill: (x, y) => if y == 0 { gray.lighten(80%) },
      table.header(
        [*Tecnica*], [Astrazione], [Tipo di Clone], [Resistenza Offuscamento], [Spiegabilità]
      ),

      [*string-based*],
      [Testo],
      [Tipo 1],
      [Bassa],
      [Altissima],

      [*token-based*],
      [Lessicale],
      [Tipo 1, 2],
      [Media],
      [Alta],

      [*tree-based*],
      [Sintattica],
      [Tipo 1, 2, 3],
      [Alta],
      [Media],

      [*graph-based*],
      [Semantica],
      [Tipo 1-3,4 (Parziale)],
      [Molto Alta],
      [Bassa],

      [*learning-based*],
      [Vettoriale],
      [Tutti],
      [Massima],
      [*Nulla (Black Box)*],

      [*LLM-based*],
      [Linguistica],
      [Tutti + cross-language],
      [Massima],
      [Altissima]

    )
  ]
)
