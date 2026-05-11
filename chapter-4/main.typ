In questa sezione verranno analizzati i principali strumenti di rilevazione del plagio del codice sorgente (SCDP) sviluppati nel corso degli anni. La rassegna si concentra sulle soluzioni accademiche e open source che hanno definito lo standard del settore, per poi esplorare piattaforme commerciali più recenti. Ogni strumento verrà valutato in base alle tecniche algoritmiche adottate (discusse nel Capitolo 3), ai linguaggi supportati e alla robustezza contro le tecniche di offuscamento.

== Strumenti Accademici e Open Source
Il panorama degli strumenti accademici è dominato da soluzioni storiche che, nonostante l'età, rimangono i punti di riferimento per l'analisi statica del plagio. Questi tool sono tipicamente gratuiti per uso educativo e si basano su analisi testuale o token-based

=== JPlag #footnote[#link("https://github.com/jplag/JPlag/")]
Sviluppato originariamente presso l'Università di Karlsruhe (Germania) da Prechelt, Malpohl e Philippsen e descritto formalmente nel 2002 @jplagFinding, JPlag è considerato il principale strumento accademico open source per la rilevazione del plagio. Creato nel 1996, è ancora oggi attivamente sviluppato e manutenuto presso il Karlsruhe Institute of Technology (KIT), e supporta Java, C, C++, C\#, Python, JavaScript, TypeScript, Go, Rust, Kotlin, Swift, Scala e altri linguaggi. 

Internamente, JPlag opera in due fasi principali: prima effettua il parsing del programma, estraendo un sottoinsieme dei nodi del parse tree sotto forma di token, linearizzando così la struttura ad albero; successivamente applica tecniche di normalizzazione per aumentare la resilienza contro l'offuscamento @saglam2024obfuscation. Per il confronto delle sequenze, JPlag utilizza l'algoritmo *Running Karp-Rabin Greedy String Tiling (RK-GST)*, che cerca di coprire la sequenza di token di un programma con le sottosequenze contigue più lunghe trovate nell'altro, indipendentemente dalla loro posizione originale.

Un aspetto cruciale che distingue JPlag dagli altri strumenti storici è il suo sviluppo continuo in risposta alle minacce emergenti. La diffusione di tecniche di offuscamento automatico mette in discussione l'assunzione che eludere un rilevatore di plagio richieda più sforzo del completare un compito di programmazione, minacciando i rilevatori privi di resilienza e, in ultima analisi, l'integrità accademica @saglam2024obfuscation. Per rispondere a questa sfida, il team di KIT ha introdotto due meccanismi di difesa complementari: la Token Sequence Normalization (TSN), progettata per neutralizzare attacchi basati sull'inserimento di codice morto e sul riordinamento di istruzioni, e il Subsequence Match Merging (SMM), una tecnica euristica che contrasta attacchi volti a interrompere la contiguità delle corrispondenze tra sequenze di token. I risultati dimostrano che JPlag supera significativamente gli strumenti concorrenti in termini di resilienza: nei benchmark contro tecniche di offuscamento automatico, le istanze di plagio raggiungono una similarità mediana del 100%, contro il 26.7% di Dolos e il 7.5% di MOSS @saglam2024obfuscation.

JPlag è open source e può essere eseguito localmente, risultando conforme al GDPR — un aspetto rilevante per i contesti accademici europei che trattano dati sensibili degli studenti. L'interfaccia grafica include una vista di distribuzione delle similarità per identificare outlier sospetti, una vista di confronto dettagliata per analizzare i segmenti di codice corrispondenti, e una cluster view che aiuta a comprendere dinamiche di plagio collettivo tra submission multiple.

Nonostante i continui progressi, l'alterazione semantica profonda e il cambio radicale di paradigma algoritmico (cloni di Tipo 4) restano un limite strutturale intrinseco dell'approccio token-based. Inoltre, i meccanismi di difesa attuali non coprono ancora in modo efficace il plagio generato interamente da AI sulla base della descrizione del compito, una minaccia distinta dall'offuscamento classico e ancora aperta nella letteratura.

=== SIM #footnote[https://dickgrune.com/Programs/similarity_tester/]
Sviluppato originariamente da Dick Grune @grune1989software alla fine degli anni '80 e formalizzato in successivi studi, SIM rappresenta uno degli strumenti più longevi nell'ambito della rilevazione del plagio. Nonostante l'età, rimane un punto di riferimento frequente negli studi comparativi.

Lo strumento supporta diversi linguaggi di programmazione, tra cui C, C++, Java, Pascal ed il linguaggio naturale. Anche SIM lavora principalmente con la tokenizzazione e attraverso l'utilizzo di _flex_ fornisce degli analizzatori lessicali dedicati ai diversi linguaggi @gondaliya2014source. In questa fase gli elementi irrilevanti come spazi e commenti vengono rimossi e le istruzioni vengono convertite in una sequenza lineare di token. Successivamente SIM applica un algoritmo di allineamento di stringhe (basato sulla ricerca della Longest Common Subsequence) per confrontare i flussi di token generati, individuando le sequenze contigue più lunghe condivise tra i due file.

Dal punto di vista prestazionale, l'implementazione dell'allineamento rende SIM computazionalmente molto efficiente e altamente preciso nell'identificare plagi di _Tipo 1_ e _Tipo 2_ (come il renaming delle variabili). Tuttavia, l'efficacia dell'algoritmo dipende dal mantenimento dell'ordine globale delle istruzioni: tecniche di offuscamento strutturale, come lo spostamento di blocchi di codice indipendenti, l'inversione di funzioni o l'inserimento di codice morto, spezzano le sequenze contigue e causano un drastico calo della similarità rilevata.

=== Plaggie
Plaggie è uno strumento open source sviluppato presso l'Helsinki University of Technology nel 2006 @ahtiainen2006plaggie, concepito come alternativa libera e autonoma a MOSS per l'uso accademico. A differenza di MOSS, che opera come servizio remoto richiedendo l'invio del codice a server di Stanford, Plaggie è distribuito come applicazione Java standalone eseguibile localmente, il che lo rende particolarmente adatto a contesti in cui la privacy dei dati degli studenti è una preoccupazione rilevante.

L'architettura di Plaggie segue un approccio rigorosamente token-based, concettualmente analogo a quello di JPlag. Il processo si articola in due fasi principali: nella prima, il codice sorgente viene analizzato da un lexer specifico per il linguaggio che converte il testo in una sequenza lineare di token astratti, eliminando commenti, spazi bianchi e normalizzando gli identificatori. Nella seconda fase, le sequenze di token vengono confrontate tramite l'algoritmo Greedy String Tiling (GST), che individua le sottostringhe contigue più lunghe comuni tra coppie di programmi, indipendentemente dalla loro posizione nel file originale.

Una caratteristica distintiva di Plaggie rispetto a JPlag è la sua focalizzazione quasi esclusiva su Java come linguaggio target, scelta che ha permesso agli autori di ottimizzare il lexer per la struttura sintattica del linguaggio, migliorando la qualità della tokenizzazione per costrutti tipici della programmazione orientata agli oggetti (classi, interfacce, generics).

In termini di robustezza, Plaggie condivide i punti di forza e i limiti strutturali dell'approccio token-based descritto nella sezione 3.3. Risulta efficace contro l'offuscamento lessicale (renaming di variabili e metodi) e contro semplici riordinamenti di istruzioni indipendenti, poiché il processo di normalizzazione rende queste trasformazioni trasparenti all'algoritmo. Tuttavia, come tutti i sistemi basati su GST puro, rimane vulnerabile all'inserimento massiccio di codice "rumore" che spezza la contiguità delle sequenze di token, e non è in grado di rilevare cloni di Tipo 4 (variazioni semantiche profonde o cambi di paradigma algoritmico).

Dal punto di vista della scalabilità, la complessità dell'algoritmo GST — superiore a quella del Winnowing utilizzato da MOSS — rende Plaggie meno performante su corpus di grandi dimensioni. Lo strumento è stato concepito per l'analisi di singoli corsi universitari (decine o centinaia di submission), non per scansioni su repository di scala industriale.

Sebbene Plaggie non abbia avuto lo stesso sviluppo continuo di JPlag e non sia più attivamente manutenuto, mantiene una presenza negli studi comparativi della letteratura come punto di riferimento storico per la valutazione di strumenti token-based su dataset Java. La sua semplicità architetturale e la disponibilità del codice sorgente lo rendono uno strumento utile per studi di benchmark e per contesti didattici in cui si desideri analizzare il funzionamento interno di un rilevatore di plagio.

=== Dolos #footnote[https://dolos.ugent.be]
Dolos è un progetto di ricerca attivo sviluppato dal Team Dodona dell'università di Gent (Belgio), con la prima pubblicazione scientifica di riferimento risalente al 2022. Il codice sorgente è rilasciato sotto licenza MIT, rendendolo liberamente utilizzabile e modificabile. Rispetto agli strumenti storici come MOSS e JPlag, Dolos nasce con una filosofia esplicitamente orientata all'usabilità: la complessità di strumenti come JPlag e MOSS ostacola spesso la loro adozione diffusa @maertens2023dolos, e Dolos si propone di abbassare questa barriera.

L'elemento tecnico più distintivo di Dolos rispetto agli altri strumenti token-based è l'utilizzo di tree-sitter come motore di parsing. Tree-sitter è una libreria di parsing incrementale che genera AST per decine di linguaggi di programmazione attraverso modelli generici, rendendo Dolos language-agnostic @maertens2022dolos: aggiungere il supporto a un nuovo linguaggio non richiede la scrittura di un lexer dedicato, ma semplicemente l'integrazione del relativo modello tree-sitter. Una volta costruito l'AST, Dolos estrae una sequenza di token dalla visita dell'albero e applica l'algoritmo di Winnowing per il calcolo delle fingerprint, in modo concettualmente analogo a MOSS ma con il vantaggio di operare su token derivati dalla struttura sintattica piuttosto che sul testo grezzo.

Una caratteristica che distingue nettamente Dolos dagli altri strumenti open source è l'attenzione all'esperienza utente. L'ecosistema Dolos include una web app self-hostable, un'API JSON, una CLI, una libreria JavaScript e un container Docker preconfigurato @maertens2024discovering. Le dashboard analitiche offrono visualizzazioni interattive che permettono di identificare immediatamente i casi sospetti e, soprattutto, di rilevare fenomeni di plagio di gruppo attraverso un sistema di clustering delle submission: una funzionalità assente negli strumenti più datati, dove l'analisi è sempre e solo su coppie di file.

Piattaforme come Codio #footnote[https://www.codio.com] hanno abbandonato MOSS e JPlag in favore di una istanza self-hosted di Dolos @maertens2024discovering, citando esplicitamente la semplicità di integrazione e la qualità dei report come motivazioni principali. Sul piano delle prestazioni, i benchmark sul dataset SOCO mostrano che Dolos supera gli strumenti dei concorrenti nella rilevazione di casi di plagio @maertens2022dolos.

Come tutti gli approcci basati su fingerprinting e analisi token/AST, Dolos rimane vulnerabile alle trasformazioni semantiche profonde (cloni di Tipo 4) e al plagio generato da strumenti di AI generativa. Inoltre, pur essendo self-hostable, l'istanza pubblica ospitata da Ghent University garantisce una retention dei dati di soli 30 giorni, il che può rappresentare un limite per contesti che richiedono un archivio storico delle analisi.

=== Oreo #footnote[#link("https://github.com/Mondego/oreo")]
Oreo è uno strumento open source per la rilevazione di cloni nel codice sorgente che adotta un approccio ibrido, combinando metriche software, information retrieval e machine learning in una pipeline sequenziale @saini2018oreo. A differenza degli strumenti token-based e tree-based discussi nelle sezioni precedenti, Oreo non si limita ad analizzare la forma sintattica del codice, ma introduce un livello di comprensione semantica attraverso un modello di deep learning addestrato su coppie di frammenti, avvicinandosi alla categoria learning-based descritta nella Sezione 3.6.

L'obiettivo dichiarato di Oreo è superare il limite comune a quasi tutti gli strumenti discussi finora: la cosiddetta _Twilight Zone_, ovvero quello spettro di cloni che, pur mantenendo ancora qualche similarità sintattica, risultano estremamente difficili da rilevare — collocandosi al confine tra il Tipo 3 e il Tipo 4. Questa zona grigia rappresenta esattamente quei casi di plagio strutturale avanzato che i metodi token-based e tree-based non riescono a gestire in modo affidabile.

Per affrontare questo problema, Oreo introduce un filtro semantico denominato Action filter che scarta un gran numero di coppie di codice prive di similarità semantica, risolvendo il problema dell'esplosione combinatoria tipico dei confronti a coppie su grandi repository. Le coppie che superano questo filtro vengono poi validate da un modello di deep learning che ne verifica la similarità strutturale, aumentando la precisione complessiva della rilevazione. L'approccio è quindi ibrido per costruzione: combina metriche software, information retrieval e machine learning in una pipeline sequenziale, dove ogni fase riduce lo spazio di ricerca per la fase successiva.

La valutazione della recall è stata condotta su BigCloneBench #footnote[#link("https://github.com/clonebench/BigCloneBench")], mentre la precision è stata stimata tramite ispezione manuale. I risultati dimostrano che Oreo raggiunge valori elevati su entrambe le metriche, con una capacità di rilevazione che si estende significativamente oltre i cloni con similarità sintattica moderata o debole, mantenendo al contempo buone proprietà di scalabilità.

Dal punto di vista pratico, per eseguire Oreo è necessario generare un file di input attraverso il Metric Calculator, uno strumento fornito insieme al sistema che scansiona le directory di progetto cercando file .java e ne calcola le metriche dei metodi. Questo rappresenta il limite più critico di Oreo in relazione agli obiettivi del presente studio: il supporto è esclusivamente per Java, il che esclude qualsiasi confronto diretto con strumenti multi-linguaggio come JPlag o MOSS su dataset eterogenei. Nei test sperimentali della Sezione 5, questo vincolo sarà discusso nella scelta del dataset e del linguaggio di riferimento.

=== CodeBERT/GraphCodeBERT
CodeBERT e GraphCodeBERT sono modelli Transformer pre-addestrati sviluppati da Microsoft Research per la comprensione del codice sorgente. A differenza di tutti gli strumenti discussi finora, non si tratta di tool autonomi con un'interfaccia di input/output pronta all'uso: sono modelli linguistici i cui pesi sono rilasciati pubblicamente su HuggingFace #footnote[#link("https://huggingface.co")] (`microsoft/codebert-base`, `microsoft/graphcodebert-base`) e che richiedono la costruzione di una pipeline sperimentale per essere applicati al task della SCPD.

CodeBERT è un modello multi-linguaggio pre-addestrato su coppie codice/linguaggio naturale in sei linguaggi di programmazione — Python, Java, JavaScript, PHP, Ruby e Go — e supporta task downstream come la ricerca di codice @feng2020codebert, la generazione di documentazione e la clone detection. Per quest'ultimo task, il modello viene sottoposto a fine-tuning su un dataset etichettato di coppie di frammenti — tipicamente BigCloneBench — in una configurazione di classificazione binaria: la coppia viene passata sequenzialmente attraverso CodeBERT come encoder, i due vettori risultanti vengono concatenati e passati a un classificatore shallow per produrre il verdetto finale.

GraphCodeBERT rappresenta l'evoluzione diretta di CodeBERT. Mentre CodeBERT opera su coppie codice/linguaggio naturale sfruttando principalmente informazioni lessicali e sintattiche, GraphCodeBERT incorpora esplicitamente la struttura del grafo del flusso di dati durante il pre-addestramento, catturando le relazioni semantiche tra le variabili del programma @guo2021graphcodebert. Microsoft fornisce una pipeline ufficiale per il fine-tuning di GraphCodeBERT sul task di clone detection, basata sul dataset BigCloneBench e configurabile tramite script Python. Questa pipeline costituisce il punto di partenza per l'utilizzo sperimentale del modello nel presente studio.

Il vantaggio principale di entrambi i modelli rispetto a Oreo è la copertura multi-linguaggio: essendo stati pre-addestrati su sei linguaggi simultaneamente, possono essere fine-tuned su dataset eterogenei senza essere vincolati a Java. Tuttavia, studi empirici evidenziano che la capacità di generalizzazione di CodeBERT decresce significativamente quando il modello viene valutato su frammenti di codice con funzionalità diverse da quelle usate durante l'addestramento, con un calo marcato nell'F1 score @sonnekalb2022generalizability. Questo limite, unito alla necessità di risorse computazionali significative per il fine-tuning, rappresenta il principale punto di attenzione per l'utilizzo di questi modelli in contesti accademici reali.

=== CodeLlama #footnote[#link("https://ollama.com/library/codellama")] e StarCoder2 #footnote[#link("https://ollama.com/library/starcoder2")]
CodeLlama e StarCoder2 sono Large Language Models open source specializzati per il codice, self-hostabili localmente, e rappresentano i candidati naturali per applicare l'approccio LLM-based descritto nella Sezione 3.7 in un contesto accademico che non dipenda da API commerciali esterne.

Come per CodeBERT e GraphCodeBERT, non si tratta di strumenti autonomi per la SCPD: i loro pesi sono rilasciati pubblicamente su HuggingFace e il loro utilizzo per la rilevazione del plagio richiede la costruzione di una pipeline sperimentale basata su prompt engineering, come discusso nella Sezione 3.7.1.

CodeLlama è una famiglia di LLM sviluppata da Meta AI nel 2023, costruita tramite fine-tuning di Llama 2 su codice sorgente. La famiglia è disponibile in taglie da 7B, 13B, 34B e 70B parametri, addestrate su sequenze di 16.384 token con supporto a contesti fino a 100.000 token. Per il task della SCPD la variante rilevante è CodeLlama-Instruct, allineata tramite instruction tuning per seguire istruzioni in linguaggio naturale, rendendola adatta a ricevere prompt strutturati e restituire risposte argomentate. I modelli sono rilasciati sotto una licenza custom di Meta che ne permette l'uso commerciale e la ricerca, e sono eseguibili localmente su singola GPU a partire dalla variante 7B.

== Strumenti Commerciali e Piattaforme Integrate
=== MOSS (Measure Of Software Similarity) #footnote[https://theory.stanford.edu/~aiken/moss/]
Sviluppato nel 1997 all'Università di Stanford, MOSS è probabilmente lo strumento di rilevazione del plagio più diffuso a livello accademico.

MOSS utilizza un approccio basato sui token. Dopo aver effettuato una procedura di _lexing_ sul codice e rimosso spazi e commenti, lo strumento non utilizza algoritmi di string matching completi (come GST), ma implementa una tecnica proprietaria basata sul *Winnowing* (Document Fingerprinting). Il codice viene diviso in _k-grammi_ di token; di questi viene calcolato l'hash e solo un sottoinsieme selezionato (la fingerprint) viene memorizzato e confrontato.

Questo approccio rende MOSS estremamente scalabile e veloce, permettendo di confrontare migliaia di file in pochi secondi. È robusto contro l'offuscamento lessicale (renaming) e il riordinamento del codice, poiché le "impronte" sono invarianti rispetto alla posizione. Un numero molto vasto di linguaggi è supportato da MOSS, tra questi C, C++, Java e Python.

MOSS viene fornito come servizio online (tramite script di invio al server di Stanford) e non è open-source. Inoltre, come tutti i sistemi token-based, può essere ingannato da pesanti modifiche strutturali o dall'inserimento massiccio di codice "rumore" che altera i _k-grammi_. Inoltre, studi recenti hanno rilevato una grave criticità legata all'Intelligenza Artificiale generative: MOSS può essere facilmente eluso utilizzando LLM pre-addestrati. È stato dimostrato, ad esempio, che studenti che utilizzano modelli come GPT-J per completare assegnamenti di programmazione introduttiva riescono a bypassare completamente i controlli di MOSS @biderman2022fooling. Questo accade poiché il codice generato da tali modelli risulta sufficientemente diversificato nella struttura e manca di quei "segnali" tipici che i sistemi algoritmici tradizionali utilizzano per identificare il plagio manuale.
=== CodeMatch
CodeMatch è uno strumento integrato nella suite commerciale CodeSuite, sviluppata originariamente da Bob Zeidman e attualmente gestita dalla Software Analysis & Forensic Engineering (S.A.F.E.) Corporation per la rilevazione di violazioni di copyright. A differenza di molte soluzioni accademiche open source, è un software proprietario distribuito come file eseguibile binario compatibile esclusivamente con sistemi Windows. L'utilizzo è gratuito per l'analisi di volumi di codice sorgente inferiori a 1 Mbyte, mentre richiede l'acquisto di una licenza per quantità superiori.

L'approccio algoritmico di CodeMatch si basa sull'analisi simultanea di quattro diversi livelli di similarità. Per identificare le copie, lo strumento misura: la correlazione delle singole istruzioni (statement correlation), la correlazione di commenti e stringhe di testo, la correlazione degli identificatori e la correlazione delle sequenze di istruzioni.

Lo strumento permette il confronto automatico di migliaia di file contenuti in cartelle e sottocartelle. Tra le sue caratteristiche principali include la generazione di report che classificano le coppie di file più correlate, insieme ai loro elementi specifici che hanno segnalato il plagio. Lo strumento è estremamente versatile e supporta ben 36 linguaggi di programmazione.

Test empirici condotti in letteratura su dataset di riferimento (come algoritmi in Java e in C modificati con varie tecniche di plagio) confermano una buona accuratezza generale dello strumento @plagDetectionSurvey. Tuttavia, mostra una vulnerabilità specifica contro l'offuscamento lessicale (il 3° tipo di plagio), ovvero la ridenominazione massiccia degli identificatori. Poiché i suoi algoritmi tentano di trovare corrispondenze dirette tra i nomi delle variabili, un'alterazione sistematica di questi elementi abbassa inevitabilmente le percentuali di rilevamento. Questo limite diventa particolarmente marcato quando si analizzano file di grandi dimensioni contenenti un elevato numero di identificatori

=== Codequiry
Codequiry si posiziona come una soluzione commerciale di fascia enterprise progettata per superare i limiti degli strumenti accademici tradizionali attraverso un approccio multi-livello alla rilevazione della similarità. A differenza di tool come MOSS, Codequiry adotta un motore di analisi proprietario denominato Zeus, basato sulla tecnologia Abstract Syntax Tree (AST), che permette di ignorare le modifiche puramente estetiche (offuscamento lessicale) come la ridenominazione di variabili, il cambiamento della formattazione o la manipolazione dei commenti per concentrarsi esclusivamente sulla struttura logica del programma @codequiry_checker_2026.

Il sistema opera attraverso una "tripletta" di controlli simultanei: il Peer Check per il confronto interno tra le sottomissioni di un gruppo, il Database Check per il riscontro con repository storici e il Web Check. Quest'ultimo rappresenta uno dei punti di forza dello strumento, in quanto si avvale di un livello di indicizzazione su scala globale che interroga oltre 1 trilione (1T+) di sorgenti di codice e più di 100 milioni di repository pubblici, inclusi GitHub, GitLab, Stack Overflow e portali di tutoring come Chegg.

Dal punto di vista dell'efficacia, uno studio tecnico condotto dalla stessa piattaforma riporta metriche di performance elevate su un campione di 100 sottomissioni: un'accuratezza complessiva del 95%, con una precisione del 95,5% e un tasso di Recall (recupero) del 98,9%. Oltre ai report testuali, Codequiry introduce strumenti di visualizzazione avanzati, come i grafici di clustering 2D, che permettono ai docenti di identificare istantaneamente pattern di collaborazione illecita o reti di plagio all'interno di classi numerose @codequiry_effectiveness_2023.

=== LLM Commerciali: La famiglia GPT
L’introduzione di modelli linguistici di grandi dimensioni, in particolare GPT-4o (Generative Pre-trained Transformer) di OpenAI, rappresenta un cambio di paradigma rispetto ai sistemi di rilevazione deterministici discussi finora. A differenza degli strumenti basati su token o grafi, GPT non richiede una pipeline di analisi statica predefinita; la sua capacità di identificare il plagio emerge dalla vasta pre-esposizione a miliardi di parametri che includono codice sorgente e documentazione tecnica in molteplici linguaggi.

Il compito di Source Code Plagiarism Detection (SCPD) viene affrontato da questi modelli prevalentemente in modalità zero-shot, ovvero senza necessità di un addestramento specifico sul problema, ma attraverso l'utilizzo di prompt strutturati in linguaggio naturale che descrivono il task e i frammenti da confrontare.

L'efficacia di questo approccio è stata validata empiricamente @llm2024scpd. I risultati mostrano che GPT-4o è in grado di raggiungere un'accuratezza complessiva del 78,70%, con un F1 score di 86,97%. Il valore aggiunto di questo modello risiede nella capacità di superare le barriere sintattiche che limitano strumenti come JPlag o MOSS: GPT-4o dimostra prestazioni superiori nel rilevamento dei cloni di Tipo 4 (variazioni semantiche profonde), riuscendo a identificare la similarità logica anche quando l'algoritmo viene completamente riscritto o tradotto tra linguaggi di programmazione differenti (plagio cross-language).

Tuttavia, l'adozione di GPT in ambito accademico e legale deve affrontare sfide strutturali. Sebbene il modello offra una spiegabilità nativa — producendo ragionamenti in linguaggio naturale che motivano il verdetto di plagio — esso rimane un sistema non deterministico. Lo stesso prompt può generare risultati leggermente diversi se eseguito più volte, complicando la piena riproducibilità degli esperimenti richiesti in ambito forense. Inoltre, persiste il rischio di falsi positivi, poiché il modello tende a segnalare come sospetti alcuni pattern di implementazione estremamente comuni, come gli algoritmi di ordinamento standard, anche in assenza di un reale riutilizzo non autorizzato.
== Tabella Comparativa degli Strumenti
A completamento della rassegna degli strumenti, vengono presentate due tabelle di sintesi. La prima si focalizza sulle soluzioni accademiche e open source, mentre la seconda analizza le piattaforme commerciali e i modelli linguistici di grandi dimensioni. Questo confronto permette di valutare rapidamente il compromesso tra potenza algoritmica, facilità d'uso e capacità di spiegazione del verdetto.

#figure(
  caption: [Confronto tra strumenti accademici e open source],
  block(width: 100%)[
    #set text(size: 7.5pt)
    #table(
      columns: (0.8fr, 1.2fr, 1fr, 0.8fr, 1.2fr),
      inset: 3pt,
      align: horizon,
      fill: (x, y) => if y == 0 { gray.lighten(80%) },
      table.header(
        [*Tool*], [*Linguaggi*], [*Tipi di Clone*], [*Spieg.*], [*Esperienza Utente*]
      ),
      [*JPlag*], [19], [Tipo 1, 2, 3 (Parziale)], [Alta], [UI locale, grafici similarità e cluster],
      [*SIM*], [9], [Tipo 1, 2], [Alta], [CLI, output testuale semplice],
      [*Plaggie*], [1 (Java)], [Tipo 1, 2], [Alta], [CLI standalone, report di base],
      [*Dolos*], [Agnostico (Tree-sitter)], [Tipo 1, 2, 3 (Parziale)], [Alta], [Web app, dashboard, grafici 2D],
      [*Oreo*], [Java], [Tipo 1, 2, 3 e "Twilight Zone"], [Nulla], [CLI e modello Deep Learning, niente report oltre la percentuale di similarità],
      [*CodeBERT*], [6], [Tutti], [Nulla], [Assente, richiede la creazione autonoma di una pipeline],
      [*CodeLlama*], [Multi-linguaggio (Universale)], [Tutti (Tipo 1-4)], [Altissima], [Output testuale: Richiede pipeline per generare report]
    )
  ]
)

#figure(
  caption: [Confronto tra strumenti commerciali],
  block(width: 100%)[
    #set text(size: 7.5pt)
    #table(
      columns: (0.8fr, 1.2fr, 1fr, 0.8fr, 1.2fr),
      inset: 3pt,
      align: horizon,
      fill: (x, y) => if y == 0 { gray.lighten(80%) },
      table.header(
        [*Tool*], [*Linguaggi*], [*Tipi di Clone*], [*Spieg.*], [*Esperienza Utente*]
      ),
      [*MOSS*], [25], [Tipo 1, 2, 3 (Parziale)], [Alta], [Servizio web, report HTML remoto],
      [*CodeMatch*], [50+], [Tipo 1, 2, 3], [Altissima], [App Windows, report multi-livello forensic-oriented],
      [*Codequiry*], [65+], [Tutti (1-4)], [Alta], [Altissima (SaaS). Dashboard interattive, funzionalità di esportazione],
      [*GPT-4o*], [Universale (Cross-language)], [Tutti (Tipo 1-4)], [Altissima], [Chat/API, ragionamento discorsivo]
    )
  ]
)