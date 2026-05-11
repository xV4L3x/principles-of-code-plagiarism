#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(languages: codly-languages, number-format: none, display-icon: false, display-name: false, zebra-fill: none, fill: rgb("#f5f5f5"))
#show raw.where(block: false): it => {
  box(fill: rgb("#f5f5f5"), inset: (x: 3pt, y: 2pt), radius: 2pt, it)
}

// --- 1. CONFIGURAZIONE GENERALE ---
#set page(
  paper: "a4",
  margin: (x: 1.5cm, y: 2cm),
  footer: context [
    #align(center)[
      #counter(page).display()
    ]
  ]
)

#set text(
  font: "New Computer Modern", // Simile al default di LaTeX
  size: 10pt,
  lang: "it"
)

#set par(
  justify: true, // Giustifica il testo (fondamentale per le colonne)
  leading: 0.65em,
)

// Numerazione delle sezioni (1.1, 1.2, ecc.)
#set heading(numbering: "1.1")

// --- 2. INTESTAZIONE (Titolo, Autori, Abstract) ---
// Questo blocco rimane a colonna singola
#align(center)[
  #text(17pt, weight: "bold")[Stato dell'arte degli strumenti e delle tecniche di rilevazione del plagio del codice sorgente]
  
  #v(1em) // Spazio verticale
  
  #text(12pt, weight: "regular")[Valerio Pio De Nicola]
  
  #text(style: "italic")[
    Dipartimento di Scienze, Università di Bologna \
    valeriopio.denicola\@studio.unibo.it
  ]
]

#v(2em)

// Abstract stilizzato
#align(center)[
  #block(width: 85%)[ // Restringe l'abstract all'85% della pagina
    #text(weight: "bold")[Abstract] \
    #v(0.5em)
    #text(size: 9pt)[
      Qui ci andrà l'abstract
    ]
  ]
]

#v(3em)

// --- 3. CONFIGURAZIONE DUE COLONNE ---
// Tutto ciò che segue questo comando sarà diviso in 2 colonne
#show: rest => columns(2, gutter: 1em, rest)


// --- 4. CONTENUTO DEL PAPER ---

= Introduzione
#include "chapter-1/main.typ"
= Il Plagio nel Codice Sorgente: Tassonomia e Offuscamento
#include "chapter-2/main.typ"
= Tecniche e Algoritmi di Rilevamento (Approccio Teorico)
#include "chapter-3/main.typ"
= Rassegna degli Strumenti (State of the Art)
#include "chapter-4/main.typ"
= Metodologie di Valutazione e Dataset
== Dataset di Riferimento
== Metriche di Performance
== Robustezza
= Nuove Sfide: L'Era dell'AI Generativa
== Il "Plagio" da AI
== Rilevamento AI-Generated Code
== Differenza tra rilevazione di plagio (copia) e rilevazione di AI (generazione)
= Conclusioni e Sviluppi Futuri
#bibliography("references.bib", style: "ieee")
