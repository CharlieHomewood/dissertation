
#let script-size = 7.97224pt
#let footnote-size = 8.50012pt
#let small-size = 9.24994pt
#let normal-size = 10.00002pt
#let large-size = 11.74988pt

// This function gets your whole document as its `body` and formats
// it as an article in the style of the American Mathematical Society.
#let ams-article(
  // The article's title.
  title: "Paper title",

  // An array of authors. For each author you can specify a name,
  // department, organization, location, and email. Everything but
  // but the name is optional.
  authors: (),

  // Your article's abstract. Can be omitted if you don't have one.
  abstract: none,

  // The article's paper size. Also affects the margins.
  paper-size: "a4",

  // The path to a bibliography file if you want to cite some external
  // works.
  bibliography-file: none,

  // The document's content.
  body,
) = {
  // Formats the author's names in a list with commas and a
  // final "and".
  let names = authors.map(author => author.name)
  let author-string = if authors.len() == 2 {
    names.join(" and ")
  } else {
    names.join(", ", last: ", and ")
  }

  // Set document metadata.
  set document(title: title, author: names)

  // Set the body font. AMS uses the LaTeX font.
  set text(size: normal-size, font: "New Computer Modern")

  // Configure the page.
  set page(
    paper: paper-size,
    // The margins depend on the paper size.
    margin: 
      (
        top: 85pt,
        left: 85pt,
        right: 85pt,
        bottom: 85pt,
      ),

    // The page header should show the page number and list of
    // authors, except on the first page. The page number is on
    // the left for even pages and on the right for odd pages.
    header-ascent: 14pt,
    header: locate(loc => {
      let i = counter(page).at(loc).first()
      if i == 1 { return }
      set text(size: script-size)
      grid(
        columns: (6em, 1fr, 6em),
        if calc.even(i) [#i],
        align(center, upper(title)),
        if calc.odd(i) { align(right)[#i] }
      )
    }),

    // On the first page, the footer should contain the page number.
    footer-descent: 12pt,
    footer: locate(loc => {
      let i = counter(page).at(loc).first()
      if i == 1 {
        align(center, text(size: script-size, [#i]))
      }
    })
  )

  // Configure headings.
  set heading(numbering: "1.")
  show heading: it => {
    // Create the heading numbering.
    let number = if it.numbering != none {
      counter(heading).display(it.numbering)
      h(7pt, weak: true)
    }

    // Level 1 headings are centered and smallcaps.
    // The other ones are run-in.
    set text(size: normal-size, weight: 700)
    if it.level == 1 {
      set align(center)
      set text(size: large-size)
      smallcaps[
        #v(20pt)
        #number
        #it.body
        #v(5pt)
      ]
      counter(figure.where(kind: "theorem")).update(0)
    } else {
      set par(first-line-indent: 0em)
      set align(left)
      v(22pt, weak: true)
      number
      let styled = if it.level == 2 { strong } else { emph }
      styled(it.body)
      h(7pt, weak: true)
    }
  }

  // Configure lists and links.
  set list(indent: 12pt, body-indent: 5pt)
  set enum(indent: 12pt, body-indent: 5pt)
  show list: it => {
    it
    par(text(size: 0pt, ""))
  }
  show enum: it => {
    it
    par(text(size: 0pt, ""))
  }
  show link: it => {
    set text(font: "New Computer Modern Mono", fill: blue.darken(20%))
    underline(it)
  }

  // Configure equations.
  show math.equation: set block(below: 20pt, above: 20pt)
  show math.equation: set text(weight: 400)

  // Configure citation and bibliography styles.
  set bibliography(style: "apa", title: "References")

  show figure: it => {
    show: pad.with(x: 23pt)
    set align(center)

    v(12.5pt, weak: true)

    // Display the figure's body.
    it.body

    // Display the figure's caption.
    if it.has("caption") {
      v(if it.has("gap") { it.gap } else { 17pt }, weak: true)
      it.caption
    }

    v(15pt, weak: true)
  }

  // Theorems.
  show figure.where(kind: "theorem"): it => block(above: 11.5pt, below: 11.5pt, {
    strong({
      it.supplement
      if it.numbering != none {
        [ ]
        counter(heading).display()
        it.counter.display(it.numbering)
      }
      [.]
    })
    emph(it.body)
  })

  v(0.5fr)
  // Display the title and authors.
  align(center, {
    text(size: 16pt, weight: 700, title)
    v(25pt, weak: true)
    text(size: 10pt, author-string)
    v(25pt, weak: true)
  })

  // Configure paragraph properties.
  set par(first-line-indent: 1.2em, justify: true, leading: 0.58em)
  show par: set block(spacing: 1.5em)

  v(30pt)
  // Display the abstract
  if abstract != none {

    align(center)[
      #text(size: large-size, weight: 700, "Abstract")  // centered bold header
    ]

    v(20pt, weak: true)
    set text(small-size)
    show: pad.with(x: 35pt)
    abstract
  }

  v(30pt)
  {
    set par(first-line-indent: 0em, justify: false)
    show par: set block(spacing: 0.5em)

    align(center)[
      #block(width: 70%)[
        #outline(title: "Table of Contents", depth: 3, indent: 2em)
      ]
    ]
  }

  v(1fr)
  pagebreak()

  // Display the article's contents.
  body

  // Display the bibliography, if any is given.
  if bibliography-file != none {
    show: page({
      v(20pt)
      show bibliography: set text(8.5pt)
      show bibliography: pad.with(x: 0.5pt)
      bibliography(bibliography-file)
    })
  }

}

// The ASM template also provides a theorem function.
#let theorem(body, numbered: true) = block(
  above: 20pt, below: 20pt,
  {
    figure(
      body,
      kind: "theorem",
      supplement: [Theorem],
      numbering: if numbered { "1" }
    )
  }
)

// And a function for a proof.
#let proof(body) = block(spacing: 11.5pt, above: 20pt, below: 20pt, {
  emph[Proof.]
  [ ] + body
  h(1fr)
  box(scale(160%, origin: bottom + right, sym.square.stroked))
})
