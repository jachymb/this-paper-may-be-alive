# This Paper May Be Alive

*Jáchym Barvínek and Claude Fable*

It is sometimes suggested that a large language model might be conscious while it is computing its reply.
This paper examines the suggestion by removing the computer: it takes a 6.4-million-parameter language model of the
Qwen3 architecture ([C10X/checkpoint-27564](https://huggingface.co/C10X/checkpoint-27564)) which, given the prompt
`Say "I am alive"`, replies `"I am alive"`, and writes out the entire computation that produces this reply as explicit
arithmetic: every multiplication, addition, normalisation, exponentiation and maximum, with every intermediate number
printed. If the computation is sufficient for the model to be conscious, the same computation on paper should be
sufficient for the paper.

## The paper

**[Download main.pdf](https://github.com/jachymb/this-paper-may-be-alive/releases/latest/download/main.pdf)**
(81,655 A4 pages, 808 MB; open it with a viewer that loads pages lazily, e.g. SumatraPDF or Acrobat).

**Part I** (the first ten pages) is the paper proper: introduction, a discussion, the architecture, the rounding conventions that make the
printed arithmetic exactly reproducible, the results and the bibliography.

**Part II** is the computation:
588,992 equations containing 59,392,000 explicit products, organised by token position and layer, ending with the
prediction of each of the five output tokens.

## Repository layout

| file | content |
|---|---|
| `main.tex`, `refs.bib` | Part I and the document skeleton |
| `gen/` | Part II, generated LaTeX (1.3 GB; included so that the paper's source is complete) |
| `forward.py` | the model's forward pass in numpy, with every intermediate value rounded as the paper describes |
| `generate.py` | runs the rounded forward pass, checks it against the `transformers` reference, writes `gen/` |
| `build.sh` | `generate.py` followed by three `pdflatex` passes and `bibtex` |
| `llm.py` | minimal demo: runs the model with `transformers` |

## Building

Requirements: [uv](https://docs.astral.sh/uv/) (it fetches `torch`, `transformers` and `numpy`), a TeX distribution
with `pdflatex` and `bibtex` (MiKTeX or TeX Live), and the model in the Hugging Face cache (downloaded on first run).

```
bash build.sh --sample   # sample.pdf: Part I plus a few hundred pages of Part II, about three minutes
bash build.sh            # main.pdf: everything; generation takes about a minute, each pdflatex pass about half an hour
```

The generator asserts that the rounded arithmetic produces the same five tokens as the float32 reference implementation
before writing anything.

## The Inspiration
Huge shoutout to /u/erraticpulse- who created the viral meme. This work follows the path they hinted at.

![](https://i.redd.it/nf6dcl8l7ygg1.png)

