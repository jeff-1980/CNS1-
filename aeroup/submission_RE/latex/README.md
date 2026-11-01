# LaTeX source (Elsevier elsarticle)

Compile: `pdflatex manuscript.tex` twice (no BibTeX step; the reference list is written as `thebibliography`
in order of first citation, Elsevier numbered style). Needs the standard TeX Live packages elsarticle, lineno,
booktabs, adjustbox, xurl, hyperref, amsmath.

`manuscript.tex` is generated from `manuscript_v5_keyed.md` by `analysis/build_v5.py` with the verified
reference strings in `analysis/refs_v5.py`; edit the Markdown source and rebuild rather than editing the .tex by hand.
