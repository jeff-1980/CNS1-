"""v6: as build_v5, plus (i) every table and figure placed after the paragraph that first cites it, (ii) double
spacing via template_v6.tex, (iii) --anon writes the double-blind manuscript (no author block, no CRediT/competing
interest/funding/acknowledgements, code link withheld).
Usage: python build_v6.py <keyed.md> <out_numbered.md> <out.tex> [--anon]"""
import re
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from refs_v5 import REFS  # noqa: E402

SRC, OUT_MD, OUT_TEX = sys.argv[1:4]
ANON = "--anon" in sys.argv[4:]
s = open(SRC, encoding="utf-8").read()
CITE = re.compile(r"\[@([^\]]+)\]")
order = []
for m in CITE.finditer(s):
    for k in m.group(1).split(","):
        if k not in order:
            order.append(k)
missing = [k for k in order if k not in REFS]
assert not missing, missing
unused = [k for k in REFS if k not in order]
assert not unused, unused
num = {k: i + 1 for i, k in enumerate(order)}


def compress(ns):
    ns = sorted(set(ns))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(f"{ns[i]}–{ns[j]}" if j - i >= 2 else ", ".join(str(x) for x in ns[i:j + 1]))
        i = j + 1
    return "[" + ",".join(out).replace(", ", ",") + "]"


# ---------- numbered Markdown ----------
md = CITE.sub(lambda m: compress([num[k] for k in m.group(1).split(",")]), s)
md = md.rstrip() + "\n\n## References\n\n" + "\n\n".join(f"[{num[k]}] {REFS[k]}" for k in order) + "\n"
open(OUT_MD, "w", encoding="utf-8").write(md)
print("references:", len(order))

# ---------- LaTeX ----------
t = CITE.sub(lambda m: "\\cite{" + ",".join(m.group(1).split(",")) + "}", s)


def cut(text, a, b=None):
    i = text.index(a)
    j = text.index(b, i) if b else len(text)
    return text[i:j]


title = t.splitlines()[0].lstrip("# ").strip()
highlights = [h[2:].strip() for h in cut(t, "## Highlights", "## Abstract").splitlines() if h.startswith("- ")]
abstract = cut(t, "## Abstract", "## Keywords").split("\n", 1)[1].strip()
keywords = cut(t, "## Keywords", "## 1. Introduction").split("\n", 1)[1].strip().split("; ")
main = cut(t, "## 1. Introduction", "## CRediT")
back = cut(t, "## CRediT", "## Table 1.")
if ANON:
    back = cut(back, "## Data and code availability")
    back = re.sub(r"available at https://github\.com/\S+ \(version used for this paper: release v1\.0\.0, commit dfcc49c\)\.",
                  "available in a public GitHub repository (release v1.0.0); the link is withheld for double-blind review and is given on the title page.", back)
    assert "github.com" not in back, back
tables = cut(t, "## Table 1.", "## Figure captions")
figs = cut(t, "## Figure captions")

# math for the LaTeX path
MATH = [
    ("*C* = *D*~*m*~ / *E*,          (1)", "\\begin{equation}\nC = D_{m} / E ,\n\\label{eq:C}\n\\end{equation}"),
    ("*T* = *P* / *ω*,   *ω* = 2π*n* / 60          (2)",
     "\\begin{equation}\nT = P/\\omega , \\qquad \\omega = 2\\pi n/60\n\\label{eq:T}\n\\end{equation}"),
    ("*E* = Σ*P*~obs~ / Σ*P*~exp~ − 1", "$E = \\sum P_\\mathrm{obs} / \\sum P_\\mathrm{exp} - 1$"),
    ("*D*~*m*~ = (mean *T*~obs~^*m*^ / mean *T*~exp~^*m*^)^1/*m*^ − 1",
     "$D_{m} = \\left( \\overline{T_\\mathrm{obs}^{\\,m}} \\big/ \\overline{T_\\mathrm{exp}^{\\,m}} \\right)^{1/m} - 1$"),
    ("Δ*T*/*T* = (2/3) Δ*P*/*P* and Δ*ω*/*ω* = (1/3) Δ*P*/*P*",
     "$\\Delta T/T = \\tfrac{2}{3}\\,\\Delta P/P$ and $\\Delta\\omega/\\omega = \\tfrac{1}{3}\\,\\Delta P/P$"),
    ("(Δ*ω*/*ω* = (1/3) Δ*P*/*P*)", "($\\Delta\\omega/\\omega = \\tfrac{1}{3}\\,\\Delta P/P$)"),
    ("*D*~*m*~", "$D_{m}$"), ("*T*~1~", "$T_1$"), ("*Kω*^2^", "$K\\omega^2$"), ("*Kω*^3^", "$K\\omega^3$"),
    ("K*ω*^2^", "$K\\omega^2$"), ("(1 + u)^1/3^", "$(1+u)^{1/3}$"), ("T^m^", "$T^m$"), ("D_m", "$D_{m}$"),
    ("Kω²", "$K\\omega^2$"), ("*m*", "$m$"),
]


def domath(x):
    for a, b in MATH:
        x = x.replace(a, b)
    return x


def pandoc(x):
    r = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars+raw_tex-auto_identifiers", "-t", "latex",
                        "--wrap=preserve"], input=x, capture_output=True, text=True, check=True)
    return r.stdout


def sections(x):
    x = re.sub(r"^## (\d+)\. (.*)$", r"# \2", x, flags=re.M)
    x = re.sub(r"^### \d+\.\d+ (.*)$", r"## \1", x, flags=re.M)
    return x


main_tex = pandoc(sections(domath(main)))
main_tex = main_tex.replace("\\section{", "\\section{").replace("\\begin{itemize}\n\\tightlist", "\\begin{itemize}")

# back matter: unnumbered sections
back_tex = pandoc(domath(re.sub(r"^## ", "# ", back, flags=re.M))).replace("\\section{", "\\section*{")
back_tex = re.sub(r"(?<!\{)(https?://[^\s)]+?)(\.?)(?=[\s)])", r"\\url{\1}\2", back_tex)

# tables -> pandoc longtables wrapped as floats is awkward; build simple tabulars instead
def md_table(block):
    rows = [r for r in block.splitlines() if r.startswith("|")]
    rows = [r for r in rows if not re.match(r"^\|[-| ]+\|$", r)]
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    return cells


def esc(c):
    c = domath(c)
    c = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", c)
    c = re.sub(r"\^([ab])\^", r"$^{\\mathrm{\1}}$", c)
    return c.replace("%", "\\%").replace("−", "$-$")


def table_tex(block, label):
    head = block.splitlines()[0].lstrip("# ").strip()
    cap = head.split(". ", 1)[1]
    cells = md_table(block)
    ncol = len(cells[0])
    note = "\n".join(l for l in block.splitlines()[1:] if l.strip() and not l.startswith("|"))
    body = []
    for i, r in enumerate(cells):
        body.append(" & ".join(esc(c) for c in r) + " \\\\")
        if i == 0:
            body.append("\\midrule")
    out = ("\\begin{table}[htbp]\n\\centering\n\\caption{" + esc(cap) + "}\n\\label{" + label + "}\n\\small\n"
           "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{" + "l" * ncol + "}\n\\toprule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n")
    if note:
        out += "\n\\par\\smallskip\\parbox{0.95\\textwidth}{\\footnotesize " + esc(note.replace("_", "\\_")) + "}\n"
    return out + "\\end{table}\n"


t1 = cut(tables, "## Table 1.", "## Table 2.")
t2 = cut(tables, "## Table 2.")
tables_tex = table_tex(t1, "tab:install") + "\n" + table_tex(t2, "tab:effects")

# figures
fig_tex = []
for m in re.finditer(r"!\[\]\(figs/(\w+)\.png\)\n\*\*Fig\. (\d+)\.\*\* (.*)", figs):
    stem, n, cap = m.groups()
    cap_tex = pandoc(domath(cap)).strip()
    fig_tex.append("\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/" + stem +
                   ".pdf}\n\\caption{" + cap_tex + "}\n\\label{fig:" + n + "}\n\\end{figure}\n")
assert len(fig_tex) == 7, len(fig_tex)


def bibesc(x):
    x = x.replace("&", "\\&").replace("%", "\\%")
    x = re.sub(r"(https?://\S+?)\.?$", lambda m: "\\url{" + m.group(1) + "}.", x)
    return x.replace(" – ", " -- ").replace("–", "--")


# ---------- place floats after the paragraph that first cites them ----------
def place(body, floats):
    paras = body.split("\n\n")
    inserts = {}
    for pat, ftex in floats:
        hit = None
        for i, p in enumerate(paras):
            if re.search(pat, p):
                hit = i
                break
        assert hit is not None, pat
        depth = 0  # move past an open list so the float is not placed inside it
        for q in paras[:hit + 1]:
            depth += q.count("\\begin{itemize}") + q.count("\\begin{enumerate}")
            depth -= q.count("\\end{itemize}") + q.count("\\end{enumerate}")
        while depth > 0 and hit + 1 < len(paras):
            hit += 1
            depth += paras[hit].count("\\begin{itemize}") - paras[hit].count("\\end{itemize}")
        inserts.setdefault(hit, []).append(ftex)
    out = []
    for i, p in enumerate(paras):
        out.append(p)
        out.extend(inserts.get(i, []))
    return "\n\n".join(out)


tab_list = [(r"Table~?\s?1(?![0-9])", table_tex(t1, "tab:install")), (r"Table~?\s?2(?![0-9])", table_tex(t2, "tab:effects"))]
fig_list = [(rf"Figs?\.~?\s?{i + 1}(?![0-9])", f) for i, f in enumerate(fig_tex)]
main_tex = place(main_tex, tab_list + fig_list)
for f in fig_tex + [x for _, x in tab_list]:
    assert main_tex.count(f) == 1
main_tex = main_tex.replace("\\begin{figure}[htbp]", "\\begin{figure}[!htbp]").replace("\\begin{table}[htbp]", "\\begin{table}[!htbp]")

bib = "\n".join(f"\\bibitem{{{k}}} {bibesc(REFS[k])}" for k in order)
hl = "\n".join("\\item " + domath(h).replace("%", "\\%") for h in highlights)
tex = open(__file__.rsplit("/", 1)[0] + "/template_v6.tex", encoding="utf-8").read()
if ANON:
    tex = re.sub(r"\\author\{.*?\n\n", "", tex, flags=re.S)
    tex = tex.replace("[hidelinks]{hyperref}", "[hidelinks,pdfauthor={}]{hyperref}")
for a, b in [("@@TITLE@@", title), ("@@ABSTRACT@@", pandoc(domath(abstract)).strip()),
             ("@@KEYWORDS@@", " \\sep ".join(keywords)), ("@@HIGHLIGHTS@@", hl), ("@@MAIN@@", main_tex),
             ("@@BACK@@", back_tex), ("@@BIB@@", bib), ("@@TABLES@@", ""), ("@@FIGURES@@", "")]:
    tex = tex.replace(a, b)
open(OUT_TEX, "w", encoding="utf-8").write(tex)
print("wrote", OUT_MD, OUT_TEX)
