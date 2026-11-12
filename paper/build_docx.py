"""Build paper/Final_IEEE_Paper.docx, an editable Word copy of paper/paper.tex.

Run after paper/build_paper.py (it needs paper.tex and paper.aux). Steps:
  1. the two TikZ diagrams are compiled on their own and saved as PNG;
  2. citations, table/figure/algorithm references and captions get their final numbers from paper.aux;
  3. IEEE-only constructs (author blocks, symbol lists, algorithmic, minipages) become plain LaTeX;
  4. pandoc converts the result to .docx (equations become Word equations);
  5. python-docx adds the author block and sets Times New Roman.
Usage: python paper/build_docx.py
"""
import re
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

PAPER = Path(__file__).resolve().parent
FIG = PAPER.parent / "figures"
OUT = PAPER / "Final_IEEE_Paper.docx"
AUTHORS = [("Sehjodh Grewal", "sg2827@srmist.edu.in"), ("Devanshu Kumar", "dk5504@srmist.edu.in"),
           ("Deepankar", "dz9629@srmist.edu.in"), ("Tanmay Singh", "ts4494@srmist.edu.in"),
           ("Dr. Banu Priya Prathaban", "banuprip2@srmist.edu.in")]
AFFIL = ["Department of Networking and Communications, School of Computing",
         "SRM Institute of Science & Technology", "Kattankulathur, Chennai, India"]


def braced(s, i):
    """Return the index just after the brace group that starts at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return j + 1
    raise ValueError("unbalanced braces")


def strip_wrapper(s, head):
    """Remove `head{...}` wrappers such as \\resizebox{\\linewidth}{!}{...}, keeping the contents."""
    while head in s:
        i = s.index(head) + len(head) - 1
        j = braced(s, i)
        s = s[:s.index(head)] + s[i + 1:j - 1] + s[j:]
    return s


def tikz_pngs(tex, work):
    pre = ("\\documentclass[border=6pt]{standalone}\n\\usepackage{amsmath,amssymb}\n\\usepackage{tikz}\n"
           "\\usetikzlibrary{arrows.meta,positioning,fit,calc}\n\\begin{document}\n")
    out = []
    for k, m in enumerate(re.finditer(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tex, re.S), 1):
        f = work / f"tikz{k}.tex"
        f.write_text(pre + m.group(0) + "\n\\end{document}\n")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f.name], cwd=work, check=True,
                       capture_output=True)
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", f"tikz{k}.pdf", f"tikz{k}"], cwd=work,
                       check=True)
        out.append((m.group(0), work / f"tikz{k}.png"))
    return out


def convert(tex, work):
    aux = (PAPER / "paper.aux").read_text()
    labels = {k: v for k, v in re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux)}
    keys = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    num = {k: i + 1 for i, k in enumerate(keys)}

    body = tex[tex.index("\\begin{document}") + len("\\begin{document}"):tex.index("\\end{document}")]
    title = re.search(r"\\title\{(.*?)\}\n", body, re.S).group(1)
    body = body[body.index("\\maketitle") + len("\\maketitle"):]

    for code, png in tikz_pngs(body, work):
        body = body.replace(code, f"\\includegraphics{{{png}}}")
    body = body.replace("\\graphicspath{{../figures/}}", "")
    body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]+\.png)\}",
                  lambda m: "\\includegraphics{" + str(PAPER / m.group(2) if (PAPER / m.group(2)).exists()
                                                       else FIG / m.group(2)) + "}", body)
    for head in ("\\resizebox{\\columnwidth}{!}{", "\\resizebox{\\linewidth}{!}{", "\\resizebox{\\textwidth}{!}{"):
        body = strip_wrapper(body, head)
    body = re.sub(r"\\begin\{minipage\}(\[[^\]]*\])?\{[^}]*\}", "", body).replace("\\end{minipage}", "")

    body = re.sub(r"\\cite\{([^}]+)\}", lambda m: ", ".join(f"[{num[k.strip()]}]" for k in m.group(1).split(",")), body)
    body = re.sub(r"\\(?:eq)?ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), body)

    # number the captions the IEEE way: "TABLE I: ...", "Fig. 1. ...", "Algorithm 1: ..."
    def number_captions(env, fmt):
        nonlocal body
        def fix(m):
            blk = m.group(0)
            lab = re.search(r"\\label\{([^}]+)\}", blk)
            n = labels.get(lab.group(1), "") if lab else ""
            return re.sub(r"\\caption\{", lambda _: "\\caption{" + fmt.format(n), blk, count=1)
        body = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", fix, body, flags=re.S)
    number_captions("table\\*?", "TABLE {}: ")
    number_captions("figure\\*?", "Fig. {}. ")

    def algo(m):
        blk = m.group(0)
        cap = re.search(r"\\caption\{(.*?)\}\n", blk).group(1)
        lines = re.search(r"\\begin\{algorithmic\}(?:\[\d\])?(.*?)\\end\{algorithmic\}", blk, re.S).group(1)
        items = []
        for ln in lines.strip().split("\n"):
            ln = ln.strip()
            ln = re.sub(r"^\\REQUIRE\s*", r"\\textbf{Require:} ", ln)
            ln = re.sub(r"^\\STATE\s*", "", ln)
            ln = re.sub(r"^\\FOR\{(.*)\}$", r"\\textbf{for} \1 \\textbf{do}", ln)
            ln = re.sub(r"^\\ENDFOR$", r"\\textbf{end for}", ln)
            ln = re.sub(r"^\\RETURN\s*", r"\\textbf{return} ", ln)
            ln = ln.replace("\\TO", "to")
            items.append("\\item " + ln)
        return (f"\n\\noindent\\textbf{{Algorithm {labels.get('alg:train', '1')}: {cap}}}\n"
                "\\begin{enumerate}\n" + "\n".join(items) + "\n\\end{enumerate}\n")
    body = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", algo, body, flags=re.S)

    body = body.replace("\\begin{symlist}", "\\noindent where:\n\\begin{description}").replace(
        "\\end{symlist}", "\\end{description}")
    body = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}",
                  r"\\noindent\\textbf{\\textit{Index Terms}}---\1", body, flags=re.S)
    bib = re.search(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}", body, re.S)
    refs = re.split(r"\\bibitem\{[^}]+\}", bib.group(1))[1:]
    body = body.replace(bib.group(0), "\\section*{References}\n" + "\n\n".join(
        f"[{i + 1}] {r.strip()}" for i, r in enumerate(refs)))
    for a, b in [("\\textdegree{}", "°"), ("\\textdegree", "°"), ("\\par", "\n\n"), ("\\footnotesize", ""),
                 ("\\small", ""), ("\\centering", ""), ("~", " ")]:
        body = body.replace(a, b)
    body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
    body = re.sub(r"\\vspace\*?\{[^}]*\}", "", body)
    return ("\\documentclass{article}\n\\usepackage{amsmath,amssymb,graphicx,booktabs}\n"
            f"\\title{{{title}}}\n\\begin{{document}}\n\\maketitle\n" + body + "\n\\end{document}\n")


def finish(docx):
    d = Document(docx)
    for st in d.styles:
        try:
            st.font.name = "Times New Roman"
            if st.name in ("Normal", "Body Text", "First Paragraph", "Compact"):
                st.font.size = Pt(10)
        except (AttributeError, ValueError):
            pass
    title = d.paragraphs[0]
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in title.runs:
        r.font.size, r.font.name = Pt(20), "Times New Roman"
    # author block: 3 + 2 names, as in the PDF
    tbl = d.add_table(rows=2, cols=3)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cells = [tbl.cell(0, 0), tbl.cell(0, 1), tbl.cell(0, 2), tbl.cell(1, 0), tbl.cell(1, 1)]
    tbl.cell(1, 2).merge(tbl.cell(1, 2))
    for cell, (name, mail) in zip(cells, AUTHORS):
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(name); r.bold = True; r.font.size = Pt(10)
        for line in AFFIL + [mail] + (["Corresponding Author"] if name.startswith("Dr.") else []):
            q = cell.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = q.add_run(line); rr.font.size = Pt(8)
            rr.italic = line in AFFIL[:2] or line == "Corresponding Author"
    title._p.addnext(tbl._tbl)
    for p in d.paragraphs:
        if p.text.startswith(("TABLE ", "Fig. ", "Algorithm ")):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for t in d.tables[1:]:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
    d.core_properties.author = "Deepankar"
    d.core_properties.title = "Beyond Development: Explaining Climate Resilience Across Indian Districts"
    d.save(docx)


def main():
    tex = (PAPER / "paper.tex").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "paper_docx.tex").write_text(convert(tex, work))
        subprocess.run(["pandoc", "paper_docx.tex", "-f", "latex", "-o", str(OUT)], cwd=work, check=True)
    finish(OUT)
    print("written", OUT.relative_to(PAPER.parent))


if __name__ == "__main__":
    main()
