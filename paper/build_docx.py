"""Build paper/Final_IEEE_Paper.docx, an editable Word copy of paper/paper.tex.

Run after paper/build_paper.py (it needs paper.tex and paper.aux). Steps:
  1. the two TikZ diagrams are compiled on their own and saved as PNG;
  2. citations, table/figure/algorithm references and captions get their final numbers from paper.aux;
  3. IEEE-only constructs (author blocks, symbol lists, algorithmic, minipages) become plain LaTeX;
  4. pandoc converts the result to .docx (equations become Word equations);
  5. python-docx lays it out like the PDF: title and the 3 + 2 author block across the page, the body in two
     columns, Tables IV-V across the page as in the PDF, Times New Roman, IEEE headings and caption styles.
Usage: python paper/build_docx.py
"""
import re
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

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
    for head in ("\\parbox{0.95\\columnwidth}{", "\\resizebox{\\columnwidth}{!}{", "\\resizebox{\\linewidth}{!}{", "\\resizebox{\\textwidth}{!}{"):
        body = strip_wrapper(body, head)
    body = re.sub(r"\\begin\{minipage\}(\[[^\]]*\])?\{[^}]*\}", "", body).replace("\\end{minipage}", "")

    # IEEE section numbers: I., II., ... and A., B., ... inside each section
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    sec = sub = 0
    def heading(m):
        nonlocal sec, sub
        if m.group(1) == "section":
            sec, sub = sec + 1, 0
            return f"\\section*{{{roman[sec - 1]}. {m.group(2)}}}"
        sub += 1
        return f"\\subsection*{{{chr(64 + sub)}. {m.group(2)}}}"
    body = re.sub(r"\\(section|subsection)\{([^}]*)\}", heading, body)

    # equation numbers (pandoc drops them): written at the right of each equation
    def eqnum(m):
        env, inner = m.group(1), m.group(2)
        if env == "align":
            lines = [ln for ln in re.split(r"\\\\\s*\n", inner.strip())]
            out = []
            for ln in lines:
                out.append(re.sub(r"\\label\{[^}]+\}", "", ln).strip())
            first = int(labels.get("eq:b", "3"))
            out = [ln.replace("&", "").rstrip(" \\\\") for ln in out]
            return "\n".join(f"\\[\n{ln}\n\\]" for ln in out)
        lab = re.search(r"\\label\{([^}]+)\}", inner)
        n = labels.get(lab.group(1), "") if lab else ""
        inner = re.sub(r"\\label\{[^}]+\}", "", inner).strip()
        return f"\\[\n{inner}\n\\]"
    body = re.sub(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", eqnum, body, flags=re.S)

    body = re.sub(r"\\cite\{([^}]+)\}", lambda m: ", ".join(f"[{num[k.strip()]}]" for k in m.group(1).split(",")), body)
    body = re.sub(r"\\(?:eq)?ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), body)

    # a table* holding two tables side by side (Tables IV and V) becomes two tables, one caption each
    def split_star(m):
        inner = m.group(1)
        parts = re.split(r"(?=\\caption\{)", inner)
        return "".join(f"\\begin{{table}}\n{part}\n\\end{{table}}\n" for part in parts if "\\caption{" in part)
    body = re.sub(r"\\begin\{table\*\}(?:\[[^\]]*\])?(.*?)\\end\{table\*\}", split_star, body, flags=re.S)

    # number the captions the IEEE way: "TABLE I: ...", "Fig. 1. ...", "Algorithm 1: ..."
    def number_captions(env, fmt):
        nonlocal body
        def fix(m):
            blk = m.group(0)
            lab = re.search(r"\\label\{([^}]+)\}", blk)
            n = labels.get(lab.group(1), "") if lab else ""
            return re.sub(r"\\caption\{", lambda _: "\\caption{" + fmt.format(n), blk, count=1)
        body = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", fix, body, flags=re.S)
    number_captions("table\\*?", "TABLE {}\\\\ ")
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
                 ("\\small", ""), ("\\scriptsize", ""), ("$<$", "<"), ("$>$", ">"),
                 ("$^{*}$", "*"), ("$\\downarrow$", "\u2193"), ("$\\rightarrow$", "\u2192"), ("\\centering", ""), ("~", " ")]:
        body = body.replace(a, b)
    body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
    body = re.sub(r"\\vspace\*?\{[^}]*\}", "", body)
    return ("\\documentclass{article}\n\\usepackage{amsmath,amssymb,graphicx,booktabs}\n"
            f"\\title{{{title}}}\n\\begin{{document}}\n\\maketitle\n" + body + "\n\\end{document}\n")


def _sect(p_el, cols, last=False):
    """Make `p_el` (a paragraph) end a continuous section with `cols` columns; `last` edits the body sectPr."""
    sp = p_el if last else OxmlElement("w:sectPr")
    for old in list(sp):
        sp.remove(old)
    typ = OxmlElement("w:type"); typ.set(qn("w:val"), "continuous")
    sz = OxmlElement("w:pgSz"); sz.set(qn("w:w"), "12240"); sz.set(qn("w:h"), "15840")
    mar = OxmlElement("w:pgMar")
    for k, v in dict(top="1080", bottom="1440", left="900", right="900", header="720", footer="720", gutter="0").items():
        mar.set(qn(f"w:{k}"), v)
    c = OxmlElement("w:cols"); c.set(qn("w:num"), str(cols)); c.set(qn("w:space"), "360")
    for el in (typ, sz, mar, c):
        sp.append(el)
    if not last:
        ppr = p_el.get_or_add_pPr()
        ppr.append(sp)


def _section_break_before(el, cols):
    """Insert an empty paragraph before `el` that ends the current section (with `cols` columns)."""
    p = OxmlElement("w:p")
    el.addprevious(p)
    from docx.text.paragraph import Paragraph
    _sect(Paragraph(p, None)._p, cols)


def _font(par, size=None, bold=None, italic=None, caps=False):
    for r in par.runs:
        r.font.name = "Times New Roman"
        if size:
            r.font.size = Pt(size)
        if bold is not None:
            r.bold = bold
        if italic is not None:
            r.italic = italic
        if caps:
            r.font.small_caps = True


def finish(docx):
    d = Document(docx)
    for st in d.styles:
        try:
            st.font.name = "Times New Roman"
            st.font.color.rgb = None
            if st.name in ("Normal", "Body Text", "First Paragraph", "Compact"):
                st.font.size = Pt(10)
                st.paragraph_format.space_after = Pt(3)
        except (AttributeError, ValueError):
            pass
    body = d.element.body
    paras = d.paragraphs
    # title
    title = paras[0]
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _font(title, 24, False)
    # author block: 3 + 2 names, as in the PDF
    tbl = d.add_table(rows=1, cols=3)
    tbl2 = d.add_table(rows=1, cols=2)
    for tb in (tbl, tbl2):
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        tb.autofit = False
        for c in tb.rows[0].cells:
            c.width = Inches(2.3)
    cells = list(tbl.rows[0].cells) + list(tbl2.rows[0].cells)
    for cell, (name, mail) in zip(cells, AUTHORS):
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(name); r.bold = True; r.font.size = Pt(11); r.font.name = "Times New Roman"
        for line in AFFIL + [mail] + (["Corresponding Author"] if name.startswith("Dr.") else []):
            q = cell.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.space_after = Pt(0)
            rr = q.add_run(line); rr.font.size = Pt(9); rr.font.name = "Times New Roman"
            rr.italic = line in AFFIL[:2] or line == "Corresponding Author"
    # 3 names, then 2 names centred under them, as in the PDF
    title._p.addnext(tbl._tbl)
    gap = OxmlElement("w:p"); tbl._tbl.addnext(gap)
    gap.addnext(tbl2._tbl)
    # end the one-column title section right after the author block
    endp = OxmlElement("w:p"); tbl2._tbl.addnext(endp)
    _sect(endp, 1)
    # abstract: bold 9 pt "Abstract—..." as in IEEE
    for i, p in enumerate(d.paragraphs):
        if p.text.strip() == "Abstract" and p.style.name.startswith("Abstract Title"):
            nxt = d.paragraphs[i + 1]
            nxt.runs[0].text = "Abstract—" + nxt.runs[0].text if nxt.runs else nxt.text
            p._p.getparent().remove(p._p)
            break
    for p in d.paragraphs:
        st = p.style.name
        if st == "Abstract":
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = p.paragraph_format.right_indent = Inches(0)
            _font(p, 9, True)
            for r in p.runs:
                r.italic = r.text.startswith("Abstract") and False
        elif p.text.startswith("Index Terms"):
            _font(p, 9, True)
        elif st == "Heading 1":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _font(p, 10, False, False, caps=True)
            p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(8), Pt(4)
        elif st == "Heading 2":
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _font(p, 10, False, True)
            p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(6), Pt(3)
        elif st in ("Table Caption",) or p.text.startswith("TABLE "):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _font(p, 8, False, False, caps=True)
        elif st in ("Image Caption", "Caption") or re.match(r"^Fig\. \d+\. ", p.text):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _font(p, 8)
        elif re.match(r"^Algorithm \d+: ", p.text):
            _font(p, 9, True)
        elif re.match(r"^\[\d+\] ", p.text):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _font(p, 8)
        elif st in ("Body Text", "First Paragraph", "Normal", "Compact"):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # display equations: centred, with the PDF's number (1)-(8) at the right of the column
    M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    eqs = [p for p in d.paragraphs if p._p.find(f"{M}oMathPara") is not None]
    for k, p in enumerate(eqs, 1):
        mp = p._p.find(f"{M}oMathPara")
        maths = mp.findall(f"{M}oMath")
        for el in maths:
            mp.addprevious(el)
        mp.getparent().remove(mp)
        pf = p.paragraph_format
        pf.tab_stops.add_tab_stop(Inches(1.72), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Inches(3.42), WD_TAB_ALIGNMENT.RIGHT)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        first = p._p.find(f"{M}oMath")
        tab1 = OxmlElement("w:r"); tab1.append(OxmlElement("w:tab")); first.addprevious(tab1)
        last = p._p.findall(f"{M}oMath")[-1]
        r = p.add_run(f"\t({k})"); r.font.name = "Times New Roman"; r.font.size = Pt(10)
        last.addnext(r._r)
    # pictures: one column wide (3.45 in), like the PDF
    for shp in d.inline_shapes:
        ratio = shp.height / shp.width
        shp.width = Inches(3.45); shp.height = int(Inches(3.45) * ratio)
    # tables: 8 pt, centred, sized to the column (Tables IV-V: to the page) by text length per column
    def cap_of(tb):
        prev = tb._tbl.getprevious()
        return "".join(x.text or "" for x in prev.iter(qn("w:t"))) if prev is not None else ""
    for tb in d.tables[2:]:
        cap = cap_of(tb)
        wide = cap.startswith(("TABLE IV", "TABLE V")) and not cap.startswith(("TABLE VI", "TABLE VII", "TABLE VIII"))
        total = 10080 if wide else 4960
        lens = [min(22 if j == 0 else 16, max(4, max(len(c.text) for c in col.cells)))
                for j, col in enumerate(tb.columns)]
        ws = [int(total * x / sum(lens)) for x in lens]
        tblPr = tb._tbl.tblPr
        for tag in ("w:tblW", "w:tblLayout"):
            for old in tblPr.findall(qn(tag)):
                tblPr.remove(old)
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(total)); tw.set(qn("w:type"), "dxa")
        tblPr.insert(1, tw)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
        tblPr.insert(3, lay)
        for old in tblPr.findall(qn("w:tblCellMar")):
            tblPr.remove(old)
        mar = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            m_ = OxmlElement(f"w:{side}"); m_.set(qn("w:w"), "40"); m_.set(qn("w:type"), "dxa"); mar.append(m_)
        look = tblPr.find(qn("w:tblLook"))
        look.addprevious(mar) if look is not None else tblPr.append(mar)
        for gc, w in zip(tb._tbl.tblGrid.findall(qn("w:gridCol")), ws):
            gc.set(qn("w:w"), str(w))
        for row in tb.rows:
            for c, w in zip(row.cells, ws):
                c.width = w * 635
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        for row in tb.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    _font(p, 8)
                    p.paragraph_format.space_after = Pt(0)
    # Tables IV and V span the page in the PDF: put them in a one-column section
    caps = [p for p in d.paragraphs if p.text.startswith(("TABLE IV", "TABLE V"))
            and not p.text.startswith(("TABLE VI", "TABLE VII", "TABLE VIII"))]
    if len(caps) == 2:
        _section_break_before(caps[0]._p, 2)
        el = caps[1]._p
        while el is not None and el.tag != qn("w:tbl"):
            el = el.getnext()
        endp = OxmlElement("w:p"); el.addnext(endp)
        _sect(endp, 1)
    # the rest of the body is in two columns
    _sect(body.sectPr if body.sectPr is not None else body.get_or_add_sectPr(), 2, last=True)
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
