"""Build paper/IEEE_Paper.docx as an exact copy of the PDF layout: each PDF page is placed as a 300-dpi image on its
own Word page (US Letter, no margins), so Word shows the paper exactly as the PDF does.
The editable Word version (pandoc) is paper/IEEE_Paper_editable.docx, built by paper/build_docx.py.
Usage: python paper/build_docx_exact.py
"""
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Inches, Pt

PAPER = Path(__file__).resolve().parent


def main():
    pdf = PAPER / "IEEE_Paper.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "300", "-png", str(pdf), f"{tmp}/p"], check=True)
        pages = sorted(Path(tmp).glob("p-*.png"))
        d = Document()
        sec = d.sections[0]
        sec.page_width, sec.page_height = Inches(8.5), Inches(11)
        sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(0)
        sec.header_distance = sec.footer_distance = Inches(0)
        normal = d.styles["Normal"].paragraph_format
        normal.space_before = normal.space_after = Pt(0)
        normal.line_spacing = 1.0
        for k, png in enumerate(pages):
            p = d.add_paragraph()
            run = p.add_run()
            run.add_picture(str(png), width=Inches(8.48), height=Inches(10.97))
            if k < len(pages) - 1:
                run.add_break(WD_BREAK.PAGE)
        d.core_properties.author = "Deepankar"
        d.core_properties.title = "Beyond Development: Explaining Climate Resilience Across Indian Districts"
        d.save(PAPER / "IEEE_Paper.docx")
    print("written paper/IEEE_Paper.docx with", len(pages), "pages")


if __name__ == "__main__":
    main()
