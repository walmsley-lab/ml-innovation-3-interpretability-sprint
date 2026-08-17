"""Render report.md to a print-ready PDF.

    PYTHONPATH=src python scripts/build_pdf.py [--in report.md] [--out report.pdf]

Two things need handling that a plain Markdown-to-HTML pass gets wrong.

**Math.** The report uses ``$...$`` inline math. WeasyPrint has no math engine,
so the LaTeX fragments are rewritten to Unicode and HTML sub/superscripts
instead. The expressions here are simple (norms, means, a few operators), so
this is faithful; anything genuinely multi-line would not be.

**Tables.** Appendix A is raw HTML with explicit column widths, because
rendered Markdown sizes each table to its own content and the three claim
tables would otherwise put their divider in three different places. Markdown
passes that block through untouched, and the stylesheet only has to avoid
fighting it. Table styling follows the usual academic convention: horizontal
rules, no vertical rules.
"""
from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import markdown
from PIL import Image, ImageChops
from weasyprint import CSS, HTML

MACROS = {
    r"\pm": "±", r"\neq": "≠", r"\nabla": "∇", r"\times": "×",
    r"\le": "≤", r"\ge": "≥", r"\approx": "≈", r"\cdot": "·",
    r"\rightarrow": "→", r"\arg\max": "argmax", r"\cos": "cos",
    r"\log": "log", r"\ ": " ", r"\,": " ",
}


def demath(expr: str) -> str:
    """One inline math expression to HTML. Not a LaTeX engine, deliberately."""
    e = expr
    e = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", e)
    e = re.sub(r"\\text\{([^}]*)\}", r"\1", e)
    for k, v in sorted(MACROS.items(), key=lambda kv: -len(kv[0])):
        e = e.replace(k, v)
    e = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", e)
    e = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", e)
    e = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", e)
    e = re.sub(r"\^([A-Za-z0-9])", r"<sup>\1</sup>", e)
    e = e.replace("'", "′")
    e = re.sub(r"\\([A-Za-z]+)", r"\1", e)     # any macro left: drop backslash
    return f'<span class="math">{e}</span>'


def trim(src: Path, out_dir: Path) -> Path:
    """Copy an image with uniform white margins removed.

    The single-panel figure reserves a blank half-canvas so its panel matches
    the width of a panel in the two-across figures. On screen that keeps panel
    width consistent; on a portrait page it throws away half the text column
    and the figure prints too small to read. Trimming the dead space at build
    time fixes the print layout without altering the committed figure.
    """
    im = Image.open(src).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        pad = 6
        box = (max(0, box[0] - pad), max(0, box[1] - pad),
               min(im.width, box[2] + pad), min(im.height, box[3] + pad))
        im = im.crop(box)
    dst = out_dir / src.name
    im.save(dst)
    return dst


CSS_TEXT = """
@page {
  size: letter;
  margin: 0.85in 0.9in 0.95in 0.9in;
  @bottom-center { content: counter(page); font-size: 9pt; color: #666; }
}
body { font-family: "Source Serif 4","Noto Serif","DejaVu Serif",Georgia,serif;
       font-size: 10.2pt; line-height: 1.42; color: #111; hyphens: auto;
       text-align: justify; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 0.15em 0;
     text-align: left; }
h1 + p { text-align: left; font-size: 11pt; margin-top: 0; }
h2 { font-size: 13pt; margin: 1.5em 0 0.4em; text-align: left;
     border-bottom: 0.6pt solid #bbb; padding-bottom: 0.15em; }
h3 { font-size: 11pt; margin: 1.15em 0 0.3em; text-align: left; }
h2, h3 { break-after: avoid; }
p { margin: 0 0 0.55em; }
code { font-family: "DejaVu Sans Mono",monospace; font-size: 0.86em;
       background: #f4f4f4; padding: 0 2px; }
.math { font-family: "DejaVu Serif",Georgia,serif; white-space: nowrap; }

/* Academic table convention: horizontal rules only, never vertical. */
table { width: 100%; border-collapse: collapse; margin: 0.6em 0 0.9em;
        font-size: 9.1pt; text-align: left; break-inside: avoid; }
thead th { border-top: 1pt solid #333; border-bottom: 0.6pt solid #333;
           padding: 4px 6px; text-align: left; font-weight: 600; }
tbody td { border-bottom: 0.35pt solid #ccc; padding: 3.5px 6px;
           vertical-align: top; text-align: left; }
tbody tr:last-child td { border-bottom: 1pt solid #333; }

img { max-width: 100%; height: auto; display: block; margin: 0.6em auto 0.2em; }

/* Figure captions are written as blockquotes in the source. */
blockquote { margin: 0.1em 0 1.1em; padding: 0 0.4em; font-size: 8.9pt;
             color: #333; line-height: 1.35; text-align: left;
             break-before: avoid; }
blockquote p { margin: 0; }

ol, ul { margin: 0.2em 0 0.6em 1.1em; padding: 0; }
li { margin: 0 0 0.25em; }
a { color: #14507d; text-decoration: none; word-break: break-all; }
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=Path, default=Path("report.md"))
    ap.add_argument("--out", dest="dst", type=Path, default=Path("report.pdf"))
    args = ap.parse_args()

    text = args.src.read_text()
    tmp = Path(tempfile.mkdtemp(prefix="reportfigs-"))
    def _trim(m):
        src = (args.src.resolve().parent / m.group(2))
        if not src.exists():
            return m.group(0)
        return f"![{m.group(1)}]({trim(src, tmp).as_posix()})"
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _trim, text)
    # Math first: it must not be mangled by the Markdown inline parser.
    text = re.sub(r"\$([^$\n]+)\$", lambda m: demath(m.group(1)), text)

    body = markdown.markdown(
        text, extensions=["tables", "attr_list", "sane_lists"],
        output_format="html5")
    html = f"<!doctype html><html><head><meta charset='utf-8'>" \
           f"<title>{args.src.stem}</title></head><body>{body}</body></html>"

    HTML(string=html, base_url=str(args.src.resolve().parent)).write_pdf(
        args.dst, stylesheets=[CSS(string=CSS_TEXT)])
    print(f"wrote {args.dst} ({args.dst.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
