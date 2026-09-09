"""
Build step for the Assignment 1 page. Run from anywhere:  python3 build.py

1. Copies the current contents of code/*.py into the <code data-src="..."> blocks
   in index.html, so the code on the page is always the code you ran.
2. Prints index.html to SYDE572_A1.pdf with headless Google Chrome.
   The PDF is the same page as the website, with all four tabs one after another
   and every collapsible expanded.

Then commit and push; the site and the PDF will match.
"""

import html
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "index.html"
PDF = HERE / "SYDE572_A1.pdf"

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

CODE_BLOCK = re.compile(r'(<code[^>]*\bdata-src="([^"]+)"[^>]*>)(.*?)(</code>)', re.S)


def sync_code():
    page = PAGE.read_text(encoding="utf-8")

    def fill(m):
        src = HERE / m.group(2)
        if src.exists():
            body = src.read_text(encoding="utf-8").rstrip() + "\n"
            print(f"  code  {m.group(2)}  ({body.count(chr(10))} lines)")
        else:
            body = f"# {m.group(2)} does not exist yet\n"
            print(f"  code  {m.group(2)}  MISSING")
        return m.group(1) + html.escape(body, quote=False) + m.group(4)

    PAGE.write_text(CODE_BLOCK.sub(fill, page), encoding="utf-8")


def report_missing():
    page = PAGE.read_text(encoding="utf-8")
    missing = [s for s in re.findall(r'<img[^>]*\bsrc="([^"]+)"', page)
               if not s.startswith("http") and not (HERE / s).exists()]
    todos = page.count('class="todo"')
    for s in missing:
        print(f"  image missing: {s}")
    print(f"  {todos} placeholder note(s) still on the page")


def build_pdf():
    chrome = next((p for p in CHROME_PATHS if os.path.exists(p)), None)
    if chrome is None:
        sys.exit("Chrome not found. Open index.html in a browser and use Print > Save as PDF instead.")
    url = PAGE.as_uri() + "?pdf"
    subprocess.run([
        chrome, "--headless=new", "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=20000",       # wait for fonts, KaTeX and highlighting to finish
        f"--print-to-pdf={PDF}", url,
    ], check=True, capture_output=True)
    print(f"  pdf   {PDF.name}  ({PDF.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    print("Syncing code blocks")
    sync_code()
    print("Checking page")
    report_missing()
    print("Building PDF")
    build_pdf()
