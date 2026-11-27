"""Turn the rendered slide PNGs into a PowerPoint file.

This script does exactly one thing: it wraps `png/slide-NN.png` into a 16:9
`.pptx`, one full-bleed image per slide.

Why images rather than real text boxes: the slides are authored in
`index.html`, where the diagrams are hand-written inline SVG. Rebuilding those
diagrams with python-pptx shapes would produce a second, drifting copy of every
figure. Exporting the rendered page keeps one source of truth.

The consequence is stated plainly: **the text in the .pptx is not editable.**
To change any wording or number, edit `index.html`, re-export the PNGs, and run
this script again.

    python docs/slides/build_pptx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PNG_DIR = HERE / "png"
OUT = HERE / "screening-slides-zh.pptx"

#: 16:9 at PowerPoint's standard widescreen size.
SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5


def main() -> int:
    try:
        from pptx import Presentation
        from pptx.util import Inches
    except ImportError:
        print("python-pptx is required: python -m pip install python-pptx",
              file=sys.stderr)
        return 1

    pngs = sorted(PNG_DIR.glob("slide-*.png"))
    if not pngs:
        print(f"no slide PNGs in {PNG_DIR}. Open index.html and export them "
              "first.", file=sys.stderr)
        return 1

    # Every page must be the same size, or PowerPoint will letterbox some of
    # them differently and the deck will look inconsistent on a projector.
    try:
        from PIL import Image

        sizes = {Image.open(p).size for p in pngs}
        if len(sizes) != 1:
            print(f"PNGs have mixed sizes {sorted(sizes)}; re-export them at a "
                  "single resolution.", file=sys.stderr)
            return 1
        width, height = sizes.pop()
        ratio = width / height
        if abs(ratio - SLIDE_W_IN / SLIDE_H_IN) > 0.01:
            print(f"PNGs are {width}x{height} (ratio {ratio:.3f}), not 16:9; "
                  "the deck would be stretched.", file=sys.stderr)
            return 1
    except ImportError:
        print("Pillow not available; skipping the size check.", file=sys.stderr)

    deck = Presentation()
    deck.slide_width = Inches(SLIDE_W_IN)
    deck.slide_height = Inches(SLIDE_H_IN)
    blank = deck.slide_layouts[6]        # the layout with no placeholders

    for png in pngs:
        slide = deck.slides.add_slide(blank)
        slide.shapes.add_picture(str(png), 0, 0,
                                 width=Inches(SLIDE_W_IN),
                                 height=Inches(SLIDE_H_IN))

    deck.save(OUT)
    print(f"Wrote {OUT} ({len(pngs)} slides, {OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
