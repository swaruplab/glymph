"""Text-fit check with the real fonts: wraps every text box the way PowerPoint does (greedy word wrap, real Georgia and
Avenir Next advance widths, letter spacing, bullet indents, line spacing, space after) and reports boxes whose text needs
more height than the box has, or a word wider than the box. LibreOffice renders can substitute fonts, so they are not a
safe guide to text fit; this check is.
Usage: python fit_check.py deck.pptx [--all]   (--all prints every box with its measured fill)"""
import sys
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from PIL import ImageFont

GE = "/System/Library/Fonts/Supplemental/"
AV = "/System/Library/Fonts/Avenir Next.ttc"
AV_INDEX = {"Regular": 7, "Italic": 4, "Bold": 0, "Bold Italic": 1, "Medium": 5, "Medium Italic": 6, "Demi Bold": 2}
PX = 20  # measurement scale: pixels per point
_fonts = {}


def face(name, bold, italic):
    key = (name, bold, italic)
    if key not in _fonts:
        if name.startswith("Georgia"):
            fn = {(0, 0): "Georgia.ttf", (1, 0): "Georgia Bold.ttf", (0, 1): "Georgia Italic.ttf", (1, 1): "Georgia Bold Italic.ttf"}
            _fonts[key] = (GE + fn[(int(bold), int(italic))], 0)
        elif name.startswith("Avenir Next"):
            base = "Medium" if name.endswith("Medium") else ("Bold" if bold else "Regular")
            style = (base + " Italic") if italic else base
            _fonts[key] = (AV, AV_INDEX["Italic" if style == "Regular Italic" else style])
        else:  # Arial and anything else
            fn = {(0, 0): "Arial.ttf", (1, 0): "Arial Bold.ttf", (0, 1): "Arial Italic.ttf", (1, 1): "Arial Bold Italic.ttf"}
            _fonts[key] = (GE + fn[(int(bold), int(italic))], 0)
    return _fonts[key]


def font(name, size, bold, italic):
    path, idx = face(name, bold, italic)
    k = (path, idx, size)
    if k not in _fonts:
        # BASIC layout: no kerning, as PowerPoint lays text out unless kerning is switched on for the run
        _fonts[k] = ImageFont.truetype(path, int(round(size * PX)), index=idx, layout_engine=ImageFont.Layout.BASIC)
    return _fonts[k]


def width_pt(text, f, spc_pt):
    return f.getlength(text) / PX + spc_pt * len(text)


def para_lines(p, box_w):
    """Greedy word wrap of one paragraph; returns (n_lines, max_font_size, widest_word_pt, notes). Renderers lay text out a
    few percent wider than these advance widths, so the usable width keeps a 3% safety margin. notes flags a single word
    left alone on the last line (an orphan) and a hyphenated word that renderers would split across lines."""
    pPr = p._p.pPr
    indent = 0.0
    if pPr is not None and pPr.get("marL"):
        indent = int(pPr.get("marL")) / 12700
    avail = (box_w - indent) * 0.97
    words, cur, sizes = [], [], []
    from pptx.text.text import _Run
    for child in p._p:  # runs and forced line breaks (a:br), in document order
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "br":
            if cur:
                words.append(("w", cur))
                cur = []
            words.append(("br", 0.0))
            continue
        if tag != "r":
            continue
        r = _Run(child, p)
        f_ = r.font
        size = f_.size.pt if f_.size else 18
        sizes.append(size)
        fnt = font(f_.name or "Avenir Next", size, bool(f_.bold), bool(f_.italic))
        rPr = r._r.rPr
        spc = int(rPr.get("spc")) / 100 if (rPr is not None and rPr.get("spc")) else 0.0
        for ch in r.text:
            if ch in "\x0b\n":  # forced line break inside a paragraph
                if cur:
                    words.append(("w", cur))
                    cur = []
                words.append(("br", 0.0))
            elif ch == " ":
                if cur:
                    words.append(("w", cur))
                    cur = []
                words.append(("s", width_pt(" ", fnt, spc)))
            else:
                cur.append((ch, fnt, spc))
    if cur:
        words.append(("w", cur))
    if not sizes:
        return 0, 0, 0, []
    lines, x, widest = 1, 0.0, 0.0
    pending_space = 0.0
    line_words, notes = [0], []
    for kind, v in words:
        if kind == "br":
            lines += 1
            line_words.append(0)
            x, pending_space = 0.0, 0.0
            continue
        if kind == "s":
            pending_space += v
            continue
        w = sum(width_pt(c, f, s) for c, f, s in v)
        widest = max(widest, w)
        if x > 0 and x + pending_space + w > avail + 0.5:
            word = "".join(c for c, _, _ in v)
            if "-" in word[1:-1]:
                head = word.split("-")[0] + "-"
                if x + pending_space + sum(width_pt(c, f, s) for c, f, s in v[:len(head)]) <= avail:
                    notes.append(f"HYPHEN-SPLIT '{word}'")
            lines += 1
            line_words.append(0)
            x = w
        else:
            x += pending_space + w
        line_words[-1] += 1
        pending_space = 0.0
    if lines > 1 and line_words[-1] == 1 and len(words) > 3:
        notes.append("ORPHAN")
    return lines, max(sizes), widest, notes


def check(path, show_all=False):
    prs = Presentation(path)
    issues = 0
    for si, sl in enumerate(prs.slides, 1):
        stack = list(sl.shapes)
        while stack:
            sh = stack.pop()
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                stack.extend(sh.shapes)
                continue
            if not sh.has_text_frame or not sh.text_frame.text.strip():
                continue
            tf = sh.text_frame
            box_w, box_h = sh.width / 12700, sh.height / 12700
            need, widest, notes = 0.0, 0.0, []
            for p in tf.paragraphs:
                n, size, wd, nt = para_lines(p, box_w)
                notes += nt
                if n == 0:
                    continue
                widest = max(widest, wd)
                ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
                need += n * size * 1.2 * ls + (p.space_after.pt if p.space_after is not None else 0)
            snippet = tf.text.replace("\n", " / ")[:70]
            over = need > box_h + 3
            wide = widest > box_w + 1
            if notes:
                print(f"slide {si:2d}  polish     {', '.join(notes)}  | {tf.text.replace(chr(10), ' / ')[:70]}")
            if over or wide or show_all:
                flag = ("OVERFLOW " if over else "") + ("WIDE-WORD " if wide else "")
                print(f"slide {si:2d}  {flag or 'ok':10s} need {need / 72:5.2f}in  box {box_h / 72:5.2f}in  w {box_w / 72:5.2f}in  | {snippet}")
                issues += bool(over or wide)
    print(f"{path}: {issues} text boxes need attention")
    return issues


if __name__ == "__main__":
    sys.exit(1 if check(sys.argv[1], "--all" in sys.argv) else 0)


def lines_of(text, size, width_in, name="Avenir Next", bold=False, italic=False, spc=0.0):
    """The lines PowerPoint would make of one single-run paragraph (same model and 3% margin as the check)."""
    f = font(name, size, bold, italic)
    avail = width_in * 72 * 0.97
    out, cur, x = [], [], 0.0
    sp = width_pt(" ", f, spc)
    for w_ in text.split(" "):
        ww = width_pt(w_, f, spc)
        if cur and x + sp + ww > avail + 0.5:
            out.append(" ".join(cur))
            cur, x = [w_], ww
        else:
            x += (sp if cur else 0) + ww
            cur.append(w_)
    out.append(" ".join(cur))
    return out
