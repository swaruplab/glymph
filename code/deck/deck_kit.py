"""Deck kit for the Glymph talk: theme tokens, figure handling, slide frame, animation hooks, and the final checks and
save. Copied from ~/.claude/deck_kit (the lab's deck system) on 2026-11-17; only the acts, the figure search paths, the
output folder and the naming rule differ. build_glymph_deck.py does `from deck_kit import *`; the theme comes from argv[1].
Themes: light (white, Georgia + Avenir Next, champagne gold) and dark (midnight navy, ivory ink, brighter gold). Colours are
semantic tokens (bg, surface, ink, gold, onaccent, feature ...) so every slide reads in either theme.
Figures: reports/deck_figures/<theme>/ (the deck's own plots, rendered per theme by make_deck_figures.py with a transparent
background). Naming: one name per thing on slides, charts and notes; the internal names in ALIAS (code and table names such
as kappa, fpr, RMSLE, g031) may appear only on a slide made with keep_alias=True, where they are introduced."""
import os
import re
import sys
import zipfile
from xml.sax.saxutils import escape
import numpy as np
from PIL import Image, ImageFilter
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
from pptx.oxml.ns import qn
from lxml import etree
from pptx_motion import Motion, set_transition
import fit_check

THEME = sys.argv[1] if len(sys.argv) > 1 else "light"
assert THEME in ("light", "dark"), THEME
DARK = THEME == "dark"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PRES = os.path.join(ROOT, "reports")
DIRS = [os.path.join(PRES, "deck_figures", THEME), os.path.join(PRES, "deck_figures"), HERE]
CACHE = os.path.join(PRES, ".deck_cache", THEME)
os.makedirs(CACHE, exist_ok=True)

HEAD, BODY, NUM = "Georgia", "Avenir Next", "Avenir Next Medium"  # NUM: lining figures for data
THEMES = {
    "light": dict(bg="FFFFFF", bg2="FFFFFF", surface="FFFFFF", card="F7F8FA", hair="E4E7EC",
                  ink="16233A", ink2="3B4658", muted="6E7887", faint="A3AAB5",
                  gold="B08D57", goldtxt="8A6A35", goldpale="E9DCC3", goldtint="FBF8F2", goldline="E7DBC4", numeral="E9DCC3",
                  onaccent="FFFFFF", feature="16233A", onfeature="FFFFFF", onfeature2="E8EBF0", featureaccent="E9DCC3",
                  track="E1E4EA", shadow="16233A", warm="C2872B", cool="3A6FB5", rna="2A78D6", atac="EB6834", green="1BAF7A"),
    # dark: chart pair BF8536 / 5B93E6 validated on 0C121D (dataviz validate_palette.js, all checks pass)
    "dark": dict(bg="0E1524", bg2="0A0F19", surface="151D2C", card="1A2333", hair="27324A",
                 ink="F3EEE4", ink2="C5CBD6", muted="8C96A8", faint="5D6879",
                 gold="D2AE6D", goldtxt="E0C389", goldpale="7A6642", goldtint="1B1A17", goldline="5C4B2E", numeral="2C2A24",
                 onaccent="0C121D", feature="1D2940", onfeature="F3EEE4", onfeature2="C5CBD6", featureaccent="E0C389",
                 track="263049", shadow="000000", warm="BF8536", cool="5B93E6", rna="3987E5", atac="D95926", green="199E70"),
}
C = THEMES[THEME]
W, H, LM, CW = 13.333, 7.5, 0.65, 12.03
ACTS = ["The gap", "The idea", "The architecture", "The benchmark"]


def rgb(c):
    return RGBColor.from_string(C.get(c, c))


def path(name):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def trimmed(name, pad=10):
    """Crop a figure to its content and cache it. Transparent plots (dark theme) are cropped by alpha and keep it;
    everything else is composited onto white and cropped against the corner colour."""
    src = path(name)
    stem, ext = os.path.splitext(os.path.basename(name))
    im = Image.open(src)
    rgba = im.mode in ("RGBA", "LA", "P")
    if rgba:
        im = im.convert("RGBA")
    a = np.asarray(im)
    keep_alpha = DARK and rgba and a[3, 3, 3] < 10
    dst = os.path.join(CACHE, "trim_" + stem + (".png" if (keep_alpha or ext.lower() == ".png") else ext))
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    if keep_alpha:
        ys, xs = np.where(a[..., 3] > 8)
    else:
        if rgba:
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        a = np.asarray(im).astype(int)
        ys, xs = np.where(np.abs(a - a[3, 3]).sum(-1) > 30)
    if len(xs):
        im = im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width), min(ys.max() + pad, im.height)))
    im.save(dst, **({} if dst.endswith(".png") else {"quality": 93}))
    return dst


def backdrop(p):
    """'clear' (transparent), 'light' (needs a plate in the dark theme) or 'dark' (photo on black), from the corners."""
    im = Image.open(p)
    if im.mode == "RGBA" and np.asarray(im)[3, 3, 3] < 10:
        return "clear", None
    a = np.asarray(im.convert("RGB")).astype(int)
    corners = np.array([a[3, 3], a[3, -4], a[-4, 3], a[-4, -4]])
    lum = (corners @ np.array([0.299, 0.587, 0.114])).mean()
    hexcol = "%02X%02X%02X" % tuple(np.median(corners, 0).astype(int))
    return ("light" if lum > 200 else "dark"), hexcol


def icon(name, color="FFFFFF", alpha=1.0):
    """Phosphor Light icon recoloured from the gold master PNG. White icons sit on coloured circles, where the light
    strokes thin out, so their strokes are thickened slightly (5 px max filter on the 512 px master)."""
    h = C.get(color, color)
    bold = h.upper() == "FFFFFF"
    dst = os.path.join(CACHE, f"icon_{name}_{h}_{int(alpha * 100)}{'_b' if bold else ''}.png")
    if not os.path.exists(dst):
        im = Image.open(path(f"icons/{name}_gold.png")).convert("RGBA")
        if bold:
            im.putalpha(im.getchannel("A").filter(ImageFilter.MaxFilter(5)))
        a = np.asarray(im).copy()
        a[..., 0], a[..., 1], a[..., 2] = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        a[..., 3] = (a[..., 3] * alpha).astype("uint8")
        Image.fromarray(a).save(dst)
    return dst


def logo(name):
    """Lab logo; in the dark theme its navy lettering is lifted toward white so it reads on the dark slide."""
    src = path(name)
    if not DARK:
        return src
    dst = os.path.join(CACHE, "logo_" + os.path.basename(name))
    if not os.path.exists(dst):
        a = np.asarray(Image.open(src).convert("RGBA")).astype(float)
        lum = a[..., :3] @ np.array([0.299, 0.587, 0.114])
        m = (a[..., 3] > 0) & (lum < 120)
        a[m, :3] = a[m, :3] * 0.3 + np.array([214, 226, 245]) * 0.7
        Image.fromarray(a.clip(0, 255).astype("uint8")).save(dst)
    return dst


# ---------------------------------------------------------------- primitives (work on slide.shapes or group.shapes)
def tb(shapes, x, y, w, h, paras, size=16, color="ink2", font=BODY, bold=False, italic=False, align="l", anchor="t",
       after=0, line=None, name=None, bullet=False):
    box_ = shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box_.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    for i, p in enumerate(paras if isinstance(paras, list) else [paras]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if after:
            para.space_after = Pt(after)
        if line:
            para.line_spacing = line
        if bullet:  # real bullets with a hanging indent, gold marker
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.22))))
            pPr.set("indent", str(-int(Inches(0.22))))
            etree.SubElement(etree.SubElement(pPr, qn("a:buClr")), qn("a:srgbClr"), val=C["gold"])
            etree.SubElement(pPr, qn("a:buFont"), typeface="Arial")
            etree.SubElement(pPr, qn("a:buChar"), char="•")
        for r in (p if isinstance(p, list) else [p]):
            t, st = (r, {}) if isinstance(r, str) else r
            for j, piece in enumerate(t.split("\v")):  # "\v" in a label is a forced line break (a real a:br element)
                if j:
                    para.add_line_break()
                run = para.add_run()
                run.text = piece
                f = run.font
                f.name = st.get("font", font)
                f.size = Pt(st.get("size", size))
                f.bold = st.get("bold", bold)
                f.italic = st.get("italic", italic)
                f.color.rgb = rgb(st.get("color", color))
                if st.get("spc"):
                    run._r.get_or_add_rPr().set("spc", str(int(st["spc"] * 100)))
    if name:
        box_.name = name
    return box_


def soft_shadow(sh, blur=12, dist=3, alpha=13):
    if DARK:  # a deeper, softer shadow reads as depth on the dark slide
        blur, dist, alpha = blur * 1.6, dist * 1.4, min(alpha * 3, 55)
    spPr = sh._element.spPr
    for e in spPr.findall(qn("a:effectLst")):
        spPr.remove(e)
    eff = etree.SubElement(spPr, qn("a:effectLst"))
    sd = etree.SubElement(eff, qn("a:outerShdw"), blurRad=str(int(blur * 12700)), dist=str(int(dist * 12700)),
                          dir="5400000", algn="t", rotWithShape="0")
    clr = etree.SubElement(sd, qn("a:srgbClr"), val=C["shadow"])
    etree.SubElement(clr, qn("a:alpha"), val=str(int(alpha * 1000)))


def plain(sh):
    """No theme effects. python-pptx's default theme gives shapes and connectors a 38% black drop shadow through
    p:style/effectRef; an explicit empty effectLst overrides it in PowerPoint, and idx 0 also in LibreOffice previews."""
    spPr = sh._element.spPr
    if spPr.find(qn("a:effectLst")) is None:
        spPr.append(spPr.makeelement(qn("a:effectLst"), {}))
    st = sh._element.find(qn("p:style"))
    if st is not None:
        st.find(qn("a:effectRef")).set("idx", "0")


def box(shapes, x, y, w, h, fill=None, line=None, lw=0.75, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12, shadow=False,
        name=None, dash=None):
    sh = shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(lw)
        if dash:
            sh.line.dash_style = dash
    else:
        sh.line.fill.background()
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    sh.shadow.inherit = False
    plain(sh)
    if shadow:
        soft_shadow(sh)
    if name:
        sh.name = name
    return sh


def text_height(lines, size, font=BODY, bold=False, italic=False, width=CW, line=1.0):
    """Height in inches that PowerPoint needs for these paragraphs (list of strings) at this size and box width, using the
    real font metrics (fit_check): lets slides stack text blocks by their measured height instead of guessing."""
    total = 0.0
    for t in lines:
        f = fit_check.font(font, size, bold, italic)
        words = t.split(" ")
        n, x = 1, 0.0
        for w_ in words:
            ww = sum(fit_check.width_pt(c, f, 0) for c in w_)
            sp = fit_check.width_pt(" ", f, 0)
            if x > 0 and x + sp + ww > width * 72 + 0.5:
                n, x = n + 1, ww
            else:
                x += (sp if x > 0 else 0) + ww
        total += n * size * 1.2 * line / 72
    return total


def round_corners(pc, r):
    geom = pc._element.spPr.find(qn("a:prstGeom"))
    geom.set("prst", "roundRect")
    av = geom.find(qn("a:avLst"))
    if av is None:
        av = etree.SubElement(geom, qn("a:avLst"))
    etree.SubElement(av, qn("a:gd"), name="adj", fmla=f"val {int(r * 100000)}")


def pic(shapes, name, x, y, w, h, align="c", valign="m", trim=True, rounded=None, shadow=False, raw=False, plate=None):
    """Fit a figure in the box. Dark theme: white-background figures get a light plate (a mounted print) that fills the
    box; photos on black get a hairline frame; transparent plots sit directly on the slide. Returns the shape to animate."""
    p = name if raw else (trimmed(name) if trim else path(name))
    kind, edge = backdrop(p)
    if plate is None:
        plate = DARK and kind == "light"
    pad = 0.11 if plate else 0
    iw, ih = Image.open(p).size
    sc = min((w - 2 * pad) / iw, (h - 2 * pad) / ih)
    pw, ph = iw * sc, ih * sc
    tw, th = pw + 2 * pad, ph + 2 * pad
    px = x + {"l": 0, "c": (w - tw) / 2, "r": w - tw}[align] + pad
    py = y + {"t": 0, "m": (h - th) / 2, "b": h - th}[valign] + pad
    if plate:
        g = shapes.add_group_shape()
        box(g.shapes, px - pad, py - pad, tw, th, fill=edge, radius=min(0.06, 0.25 / min(tw, th)), shadow=True)
        g.shapes.add_picture(p, Inches(px), Inches(py), Inches(pw), Inches(ph))
        return g
    pc = shapes.add_picture(p, Inches(px), Inches(py), Inches(pw), Inches(ph))
    if rounded:
        round_corners(pc, rounded)
    if DARK and kind == "dark":
        pc.line.color.rgb = rgb("hair")
        pc.line.width = Pt(0.75)
    elif shadow and not DARK:
        soft_shadow(pc, blur=16, dist=4, alpha=16)
    return pc


def arrow(shapes, x1, y1, x2, y2, color="gold", w=1.5, head=True, dash=None):
    c = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(w)
    if dash:
        c.line.dash_style = dash
    if head:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    plain(c)
    return c


def icon_circle(shapes, x, y, d, name, fill="gold", color="onaccent", line=None, frac=0.56):
    g = shapes.add_group_shape()
    box(g.shapes, x, y, d, d, fill=fill, line=line, lw=1.25, kind=MSO_SHAPE.OVAL)
    s = d * frac
    g.shapes.add_picture(icon(name, color), Inches(x + (d - s) / 2), Inches(y + (d - s) / 2), Inches(s), Inches(s))
    return g


def chip(shapes, x, y, w, h, big, label, big_size=26, label_size=12, big_color="ink", fill="surface", line="hair",
         shadow=True, align="l"):
    g = shapes.add_group_shape()
    box(g.shapes, x, y, w, h, fill=fill, line=line, radius=0.1, shadow=shadow)
    pad = 0.2
    bh = big_size / 72 * 1.25
    tb(g.shapes, x + pad, y + 0.13, w - 2 * pad, bh, big, size=big_size, color=big_color, font=NUM, align=align)
    tb(g.shapes, x + pad, y + 0.15 + bh, w - 2 * pad, h - bh - 0.25, label, size=label_size, color="muted", align=align,
       line=1.05)
    return g


def dot_row(shapes, x, y, w, head, text, num=None, d=0.36, head_size=17, size=14, icon_name=None):
    """Numbered (or icon) marker + bold lead + text. Returns the group."""
    g = shapes.add_group_shape()
    if icon_name:
        icon_circle(g.shapes, x, y, d, icon_name)
    else:
        box(g.shapes, x, y, d, d, fill="gold", kind=MSO_SHAPE.OVAL)
        tb(g.shapes, x, y, d, d, str(num), size=13, color="onaccent", font=HEAD, align="c", anchor="m")
    paras = [[(head, {"font": HEAD, "size": head_size, "color": "ink"})]] if head else []
    if text:
        paras.append([(text, {"size": size, "color": "ink2"})])
    tb(g.shapes, x + d + 0.2, y - 0.02, w - d - 0.2, 1.0, paras, after=2, line=1.05)
    return g


POLY_K = 1000.0  # path units per inch; see the note below


def polygon(shapes, pts, fill=None, line=None, lw=1.0):
    """Closed freeform polygon; pts in inches.

    python-pptx writes freeform path coordinates as INTEGERS in whatever unit `scale` declares, so
    passing scale=914400 (one local unit = one inch) rounds every point to the nearest inch and any
    curve collapses onto a lattice. Found 2026-10-29 while drawing jigsaw pieces, whose arcs came
    out as square notches. Work in thousandths of an inch instead.
    """
    ff = shapes.build_freeform(pts[0][0] * POLY_K, pts[0][1] * POLY_K, scale=914400.0 / POLY_K)
    ff.add_line_segments([(x * POLY_K, y * POLY_K) for x, y in pts[1:]], close=True)
    sh = ff.convert_to_shape(0, 0)
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(lw)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    plain(sh)
    return sh


def clear_chart_bg(ch):
    """No fill on the chart and plot areas, so charts sit on the slide in either theme."""
    cs = ch._chartSpace
    chart_el = cs.find(qn("c:chart"))
    sp = cs.find(qn("c:spPr"))
    if sp is None:
        sp = etree.Element(qn("c:spPr"))
        chart_el.addnext(sp)
    sp.clear()
    etree.SubElement(sp, qn("a:noFill"))
    etree.SubElement(etree.SubElement(sp, qn("a:ln")), qn("a:noFill"))
    pa = chart_el.find(qn("c:plotArea"))
    sp2 = pa.find(qn("c:spPr"))
    if sp2 is None:
        sp2 = etree.Element(qn("c:spPr"))
        ext = pa.find(qn("c:extLst"))
        ext.addprevious(sp2) if ext is not None else pa.append(sp2)
    sp2.clear()
    etree.SubElement(sp2, qn("a:noFill"))


def style_chart(ch, title=None, title_size=13, title_font=BODY, title_color="muted"):
    ch.font.name = BODY
    ch.font.size = Pt(12)
    ch.font.color.rgb = rgb("ink2")
    ch.has_legend = False
    if title:
        ch.has_title = True
        ch.chart_title.text_frame.text = title
        f = ch.chart_title.text_frame.paragraphs[0].runs[0].font
        f.size, f.bold, f.name = Pt(title_size), False, title_font
        f.color.rgb = rgb(title_color)
    clear_chart_bg(ch)


# ---------------------------------------------------------------- slide frames
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
SLIDES = []
ALIAS = re.compile(r"\b(kappa|fpr|FPR|RMSLE|chibar|eps|g031\w*|glymph_\w+|cellbender_\w+|decontx_bg)\b")


class Slide:
    def __init__(self, kicker=None, title=None, act=None, notes="", transition="morph", footer=True, keep_alias=False):
        self.s = prs.slides.add_slide(BLANK)
        self.sh = self.s.shapes
        self.n = len(prs.slides)
        self.m = Motion(self.s)
        self.t = 0
        self.notes = notes
        self.transition = transition
        self.keep_alias = keep_alias
        bg = self.s.background.fill
        if DARK:  # a faint top-to-bottom fall-off gives the midnight background some depth
            bg.gradient()
            bg.gradient_angle = 90
            st = bg.gradient_stops
            st[0].color.rgb, st[0].position = rgb("bg"), 0.0
            st[1].color.rgb, st[1].position = rgb("bg2"), 1.0
        else:
            bg.solid()
            bg.fore_color.rgb = rgb("bg")
        if footer:
            tb(self.sh, LM, 7.02, 6, 0.25, "Swarup Lab  ·  UCI MIND", size=9, color="faint", name="!!foot")
            tb(self.sh, W - LM - 1.0, 7.0, 1.0, 0.25, f"{self.n:02d}", size=10, color="muted", font=HEAD, align="r",
               name="!!page")
        if act is not None:
            for i in range(len(ACTS)):
                fill = "gold" if i == act else ("goldpale" if i < act else None)
                box(self.sh, W - LM - 1.02 + i * 0.22, 0.5, 0.11, 0.11, fill=fill, line=None if fill else "faint",
                    lw=0.75, kind=MSO_SHAPE.OVAL, name=f"!!dot{i}")
        if kicker:
            tb(self.sh, LM, 0.42, 10.5, 0.3, [[(kicker.upper(), {"spc": 2})]], size=10.5, color="goldtxt", bold=True,
               name="!!kicker")
        if title:
            assert len(title) <= 48, f"title too long ({len(title)}): {title}"
            tb(self.sh, LM, 0.66, CW, 0.75, title, size=34, color="ink", font=HEAD, name="!!title")
        SLIDES.append(self)

    def anim(self, shape, effect="rise", gap=140, group=0, at=None, dur=None):
        if at is None:
            at = self.t
            self.t += gap
        self.m.add(shape, effect, group=group, delay=at, dur=dur)
        return shape

    def takehome(self, text, y=6.2):
        assert len(text) <= 100, f"take-home too long ({len(text)}): {text}"
        g = self.sh.add_group_shape()
        box(g.shapes, LM, y, CW, 0.62, fill="goldtint", line="goldline", radius=0.5, shadow=True)
        icon_circle(g.shapes, LM + 0.12, y + 0.11, 0.4, "Key")
        tb(g.shapes, LM + 0.7, y, CW - 0.9, 0.62,
           [[("TAKE-HOME", {"size": 10, "bold": True, "color": "goldtxt", "spc": 1.5}), ("     " + text, {"size": 15, "color": "ink"})]],
           anchor="m")
        self.anim(g, "rise", group=1, at=0, dur=600)
        return g

    def finish(self):
        self.m.apply()
        set_transition(self.s, self.transition, dur=900 if self.transition == "morph" else 700)
        clicks = sorted(g for g in self.m.groups if g > 0)
        auto = 0 in self.m.groups
        build = ("▸ Build: " + ("content builds automatically after the transition" if auto else "no automatic build")
                 + (f"; {len(clicks)} click{'s' if len(clicks) > 1 else ''}"
                    + (" reveals the take-home." if len(clicks) == 1 else " (see notes) before advancing.") if clicks else "."))
        body = re.sub(r"(?<!\n)\n(?!\n)", " ", self.notes.strip())  # source line wraps -> spaces; blank lines stay paragraphs
        self.s.notes_slide.notes_text_frame.text = build + "\n\n" + body


def divider(num, title, sub, act, watermark, notes, aside=None):
    S = Slide(act=act, notes=notes)
    S.anim(tb(S.sh, LM, 1.0, 5, 2.55, f"{num:02d}", size=150, color="numeral", font=HEAD, name="!!numeral"), "fade", dur=900)
    S.anim(tb(S.sh, LM + 0.05, 3.55, 8, 0.4, [[(f"CHAPTER {num:02d}", {"spc": 3})]], size=11, color="goldtxt", bold=True), "fade")
    S.anim(tb(S.sh, LM, 3.95, 8.4, 0.9, title, size=48, color="ink", font=HEAD), "rise", gap=200)
    S.anim(tb(S.sh, LM + 0.02, 4.95, 7.8, 0.9, [[(s_, {})] for s_ in (sub if isinstance(sub, list) else [sub])], size=20, color="muted",
              line=1.1), "rise", gap=200)
    if aside:
        S.anim(tb(S.sh, LM + 0.02, 6.05, 7.8, 0.4, aside, size=13, color="goldtxt", italic=True), "fade")
    p = S.sh.add_picture(icon(watermark, "gold", 0.12 if DARK else 0.16), Inches(8.7), Inches(1.35), Inches(4.1), Inches(4.1))
    S.anim(p, "zoom", at=150, dur=1100)
    S.finish()
    return S


def fig_right_chips(S, figname, chips, take, fig_w=8.85, chip_w=2.9, y0=1.68, h=4.3, chip_h=1.22, big_size=24,
                    rounded=None, fmt=None):
    """Figure on the left, a column of stat chips (inserts) on the right, take-home on click."""
    S.anim(pic(S.sh, figname, LM, y0, fig_w, h, rounded=rounded), "fade", gap=300, dur=800)
    x = W - LM - chip_w
    gap = (h - len(chips) * chip_h) / max(len(chips) - 1, 1)
    for i, (big, label) in enumerate(chips):
        S.anim(chip(S.sh, x, y0 + i * (chip_h + gap), chip_w, chip_h, big, fmt(label, chip_w - 0.4, 12) if fmt else label,
                    big_size=big_size), "rise", gap=160)
    if take:
        S.takehome(take)


# ---------------------------------------------------------------- checks and save
def _all_text(sl):
    texts = []
    stack = list(sl.shapes)
    while stack:
        sh = stack.pop()
        if sh.shape_type == 6:  # group
            stack.extend(sh.shapes)
        elif sh.has_text_frame:
            texts.append(sh.text_frame.text)
        elif getattr(sh, "has_chart", False) and sh.has_chart:
            texts += [str(c) for c in sh.chart.plots[0].categories]
    texts.append(sl.notes_slide.notes_text_frame.text)
    return "\n".join(texts)


def fix_app_xml(pptx, titles):
    """python-pptx leaves docProps/app.xml at 0 slides; write the real slide/note counts and slide titles."""
    tmp = pptx + ".tmp"
    with zipfile.ZipFile(pptx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/app.xml":
                x = data.decode("utf8")
                x = re.sub(r"<Slides>\d+</Slides>", f"<Slides>{len(titles)}</Slides>", x)
                x = re.sub(r"<Notes>\d+</Notes>", f"<Notes>{len(titles)}</Notes>", x)
                x = re.sub(r"<PresentationFormat>[^<]*</PresentationFormat>", "<PresentationFormat>Widescreen</PresentationFormat>", x)
                x = re.sub(r"<vt:lpstr>Slide Titles</vt:lpstr></vt:variant><vt:variant><vt:i4>\d+</vt:i4>",
                           f"<vt:lpstr>Slide Titles</vt:lpstr></vt:variant><vt:variant><vt:i4>{len(titles)}</vt:i4>", x)
                parts = "".join(f"<vt:lpstr>{escape(t)}</vt:lpstr>" for t in ["Office Theme"] + titles)
                x = re.sub(r"<TitlesOfParts>.*?</TitlesOfParts>",
                           f'<TitlesOfParts><vt:vector size="{len(titles) + 1}" baseType="lpstr">{parts}</vt:vector></TitlesOfParts>', x)
                data = x.encode("utf8")
            zout.writestr(item, data)
    os.replace(tmp, pptx)


def finalize(stem, n_slides, title, subject):
    """Check naming and punctuation on every slide and its notes, then save <stem>[_dark].pptx with real app.xml counts."""
    assert len(prs.slides) == n_slides, len(prs.slides)
    for i, sl in enumerate(prs.slides):  # no internal names (only where kept), no em dashes
        blob = _all_text(sl)
        assert "\u2014" not in blob, f"em dash on slide {i + 1}"
        if not SLIDES[i].keep_alias:
            assert not ALIAS.search(blob), f"internal name on slide {i + 1}: {ALIAS.findall(blob)}"
    cp = prs.core_properties
    cp.title, cp.subject = title, subject
    cp.author = cp.last_modified_by = "Vivek Swarup"
    out = os.path.join(PRES, stem + ("_dark" if DARK else "") + ".pptx")
    prs.save(out)
    titles = []
    for i, sl in enumerate(prs.slides):
        t = next((sh.text_frame.text.replace("\n", " ") for sh in sl.shapes if sh.name in ("!!title", "!!numeral")), None)
        titles.append(title if i == 0 else (t or f"Slide {i + 1}"))
    fix_app_xml(out, titles)
    print("saved", out, len(prs.slides), "slides")
    assert fit_check.check(out) == 0, "text does not fit with the real fonts (see the list above)"
