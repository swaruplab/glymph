#!/usr/bin/env python3
"""Package and animation gates for a built deck (the lab's deck rules, sections 9 and 10).

    python3 code/deck/check_package.py reports/Glymph_overview_dark.pptx

OPC package: no duplicate Override in [Content_Types].xml, every part has a content type, every internal relationship
target exists, every r:id / r:embed / r:link used in a part resolves in that part's relationships, and docProps/app.xml
carries the real slide and note counts.
Animation: every animated shape id exists on its slide, timing node ids are unique within a slide, the transition comes
before the timing element, and names starting with "!!" (Morph anchors) are unique within a slide.
Exits non-zero on any failure.
"""
import posixpath
import re
import sys
import zipfile

from lxml import etree

path = sys.argv[1]
z = zipfile.ZipFile(path)
names = set(z.namelist())
fail = []
CT = "{http://schemas.openxmlformats.org/package/2006/content-types}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"

ct = etree.fromstring(z.read("[Content_Types].xml"))
over = [o.get("PartName") for o in ct.findall(CT + "Override")]
dup = {o for o in over if over.count(o) > 1}
if dup:
    fail.append(f"duplicate Override entries: {sorted(dup)}")
defaults = {d.get("Extension").lower() for d in ct.findall(CT + "Default")}
for n in names:
    if n == "[Content_Types].xml" or n.endswith("/"):
        continue
    if "/" + n not in over and n.rsplit(".", 1)[-1].lower() not in defaults:
        fail.append(f"no content type for {n}")
for o in over:
    if o.lstrip("/") not in names:
        fail.append(f"Override for a missing part: {o}")

n_slides = n_notes = 0
for n in sorted(names):
    if not n.endswith(".rels"):
        continue
    base = posixpath.dirname(posixpath.dirname(n))
    part = posixpath.join(base, posixpath.basename(n)[:-5])
    ids = {}
    for r in etree.fromstring(z.read(n)).findall(REL + "Relationship"):
        ids[r.get("Id")] = r
        if r.get("TargetMode") == "External":
            continue
        tgt = r.get("Target")
        full = tgt.lstrip("/") if tgt.startswith("/") else posixpath.normpath(posixpath.join(base, tgt))
        if full not in names:
            fail.append(f"{n}: target missing: {tgt}")
    if part in names and part.endswith(".xml"):
        xml = z.read(part).decode("utf8")
        for attr in ("id", "embed", "link", "pict", "dm", "lo", "qs", "cs"):
            for rid in set(re.findall(rf'\br:{attr}="([^"]+)"', xml)):
                if rid not in ids:
                    fail.append(f"{part}: r:{attr}={rid} has no relationship")
for n in names:
    n_slides += bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", n))
    n_notes += bool(re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n))
app = z.read("docProps/app.xml").decode("utf8")
for tag, want in (("Slides", n_slides), ("Notes", n_notes)):
    m = re.search(rf"<{tag}>(\d+)</{tag}>", app)
    if not m or int(m.group(1)) != want:
        fail.append(f"docProps/app.xml {tag} = {m.group(1) if m else None}, the package has {want}")

n_anim = n_morph = 0
for n in sorted(x for x in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", x)):
    sld = etree.fromstring(z.read(n))
    shape_ids, anchors = set(), []
    for el in sld.iter(P + "cNvPr"):
        shape_ids.add(el.get("id"))
        if (el.get("name") or "").startswith("!!"):
            anchors.append(el.get("name"))
    for a in {x for x in anchors if anchors.count(x) > 1}:
        fail.append(f"{n}: Morph anchor name used twice: {a}")
    kids = [etree.QName(c).localname for c in sld]
    if "timing" in kids:
        tr = [i for i, k in enumerate(kids) if k in ("transition", "AlternateContent")]
        if tr and kids.index("timing") < tr[-1]:
            fail.append(f"{n}: timing comes before the transition")
        tn = [c.get("id") for c in sld.iter(P + "cTn")]
        if len(tn) != len(set(tn)):
            fail.append(f"{n}: timing node ids are not unique")
        for t in sld.iter(P + "spTgt"):
            n_anim += 1
            if t.get("spid") not in shape_ids:
                fail.append(f"{n}: animation targets shape {t.get('spid')}, which is not on the slide")
        for b in list(sld.iter(P + "bldP")) + list(sld.iter(P + "bldGraphic")):
            if b.get("spid") not in shape_ids:
                fail.append(f"{n}: build list names shape {b.get('spid')}, which is not on the slide")
    n_morph += b"morph" in z.read(n)
print(f"{path}: {n_slides} slides, {n_notes} notes, {n_anim} animation targets, {n_morph} slides with Morph")
for f in fail:
    print("FAIL", f)
print("package and animation checks:", "FAILED" if fail else "passed")
sys.exit(1 if fail else 0)
