"""Slide transitions (Morph with a Fade fallback) and entrance animations for python-pptx slides.

PowerPoint's timing XML is written by hand here, following the structure PowerPoint itself saves:
a main sequence of click groups, each holding effects that start "with previous" at a staggered delay.
Group 0 starts automatically once the slide transition ends; later groups start on a click.

Usage:
    m = Motion(slide)
    m.add(shape, "rise", delay=200)          # group 0: plays on slide entry
    m.add(card, "fade", group=1)             # group 1: plays on the first click
    m.apply(); set_transition(slide, "morph")
"""
from lxml import etree

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
P159 = "http://schemas.microsoft.com/office/powerpoint/2015/09/main"

# effect name -> (presetID, presetSubtype, default duration in ms)
EFFECTS = {
    "fade": (10, 0, 600),
    "rise": (42, 0, 750),     # Float In: fade while drifting up
    "zoom": (53, 16, 550),    # Zoom (object centre) with fade
    "wipe_l": (22, 8, 650),   # Wipe from left
    "wipe_d": (22, 1, 650),   # Wipe from top (draws downward)
    "wipe_u": (22, 4, 650),   # Wipe from bottom (draws upward)
}
WIPE_FILTER = {"wipe_l": "wipe(left)", "wipe_d": "wipe(up)", "wipe_u": "wipe(down)"}


def _tgt(spid):
    return f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>'


def _set_visible(cid, spid):
    return (f'<p:set><p:cBhvr><p:cTn id="{cid}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
            f'</p:cTn>{_tgt(spid)}<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr>'
            f'<p:to><p:strVal val="visible"/></p:to></p:set>')


def _fade(cid, spid, dur):
    return (f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{cid}" dur="{dur}"/>{_tgt(spid)}'
            f'</p:cBhvr></p:animEffect>')


def _anim(cid, spid, dur, attr, v0, v1, decel=True):
    def val(v):
        return f'<p:fltVal val="{v}"/>' if isinstance(v, (int, float)) else f'<p:strVal val="{v}"/>'
    dec = ' decel="100000"' if decel else ""
    return (f'<p:anim calcmode="lin" valueType="num"><p:cBhvr additive="base"><p:cTn id="{cid}" dur="{dur}" fill="hold"{dec}/>'
            f'{_tgt(spid)}<p:attrNameLst><p:attrName>{attr}</p:attrName></p:attrNameLst></p:cBhvr><p:tavLst>'
            f'<p:tav tm="0"><p:val>{val(v0)}</p:val></p:tav><p:tav tm="100000"><p:val>{val(v1)}</p:val></p:tav>'
            f'</p:tavLst></p:anim>')


class Motion:
    def __init__(self, slide):
        self.slide = slide
        self.groups = {}  # group index -> list of (spid, effect, delay, dur, is_sp)

    def add(self, shape, effect="fade", group=0, delay=0, dur=None):
        if effect not in EFFECTS:
            raise ValueError(effect)
        kind = etree.QName(shape._element).localname  # sp, pic, grpSp, cxnSp or graphicFrame
        self.groups.setdefault(group, []).append(
            (shape.shape_id, effect, int(delay), int(dur or EFFECTS[effect][2]), kind))
        return shape

    def _effect(self, ids, spid, effect, delay, dur, node_type):
        preset, sub, _ = EFFECTS[effect]
        cid = next(ids)
        kids = [_set_visible(next(ids), spid)]
        if effect == "fade":
            kids.append(_fade(next(ids), spid, dur))
        elif effect == "rise":
            kids.append(_fade(next(ids), spid, dur))
            kids.append(_anim(next(ids), spid, dur, "ppt_x", "#ppt_x", "#ppt_x"))
            kids.append(_anim(next(ids), spid, dur, "ppt_y", "#ppt_y+.04", "#ppt_y"))
        elif effect == "zoom":
            kids.append(_anim(next(ids), spid, dur, "ppt_w", 0, "#ppt_w"))
            kids.append(_anim(next(ids), spid, dur, "ppt_h", 0, "#ppt_h"))
            kids.append(_fade(next(ids), spid, dur))
        else:  # wipes
            kids.append(f'<p:animEffect transition="in" filter="{WIPE_FILTER[effect]}"><p:cBhvr>'
                        f'<p:cTn id="{next(ids)}" dur="{dur}"/>{_tgt(spid)}</p:cBhvr></p:animEffect>')
        return (f'<p:par><p:cTn id="{cid}" presetID="{preset}" presetClass="entr" presetSubtype="{sub}" fill="hold" '
                f'grpId="0" nodeType="{node_type}"><p:stCondLst><p:cond delay="{delay}"/></p:stCondLst>'
                f'<p:childTnLst>{"".join(kids)}</p:childTnLst></p:cTn></p:par>')

    def apply(self):
        if not self.groups:
            return
        counter = iter(range(3, 100000))
        ids = counter
        pars = []
        order = sorted(self.groups)
        for gi in order:
            items = sorted(self.groups[gi], key=lambda t: t[2])
            auto = gi == 0
            outer, inner = next(ids), next(ids)
            effects = []
            for k, (spid, eff, delay, dur, _) in enumerate(items):
                node = ("afterEffect" if auto else "clickEffect") if k == 0 else "withEffect"
                effects.append(self._effect(ids, spid, eff, delay, dur, node))
            start = '<p:cond delay="indefinite"/>' + ('<p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond>' if auto else "")
            pars.append(f'<p:par><p:cTn id="{outer}" fill="hold"><p:stCondLst>{start}</p:stCondLst><p:childTnLst>'
                        f'<p:par><p:cTn id="{inner}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
                        f'<p:childTnLst>{"".join(effects)}</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>')
        seen, bld = set(), []
        for gi in order:
            for spid, _, _, _, kind in self.groups[gi]:
                if spid in seen:
                    continue
                seen.add(spid)
                if kind == "sp":
                    bld.append(f'<p:bldP spid="{spid}" grpId="0" animBg="1"/>')
                elif kind == "graphicFrame":  # charts build as one object
                    bld.append(f'<p:bldGraphic spid="{spid}" grpId="0"><p:bldAsOne/></p:bldGraphic>')
        xml = (f'<p:timing xmlns:p="{P}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
               f'<p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq">'
               f'<p:childTnLst>{"".join(pars)}</p:childTnLst></p:cTn>'
               f'<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
               f'<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
               f'</p:seq></p:childTnLst></p:cTn></p:par></p:tnLst>'
               f'{"<p:bldLst>" + "".join(bld) + "</p:bldLst>" if bld else ""}</p:timing>')
        sld = self.slide._element
        for old in sld.findall(f"{{{P}}}timing"):
            sld.remove(old)
        _insert_after_transition(sld, etree.fromstring(xml))


def _insert_after_transition(sld, el):
    """Children of p:sld run cSld, clrMapOvr, transition (or its mc:AlternateContent), timing, extLst."""
    anchor = None
    for child in sld:
        tag = etree.QName(child).localname
        if tag in ("cSld", "clrMapOvr", "transition", "AlternateContent"):
            anchor = child
    if anchor is None:
        sld.insert(0, el)
    else:
        anchor.addnext(el)


def set_transition(slide, kind="morph", dur=900):
    sld = slide._element
    for child in list(sld):
        if etree.QName(child).localname in ("transition", "AlternateContent"):
            sld.remove(child)
    if kind == "morph":
        choice = (f'<mc:Choice xmlns:p159="{P159}" xmlns:p14="{P14}" Requires="p159">'
                  f'<p:transition spd="slow" p14:dur="{dur}"><p159:morph option="byObject"/></p:transition></mc:Choice>')
    else:
        choice = (f'<mc:Choice xmlns:p14="{P14}" Requires="p14">'
                  f'<p:transition spd="slow" p14:dur="{dur}"><p:fade/></p:transition></mc:Choice>')
    xml = (f'<mc:AlternateContent xmlns:mc="{MC}" xmlns:p="{P}">{choice}'
           f'<mc:Fallback><p:transition spd="slow"><p:fade/></p:transition></mc:Fallback></mc:AlternateContent>')
    el = etree.fromstring(xml)
    anchor = None
    for child in sld:
        if etree.QName(child).localname in ("cSld", "clrMapOvr"):
            anchor = child
    anchor.addnext(el)
