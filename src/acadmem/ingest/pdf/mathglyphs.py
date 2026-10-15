"""Recover equations that a PDF draws as vector paths instead of text.

Chrome/Skia prints KaTeX-style maths (as in the rebuilt worked-solution guides) as filled outlines, one path per
glyph, with no text layer, so every text extractor (pymupdf, pymupdf4llm, pdftotext) silently drops the equations.

The recovery is deterministic and fast (no OCR model):

1. every filled path is matched, by scale-free outline, against ``math_lexicon.json`` (a survey of the glyphs those
   PDFs use, labelled once; see ``eval/build_glyph_lexicon.py``);
2. glyphs and thin filled rules (fraction bars, minus signs, square-root vinculums) that touch are grouped into
   formulas;
3. each group is linearised to LaTeX-ish text (fractions, roots, sub/superscripts, big operators with limits,
   accents, matrices);
4. :func:`recover_math` writes that text back into the (in-memory) document at the formula's position so any text
   extractor reading the patched document sees it in the right place.

A page with fewer than ``MIN_GLYPHS`` recognised glyphs is left alone, so ordinary PDFs are unaffected.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

MIN_GLYPHS = 3
MATCH_TOL = 0.05
MINUS_THICKNESS = 0.52  # thickness of a full-size minus; fraction bars are ~0.78
MINUS_MAX_THICKNESS = 0.65
LEXICON_PATH = Path(__file__).with_name("math_lexicon.json")


def outline_key(items, rect=None) -> tuple[str, list[float]]:
    """Scale-free outline of a drawing: (operator string, flattened normalised points)."""
    ops: list[str] = []
    pts: list[tuple[float, float]] = []
    for it in items:
        ops.append(it[0])
        for q in it[1:]:
            if hasattr(q, "x") and hasattr(q, "y") and not hasattr(q, "x0"):
                pts.append((q.x, q.y))
            elif hasattr(q, "x0") and hasattr(q, "ul"):
                pts.extend((c.x, c.y) for c in (q.ul, q.ur, q.ll, q.lr))
            elif hasattr(q, "x0"):
                pts += [(q.x0, q.y0), (q.x1, q.y0), (q.x0, q.y1), (q.x1, q.y1)]
    if rect is None:
        return "".join(ops), []
    s = max(rect.width, rect.height)
    vec = [c for (px, py) in pts for c in ((px - rect.x0) / s, (py - rect.y0) / s)]
    return "".join(ops), vec


@dataclass
class Token:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str = ""
    kind: str = "glyph"  # glyph | delim | op | sqrt | acc | dot | bar | comp
    r: float | None = None  # size relative to the lexicon's canonical size (None: never a script)
    label: str = ""
    cx: float = field(init=False)
    cy: float = field(init=False)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.cx = (self.x0 + self.x1) / 2
        self.cy = (self.y0 + self.y1) / 2

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


@lru_cache(maxsize=1)
def load_lexicon() -> dict[str, list[tuple[list[float], str, float]]]:
    by_key: dict[str, list[tuple[list[float], str, float]]] = defaultdict(list)
    for e in json.loads(LEXICON_PATH.read_text(encoding="utf-8")):
        by_key[e["key"]].append((e["vec"], e["label"], float(e.get("h") or 0)))
    return by_key


def _match(key: str, vec: list[float], lexicon) -> tuple[str, float] | None:
    best = None
    for lvec, label, h in lexicon.get(key, ()):
        if len(lvec) != len(vec):
            continue
        d = max(abs(a - b) for a, b in zip(lvec, vec))
        if d < MATCH_TOL and (best is None or d < best[0]):
            best = (d, label, h)
    return (best[1], best[2]) if best else None


_DELIMS = {"D(": "(", "D)": ")", "D[": "[", "D]": "]", "D{": "\\{", "D}": "\\}", "D|": "|"}


def _is_bar(x) -> bool:
    r = x["rect"]
    return x["type"] in ("f", "fs") and 1.5 <= r.width <= 140 and r.height <= 1.6 and r.width >= 3 * max(r.height, 0.2)


def page_tokens(page) -> list[Token]:
    lexicon = load_lexicon()
    tokens: list[Token] = []
    for x in page.get_drawings():
        r = x["rect"]
        if x["type"] not in ("f", "fs"):
            continue
        if _is_bar(x) and len(x["items"]) <= 12:
            tokens.append(Token(r.x0, r.y0, r.x1, max(r.y1, r.y0 + 0.4), kind="bar"))
            continue
        s = max(r.width, r.height)
        if s < 0.3 or s > 60:
            continue
        key, vec = outline_key(x["items"], r)
        if not vec:
            continue
        hit = _match(key, vec, lexicon)
        if not hit or not hit[0]:
            continue
        label, canon = hit
        t = Token(r.x0, r.y0, r.x1, r.y1, label=label)
        t.r = (r.height / canon) if canon else None
        if label in _DELIMS:
            t.kind, t.text, t.r = "delim", _DELIMS[label], None
        elif label == "||":
            t.text = "\\|"
        elif label.startswith("OP:"):
            t.kind, t.text = "op", label[3:]
        elif label == "SQRT":
            t.kind, t.text = "sqrt", "\\sqrt"
        elif label.startswith("ACC:"):
            t.kind, t.text = "acc", label[4:]
        elif label == "DOT":
            t.kind = "dot"
        elif label == "DDOT":
            t.kind = "dot"
        else:
            t.text = label
        tokens.append(t)
    return tokens


def group_tokens(tokens: list[Token], pad_x: float = 9.0, pad_y: float = 2.5, bar_pad_y: float = 8.0) -> list[list[Token]]:
    """Union tokens whose padded boxes touch; groups with no real glyph (just rules) are dropped."""
    tokens = _delimiters(list(tokens))
    n = len(tokens)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cell = 24.0
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, t in enumerate(tokens):
        for gx in range(int((t.x0 - pad_x) // cell), int((t.x1 + pad_x) // cell) + 1):
            for gy in range(int((t.y0 - bar_pad_y) // cell), int((t.y1 + bar_pad_y) // cell) + 1):
                grid[(gx, gy)].append(i)
    for members in grid.values():
        for a in range(len(members)):
            ta = tokens[members[a]]
            for b in range(a + 1, len(members)):
                tb = tokens[members[b]]
                py = bar_pad_y if "bar" in (ta.kind, tb.kind) else pad_y
                px = pad_x
                if ta.x0 - px <= tb.x1 and tb.x0 - px <= ta.x1 and ta.y0 - py <= tb.y1 and tb.y0 - py <= ta.y1:
                    parent[find(members[a])] = find(members[b])
    # tall brackets (matrices, big parentheses) pull in everything between them
    d = _dim(tokens)
    for i, t in enumerate(tokens):
        if t.kind == "delim" and t.h > 2.4 * d:
            for j, u in enumerate(tokens):
                if u is not t and t.x0 - 0.5 <= u.cx and u.cy >= t.y0 - 1 and u.cy <= t.y1 + 1:
                    mates = [v for v in tokens if v.kind == "delim" and v is not t and v.x0 > t.x1 and abs(v.y0 - t.y0) < 4 and abs(v.y1 - t.y1) < 4]
                    if mates and u.cx <= min(mates, key=lambda v: v.x0).x1 + 0.5:
                        parent[find(i)] = find(j)
    groups: dict[int, list[Token]] = defaultdict(list)
    for i, t in enumerate(tokens):
        groups[find(i)].append(t)
    return [g for g in groups.values() if sum(1 for t in g if t.kind != "bar") >= 1]


# ---------------------------------------------------------------------------------------------------------------
# Linearisation
# ---------------------------------------------------------------------------------------------------------------


def _union(ts: list[Token], text: str, kind: str = "comp", r: float | None = None) -> Token:
    return Token(min(t.x0 for t in ts), min(t.y0 for t in ts), max(t.x1 for t in ts), max(t.y1 for t in ts), text, kind, r)


def _dim(ts: list[Token]) -> float:
    hs = sorted(t.h for t in ts if t.kind in ("glyph", "comp") and t.h > 1)
    return hs[len(hs) // 2] if hs else 8.0


def _fractions(ts: list[Token]) -> list[Token]:
    changed = True
    while changed:
        changed = False
        d = _dim(ts)
        for bar in sorted((t for t in ts if t.kind == "bar" and t.w > 3 and t.h >= MINUS_MAX_THICKNESS), key=lambda t: -t.w):
            reach = 3.0 * d
            above = [t for t in ts if t is not bar and bar.x0 - 1 <= t.cx <= bar.x1 + 1 and t.cy < bar.cy and bar.y0 - t.y1 < reach and t.y1 <= bar.y1 + 0.5]
            below = [t for t in ts if t is not bar and bar.x0 - 1 <= t.cx <= bar.x1 + 1 and t.cy > bar.cy and t.y0 - bar.y1 < reach and t.y0 >= bar.y0 - 0.5]
            if not above or not below:
                continue
            # a rule that sits right on top of a glyph with nothing above is an overline, not a fraction
            inner = set(map(id, above)) | set(map(id, below))
            comp = _union(above + below + [bar], "\\frac{%s}{%s}" % (_linearise(above), _linearise(below)))
            ts = [t for t in ts if id(t) not in inner and t is not bar] + [comp]
            changed = True
            break
    return ts


def _roots(ts: list[Token]) -> list[Token]:
    for rad in [t for t in ts if t.kind == "sqrt"]:
        bars = [b for b in ts if b.kind == "bar" and abs(b.x0 - rad.x1) < 3.5 and rad.y0 - 2 <= b.y0 <= rad.y0 + 0.3 * rad.h + 2]
        if not bars:
            continue
        bar = min(bars, key=lambda b: abs(b.x0 - rad.x1))
        content = [t for t in ts if t not in (bar, rad) and bar.x0 - 1 <= t.cx <= bar.x1 + 1 and bar.y1 - 1 <= t.cy <= rad.y1 + 2]
        comp = _union(content + [bar, rad], "\\sqrt{%s}" % _linearise(content))
        ids = set(map(id, content)) | {id(bar), id(rad)}
        ts = [t for t in ts if id(t) not in ids] + [comp]
    return ts


def _overlines(ts: list[Token]) -> list[Token]:
    for bar in [t for t in ts if t.kind == "bar"]:
        under = [t for t in ts if t is not bar and bar.x0 - 1 <= t.cx <= bar.x1 + 1 and -0.5 <= t.y0 - bar.y1 < 3.0 and t.y1 > bar.y1]
        if under and bar.w < 3.2 * max(sum(t.w for t in under), 4):
            above = [t for t in ts if t is not bar and bar.x0 - 1 <= t.cx <= bar.x1 + 1 and t.cy < bar.cy and bar.y0 - t.y1 < 2]
            if above:
                continue
            comp = _union(under + [bar], "\\overline{%s}" % _linearise(under), r=max((t.r or 0) for t in under) or None)
            ids = set(map(id, under)) | {id(bar)}
            ts = [t for t in ts if id(t) not in ids] + [comp]
    return ts


def _minus(ts: list[Token]) -> list[Token]:
    for t in ts:
        if t.kind == "bar":
            t.kind, t.text, t.r = "glyph", "-", (t.h / MINUS_THICKNESS if t.w <= 9.5 else None)
    return ts


def _accents(ts: list[Token]) -> list[Token]:
    for a in [t for t in ts if t.kind in ("acc", "dot")]:
        targets = [t for t in ts if t is not a and t.kind in ("glyph", "comp") and a.cx >= t.x0 - 1.5 and a.cx <= t.x1 + 3 and 0 <= t.y0 - a.y1 < 0.5 * max(t.h, 5) and a.cy < t.y0 + 0.2 * t.h]
        if not targets:
            continue
        tgt = min(targets, key=lambda t: t.y0 - a.y1)
        if a.kind == "acc":
            name = a.text
        else:
            others = [d for d in ts if d is not a and d.kind == "dot" and abs(d.cx - a.cx) < 4 and abs(d.cy - a.cy) < 1.5]
            name = "\\ddot" if a.label == "DDOT" or others else "\\dot"
        if a.kind == "dot" and tgt.kind == "glyph" and not tgt.text.isalpha() and not tgt.text.startswith("\\"):
            continue
        comp = _union([tgt, a], "%s{%s}" % (name, tgt.text), r=tgt.r)
        ts = [t for t in ts if t is not a and t is not tgt] + [comp]
    return ts


def _delimiters(ts: list[Token]) -> list[Token]:
    """Merge stacked delimiter pieces (tall brackets are drawn as several paths) into one token."""
    ds = [t for t in ts if t.kind == "delim"]
    ds.sort(key=lambda t: (round(t.cx), t.y0))
    used: set[int] = set()
    out: list[Token] = []
    for t in ds:
        if id(t) in used:
            continue
        chain = [t]
        used.add(id(t))
        grew = True
        while grew:
            grew = False
            for u in ds:
                if id(u) in used:
                    continue
                if any(u.x0 < c.x1 + 0.3 and c.x0 < u.x1 + 0.3 and u.y0 <= c.y1 + 1.8 and c.y0 <= u.y1 + 1.8 for c in chain):
                    chain.append(u)
                    used.add(id(u))
                    grew = True
        text = next((c.text for c in chain if c.text != "|"), "|")
        out.append(_union(chain, text, kind="delim"))
    return [t for t in ts if t.kind != "delim"] + out


def _matrices(ts: list[Token]) -> list[Token]:
    d = _dim(ts)
    lefts = [t for t in ts if t.kind == "delim" and t.text in ("(", "[", "|") and t.h > 2.4 * d]
    for left in sorted(lefts, key=lambda t: t.x0):
        rights = [t for t in ts if t.kind == "delim" and t.text in (")", "]", "|") and t.x0 > left.x1 and abs(t.y0 - left.y0) < 4 and abs(t.y1 - left.y1) < 4]
        if not rights:
            continue
        right = min(rights, key=lambda t: t.x0)
        cells = [t for t in ts if t not in (left, right) and left.x1 - 1 <= t.cx <= right.x0 + 1 and left.y0 - 2 <= t.cy <= left.y1 + 2 and t.kind != "delim"]
        top = max([c.r or 0 for c in cells] + [0.01])
        main = [t for t in cells if (t.r is None or t.r >= 0.8 * top) and (t.kind == "comp" or t.text[:1].isalnum() or t.text.startswith("\\"))]
        if len(main) < 2:
            continue
        rows: list[list[Token]] = []
        for t in sorted(main, key=lambda t: t.cy):
            if rows and abs(t.cy - sum(c.cy for c in rows[-1]) / len(rows[-1])) < 0.55 * d:
                rows[-1].append(t)
            else:
                rows.append([t])
        if len(rows) < 2:
            continue
        for t in cells:
            if all(t is not m for m in main):
                min(rows, key=lambda row: abs(row[0].cy - t.cy)).append(t)
        lines = []
        for row in rows:
            row.sort(key=lambda t: t.x0)
            cols: list[list[Token]] = [[row[0]]]
            for t in row[1:]:
                if t.x0 - max(c.x1 for c in cols[-1]) > 0.9 * d:
                    cols.append([t])
                else:
                    cols[-1].append(t)
            lines.append(" & ".join(_linearise(c) for c in cols))
        env = {"(": "pmatrix", "[": "bmatrix", "|": "vmatrix"}[left.text]
        comp = _union(cells + [left, right], "\\begin{%s} %s \\end{%s}" % (env, " \\\\ ".join(lines), env))
        ids = set(map(id, cells)) | {id(left), id(right)}
        ts = [t for t in ts if id(t) not in ids] + [comp]
    return ts


def _big_ops(ts: list[Token], r0: float) -> list[Token]:
    for op in [t for t in ts if t.kind == "op"]:
        side = op.text == "\\int"
        lo_x, hi_x = (op.x0 - 4, op.x1 + 18) if side else (op.x0 - 2, op.x1 + 2)
        lims = [t for t in ts if t is not op and t.r is not None and t.r < 0.85 * r0 and lo_x <= t.cx <= hi_x and (
            (t.cy <= op.y0 + 0.4 * op.h or t.cy >= op.y1 - 0.4 * op.h) if side else (t.y1 <= op.y0 + 1 or t.y0 >= op.y1 - 1))]
        if not lims:
            op.kind = "glyph"
            op.r = None
            continue
        up = sorted((t for t in lims if t.cy < op.cy), key=lambda t: t.x0)
        lo = sorted((t for t in lims if t.cy >= op.cy), key=lambda t: t.x0)
        text = op.text + ("_{%s}" % _linearise(lo) if lo else "") + ("^{%s}" % _linearise(up) if up else "")
        comp = _union(lims + [op], text)
        ids = set(map(id, lims)) | {id(op)}
        ts = [t for t in ts if id(t) not in ids] + [comp]
    for op in ts:
        if op.kind == "op":
            op.kind, op.r = "glyph", None
    return ts


def _braces(text: str) -> str:
    return text if len(text) == 1 or re.fullmatch(r"\\[A-Za-z]+", text) else "{%s}" % text


_OPS = {"=", "+", "-", "<", ">", "\\le", "\\ge", "\\ne", "\\approx", "\\simeq", "\\sim", "\\to", "\\pm", "\\mp", "\\times",
        "\\in", "\\Rightarrow", "\\Leftrightarrow", "\\propto", "\\ll", "\\cap", "\\oplus", "\\cdot"}


def _cls(ch: str) -> str:
    return "d" if ch.isdigit() else "a" if ch.isalpha() else ch


def _line(ts: list[Token], r0: float) -> str:
    if not ts:
        return ""
    ts = sorted(ts, key=lambda t: t.x0)
    hm = max((t.h for t in ts if t.kind in ("glyph", "comp") and (t.r is None or t.r >= 0.85 * r0)), default=8.0)
    mains = [t for t in ts if t.r is None or t.r >= 0.85 * r0 or t.kind == "delim"]
    scripts = [t for t in ts if t not in mains]
    if not mains:
        return " ".join(t.text for t in ts)
    # dots: a dot sitting on the baseline is a full stop / decimal point, a raised one is a centred dot
    base = sorted(t.y1 for t in mains if t.kind == "glyph" and t.text.isalnum())
    baseline = base[len(base) // 2] if base else max(t.y1 for t in mains)
    last = max(mains, key=lambda t: t.x1)
    for t in mains:
        if t.kind == "dot":
            # a full stop ends the line; a centred dot sits above the baseline
            t.text = "." if t is last or t.y1 >= baseline - 0.25 * hm else "\\cdot"
    attach: dict[int, list[Token]] = {id(m): [] for m in mains}
    for s in scripts:
        left = [m for m in mains if m.x0 <= s.x0 + 0.5]
        if not left:
            mains.append(s)
            attach[id(s)] = []
            continue
        attach[id(max(left, key=lambda m: m.x1 if m.x1 <= s.x1 + 0.5 else m.x0))].append(s)
    mains.sort(key=lambda t: t.x0)
    pieces: list[tuple[str, float, float, bool, str, str]] = []
    for m in mains:
        txt = m.text
        sc = attach[id(m)]
        right = m.x1
        if sc:
            if m.kind == "delim" or m.h > 1.6 * hm:
                up = [s for s in sc if s.cy < m.y0 + 0.4 * m.h]
            else:
                up = [s for s in sc if s.y1 < baseline - 0.12 * hm]
            dn = [s for s in sc if s not in up]
            for part, mark in ((sorted(dn, key=lambda s: s.x0), "_"), (sorted(up, key=lambda s: s.x0), "^")):
                if part:
                    txt += mark + _braces(_linearise(part))
                    right = max(right, max(s.x1 for s in part))
        pieces.append((txt, m.x0, right, bool(sc), m.text, m.kind))
    out = ""
    prev = None
    unary = True
    binary_op = False
    for txt, x0, x1, scripted, base_text, kind in pieces:
        is_op = base_text in _OPS and kind == "glyph" and not scripted
        if prev is not None:
            gap = x0 - prev[2]
            ptxt = prev[0]
            space = False
            if re.search(r"\\[A-Za-z]+$", ptxt) and txt[:1].isalnum():
                space = True
            elif (is_op and not (unary and base_text in "+-")) or binary_op:
                space = True
            elif ptxt[-1:] in ",;" and gap > 1.0:
                space = True
            elif gap > 0.45 * hm:
                space = True
            elif gap > 0.15 * hm and _cls(ptxt[-1:]) != _cls(txt[:1]) and _cls(ptxt[-1:]) in "da" and _cls(txt[:1]) in "da":
                space = True
            elif prev[3] and gap > 0.12 * hm and txt[:1].isalnum():
                space = True
            if space:
                out += " "
        out += txt
        binary_op = is_op and not (unary and base_text in "+-")
        unary = is_op or (base_text in ("(", "[", ",", "\\{") and kind in ("delim", "glyph"))
        prev = (txt, x0, max(x1, prev[2] if prev else x1), scripted, base_text, is_op)
    return out


def _linearise(ts: list[Token]) -> str:
    ts = [t for t in ts]
    if not ts:
        return ""
    ts = _fractions(ts)
    ts = _roots(ts)
    ts = _overlines(ts)
    ts = _minus(ts)
    ts = _delimiters(ts)
    ts = _accents(ts)
    ts = _matrices(ts)
    rs = [t.r for t in ts if t.kind == "glyph" and t.r is not None and t.text and (t.text.isalnum() or t.text.startswith("\\"))]
    r0 = max(rs) if rs else 1.0
    ts = _big_ops(ts, r0)
    return _line(ts, r0).strip()


_FUNCTIONS = re.compile(r"(log|ln|sin|cos|tan|det|exp|lim|min|max)")


def _separate_functions(text: str) -> str:
    """Roman function names have no gap to their neighbours in the outlines; put spaces around them."""
    parts = re.split(r"(\\[A-Za-z]+)", text)
    return "".join(p if p.startswith("\\") else _FUNCTIONS.sub(r" \1 ", p) for p in parts)


def linearise_group(group: list[Token]) -> tuple[str, tuple[float, float, float, float]]:
    text = _separate_functions(_linearise(list(group)))
    text = re.sub(r"\s+", " ", re.sub(r"\s+([_^])", r"\1", text)).strip()
    return text, (min(t.x0 for t in group), min(t.y0 for t in group), max(t.x1 for t in group), max(t.y1 for t in group))


# ---------------------------------------------------------------------------------------------------------------
# Writing the text back into a document
# ---------------------------------------------------------------------------------------------------------------


def _word_boxes(page) -> list[tuple[float, float, float, float]]:
    return [w[:4] for w in page.get_text("words")]


@dataclass
class Formula:
    text: str  # LaTeX-ish, without delimiters
    bbox: tuple[float, float, float, float]
    inline: bool

    @property
    def markdown(self) -> str:
        return f"${self.text}$" if self.inline else f"$$ {self.text} $$"


def drawn_by_browser(metadata: dict | None) -> bool:
    """True for PDFs printed from a browser (Skia/PDF, Chrome), which draw web-font maths as outlines.

    The glyph lexicon only describes those, and looking at other PDFs only risks mistaking table rules and
    diagrams for equations, so everything else is left to the normal text extraction.
    """
    meta = metadata or {}
    return "skia" in str(meta.get("producer") or "").lower() or "chrome" in str(meta.get("creator") or "").lower()


def recover_page(page) -> list[Formula]:
    """Replace the formula paths on ``page`` (in memory) by their recovered text; returns what was recovered."""
    import pymupdf

    tokens = page_tokens(page)
    if sum(1 for t in tokens if t.kind != "bar") < MIN_GLYPHS:
        return []
    words = _word_boxes(page)
    found: list[Formula] = []
    for group in group_tokens(tokens):
        text, bbox = linearise_group(group)
        if not text or not re.search(r"[A-Za-z0-9]", text):
            continue
        x0, y0, x1, y1 = bbox
        mid = (y0 + y1) / 2
        inline = any(
            wy0 - 1 <= mid <= wy1 + 1 and (x0 - 14 < wx1 <= x0 + 1 or x1 - 1 <= wx0 < x1 + 14) for wx0, wy0, wx1, wy1 in words
        )
        found.append(Formula(text, bbox, inline or len(text) <= 3))
    if not found:
        return []
    # take the glyph outlines away (pymupdf4llm treats dense vector art as a picture and drops text inside it)
    for f in found:
        x0, y0, x1, y1 = f.bbox
        page.add_redact_annot(pymupdf.Rect(x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5), fill=False)
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
        text=pymupdf.PDF_REDACT_TEXT_NONE,
    )
    for f in found:
        x0, y0, x1, y1 = f.bbox
        text = f.markdown
        h = y1 - y0
        size = max(5.0, min(11.0, h * 0.9 if h > 7 else 9.0))
        width = len(text) * size * 0.5
        avail = max(x1 - x0, 60.0) * 1.6
        if width > avail:
            size = max(4.0, size * avail / width)
        page.insert_text((x0, y1 - 0.2 * min(h, size * 1.2)), text, fontsize=size, fontname="helv", color=(0, 0, 0))
    return found


def recover_math(doc) -> dict[int, list[Formula]]:
    """Recover vector-drawn equations on every page of ``doc`` (modified in memory); {page index: formulas}."""
    out: dict[int, list[Formula]] = {}
    for page in doc:
        found = recover_page(page)
        if found:
            out[page.number] = found
    return out


def fill_formula_boxes(text: str, page_boxes: list[dict], formulas: list[Formula]) -> str:
    """Put back recovered formulas that pymupdf4llm's layout model left out of the page text.

    It replaces any block it calls a "formula" by an image placeholder, and some equations end up in no block at
    all. Each missing formula is inserted right after the last block that ends above it (reading order).
    """
    squashed = re.sub(r"\s+", "", text)
    boxes = sorted((b for b in page_boxes or () if b.get("pos") is not None), key=lambda b: b["pos"][0])
    inserts: list[tuple[int, float, float, str]] = []
    for f in formulas:
        if re.sub(r"\s+", "", f.markdown) in squashed or re.sub(r"\s+", "", f"${f.text}$") in squashed:
            continue
        cy = (f.bbox[1] + f.bbox[3]) / 2
        above = [b for b in boxes if b["bbox"][3] <= cy + 1 and b["class"] not in ("page-header", "page-footer")]
        pos = int(above[-1]["pos"][1]) if above else 0
        inserts.append((pos, f.bbox[1], f.bbox[0], f.markdown))
    # later positions first, so earlier offsets stay valid; same position: keep reading order
    by_pos: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    for pos, y, x, md in inserts:
        by_pos[pos].append((round(y / 4), x, md))
    for pos in sorted(by_pos, reverse=True):
        extra = "\n\n".join(md for _, _, md in sorted(by_pos[pos])) + "\n\n"
        text = text[:pos] + extra + text[pos:]
    return text
