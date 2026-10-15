"""Build src/acadmem/ingest/pdf/math_lexicon.json from the cluster dump made by the glyph survey.

Chrome/Skia prints KaTeX-style maths as filled vector paths with no text layer. The survey clusters those
paths by outline (scale-free); LABELS maps survey cluster index -> what the glyph is. Kept for reproducibility.
Usage: python eval/build_glyph_lexicon.py allc.pkl <pdf folder glob> out.json
"""
import collections
import glob
import json
import pickle
import sys

import pymupdf

sys.path.insert(0, "src")
from acadmem.ingest.pdf.mathglyphs import outline_key  # noqa: E402

L = {}
def put(labels, idxs):
    for i in idxs:
        L[i] = labels
for i, v in {
    0: "1", 1: "2", 2: "=", 3: "0", 4: "D)", 5: "D(", 7: "DOT", 8: "+", 9: ",", 10: "/", 11: "3", 12: "D|", 13: "n",
    14: "x", 16: "4", 17: "5", 18: "k", 19: "t", 20: "6", 21: "i", 22: "a", 23: "j", 24: "8", 25: "X", 26: "y",
    27: "s", 28: "v", 29: "D|", 30: "9", 31: "p", 32: "7", 33: "A", 34: "m", 35: "b", 36: "R", 37: "d", 38: "D[",
    39: "Z", 40: "T", 41: "D]", 42: "e", 43: "\\pi", 44: "r", 45: "u", 46: "C", 47: "c", 48: "N", 49: "L", 50: "s",
    51: "o", 53: "\\omega", 54: "H", 55: "E", 56: "f", 57: "\\theta", 58: "n", 59: "P",
    60: "q", 61: "V", 62: "B", 63: "\\alpha", 64: "Y", 65: "i", 66: "\\lambda", 67: "D", 68: "l", 69: "\\mu",
    70: "h", 71: "I", 72: "g", 73: "m", 74: "'", 75: "\\le", 76: "S", 77: "\\sigma", 79: "SQRT", 80: "c",
    81: "||", 82: "Q", 83: "OP:\\sum", 85: "M", 86: "g", 87: "G", 88: "a", 89: "<", 90: "w", 91: "\\Omega",
    92: "t", 93: "K", 94: "e", 95: "Z", 96: "\\beta", 97: ">", 99: "r", 100: "W", 101: "F", 102: "\\rho",
    103: "\\infty", 104: "\\ge", 105: "*", 106: "\\ell", 107: "V", 108: "\\tau", 109: "OP:\\sum", 110: "\\times",
    111: "J", 112: "OP:\\int", 113: "p", 114: "\\phi", 115: "\\delta", 116: "d", 118: "\\to", 119: "U",
    120: "\\epsilon", 121: "\\ldots", 122: "o", 123: "k", 124: "D|", 125: "h", 126: "DOT", 127: "\\simeq", 128: "l",
    129: "D(", 130: "D)", 131: "SQRT", 132: "A", 133: "X", 134: "\\Delta", 135: "F", 136: "\\pm", 137: "D{",
    138: "D}", 139: "D[", 140: "D]", 141: "O", 142: "D[", 143: "D]", 144: "H", 145: "\\approx", 146: ":",
    148: ";", 150: "\\gamma", 152: "M", 153: "Z", 155: "N", 156: "\\in", 157: "OP:\\int", 158: "ACC:\\hat",
    159: "OP:\\sum", 160: "S", 161: "\\ne", 162: "\\varepsilon", 163: "\\partial", 164: "\\kappa", 165: "\\eta",
    166: "E", 167: "\\top", 168: "1", 169: "\\sim", 170: "ACC:\\hat", 171: "u", 172: "f", 173: "P", 174: "b",
    175: "v", 176: "\\oplus", 177: "SQRT", 178: "\\Phi", 179: "\\Gamma",
    180: "\\zeta", 181: "W", 182: "W", 183: "\\sim", 184: "y", 185: "SQRT", 187: "\\xi", 188: "D(", 189: "D)",
    190: "\\vartheta", 191: "L", 192: "D|", 193: "\\circ", 195: "B", 196: "OP:\\prod", 197: "\\varphi", 199: "w",
    200: "!", 201: "D{", 202: "S", 203: "D[", 204: "D]", 205: "R", 206: "S", 208: "\\Theta", 209: "\\nabla",
    210: "R", 211: "y", 212: "x", 213: "\\Leftrightarrow", 215: "y", 216: "C", 217: "D(", 218: "D(", 219: "D)",
    220: "D)", 222: "R", 224: "\\Rightarrow", 225: "w", 226: "\\langle", 227: "\\rangle", 229: "X", 230: "DDOT",
    232: "\\Rightarrow", 233: "u", 234: "V", 235: "SQRT", 236: "G", 237: "%", 238: "\\propto", 239: "D(",
    240: "D(", 241: "D}", 242: "\\psi", 243: "I", 244: "O", 246: "D)", 247: "D(", 248: "T", 249: "K", 250: "\\ll",
    251: "K", 252: "u", 253: "\\star", 255: "\\to", 256: "h", 257: "\\Lambda", 258: "OP:\\prod", 259: "C",
    260: "N", 268: "Z", 270: "\\theta", 271: "?", 273: "T", 274: "M", 275: "I", 276: "M", 277: "\\cap",
    280: "\\mp", 281: "ACC:\\tilde", 282: "Z", 284: "Z", 286: "\\vdots", 287: "v", 288: "V", 289: "D(", 290: "D)",
    292: "E", 293: "D[", 294: "D]", 295: "q", 296: "D|", 297: "x", 298: "F", 299: "e",
}.items():
    L[i] = v

def main(allc_pkl, pdf_glob, out):
    allc = pickle.load(open(allc_pkl, "rb"))
    entries = []
    for i, lab in sorted(L.items()):
        c = allc[i]
        entries.append({"key": outline_key(c["items"])[0], "vec": [round(x, 4) for x in c["vec"]], "label": lab,
                        "hist": collections.Counter()})
    by_key = collections.defaultdict(list)
    for e in entries:
        by_key[e["key"]].append(e)
    for f in sorted(glob.glob(pdf_glob))[:30]:
        with pymupdf.open(f) as doc:
            for page in doc:
                for x in page.get_drawings():
                    if x["type"] not in ("f", "fs"):
                        continue
                    r = x["rect"]
                    s = max(r.width, r.height)
                    if s < 0.3 or s > 60:
                        continue
                    key, vec = outline_key(x["items"], r)
                    best = None
                    for e in by_key.get(key, ()):
                        d = max(abs(a - b) for a, b in zip(e["vec"], vec))
                        if d < 0.03 and (best is None or d < best[0]):
                            best = (d, e)
                    if best:
                        best[1]["hist"][round(r.height * 4) / 4] += 1
    for e in entries:
        # canonical size = the largest size that carries a real share of the glyph's uses (sub/superscript copies are
        # ~0.7x of it), so a script-heavy glyph such as "2" still gets its full-size height
        total = sum(e["hist"].values())
        big = [h for h, n in e["hist"].items() if n >= 0.08 * total]
        e["h"] = max(big) if big else 0
        del e["hist"]
    json.dump(entries, open(out, "w"), separators=(",", ":"))
    print(len(entries), "entries;", sum(1 for e in entries if not e["h"]), "unseen in sample")

if __name__ == "__main__":
    main(*sys.argv[1:4])
