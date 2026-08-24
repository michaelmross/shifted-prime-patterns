#!/usr/bin/env python3
"""
make_notes.py -- regenerate notes.html, the informal companion document.

Assumes it lives in the repository root, with the certificates in ./certificates
beside it. Paths resolve relative to this file, not the working directory, so
the script can be invoked from anywhere. An optional argument overrides the
certificate location: python make_notes.py path/to/certificates

Reads figure data from docdata.json (produced by the campaign scripts) and
emits a fully self-contained HTML page: no webfonts, no CDN, no scripts.
Every figure is inline SVG built from the certified data.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "certificates")

REQUIRED_CERTS = ["cgX10.json", "cgX30.json", "cgX100.json", "cgX300.json",
                  "cgX1000.json", "cgX10000.json", "record_X30_m69.json"]


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0].astype(np.int64)


def build_data():
    d = {}
    d["decay"] = {"X": [10, 30, 100, 300, 1000, 10000],
                  "records": [0.4, 16/69, 3/17, 2/21, 2/29, 1/74],
                  "labels": ["2/5", "16/69", "3/17", "2/21", "2/29", "1/74"],
                  "trivial": [1/25, 1/169, 1/289, 1/961, 1/7225, 1/130321]}
    P = primes_upto(10000); m, c, L = 3848, 55, 52
    S = np.unique(((P - 1) ** 2) % m); t = np.unique((c * S) % m)
    d["ring"] = {"m": m, "c": c, "L": L, "t": [int(x) for x in t],
                 "crit": [int(x) for x in t if min(x, m - x) == L]}
    P1 = primes_upto(1000); Sf = np.unique((P1 - 1) ** 2)
    pts = []
    for mm in range(2, 3001):
        r = np.unique(Sf % mm)
        if r[0] == 0:
            pts.append([mm, 0.0]); continue
        cs = np.arange(1, mm // 2 + 1, dtype=np.int64)
        q = np.minimum((cs[:, None] * r[None, :]) % mm,
                       mm - (cs[:, None] * r[None, :]) % mm).min(axis=1)
        pts.append([mm, round(float(q.max() / mm), 5)])
    d["land"] = {"pts": pts, "mult145": list(range(145, 3001, 145))}
    DIG = {0, 1, 3}
    def rep(s):
        st = {0}
        while True:
            sd = s % 8; nxt = set()
            for cc in st:
                for ai in DIG:
                    u = ai + sd + cc
                    if u % 8 in DIG:
                        nxt.add(u // 8)
            if not nxt:
                return False
            st = nxt; s //= 8
            if s == 0 and 0 in st:
                return True
    d["digits"] = {"p1": [[int(p), rep((int(p) - 1) ** 2)] for p in primes_upto(4000) if p % 4 == 1],
                   "p3": [[int(p), rep((int(p) - 1) ** 2)] for p in primes_upto(4000) if p % 4 == 3]}
    d["digests"] = {}
    for name in REQUIRED_CERTS:
        path = os.path.join(CERT_DIR, name)
        if not os.path.isfile(path):
            continue
        dig = json.load(open(path)).get("core_digest", "")
        if dig:
            d["digests"][name] = dig
    missing = [n for n in REQUIRED_CERTS if n not in d["digests"]]
    if missing:
        sys.exit(
            "make_notes.py: cannot build the page without its certificates.\n"
            f"  looked in : {os.path.abspath(CERT_DIR)}\n"
            f"  found     : {sorted(d['digests']) or '(none)'}\n"
            f"  missing   : {missing}\n"
            "The digest chips on the page are read from the certificate files so they\n"
            "cannot drift from the repository; a page generated without them would\n"
            "violate its own premise. Point the script at the right directory with\n"
            "  python make_notes.py path/to/certificates\n"
            "or restore the missing files (each is regenerable by the harness with the\n"
            "parameters recorded in the README record table).")
    return d


D = build_data()

INK = "#22252a"; MUT = "#6d6c66"; GRID = "#e5e4dd"
BLUE = "#2a78d6"; ORANGE = "#d95d2a"; GRAY = "#a3a29a"


def svg_decay():
    W, H = 680, 360
    x0, x1, y0, y1 = 64, 640, 24, 300
    def px(X): return x0 + (math.log10(X) - 0.8) / (4.2 - 0.8) * (x1 - x0)
    def py(d): return y1 - (math.log10(d) + 5.4) / (5.4 - 0.1) * (y1 - y0)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Certified density records versus the trivial family, log-log">']
    for e in range(1, 5):
        s.append(f'<line x1="{px(10**e):.0f}" y1="{y0}" x2="{px(10**e):.0f}" y2="{y1}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{px(10**e):.0f}" y="{y1+20}" text-anchor="middle" class="ax">10<tspan dy="-4" font-size="9">{e}</tspan></text>')
    for e in range(-5, 0):
        s.append(f'<line x1="{x0}" y1="{py(10**e):.0f}" x2="{x1}" y2="{py(10**e):.0f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x0-8}" y="{py(10**e)+4:.0f}" text-anchor="end" class="ax">10<tspan dy="-4" font-size="9">{e}</tspan></text>')
    g = [(X, X**-0.534) for X in (10, 10000)]
    s.append(f'<line x1="{px(g[0][0]):.1f}" y1="{py(g[0][1]):.1f}" x2="{px(g[1][0]):.1f}" y2="{py(g[1][1]):.1f}" stroke="{ORANGE}" stroke-width="1.5" stroke-dasharray="6 5" opacity="0.75"/>')
    s.append(f'<text x="{px(300):.0f}" y="{py(300**-0.534)-9:.0f}" class="ax" fill="{ORANGE}">Ruzsa shape X<tspan dy="-4" font-size="9">-0.534</tspan></text>')
    for key, col, shape in (("trivial", GRAY, "s"), ("records", BLUE, "c")):
        pts = list(zip(D["decay"]["X"], D["decay"][key]))
        path = " ".join(f'{"M" if i==0 else "L"}{px(X):.1f} {py(d):.1f}' for i, (X, d) in enumerate(pts))
        s.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2"/>')
        for X, d in pts:
            if shape == "c":
                s.append(f'<circle cx="{px(X):.1f}" cy="{py(d):.1f}" r="4.5" fill="{col}"/>')
            else:
                s.append(f'<rect x="{px(X)-4:.1f}" y="{py(d)-4:.1f}" width="8" height="8" fill="{col}"/>')
    for X, d, lab in zip(D["decay"]["X"], D["decay"]["records"], D["decay"]["labels"]):
        s.append(f'<text x="{px(X)+9:.1f}" y="{py(d)-8:.1f}" class="ax" fill="{BLUE}">{lab}</text>')
    s.append(f'<text x="{px(30):.0f}" y="{py(1/169)+22:.0f}" class="ax" fill="{MUT}">trivial 1/m&#178;</text>')
    s.append(f'<text x="{(x0+x1)//2}" y="{H-6}" text-anchor="middle" class="ax">prime threshold X</text>')
    s.append("</svg>")
    return "".join(s)


def svg_ring():
    W, H = 680, 430
    cx, cy, R = 340, 218, 158
    m, L = D["ring"]["m"], D["ring"]["L"]
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Residues on a circle mod 3848 avoiding an empty wedge at the top">']
    aw = 2 * math.pi * L / m
    Rw = R + 15
    a1, a2 = -math.pi/2 - aw, -math.pi/2 + aw
    s.append(f'<path d="M {cx} {cy} L {cx+Rw*math.cos(a1):.1f} {cy+Rw*math.sin(a1):.1f} '
             f'A {Rw} {Rw} 0 0 1 {cx+Rw*math.cos(a2):.1f} {cy+Rw*math.sin(a2):.1f} Z" '
             f'fill="{ORANGE}" opacity="0.12" stroke="{ORANGE}" stroke-width="0.5" stroke-opacity="0.5"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{GRID}" stroke-width="1"/>')
    crit = set(D["ring"]["crit"])
    for t in D["ring"]["t"]:
        a = 2 * math.pi * t / m - math.pi / 2
        x, y = cx + R * math.cos(a), cy + R * math.sin(a)
        if t in crit:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{ORANGE}"/>')
        else:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="{BLUE}" opacity="0.8"/>')
    s.append(f'<text x="{cx}" y="{cy-R-26}" text-anchor="middle" class="ax">t = 0</text>')
    s.append(f'<text x="{cx}" y="{cy-6}" text-anchor="middle" class="fig-c">&#8484;/3848</text>')
    s.append(f'<text x="{cx}" y="{cy+16}" text-anchor="middle" class="ax">density 1/74</text>')
    s.append("</svg>")
    return "".join(s)


def svg_landscape():
    W, H = 680, 330
    x0, x1, y0, y1 = 64, 648, 20, 280
    def px(mm): return x0 + mm / 3000 * (x1 - x0)
    def py(d): return y1 - d / 0.078 * (y1 - y0)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Best density per modulus for X=1000; multiples of 145 form the plateau">']
    for v, lab in ((0.0, "0"), (2/29, "2/29")):
        s.append(f'<line x1="{x0}" y1="{py(v):.0f}" x2="{x1}" y2="{py(v):.0f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x0-8}" y="{py(v)+4:.0f}" text-anchor="end" class="ax">{lab}</text>')
    for mm in (1000, 2000, 3000):
        s.append(f'<text x="{px(mm):.0f}" y="{y1+20}" text-anchor="middle" class="ax">{mm}</text>')
    mult = set(D["land"]["mult145"])
    for mm, d in D["land"]["pts"]:
        if d <= 0 or mm in mult:
            continue
        s.append(f'<circle cx="{px(mm):.1f}" cy="{py(d):.1f}" r="1.4" fill="{GRAY}" opacity="0.75"/>')
    for mm in sorted(mult):
        s.append(f'<circle cx="{px(mm):.1f}" cy="{py(2/29):.1f}" r="4" fill="{ORANGE}"/>')
    s.append(f'<text x="{px(1450):.0f}" y="{py(2/29)-12:.0f}" text-anchor="middle" class="ax" fill="{ORANGE}">multiples of the conductor 145, all exactly 2/29</text>')
    s.append(f'<text x="{(x0+x1)//2}" y="{H-6}" text-anchor="middle" class="ax">modulus m</text>')
    s.append("</svg>")
    return "".join(s)


def svg_digits():
    W = 680
    cols, r, gap = 20, 3.2, 9.4
    def block(x0, y0, items, col_on):
        out = []
        for i, (_, pat) in enumerate(items):
            x = x0 + (i % cols) * gap
            y = y0 + (i // cols) * gap
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col_on if pat else GRAY}" opacity="{1 if pat else 0.55}"/>')
        rows = (len(items) + cols - 1) // cols
        return out, y0 + rows * gap
    s = []
    left, right = 70, 400
    h1, e1 = block(left, 58, D["digits"]["p1"], BLUE)
    h2, e2 = block(right, 58, D["digits"]["p3"], BLUE)
    H = int(max(e1, e2)) + 34
    s.append(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Primes to 4000 split by residue mod 4; every pattern lies in the class 1 mod 4">')
    n1 = sum(1 for _, b in D["digits"]["p1"] if b)
    s.append(f'<text x="{left+cols*gap/2-5:.0f}" y="30" text-anchor="middle" class="fig-c">p &#8801; 1 (mod 4)</text>')
    s.append(f'<text x="{left+cols*gap/2-5:.0f}" y="46" text-anchor="middle" class="ax">{n1} of {len(D["digits"]["p1"])} admit a pattern</text>')
    s.append(f'<text x="{right+cols*gap/2-5:.0f}" y="30" text-anchor="middle" class="fig-c">p &#8801; 3 (mod 4)</text>')
    s.append(f'<text x="{right+cols*gap/2-5:.0f}" y="46" text-anchor="middle" class="ax">0 of {len(D["digits"]["p3"])} &#8212; blocked mod 8</text>')
    s += h1 + h2
    s.append("</svg>")
    return "".join(s)


def chip(name):
    return f'<span class="chip">{D["digests"][name][:16]}</span>'


CSS = """
:root{--ink:#22252a;--mut:#6d6c66;--paper:#fbfbf8;--rule:#e0dfd8;
--blue:#2a78d6;--orange:#d95d2a;--chipbg:#f0efe9}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font:17px/1.68 Charter,Georgia,'Times New Roman',serif}
main{max-width:46rem;margin:0 auto;padding:3.5rem 1.4rem 5rem}
header h1{font-size:31px;font-weight:400;line-height:1.25;margin:0 0 .4rem}
.pattern{font-size:21px;margin:1.1rem 0 1.3rem;color:var(--ink)}
.sub{color:var(--mut);margin:0 0 1rem}
.links{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;font-size:13px}
.links a{color:var(--blue);text-decoration:none;margin-right:1.2em}
.links a:hover{text-decoration:underline}
h2{font-size:22px;font-weight:400;margin:2.8rem 0 .7rem}
.eyebrow{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;font-size:12px;
letter-spacing:.14em;text-transform:uppercase;color:var(--mut);display:block;margin-bottom:.15rem}
figure{margin:1.6rem 0}
figure svg{width:100%;height:auto;display:block}
figcaption{font-size:14px;color:var(--mut);margin-top:.5rem;line-height:1.55}
.ax{font:12px ui-monospace,'SF Mono',Menlo,Consolas,monospace;fill:#6d6c66}
.fig-c{font:14px Charter,Georgia,serif;fill:#22252a}
code,pre{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;font-size:14px}
pre{background:var(--chipbg);border:1px solid var(--rule);border-radius:6px;
padding:.7rem .9rem;overflow-x:auto}
.chip{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;font-size:12px;
background:var(--chipbg);border:1px solid var(--rule);border-radius:4px;
padding:1px 6px;white-space:nowrap;color:var(--mut)}
table{border-collapse:collapse;width:100%;margin:1.2rem 0;font-size:15px}
th{font-weight:400;color:var(--mut);text-align:left;font-size:13px;
font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace}
th,td{padding:.42rem .6rem .42rem 0;border-bottom:1px solid var(--rule);vertical-align:top}
td.n,th.n{text-align:right;padding-right:1.1rem;font-variant-numeric:tabular-nums}
.rec{color:var(--blue)}
em{font-style:italic}
hr{border:none;border-top:1px solid var(--rule);margin:2.6rem 0}
footer{color:var(--mut);font-size:14px;margin-top:3rem}
a{color:var(--blue)}
@media print{body{background:#fff}}
"""

decay_fig, ring_fig, land_fig, digits_fig = svg_decay(), svg_ring(), svg_landscape(), svg_digits()

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shifted prime patterns &#8212; notes from a computational campaign</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header>
<h1>Shifted prime patterns</h1>
<p class="sub">Notes from a computational campaign &#8212; certified searches, certified absences, and the first lower-bound records.</p>
<p class="pattern"><i>a</i>,&ensp;<i>a</i> + <i>P</i><sub>1</sub>(<i>p</i>&#8722;1),&ensp;&#8230;,&ensp;<i>a</i> + <i>P</i><sub><i>k</i>&#8722;1</sub>(<i>p</i>&#8722;1),&emsp;<i>p</i> prime</p>
<p class="links">
<span style="color:var(--mut);margin-right:1.2em">Sources:</span><a href="https://github.com/michaelmross/shifted-prime-patterns">repository</a>
<a href="https://arxiv.org/abs/2608.19525">arXiv:2608.19525</a>
<a href="https://terrytao.wordpress.com/2026/08/20/quantitative-bounds-for-sets-lacking-polynomial-progressions-with-shifted-prime-difference/">blog post</a>
</p>
</header>

<h2><span class="eyebrow">&#167;1</span>The object</h2>
<p>A <em>shifted prime pattern</em> is a polynomial progression whose shift parameter is one
less than a prime, with the polynomials <i>P<sub>i</sub></i> &#8712; &#8484;[<i>y</i>] having zero
constant term. A theorem of Wooley and Ziegler says these patterns are unavoidable: every set
of positive density contains one for infinitely many primes <i>p</i>. The recent paper of
Krause, Mousavi, Tao, and Ter&#228;v&#228;inen makes that quantitative from above, with density
bounds that engage only at astronomical scales. These notes work the other side at scales a
computer can certify: <em>how far, and at what density, can avoidance actually be pushed?</em></p>
<p>The repository's harness holds <i>A</i> &#8838; [0,&#8202;<i>N</i>) as a bitmask, so one prime
costs one chain of shift-and-AND operations across the window. A pattern found is a witness; a
zero intersection is an exhaustive proof of absence, and that proof &#8212; parameters, mask
digest, per-prime table &#8212; is recorded as a certificate that <code>--verify</code>
reproduces byte for byte. Each certificate carries a <em>core digest</em> over the
mathematically meaningful content only, stable across schema changes; those are the
identifiers quoted throughout these notes.</p>

<h2><span class="eyebrow">&#167;2</span>A calibration that bit back</h2>
<p>The first structured set we tried looked like a discovery. Take base-8 integers whose
digits all lie in {{0,&#8202;1,&#8202;3}} and search for square differences
(<i>P</i>(<i>y</i>) = <i>y</i>&#178;): 435 of the 550 primes up to 4000 admit no pattern.
That is not resistance. The set reduced mod 8 is {{0,&#8202;1,&#8202;3}}, no residue there has
<i>r</i>&#8202;+&#8202;4 in the set, so no difference is &#8801; 4 (mod 8) &#8212; and
(<i>p</i>&#8722;1)&#178; &#8801; 4 (mod 8) exactly when <i>p</i> &#8801; 3 (mod 4). An entire
residue class of primes was blocked before any search ran.</p>
<figure>{digits_fig}
<figcaption>Primes to 4000 against the digit set, split by residue mod 4. Blue admits a
pattern, gray does not. The right column is empty by a single congruence; among the survivors
the hit rate (0.424) matches a length-matched random control (0.378) &#8212; no deficit at
all. A two-state carry automaton on octal digits confirms every verdict window-free.</figcaption>
</figure>
<p>The harness now runs this diagnosis automatically: a residue scan reduces <i>A</i> mod
every small modulus and reports which no-pattern verdicts are congruence artifacts. The
lesson generalizes &#8212; read the scan before calling a null result a finding.</p>

<h2><span class="eyebrow">&#167;3</span>The campaign</h2>
<p>For the two-point case <i>x</i>, <i>x</i>&#8202;+&#8202;(<i>p</i>&#8722;1)&#178;, blocking
every prime <i>p</i> &#8804; <i>X</i> means the difference set of <i>A</i> avoids
<i>S<sub>X</sub></i> = {{(<i>p</i>&#8722;1)&#178; : <i>p</i> &#8804; <i>X</i>}} &#8212; an
instance of Motzkin's difference-avoidance problem. Cantor&#8211;Gordon interval sets
{{<i>n</i> : <i>cn</i> mod <i>m</i> &lt; <i>L</i>}} give the classical periodic lower bound;
exact circulant independence numbers give more. Every winner below is certified end to end.</p>
<table>
<tr><th class="n">X</th><th class="n">&#960;(X)</th><th>trivial</th><th>certified record</th><th>core digest</th></tr>
<tr><td class="n">10</td><td class="n">4</td><td>1/25</td><td class="rec">2/5 &#8212; optimal, periods &#8804; 80</td><td>{chip('cgX10.json')}</td></tr>
<tr><td class="n">30</td><td class="n">10</td><td>1/169</td><td class="rec">16/69 &#8212; exact, beats intervals (3/13)</td><td>{chip('record_X30_m69.json')}</td></tr>
<tr><td class="n">100</td><td class="n">25</td><td>1/289</td><td class="rec">3/17</td><td>{chip('cgX100.json')}</td></tr>
<tr><td class="n">300</td><td class="n">62</td><td>1/961</td><td class="rec">2/21</td><td>{chip('cgX300.json')}</td></tr>
<tr><td class="n">1000</td><td class="n">168</td><td>1/7225</td><td class="rec">2/29</td><td>{chip('cgX1000.json')}</td></tr>
<tr><td class="n">10000</td><td class="n">1229</td><td>1/130321</td><td class="rec">1/74 &#8212; within m &#8804; 8000</td><td>{chip('cgX10000.json')}</td></tr>
</table>
<figure>{decay_fig}
<figcaption>Certified density records against the trivial congruence family, log&#8211;log.
The dashed guide is the Ruzsa square-difference shape mapped through <i>N</i> = <i>X</i>&#178;.
The quadratic structure of the shifts, not their count, controls the decay; whether the prime
restriction buys anything asymptotically appears to be an open question.</figcaption>
</figure>
<p>The X = 10&#8308; record means concretely: for every window <i>N</i> up to about 10&#8312;,
there are explicit sets of density 1/74 &#8776; 0.0135 containing no pattern
<i>x</i>, <i>x</i>&#8202;+&#8202;(<i>p</i>&#8722;1)&#178; at all &#8212; a regime where the
known upper bounds say nothing yet.</p>

<h2><span class="eyebrow">&#167;4</span>Anatomy of a record</h2>
<p>The X = 10&#8308; winner is the set {{<i>n</i> : 55<i>n</i> mod 3848 &lt; 52}}. Its whole
proof is one picture: multiply every shift (<i>p</i>&#8722;1)&#178; by 55 and reduce mod 3848,
and none of the 263 resulting residue classes comes within 52 of zero. The wedge is empty, so
no difference of two set elements ever equals a shifted-prime square.</p>
<figure>{ring_fig}
<figcaption>Each dot is one residue class 55&#8202;(<i>p</i>&#8722;1)&#178; mod 3848 over the
1,229 primes to 10&#8308;. The wedge |<i>t</i>| &lt; 52 defines the set and is empty. The
orange dots pinned to its edges are the seven <em>critical primes</em>
{{547, 5227, 6163, 7151, 7307, 8087, 8243}} that attain the margin exactly &#8212; any denser
construction must re-route precisely these.</figcaption>
</figure>

<h2><span class="eyebrow">&#167;5</span>Conductors</h2>
<p>Sweeping all moduli for the best construction reveals that the landscape is not a scatter
of independently good moduli. If the record at <i>m</i>&#8320; is
{{<i>n</i> : <i>cn</i> mod <i>m</i>&#8320; &lt; <i>L</i>}}, then at any multiple
<i>km</i>&#8320; the pair (<i>kc</i>, <i>kL</i>) defines <em>literally the same set</em>, so
every multiple inherits the record and the plateau is exact. Borrowing the language of
Dirichlet characters: plateau members are <em>induced</em>, and the minimal modulus carrying
the structure is the <em>conductor</em>. The sweep only ever needs primitive moduli.</p>
<figure>{land_fig}
<figcaption>Best density per modulus at X = 1000. The ceiling 2/29 is attained only on
multiples of the conductor 145 = 5&#183;29. The X = 10&#8308; record modulus 3848 =
2&#179;&#183;13&#183;37 is primitive in the strongest sense: every proper divisor is outright
dead. It survives by exactly one power of two &#8212; requiring 4 | <i>p</i>&#8722;1 raises
its kill-condition to <i>p</i> &#8801; 1 (mod 1924), whose least prime is 11545 &gt; X.</figcaption>
</figure>

<h2><span class="eyebrow">&#167;6</span>Open</h2>
<p>Four directions, in rough order of expected yield. A conductor-only sweep should reach
X = 10&#8309; and beyond, pinning the decay exponent that the current cap-truncated data
cannot. The seven critical primes make improving the X = 10&#8308; record a concrete
re-routing problem. Exact independence numbers past period &#8776; 80 (a proper clique
solver) would settle whether the interval family's deficit at X = 30 grows. And the genuine
polynomial progressions <i>x</i>, <i>x</i>&#8202;+&#8202;(<i>p</i>&#8722;1),
<i>x</i>&#8202;+&#8202;(<i>p</i>&#8722;1)&#178; remain untouched: there the Cayley graph
becomes a 3-uniform hypergraph, none of these pictures has an analog yet, and the harness is
the only certifier available.</p>

<hr>
<footer>
<p>Every number on this page traces to a certificate in the repository. To reproduce one:</p>
<pre>python shifted_prime_patterns.py --verify certificates/cgX10000.json</pre>
<p>A passing run means the archived result was rebuilt from its recorded parameters and
matched byte for byte, mask digest and core digest included. This page is generated by
<code>make_notes.py</code> from the same data.</p>
</footer>
</main>
</body>
</html>
"""

open(os.path.join(HERE, "notes.html"), "w").write(HTML)
print(f"notes.html written: {len(HTML):,} bytes")
