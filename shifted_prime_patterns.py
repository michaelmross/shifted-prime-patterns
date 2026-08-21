#!/usr/bin/env python3
"""
shifted_prime_patterns.py — exhaustive search for Wooley-Ziegler configurations.

For polynomials P_1, ..., P_{k-1} in Z[y] with zero constant term, and a set
A contained in [0, N), this searches for integers a and primes p such that

    a, a + P_1(p-1), ..., a + P_{k-1}(p-1)

all lie in A.  A is held as a bitmask, so the test at a single prime is one
chain of shift-and-AND operations over the whole window:

    M = AND_{o in offsets} (A >> (o - min offsets)),

and M != 0 exactly when a witness exists.  The lowest set bit of M gives the
least witness.  When M == 0 the prime is certified pattern-free over the
whole window, and that certificate is what this script records.

Reproducibility: every run emits a JSON certificate containing the SHA-256 of
the packed bitmask of A, the SHA-256 of this script, and the full per-prime
result table.  `--verify CERT.json` rebuilds everything from the recorded
parameters and compares byte-for-byte.

Usage examples
--------------
  # arithmetic progressions of length 3, spacing p-1, inside 6Z
  python3 shifted_prime_patterns.py --N 1000000 --polys "y,2y" \
      --set congruence:6 --pmax 200 --out runs/cong6

  # square differences (Sarkozy along shifted primes) in a quadratic-phase set
  python3 shifted_prime_patterns.py --N 100000000 --polys "y^2" \
      --set quad:sqrt2,0.05 --pmax 10000 --out runs/quad

  # re-verify a stored certificate
  python3 shifted_prime_patterns.py --verify runs/cong6.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np

SCHEMA = "shifted-prime-patterns/3"
CORE = "core/1"     # digest scope; frozen, so additive schema changes never move it
DEFAULT_CHUNK_BITS = 1 << 23  # 8388608 bits per construction chunk
MOD_BITS = 48                 # dyadic precision for Bohr / quadratic-phase sets
MOD = 1 << MOD_BITS


# --------------------------------------------------------------------------
# polynomials
# --------------------------------------------------------------------------

_ALLOWED_CHARS = set("0123456789y+-*^() ")


def parse_poly(src: str):
    """Parse a polynomial in y with integer coefficients and zero constant term."""
    src = src.strip()
    if not src:
        raise ValueError("empty polynomial")
    if not set(src) <= _ALLOWED_CHARS:
        bad = sorted(set(src) - _ALLOWED_CHARS)
        raise ValueError(f"illegal character(s) {bad} in polynomial {src!r}")
    expr = src.replace("^", "**")
    expr = re.sub(r"(\d)\s*(?=y)", r"\1*", expr)      # 3y   -> 3*y
    expr = re.sub(r"\)\s*(?=y)", r")*", expr)         # (2)y -> (2)*y
    expr = re.sub(r"(?<=y)\s*(?=\()", r"*", expr)     # y(   -> y*(
    code = compile(expr, "<poly>", "eval")

    def f(y):
        return eval(code, {"__builtins__": {}}, {"y": int(y)})

    if f(0) != 0:
        raise ValueError(f"polynomial {src!r} has nonzero constant term")
    for probe in (1, 2, 3, 7, 11):
        if not isinstance(f(probe), int):
            raise ValueError(f"polynomial {src!r} is not integer valued")
    return f


def parse_polys(spec: str):
    srcs = [s.strip() for s in spec.split(",") if s.strip()]
    if not srcs:
        raise ValueError("no polynomials given")
    return srcs, [parse_poly(s) for s in srcs]


# --------------------------------------------------------------------------
# set generators
# --------------------------------------------------------------------------

def _parse_alpha(tok: str) -> float:
    tok = tok.strip().lower()
    named = {
        "sqrt2": math.sqrt(2.0),
        "sqrt3": math.sqrt(3.0),
        "sqrt5": math.sqrt(5.0),
        "phi": (1.0 + math.sqrt(5.0)) / 2.0,
        "pi": math.pi,
        "e": math.e,
    }
    if tok in named:
        return named[tok]
    if "/" in tok:
        num, den = tok.split("/", 1)
        return float(num) / float(den)
    return float(tok)


def residue_images(N, chunk_fn, chunk_bits, moduli):
    """Exact image of A modulo each q.  A modulus is dropped once its image is
    all of Z/q, since it can then never yield an obstruction."""
    images = {q: np.zeros(q, dtype=bool) for q in moduli}
    live = set(moduli)
    for s in range(0, N, chunk_bits):
        if not live:
            break
        e = min(s + chunk_bits, N)
        idx = np.nonzero(np.ascontiguousarray(chunk_fn(s, e)))[0].astype(np.int64) + s
        if idx.size == 0:
            continue
        for q in list(live):
            seen = images[q]
            seen[np.unique(idx % q)] = True
            if seen.all():
                live.discard(q)
    return images


def local_scan(N, chunk_fn, chunk_bits, polys, primes, qmax,
               allow_degenerate, rows=None):
    """For each modulus q, reduce A mod q and ask whether the configuration is
    locally solvable.  If no residue r in the image admits r + P_i(d) for every
    i, then no prime with p - 1 = d can carry a pattern.  This is exact and
    holds for the whole window, independently of the search."""
    if qmax < 2:
        return None
    moduli = list(range(2, qmax + 1))
    images = residue_images(N, chunk_fn, chunk_bits, moduli)

    blocked = {}
    for q in moduli:
        seen = images[q]
        if seen.all():
            continue
        R = set(int(x) for x in np.nonzero(seen)[0])
        bad = [d for d in range(q)
               if not any(all(((r + f(d)) % q) in R for f in polys) for r in R)]
        if bad:
            blocked[q] = bad

    explained, by_modulus = {}, {}
    for p in primes:
        p, d = int(p), int(p) - 1
        for q in moduli:
            if q in blocked and d % q in blocked[q]:
                explained[p] = q
                by_modulus[q] = by_modulus.get(q, 0) + 1
                break

    contradictions = []
    if rows is not None:
        status = {r["p"]: r["status"] for r in rows}
        contradictions = sorted(p for p in explained
                                if status.get(p) == "pattern")

    return {
        "qmax": qmax,
        "blocked_shift_classes": {str(q): v for q, v in sorted(blocked.items())},
        "primes_blocked": len(explained),
        "primes_blocked_by_modulus": {str(q): n for q, n in sorted(by_modulus.items())},
        "least_blocking_modulus": {str(p): q for p, q in sorted(explained.items())},
        "contradictions": contradictions,
    }


def _ap_free(D) -> bool:
    """True when D holds no three-term arithmetic progression."""
    S = set(D)
    return not any(x != y and (x + y) % 2 == 0 and (x + y) // 2 in S
                   for x in S for y in S)


def make_generator(spec: str, N: int):
    """Return (chunk_fn, canonical_spec_dict).

    chunk_fn(start, stop) -> boolean numpy array of length stop-start,
    True exactly on the elements of A in that range.
    """
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    args = [t for t in rest.split(",") if t.strip() != ""] if rest else []

    if kind == "all":
        def fn(s, e):
            return np.ones(e - s, dtype=bool)
        return fn, {"kind": "all"}

    if kind == "congruence":
        if not args:
            raise ValueError("congruence set needs a modulus, e.g. congruence:6")
        q = int(args[0])
        if q < 1:
            raise ValueError("modulus must be positive")
        residues = sorted({int(t) % q for t in args[1:]}) or [0]
        R = np.array(residues, dtype=np.int64)

        def fn(s, e):
            return np.isin(np.arange(s, e, dtype=np.int64) % q, R)

        return fn, {"kind": "congruence", "q": q, "residues": residues}

    if kind in ("bohr", "quad"):
        if len(args) < 2:
            raise ValueError(f"{kind} set needs alpha and a threshold, "
                             f"e.g. {kind}:sqrt2,0.05")
        alpha_src = args[0].strip()
        alpha = _parse_alpha(alpha_src)
        c = float(args[1])
        if not (0.0 < c <= 0.5):
            raise ValueError("threshold must lie in (0, 0.5]")
        if N > (1 << 32):
            raise ValueError("bohr/quad generators require N <= 2^32")
        Aint = int(round(alpha * MOD)) % MOD
        thr = int(c * MOD)
        Au = np.uint64(Aint)
        Mu = np.uint64(MOD)
        MASKu = np.uint64(MOD - 1)
        thru = np.uint64(thr)
        square = (kind == "quad")

        def fn(s, e):
            n = np.arange(s, e, dtype=np.uint64)
            t = (Au * n * n) & MASKu if square else (Au * n) & MASKu
            dist = np.minimum(t, Mu - t)
            return dist < thru

        return fn, {"kind": kind, "alpha": alpha_src,
                    "alpha_numerator": Aint, "alpha_denominator": MOD,
                    "threshold": c, "threshold_int": thr}

    if kind == "cg":
        # Cantor-Gordon interval set {n : (c*n) mod m < L}, density L/m.
        # Blocks shift s exactly when |c*s|_m >= L, |x|_m = min(x mod m, m - x mod m).
        if len(args) < 3:
            raise ValueError("cg set needs m,c,L, e.g. cg:330,47,29")
        m, c, L = int(args[0]), int(args[1]), int(args[2])
        if not (2 <= m and 1 <= c < m and 1 <= L <= m):
            raise ValueError("need 2<=m, 1<=c<m, 1<=L<=m")

        def fn(s_, e_):
            n = np.arange(s_, e_, dtype=np.int64)
            return (c * n) % m < L

        return fn, {"kind": "cg", "m": m, "c": c, "L": L, "density": L / m}

    if kind == "digits":
        if len(args) < 2:
            raise ValueError("digits set needs a base and a digit set, "
                             "e.g. digits:8,0,1,3")
        b = int(args[0])
        if b < 2:
            raise ValueError("base must be at least 2")
        D = sorted({int(t) for t in args[1:]})
        if not D or min(D) < 0 or max(D) >= b:
            raise ValueError(f"digits must lie in [0,{b})")
        Darr = np.array(D, dtype=np.int64)

        def fn(s, e):
            n = np.arange(s, e, dtype=np.int64)
            ok = np.ones(n.size, dtype=bool)
            m = n.copy()
            live = (m > 0) | (n == 0)
            while live.any():
                ok &= ~live | np.isin(m % b, Darr)
                m = np.where(live, m // b, m)
                live = m > 0
            return ok

        return fn, {"kind": "digits", "base": b, "digits": D,
                    "carry_free": max(D) * 2 < b,
                    "digit_set_is_ap_free": _ap_free(D)}

    if kind == "random":
        if not args:
            raise ValueError("random set needs a density, e.g. random:0.1,42")
        delta = float(args[0])
        if not (0.0 < delta <= 1.0):
            raise ValueError("density must lie in (0, 1]")
        seed = int(args[1]) if len(args) > 1 else 0

        def fn(s, e):
            rng = np.random.default_rng([seed, s])
            return rng.random(e - s) < delta

        return fn, {"kind": "random", "density": delta, "seed": seed}

    if kind == "file":
        if not args:
            raise ValueError("file set needs a path, e.g. file:A.txt")
        path = args[0].strip()
        with open(path) as fh:
            idx = np.array([int(t) for t in fh.read().split()], dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < N)]
        flags = np.zeros(N, dtype=bool)
        flags[idx] = True
        digest = hashlib.sha256(idx.tobytes()).hexdigest()

        def fn(s, e):
            return flags[s:e]

        return fn, {"kind": "file", "path": os.path.abspath(path),
                    "elements": int(idx.size), "elements_sha256": digest}

    raise ValueError(f"unknown set kind {kind!r}")


def build_mask(N: int, chunk_fn, chunk_bits: int):
    """Pack A into a Python int bitmask.  Returns (mask, sha256, popcount)."""
    parts, popcount = [], 0
    for s in range(0, N, chunk_bits):
        e = min(s + chunk_bits, N)
        block = np.ascontiguousarray(chunk_fn(s, e)).astype(np.uint8)
        popcount += int(block.sum())
        parts.append(np.packbits(block, bitorder="little"))
    raw = np.concatenate(parts).tobytes() if parts else b""
    return int.from_bytes(raw, "little"), hashlib.sha256(raw).hexdigest(), popcount


# --------------------------------------------------------------------------
# primes and local analysis
# --------------------------------------------------------------------------

def primes_upto(n: int) -> np.ndarray:
    if n < 2:
        return np.empty(0, dtype=np.int64)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:: i] = False
    return np.nonzero(sieve)[0].astype(np.int64)


def local_analysis(q: int, residues, polys, primes, N: int,
                   allow_degenerate: bool, max_q: int = 200000):
    """Exact congruence obstruction for a periodic set A = union of residues mod q.

    A shift d is locally admissible when some r in R has r + P_i(d) in R for
    every i.  This is necessary for a pattern with p - 1 = d.  For a purely
    periodic A it is also sufficient once the window holds a full period beyond
    the span, so the least admissible prime then predicts p_min exactly.  The
    prediction applies the same degeneracy and window filters as the search so
    that the two numbers are directly comparable.
    """
    if q > max_q:
        return None
    Rset = set(int(r) % q for r in residues)
    admissible = []
    for d in range(q):
        shifts = [f(d) % q for f in polys]
        for r in Rset:
            if all(((r + s) % q) in Rset for s in shifts):
                admissible.append(d)
                break
    adm = set(admissible)

    first_admissible = None      # window-free: the periodic certificate
    for p in primes:
        p, d = int(p), int(p) - 1
        offsets = [0] + [f(d) for f in polys]
        if len(set(offsets)) < len(offsets) and not allow_degenerate:
            continue
        if d % q in adm:
            first_admissible = p
            break

    predicted, tight = None, True
    for p in primes:
        p, d = int(p), int(p) - 1
        offsets = [0] + [f(d) for f in polys]
        if len(set(offsets)) < len(offsets) and not allow_degenerate:
            continue
        span = max(offsets) - min(offsets)
        if span >= N:
            continue
        if d % q in adm:
            predicted = p
            tight = (span + q <= N)
            break
    return {
        "modulus": q,
        "admissible_shifts_mod_q": admissible,
        "admissible_count": len(admissible),
        "predicted_p_min": predicted,
        "first_admissible_p_any_window": first_admissible,
        "periodic_no_pattern_for_all_p_upto":
            (int(primes[-1]) if len(primes) and first_admissible is None else None),
        "prediction_is_tight": tight,
        "searched_to": int(primes[-1]) if len(primes) else None,
    }


# --------------------------------------------------------------------------
# the search itself
# --------------------------------------------------------------------------

def count_bits(x: int) -> int:
    if x == 0:
        return 0
    raw = x.to_bytes((x.bit_length() + 7) // 8, "little")
    return int(np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).sum())


def search(mask: int, N: int, polys, poly_srcs, primes,
           allow_degenerate: bool, count_witnesses: bool, stop_at_first: bool,
           progress: bool = True):
    rows = []
    p_min = None
    t0 = time.time()
    for j, p in enumerate(primes):
        p = int(p)
        d = p - 1
        offsets = [0] + [f(d) for f in polys]
        lo, hi = min(offsets), max(offsets)
        span = hi - lo
        degenerate = len(set(offsets)) < len(offsets)

        row = {
            "p": p, "d": d,
            "offsets": ";".join(str(o) for o in offsets),
            "span": span,
            "status": "", "witness_a": "", "witness_pattern": "",
            "witness_count": "",
        }

        if degenerate and not allow_degenerate:
            row["status"] = "skipped_degenerate"
            rows.append(row)
            continue
        if span >= N:
            row["status"] = "skipped_window"
            rows.append(row)
            continue

        M = None
        for o in offsets:
            shift = o - lo
            part = mask >> shift if shift else mask
            M = part if M is None else (M & part)
            if M == 0:
                break

        if M:
            b = (M & -M).bit_length() - 1
            a = b + lo
            row["status"] = "pattern"
            row["witness_a"] = a
            row["witness_pattern"] = ";".join(str(a + o) for o in offsets)
            if count_witnesses:
                row["witness_count"] = count_bits(M)
            if p_min is None:
                p_min = p
        else:
            row["status"] = "no_pattern"

        rows.append(row)
        if progress and (j % 200 == 199):
            print(f"  ... {j+1}/{len(primes)} primes, "
                  f"{time.time()-t0:.1f}s", file=sys.stderr)
        if stop_at_first and p_min is not None:
            break

    return rows, p_min, time.time() - t0


# --------------------------------------------------------------------------
# run / certificate
# --------------------------------------------------------------------------

def self_sha256() -> str:
    try:
        with open(os.path.abspath(__file__), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def run(params: dict, progress: bool = True):
    N = params["N"]
    chunk_bits = params.get("chunk_bits", DEFAULT_CHUNK_BITS)
    poly_srcs, polys = parse_polys(params["polys"])
    chunk_fn, set_spec = make_generator(params["set"], N)

    mask, mask_sha, popcount = build_mask(N, chunk_fn, chunk_bits)
    primes = primes_upto(params["pmax"])

    rows, p_min, elapsed = search(
        mask, N, polys, poly_srcs, primes,
        allow_degenerate=params.get("allow_degenerate", False),
        count_witnesses=params.get("count_witnesses", False),
        stop_at_first=params.get("stop_at_first", False),
        progress=progress,
    )

    local = None
    if set_spec["kind"] == "cg":
        mq, cc, LL = set_spec["m"], set_spec["c"], set_spec["L"]
        residues = [r for r in range(mq) if (cc * r) % mq < LL]
        local = local_analysis(mq, residues, polys, primes,
                               N, params.get("allow_degenerate", False))
    elif set_spec["kind"] == "congruence":
        local = local_analysis(set_spec["q"], set_spec["residues"], polys, primes,
                               N, params.get("allow_degenerate", False))

    scan = local_scan(N, chunk_fn, chunk_bits, polys, primes,
                      params.get("local_scan", 64),
                      params.get("allow_degenerate", False), rows)

    witness_elts = [max(int(x) for x in r["witness_pattern"].split(";"))
                    for r in rows if r["status"] == "pattern"]
    max_witness = max(witness_elts) if witness_elts else None

    no_pattern = [r["p"] for r in rows if r["status"] == "no_pattern"]
    tested = [r for r in rows if r["status"] in ("pattern", "no_pattern")]

    result = {
        "schema": SCHEMA,
        "params": params,
        "set": set_spec,
        "polynomials": poly_srcs,
        "mask": {
            "N": N,
            "sha256": mask_sha,
            "cardinality": popcount,
            "density": popcount / N if N else 0.0,
            "chunk_bits": chunk_bits,
        },
        "summary": {
            "primes_considered": int(len(primes)),
            "primes_tested": len(tested),
            "primes_skipped_window": sum(1 for r in rows
                                         if r["status"] == "skipped_window"),
            "primes_skipped_degenerate": sum(1 for r in rows
                                             if r["status"] == "skipped_degenerate"),
            "p_min": p_min,
            "max_witness_element": max_witness,
            "window_use": round(max_witness / N, 6) if max_witness else None,
            "primes_without_pattern": no_pattern,
            "elapsed_seconds": round(elapsed, 3),
        },
        "local_analysis": local,
        "residue_scan": scan,
        "rows": rows,
        "core_schema": CORE,
        "script_sha256": self_sha256(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    result["core_digest"] = core_digest(result)
    return result


def core_digest(result: dict) -> str:
    """SHA-256 over the mathematically meaningful content only: what was
    computed, and what came out.  Deliberately excludes timestamps, the script
    hash, timings, and every diagnostic field, so that adding output to a later
    version of this script leaves it untouched.  This, not the digest of the
    JSON file, is the stable identifier for a run.
    """
    p = result["params"]
    body = {
        "core": CORE,
        "N": p["N"],
        "polys": p["polys"],
        "set": p["set"],
        "pmax": p["pmax"],
        "allow_degenerate": p.get("allow_degenerate", False),
        "stop_at_first": p.get("stop_at_first", False),
        "chunk_bits": p.get("chunk_bits"),
        "mask_sha256": result["mask"]["sha256"],
        "rows": [[r["p"], r["status"], r["witness_a"]] for r in result["rows"]],
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def canonical(result: dict, template: dict | None = None) -> str:
    """Certificate content with volatile fields removed, for drift detection.

    `template` restricts the comparison, at both the top level and inside the
    summary, to the fields the template actually records.  A later script that
    adds new fields therefore still verifies certificates written by an earlier
    one, on their own terms.
    """
    drop = ("generated_utc", "schema", "script_sha256", "core_digest",
            "core_schema")

    def restrict(v, t):
        if isinstance(v, dict) and isinstance(t, dict):
            # a key the template recorded as null was not computed then;
            # drop it from the comparison rather than fail on new data
            return {k: restrict(v[k], t[k]) for k in v
                    if k in t and t[k] is not None}
        if isinstance(v, list) and isinstance(t, list) and t and isinstance(t[0], dict):
            return [restrict(x, t[0]) if isinstance(x, dict) else x for x in v]
        return v

    stable = {k: v for k, v in result.items() if k not in drop}
    if template is not None:
        stable = restrict(stable, {k: v for k, v in template.items()
                                   if k not in drop})
    summary = {k: v for k, v in result["summary"].items() if k != "elapsed_seconds"}
    if template is not None and "summary" in template:
        summary = {k: v for k, v in summary.items() if k in template["summary"]}
    stable["summary"] = summary
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def write_outputs(result: dict, prefix: str):
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
    with open(prefix + ".json", "w") as fh:
        json.dump(result, fh, indent=2)
    fields = ["p", "d", "offsets", "span", "status",
              "witness_a", "witness_pattern", "witness_count"]
    with open(prefix + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in result["rows"]:
            w.writerow(r)
    return prefix + ".json", prefix + ".csv"


def report(result: dict):
    s, m = result["summary"], result["mask"]
    print()
    print(f"  set          {result['set']}")
    print(f"  polynomials  {', '.join(result['polynomials'])}")
    print(f"  window       N = {m['N']:,}   |A| = {m['cardinality']:,}   "
          f"density = {m['density']:.6g}")
    print(f"  mask sha256  {m['sha256']}")
    print(f"  primes       {s['primes_tested']} tested, "
          f"{s['primes_skipped_window']} outside window, "
          f"{s['primes_skipped_degenerate']} degenerate")
    print(f"  p_min        {s['p_min']}")
    npat = s["primes_without_pattern"]
    shown = ", ".join(str(p) for p in npat[:20])
    more = f" ... (+{len(npat)-20} more)" if len(npat) > 20 else ""
    print(f"  no pattern   {len(npat)} prime(s)" + (f": {shown}{more}" if npat else ""))
    if result.get("local_analysis"):
        la = result["local_analysis"]
        print(f"  local        {la['admissible_count']}/{la['modulus']} "
              f"admissible shifts mod q, predicted p_min = {la['predicted_p_min']}")
        if la.get("periodic_no_pattern_for_all_p_upto"):
            print(f"  periodic     no admissible shift for any p <= "
                  f"{la['periodic_no_pattern_for_all_p_upto']}: holds at EVERY N")
        if la["predicted_p_min"] != s["p_min"]:
            note = ("" if la.get("prediction_is_tight", True)
                    else " (window too narrow for the prediction to be sharp)")
            print("  WARNING      local prediction and computed p_min disagree" + note)
    mw, wu = s.get("max_witness_element"), s.get("window_use")
    if mw:
        flag = "  <-- witnesses reach far into the window" if wu and wu > 0.5 else ""
        print(f"  window use   largest witness element {mw:,} = "
              f"{100*wu:.1f}% of N{flag}")
        if wu and wu > 0.5 and s["primes_without_pattern"]:
            print("  CAUTION      no-pattern verdicts may be window-limited; "
                  "re-run with a larger N")
    sc = result.get("residue_scan")
    if sc and sc["blocked_shift_classes"]:
        qs = ", ".join(f"q={q}:{len(v)}/{q}"
                       for q, v in list(sc["blocked_shift_classes"].items())[:6])
        print(f"  residue scan blocked shift classes at {qs}")
        npn = len(s["primes_without_pattern"])
        share = f" ({sc['primes_blocked']}/{npn})" if npn else ""
        print(f"  explained    {sc['primes_blocked']} prime(s) blocked by "
              f"congruence alone{share}")
        if sc["contradictions"]:
            print(f"  ERROR        blocked primes that reported a pattern: "
                  f"{sc['contradictions'][:10]}")
    print(f"  core digest  {result.get('core_digest','')[:32]}")
    print(f"  elapsed      {s['elapsed_seconds']}s")
    print()


def verify(path: str):
    with open(path) as fh:
        stored = json.load(fh)
    stored_schema = stored.get("schema", "")
    if not stored_schema.startswith("shifted-prime-patterns/"):
        print(f"FAIL  unknown schema {stored_schema!r}")
        return 1
    if stored_schema != SCHEMA:
        print(f"NOTE  certificate schema {stored_schema} verified under {SCHEMA};"
              " comparison is limited to the fields it records")
    print(f"re-running {path} ...")
    fresh = run(stored["params"], progress=False)
    ok = True
    if fresh["mask"]["sha256"] != stored["mask"]["sha256"]:
        print("FAIL  mask digest differs")
        print(f"      stored {stored['mask']['sha256']}")
        print(f"      fresh  {fresh['mask']['sha256']}")
        ok = False
    keys = set(stored.keys())
    if canonical(fresh, stored) != canonical(stored, stored):
        print("FAIL  certificate content differs")
        for section in ("set", "polynomials", "mask", "local_analysis",
                        "residue_scan", "rows"):
            if section not in keys:
                continue
            if json.dumps(stored.get(section), sort_keys=True) != \
               json.dumps(fresh.get(section), sort_keys=True):
                print(f"      section {section!r} differs")
        for key in ("p_min", "primes_without_pattern", "primes_tested",
                    "primes_skipped_window", "primes_skipped_degenerate",
                    "max_witness_element"):
            if key not in stored["summary"]:
                continue
            a, b = stored["summary"].get(key), fresh["summary"].get(key)
            if a != b:
                sa = a if not isinstance(a, list) else f"{len(a)} primes"
                sb = b if not isinstance(b, list) else f"{len(b)} primes"
                print(f"      summary.{key}: stored {sa!r} -> fresh {sb!r}")
        ok = False
    if "core_digest" in stored:
        restated = core_digest(stored)
        if restated != stored["core_digest"]:
            print("FAIL  stored core digest does not match the stored content")
            print(f"      recorded   {stored['core_digest']}")
            print(f"      recomputed {restated}")
            ok = False
        elif stored["core_digest"] == fresh["core_digest"]:
            print(f"      core digest matches: {fresh['core_digest'][:32]}")
        else:
            print("FAIL  core digest differs")
            print(f"      stored {stored['core_digest']}")
            print(f"      fresh  {fresh['core_digest']}")
            ok = False
    else:
        print(f"NOTE  certificate predates the core digest; recomputed as "
              f"{fresh['core_digest'][:32]}")
    if stored.get("script_sha256") and stored["script_sha256"] != fresh["script_sha256"]:
        print("NOTE  script has changed since the certificate was written")
    if ok:
        print("PASS  certificate reproduced byte-for-byte")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Search for a, a+P_1(p-1), ..., a+P_{k-1}(p-1) inside a set A.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""set specifications:
  all                        A = [0, N)
  congruence:q[,r1,r2,...]   A = union of residue classes mod q (default r=0)
  bohr:alpha,c               A = { n : ||alpha*n|| < c }
  quad:alpha,c               A = { n : ||alpha*n^2|| < c }
  cg:m,c,L                   A = { n : (c*n) mod m < L }  (Cantor-Gordon interval)
  digits:b,d1,d2,...         A = { n : every base-b digit of n lies in D }
  random:delta[,seed]        A = independent bits of density delta
  file:path                  A = whitespace-separated integers from a file

alpha accepts a decimal, a ratio a/b, or one of sqrt2 sqrt3 sqrt5 phi pi e.
bohr and quad use the dyadic rational round(alpha*2^48)/2^48, computed exactly
in uint64 arithmetic, so the set is the one named in the certificate.""")
    ap.add_argument("--verify", metavar="CERT.json",
                    help="re-run a stored certificate and compare")
    ap.add_argument("--N", type=int, help="window size; A is a subset of [0,N)")
    ap.add_argument("--polys", help='comma-separated, e.g. "y,2y" or "y^2,y^3"')
    ap.add_argument("--set", dest="setspec", help="set specification (see below)")
    ap.add_argument("--pmax", type=int, default=1000, help="largest prime to test")
    ap.add_argument("--out", help="output prefix; writes <prefix>.json and .csv")
    ap.add_argument("--allow-degenerate", action="store_true",
                    help="keep primes where the offsets collide or vanish")
    ap.add_argument("--count-witnesses", action="store_true",
                    help="count all witnesses per prime, not just the least")
    ap.add_argument("--stop-at-first", action="store_true",
                    help="stop once a witness prime is found")
    ap.add_argument("--local-scan", type=int, default=64, metavar="QMAX",
                    help="scan moduli 2..QMAX for exact residue obstructions "
                         "(0 disables)")
    ap.add_argument("--chunk-bits", type=int, default=DEFAULT_CHUNK_BITS,
                    help="construction chunk size (affects random sets only)")
    args = ap.parse_args(argv)

    if args.verify:
        return verify(args.verify)

    missing = [f for f, v in (("--N", args.N), ("--polys", args.polys),
                              ("--set", args.setspec)) if v is None]
    if missing:
        ap.error("missing required argument(s): " + ", ".join(missing))
    if args.N <= 0:
        ap.error("--N must be positive")

    params = {
        "N": args.N,
        "polys": args.polys,
        "set": args.setspec,
        "pmax": args.pmax,
        "allow_degenerate": args.allow_degenerate,
        "count_witnesses": args.count_witnesses,
        "stop_at_first": args.stop_at_first,
        "chunk_bits": args.chunk_bits,
        "local_scan": args.local_scan,
    }
    result = run(params)
    report(result)
    if args.out:
        j, c = write_outputs(result, args.out)
        print(f"  wrote {j}")
        print(f"  wrote {c}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
