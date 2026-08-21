#!/usr/bin/env python3
"""
cg_campaign.py -- lower-bound campaign for the configuration a, a+(p-1)^2.

For each threshold X, find the densest set in two families with
(A - A) disjoint from S_X = { (p-1)^2 : p prime, p <= X }:

  trivial   A = q*Z, q minimal with no prime p <= X, p == 1 mod rad-part;
            here q = m^2 with m minimal such that no prime <= X is 1 mod m.
  CG        A = { n : (c*n) mod m < L } over all m <= Mmax, 1 <= c <= m/2,
            with L = min_s |c*s|_m.  Density L/m.  (Cantor-Gordon 1973.)

Every winner is a periodic set, so it extends to all N with the same density,
and the harness certifies it by exhaustive intersection.
"""
import sys, time, csv
import numpy as np


def primes_upto(n):
    if n < 2:
        return []
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(x) for x in np.nonzero(s)[0]]


def trivial_family(X):
    """Minimal m with no prime <= X congruent to 1 mod m; A = m^2 Z blocks all."""
    P = primes_upto(X)
    m = 2
    while True:
        if not any(p % m == 1 for p in P):
            return m, 1.0 / (m * m)
        m += 1


def cg_best(X, Mmax, topk=5):
    S = np.array(sorted({(p - 1) ** 2 for p in primes_upto(X)}), dtype=np.int64)
    found = []
    for m in range(2, Mmax + 1):
        r = np.unique(S % m)
        if r[0] == 0:
            continue
        cs = np.arange(1, m // 2 + 1, dtype=np.int64)
        t = (cs[:, None] * r[None, :]) % m
        t = np.minimum(t, m - t)
        q = t.min(axis=1)
        i = int(q.argmax())
        found.append((q[i] / m, m, int(cs[i]), int(q[i])))
    found.sort(reverse=True)
    return found[:topk], S


def main():
    # usage: cg_campaign.py [out.csv] [Mmax] [X1 X2 ...]
    argv = sys.argv[1:]
    outfile = argv[0] if argv else "cg_campaign.csv"
    Mmax = int(argv[1]) if len(argv) > 1 else 3000
    Xs = [int(x) for x in argv[2:]] or [10, 30, 100, 300, 1000]
    print(f"# sweep cap Mmax = {Mmax}; densities are family records "
          f"WITHIN that cap and can improve if it is raised")
    out = []
    for X in Xs:
        t0 = time.time()
        top, S = cg_best(X, Mmax)
        m_triv, d_triv = trivial_family(X)
        d, m, c, L = top[0]
        out.append({
            "X": X, "num_primes": len(primes_upto(X)), "s_max": int(S.max()),
            "trivial_m": m_triv, "trivial_density": f"{d_triv:.6g}",
            "cg_m": m, "cg_c": c, "cg_L": L, "cg_density": f"{d:.6g}",
            "gain_over_trivial": f"{d / d_triv:.1f}",
            "runners_up": "; ".join(f"m={mm},c={cc},L={ll},d={dd:.4f}"
                                    for dd, mm, cc, ll in top[1:4]),
            "sweep_seconds": round(time.time() - t0, 1),
        })
        r = out[-1]
        print(f"X={X:5d}  |S|={r['num_primes']:4d}  "
              f"trivial 1/{m_triv}^2 = {d_triv:.5f}   "
              f"CG best m={m} c={c} L={L}  d={d:.5f}   "
              f"gain x{d/d_triv:.1f}   ({r['sweep_seconds']}s)")
    for r in out:
        r["Mmax"] = Mmax
    with open(outfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)


if __name__ == "__main__":
    main()
