#!/usr/bin/env python3
"""
k3_sweep.py -- lower-bound records for the length-3 polynomial progression

    x,  x + (p-1),  x + (p-1)^2,        p prime,

the model configuration of Krause-Mousavi-Tao-Teraevaeinen.  As in the k=2
campaign, candidates are Cantor-Gordon interval sets {n : cn mod m < L}.  The
pattern exists at shift d = p-1 exactly when the three points {0, cd, cd^2}
mod m fit inside a cyclic window of L consecutive residues, so each prime
carries the margin

    span(d) = m - maxgap(0, cd mod m, cd^2 mod m),

and the record at (m, c) is L = min over odd primes p <= X of span(d).
(p = 2 gives d = d^2 = 1, a degenerate repeated pattern, and is skipped --
matching the harness's default.)

Structure worth knowing before reading results:
  * blocking dies only when m | d itself, so a modulus is alive at X iff the
    least prime == 1 (mod m) exceeds X.  K(m) = m: the square-part trick that
    keeps squarefull moduli alive in the k=2 problem does not exist here.
  * span(0, u, v) >= |u|_m and >= |v|_m, so at every (m, c) the k=3 record
    dominates both two-point records: delta_3(X) >= delta_2(X) structurally.

usage examples:
  python k3_sweep.py 1000                      # threshold X, defaults
  python k3_sweep.py 10000 30000 3600 --jobs 4
"""
import argparse
import time

import numpy as np

from conductor_sweep import least_prime_1mod, prime_sieve


def scan_modulus_k3(m, Dres, best):
    """Best coprime multiplier at one modulus for the 3-point span margin.
    Dres: the distinct values (p-1) mod m over odd primes p <= X.
    Returns (m, c, L) if the modulus strictly beats `best`, else None."""
    r = Dres
    r2 = (r * r) % m
    cs = np.arange(1, m // 2 + 1, dtype=np.int64)
    cs = cs[np.gcd(cs, m) == 1]
    thr = int(best * m)
    if cs.size == 0 or thr >= (2 * m) // 3:   # span <= m - ceil(m/3) always
        return None
    runmin = np.full(cs.size, m, dtype=np.int64)
    for i in range(0, r.size, 4):
        u = (cs[:, None] * r[i:i + 4][None, :]) % m
        v = (cs[:, None] * r2[i:i + 4][None, :]) % m
        a = np.minimum(u, v)
        b = np.maximum(u, v)
        gap = np.maximum(np.maximum(a, b - a), m - b)
        np.minimum(runmin, (m - gap).min(axis=1), out=runmin)
        keep = runmin > thr
        if not keep.all():
            cs, runmin = cs[keep], runmin[keep]
            if cs.size == 0:
                return None
    j = int(runmin.argmax())
    if runmin[j] / m > best:
        return (m, int(cs[j]), int(runmin[j]))
    return None


def sweep(X, M, budget=None, seed=0.0, verbose=True, mmin=2, step=1):
    t0 = time.time()
    is_prime = prime_sieve(X)
    P = np.nonzero(is_prime)[0].astype(np.int64)
    P = P[P >= 3]                              # p = 2 is degenerate here
    D = P - 1
    cache = {}
    best, winner = seed, None
    alive = 0
    stopped_at = M
    last_done = mmin - step
    for m in range(mmin, M + 1, step):
        if budget and time.time() - t0 > budget:
            stopped_at = last_done
            break
        last_done = m
        if least_prime_1mod(m, X, is_prime, cache):
            continue                           # some p == 1 mod m: dead
        alive += 1
        hit = scan_modulus_k3(m, np.unique(D % m), best)
        if hit:
            best = hit[2] / m
            winner = hit
            if verbose:
                print(f"    new record  m={m} c={hit[1]} L={hit[2]}"
                      f"  density {best:.6f}  [{time.time()-t0:.0f}s]",
                      flush=True)
    else:
        stopped_at = last_done
    return {"X": X, "cap": M, "mmin": mmin, "step": step,
            "swept_to": stopped_at, "alive": alive, "best": best,
            "winner": winner, "seconds": round(time.time() - t0, 1)}


def _worker(args):
    X, M, budget, seed, mmin, step = args
    return sweep(X, M, budget=budget, seed=seed, verbose=False,
                 mmin=mmin, step=step)


def critical_primes(X, m, c, L):
    is_prime = prime_sieve(X)
    P = np.nonzero(is_prime)[0].astype(np.int64)
    P = P[P >= 3]
    d = (P - 1).astype(np.int64)
    u = (c * d) % m
    v = (c * ((d * d) % m)) % m
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    span = m - np.maximum(np.maximum(a, b - a), m - b)
    assert int(span.min()) == L, "winner does not verify"
    return [int(p) for p in P[span == L]]


def main():
    ap = argparse.ArgumentParser(
        description="Blind conductor sweep for x, x+(p-1), x+(p-1)^2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("X", type=int, help="prime threshold (odd primes p <= X)")
    ap.add_argument("Mmax", type=int, nargs="?", default=20000,
                    help="largest modulus; records are within this cap")
    ap.add_argument("budget", type=float, nargs="?", default=None,
                    help="wall-clock budget in seconds")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel workers over interleaved moduli")
    args = ap.parse_args()
    X, M, jobs = args.X, args.Mmax, max(1, args.jobs)

    print(f"X = {X}, cap M = {M}"
          + (f", budget {args.budget:.0f}s" if args.budget else "")
          + (f", jobs {jobs}" if jobs > 1 else ""))
    pilot = sweep(X, min(M, 2000), verbose=False)
    print(f"  pilot (M={min(M, 2000)}): density {pilot['best']:.6f}"
          f" at {pilot['winner']}  [{pilot['seconds']}s]")
    if jobs == 1:
        full = sweep(X, M, budget=args.budget, seed=pilot["best"])
    else:
        import multiprocessing as mp
        t0 = time.time()
        with mp.Pool(jobs) as pool:
            parts = pool.map(_worker, [(X, M, args.budget, pilot["best"],
                                        2 + j, jobs) for j in range(jobs)])
        hits = [q for q in parts if q["winner"]]
        top = max(hits, key=lambda q: q["best"]) if hits else parts[0]

        def stride_end(j):
            lo = 2 + j
            return lo + ((M - lo) // jobs) * jobs if lo <= M else lo - jobs
        done = all(q["swept_to"] == stride_end(j) for j, q in enumerate(parts))
        full = {"swept_to": M if done else min(q["swept_to"] for q in parts),
                "alive": sum(q["alive"] for q in parts),
                "best": top["best"], "winner": top["winner"],
                "seconds": round(time.time() - t0, 1)}
    if full["winner"] is None:
        full["winner"], full["best"] = pilot["winner"], pilot["best"]
    m, c, L = full["winner"]
    crit = critical_primes(X, m, c, L)
    print(f"  record within M<={full['swept_to']}: (m,c,L)=({m},{c},{L})"
          f"  density {L}/{m} = {full['best']:.6f}")
    print(f"  alive moduli: {full['alive']}   sweep time {full['seconds']}s")
    print(f"  critical primes ({len(crit)}): {crit[:10]}"
          + (f" ... +{len(crit) - 10} more" if len(crit) > 10 else ""))
    if full["swept_to"] < M:
        print(f"  NOTE: budget exhausted at m={full['swept_to']} < {M}")


if __name__ == "__main__":
    main()
