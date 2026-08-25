#!/usr/bin/env python3
"""
targeted_search.py -- chase the record ladder along its observed structure.

Every record conductor found by the blind sweeps factors, beyond a power of
two, entirely into primes == 1 (mod 4), and consecutive rungs share a base:
145 = 5*29, 3848 = 2^3*13*37, 3145 = 5*17*37, 5185 = 5*17*61.  This tool
tests that regularity as a search strategy: candidates are m = 2^e * B * q
with B a product of small primes == 1 (mod 4) and q a larger prime == 1
(mod 4), reaching moduli far beyond any blind cap.

The result is a LOWER BOUND like every other record here: a targeted hit is
certified identically, but no completeness-within-a-cap claim is made beyond
the blind pilot's cap, which is reported separately.

usage examples:
  python targeted_search.py 300000                     # X = 3e5, defaults
  python targeted_search.py 300000 --mcap 150000 --budget 600
"""
import argparse
import time

import numpy as np

from conductor_sweep import (K_of, critical_primes, least_prime_1mod,
                             prime_sieve, scan_modulus, spf_sieve, sweep)


def candidates(mcap, base_cap, spf):
    """Moduli 2^e * B * q: B a product of <=2 distinct primes == 1 mod 4 below
    base_cap, q a prime == 1 mod 4 exceeding B's largest factor."""
    is_p = prime_sieve(mcap)
    p1 = [int(q) for q in np.nonzero(is_p)[0] if q % 4 == 1]
    small = [q for q in p1 if q < base_cap]
    bases = {1}
    for i, a in enumerate(small):
        bases.add(a)
        for b in small[i + 1:]:
            if a * b < base_cap:
                bases.add(a * b)
    out = set()
    for B in bases:
        bigfac = 1 if B == 1 else max(int(spf[B]), B // int(spf[B]))
        for q in p1:
            if q <= bigfac:
                continue
            for e in (0, 1, 2, 3):
                m = (1 << e) * B * q
                if m > mcap:
                    break
                out.add(m)
    return sorted(out)


def _scan_worker(args):
    """Scan one interleaved sublist of candidates (module-level for Windows spawn)."""
    X, mcap, sub, seed, budget = args
    spf = spf_sieve(mcap)
    is_prime = prime_sieve(X)
    P = np.nonzero(is_prime)[0].astype(np.int64)
    Svals = (P - 1) ** 2
    cache = {}
    best, winner, alive = seed, None, 0
    t0 = time.time()
    last, trunc = None, False
    for m in sub:
        if budget and time.time() - t0 > budget:
            trunc = True
            break
        last = m
        if least_prime_1mod(K_of(m, spf), X, is_prime, cache):
            continue
        alive += 1
        hit = scan_modulus(m, Svals, best)
        if hit:
            best, winner = hit[2] / hit[0], hit
    return {"best": best, "winner": winner, "alive": alive,
            "last": last, "trunc": trunc}


def main():
    ap = argparse.ArgumentParser(
        description="Targeted conductor search along the observed "
                    "primes-==-1-mod-4 structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("X", type=int, help="prime threshold, e.g. 300000")
    ap.add_argument("--mcap", type=int, default=150000,
                    help="largest targeted modulus")
    ap.add_argument("--base-cap", type=int, default=2000,
                    help="largest structured base B")
    ap.add_argument("--pilot-cap", type=int, default=6000,
                    help="blind sweep cap used to seed the incumbent")
    ap.add_argument("--budget", type=float, default=None,
                    help="wall-clock budget in seconds for the targeted pass")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel workers over an interleaved candidate split")
    args = ap.parse_args()
    X = args.X

    print(f"X = {X}: blind pilot to M <= {args.pilot_cap} ...", flush=True)
    pilot = sweep(X, args.pilot_cap, verbose=False)
    best, winner = pilot["best"], pilot["winner"]
    print(f"  pilot record {winner}  density {best:.6f}  [{pilot['seconds']}s]",
          flush=True)

    spf = spf_sieve(args.mcap)
    cand = candidates(args.mcap, args.base_cap, spf)
    t0 = time.time()
    stopped = None
    jobs = max(1, args.jobs)
    if jobs > 1:
        import multiprocessing as mp
        subs = [cand[j::jobs] for j in range(jobs)]
        work = [(X, args.mcap, sub, best, args.budget) for sub in subs]
        with mp.Pool(jobs) as pool:
            parts = pool.map(_scan_worker, work)
        alive = sum(q["alive"] for q in parts)
        examined = alive
        hits = [q for q in parts if q["winner"]]
        if hits:
            top = max(hits, key=lambda q: q["best"])
            best, winner = top["best"], top["winner"]
        if any(q["trunc"] for q in parts):
            stopped = min(q["last"] for q in parts if q["trunc"])
        print(f"  workers scanned through candidates near m ="
              f" {[q['last'] for q in parts]}")
    else:
        is_prime = prime_sieve(X)
        P = np.nonzero(is_prime)[0].astype(np.int64)
        Svals = (P - 1) ** 2
        cache = {}
        alive = examined = 0
        for m in cand:
            if args.budget and time.time() - t0 > args.budget:
                stopped = m
                break
            if least_prime_1mod(K_of(m, spf), X, is_prime, cache):
                continue
            alive += 1
            hit = scan_modulus(m, Svals, best)
            examined += 1
            if hit:
                best = hit[2] / hit[0]
                winner = hit
                print(f"    new record  m={hit[0]} c={hit[1]} L={hit[2]}"
                      f"  density {best:.6f}  [{time.time()-t0:.0f}s]", flush=True)

    m, c, L = winner
    crit = critical_primes(X, m, c, L)
    fac, mm = [], m
    while mm > 1:
        q = int(spf[mm]) if mm < len(spf) else mm
        a = 0
        while mm % q == 0:
            mm //= q; a += 1
        fac.append(f"{q}^{a}" if a > 1 else str(q))
    print(f"  record: (m,c,L)=({m},{c},{L})  density {L}/{m} = {best:.6f}"
          f"   m = {'*'.join(fac)}")
    print(f"  targeted candidates: {len(cand)} generated, {alive} alive,"
          f" {examined} scanned  [{time.time()-t0:.0f}s]")
    print(f"  critical primes ({len(crit)}):"
          f" {crit[:10]}"
          + (f" ... +{len(crit) - 10} more" if len(crit) > 10 else ""))
    if stopped:
        print(f"  NOTE: budget exhausted at m={stopped}; larger candidates unscanned")
    print(f"  completeness: blind only to M <= {args.pilot_cap};"
          " the targeted family is a heuristic, the record a certified lower bound")


if __name__ == "__main__":
    main()
