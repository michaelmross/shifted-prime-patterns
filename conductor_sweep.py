#!/usr/bin/env python3
"""
conductor_sweep.py -- push the Cantor-Gordon record ladder to larger X.

The CG density of {n : cn mod m < L} depends only on the reduced fraction
c/m, so the sweep enumerates conductors directly: moduli m with multipliers
coprime to m.  Three accelerations over the naive sweep:

  1. aliveness by arithmetic, not by residues: writing m = prod q^a_q and
     K(m) = prod q^ceil(a_q/2), some shift (p-1)^2 vanishes mod m exactly
     when p == 1 (mod K(m)).  So m is alive at threshold X iff the least
     prime == 1 mod K(m) exceeds X, decided by stepping the arithmetic
     progression through a primality sieve -- no residue set ever built
     for dead moduli.

  2. reduced multipliers only (gcd(c,m)=1): an unreduced pair is either
     covered at the smaller modulus or identically zero.

  3. staged elimination against a running incumbent: residues are fed to
     the multiplier array a few at a time, and multipliers whose running
     minimum can no longer beat the incumbent are discarded.  A cheap
     low-cap first pass seeds the incumbent.

Records printed here are family records WITHIN the stated cap M, exactly as
in cg_campaign.py.  Winners are already in conductor (reduced) form.

usage examples:
  python conductor_sweep.py 30000                    # threshold X=30000, defaults
  python conductor_sweep.py 100000 60000 3600        # X=1e5, cap m<=60000, 1h budget
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np

_STOP = None            # multiprocessing.Event in workers; None when serial


def _init_worker(ev):
    """Pool initializer: inherit the stop event, and ignore console Ctrl+C so
    the parent alone coordinates a cooperative shutdown (Windows sends the
    console control event to every attached process)."""
    global _STOP
    _STOP = ev
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def prime_sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def spf_sieve(n):
    """Smallest prime factor for every integer up to n."""
    spf = np.zeros(n + 1, dtype=np.int64)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i::i][spf[i::i] == 0] = i
    return spf


def K_of(m, spf):
    """K(m) = prod q^ceil(a/2): the modulus whose progression 1 mod K kills m."""
    K = 1
    while m > 1:
        q = int(spf[m]); a = 0
        while m % q == 0:
            m //= q; a += 1
        K *= q ** ((a + 1) // 2)
    return K


def least_prime_1mod(k, X, is_prime, cache):
    """Least prime p <= X with p == 1 (mod k), or 0 if none (alive)."""
    if k in cache:
        return cache[k]
    p = k + 1
    hit = 0
    while p <= X:
        if is_prime[p]:
            hit = p
            break
        p += k
    cache[k] = hit
    return hit


def scan_modulus(m, Svals, best):
    """Best coprime multiplier at one modulus, pruned against the incumbent.
    Returns (m, c, L) if the modulus strictly beats `best`, else None."""
    r = np.unique(Svals % m)
    if r[0] == 0:
        return None
    cs = np.arange(1, m // 2 + 1, dtype=np.int64)
    cs = cs[np.gcd(cs, m) == 1]            # conductors only
    thr = int(best * m)                    # must exceed this to beat the record
    if thr >= (m - 1) // 2 or cs.size == 0:
        return None
    runmin = np.full(cs.size, m, dtype=np.int64)
    for i in range(0, r.size, 6):
        t = (cs[:, None] * r[i:i + 6][None, :]) % m
        np.minimum(runmin, np.minimum(t, m - t).min(axis=1), out=runmin)
        keep = runmin > thr
        if not keep.all():
            cs, runmin = cs[keep], runmin[keep]
            if cs.size == 0:
                return None
    j = int(runmin.argmax())
    if runmin[j] / m > best:
        return (m, int(cs[j]), int(runmin[j]))
    return None


def sweep(X, M, budget=None, seed=0.0, verbose=True, mmin=2, step=1,
          heartbeat=None):
    t0 = time.time()
    is_prime = prime_sieve(X)
    P = np.nonzero(is_prime)[0].astype(np.int64)
    Svals = (P - 1) ** 2
    spf = spf_sieve(M)
    cache = {}

    best = seed
    winner = None
    alive_count = 0
    stopped_at = M
    last_done = mmin - step
    for m in range(mmin, M + 1, step):
        if (budget and time.time() - t0 > budget) or (_STOP is not None
                                                      and _STOP.is_set()):
            stopped_at = last_done
            break
        last_done = m
        if heartbeat and (m - mmin) % (heartbeat * step) < step:
            print(f"    [worker m0={mmin}] at m={m}, best {best:.6f},"
                  f" {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
        K = K_of(m, spf)
        if least_prime_1mod(K, X, is_prime, cache):
            continue                       # dead: some (p-1)^2 == 0 mod m
        alive_count += 1
        hit = scan_modulus(m, Svals, best)
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
            "swept_to": stopped_at, "alive": alive_count,
            "best": best, "winner": winner, "seconds": round(time.time() - t0, 1)}


def _sweep_worker(args):
    X, M, budget, seed, mmin, step = args
    return sweep(X, M, budget=budget, seed=seed, verbose=False,
                 mmin=mmin, step=step, heartbeat=5000)


def critical_primes(X, m, c, L):
    is_prime = prime_sieve(X)
    P = np.nonzero(is_prime)[0].astype(np.int64)
    t = (c * ((P - 1) ** 2)) % m
    d = np.minimum(t, m - t)
    assert int(d.min()) == L, "winner does not verify"
    return [int(p) for p in P[d == L]]


def merge(files):
    runs = [json.load(open(f)) for f in files]
    Xs = {r["X"] for r in runs}
    if len(Xs) != 1:
        sys.exit(f"cannot merge runs with different X: {sorted(Xs)}")
    X = Xs.pop()
    cands = [r for r in runs if r.get("winner")]
    top = max(cands, key=lambda r: r["best"]) if cands else None
    print(f"merging {len(runs)} run(s) at X = {X}")
    for f, r in zip(files, runs):
        print(f"  {f}: slice [{r.get('mmin', 2)},{r['cap']}] step {r.get('step', 1)}"
              f" swept to {r['swept_to']}, best {r['best']:.6f} at {r['winner']}")
    if top is None:
        print("  no slice produced a record beating its seed;"
              " coverage recorded, incumbent stands")
        return 0
    m, c, L = top["winner"]
    print(f"  combined record: (m,c,L)=({m},{c},{L})  density {L}/{m}"
          f" = {top['best']:.6f}")
    print("  combined coverage is the union of the slices above;"
          " state caps slice-by-slice when citing")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Sweep conductors (reduced fractions c/m) for the best "
                    "Cantor-Gordon set avoiding (p-1)^2 for every prime p <= X.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("X", type=int,
                    help="prime threshold: block every prime p <= X (e.g. 30000)")
    ap.add_argument("Mmax", type=int, nargs="?", default=40000,
                    help="largest modulus to sweep; records are within this cap")
    ap.add_argument("budget", type=float, nargs="?", default=None,
                    help="wall-clock budget in seconds; the sweep stops early "
                         "and reports the cap it actually reached")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel workers; moduli are interleaved (worker j "
                         "takes m == j mod jobs) so slices stay balanced")
    ap.add_argument("--mrange", type=int, nargs=2, metavar=("A", "B"),
                    help="sweep only moduli in [A, B] (manual slice for "
                         "multi-machine runs; combine with --out and --merge)")
    ap.add_argument("--out", metavar="FILE",
                    help="write a machine-readable JSON summary of this run")
    ap.add_argument("--seed", type=float, default=0.0,
                    help="known record density to prune against (e.g. "
                         "0.000837 for X=1e6); sliced runs without this do "
                         "unpruned scans and cost several times more. A run "
                         "reports a winner only if it strictly beats the seed")
    ap.add_argument("--state", metavar="FILE",
                    help="checkpoint file: budget-truncated sessions resume "
                         "where they stopped when the same command is re-run")
    ap.add_argument("--merge", nargs="+", metavar="FILE",
                    help="merge JSON summaries from --out runs instead of "
                         "sweeping; reports the combined record and honest cap")
    args = ap.parse_args()

    if args.merge:
        return merge(args.merge)

    X, M, budget = args.X, args.Mmax, args.budget
    if X < 2:
        ap.error("X must be at least 2 (it is the prime threshold)")
    mmin = 2
    if args.mrange:
        mmin, M = args.mrange
        if mmin < 2 or M < mmin:
            ap.error("--mrange needs 2 <= A <= B")

    state = None
    if args.state and os.path.isfile(args.state):
        state = json.load(open(args.state))
        if (state["X"] != X or state["cap"] != M
                or state.get("mmin0", 2) != mmin):
            sys.exit(f"state file {args.state} is for X={state['X']},"
                     f" range [{state.get('mmin0', 2)},{state['cap']}];"
                     " refusing to mix parameters")
        if state.get("complete"):
            print(f"already complete: best {state['best']:.6f}"
                  f" at {state['winner']}  [{state['elapsed']:.0f}s total]")
            return
        print(f"resuming from m = {state['next_m']}"
              f" (best so far {state['best']:.6f}, {state['elapsed']:.0f}s spent)")
    mmin0 = mmin
    if state:
        mmin = state["next_m"]

    tag = f", slice [{mmin},{M}]" if (args.mrange or state) else ""
    jobs = max(1, args.jobs)
    print(f"X = {X}, cap M = {M}{tag}"
          + (f", budget {budget:.0f}s" if budget else "")
          + (f", jobs {jobs}" if jobs > 1 else ""))
    if state:
        pilot = {"best": state["best"],
                 "winner": tuple(state["winner"]) if state["winner"] else None,
                 "seconds": 0.0}
        pilot_cap = None
    else:
        pilot_cap = min(4000, mmin - 1) if args.mrange else min(M, 4000)
    if pilot_cap is not None and pilot_cap >= 2:
        pilot = sweep(X, pilot_cap, verbose=False)
    elif pilot_cap is not None:
        pilot = {"best": 0.0, "winner": None, "seconds": 0.0}
    if args.seed > pilot["best"]:
        pilot = {"best": args.seed, "winner": None, "seconds": pilot.get("seconds", 0.0)}
    if pilot_cap is not None:
        print(f"  pilot (M={pilot_cap}): density {pilot['best']:.6f}"
              f" at {pilot['winner']}  [{pilot['seconds']}s]")
    if jobs == 1:
        full = sweep(X, M, budget=budget, seed=pilot["best"], verbose=True,
                     mmin=mmin, step=1)
    else:
        import multiprocessing as mp
        work = [(X, M, budget, pilot["best"], mmin + j, jobs)
                for j in range(jobs)]
        t0 = time.time()
        ev = mp.Event()
        interrupted = False
        with mp.Pool(jobs, initializer=_init_worker, initargs=(ev,)) as pool:
            res = pool.map_async(_sweep_worker, work)
            try:
                parts = res.get()
            except KeyboardInterrupt:
                interrupted = True
                print("\n  interrupt received: asking workers to stop at their"
                      " current modulus...", flush=True)
                ev.set()
                parts = res.get()
        cands = [q for q in parts if q["winner"]]
        top = max(cands, key=lambda q: q["best"]) if cands else parts[0]
        # a worker is complete iff it reached the last modulus of its stride;
        # then the union covers everything up to M and the cap is M itself
        def stride_end(j):
            lo = mmin + j
            return lo + ((M - lo) // jobs) * jobs if lo <= M else lo - jobs
        done = all(q["swept_to"] == stride_end(j) for j, q in enumerate(parts))
        full = {"X": X, "cap": M, "mmin": mmin, "step": 1,
                "swept_to": M if done else min(q["swept_to"] for q in parts),
                "alive": sum(q["alive"] for q in parts),
                "best": top["best"], "winner": top["winner"],
                "seconds": round(time.time() - t0, 1)}
        print(f"  workers complete through m = "
              f"{[q['swept_to'] for q in parts]}")
    if full["winner"] is None and not args.mrange:
        full["winner"], full["best"] = pilot["winner"], pilot["best"]
    if full["winner"] is None:
        inc = pilot["winner"] if pilot["winner"] else "seed"
        print(f"  no record inside slice [{mmin},{M}]"
              f" beats the incumbent {inc}"
              f" (density {pilot['best']:.6f}); slice is coverage-only")
        crit = []
    else:
        m, c, L = full["winner"]
        crit = critical_primes(X, m, c, L)
        print(f"  record within M<={full['swept_to']}: (m,c,L)=({m},{c},{L})"
              f"  density {L}/{m} = {full['best']:.6f}")
        print(f"  critical primes ({len(crit)}):"
              f" {crit[:10]}"
          + (f" ... +{len(crit) - 10} more" if len(crit) > 10 else ""))
    print(f"  alive moduli: {full['alive']}   sweep time {full['seconds']}s")
    if full["swept_to"] < M:
        why = "interrupted" if (jobs > 1 and interrupted) else "budget exhausted"
        print(f"  NOTE: {why} at m={full['swept_to']} < {M};"
              " record is within that smaller cap")
    if args.state:
        done = full["swept_to"] >= M
        json.dump({"X": X, "cap": M, "mmin0": mmin0, "seed": args.seed,
                   "next_m": full["swept_to"] + 1,
                   "best": full["best"],
                   "winner": list(full["winner"]) if full["winner"] else None,
                   "alive": full["alive"] + (state["alive"] if state else 0),
                   "elapsed": full["seconds"] + (state["elapsed"] if state else 0.0),
                   "complete": done},
                  open(args.state, "w"), indent=1)
        print(f"  state -> {args.state}"
              + ("  (complete)" if done else
                 "  (re-run the same command to continue)"))
    if args.out:
        full["critical_primes"] = crit
        with open(args.out, "w") as fh:
            json.dump(full, fh, indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
