#!/usr/bin/env python3
"""
pool_floor.py -- the alive-pool floor, by sieving alone (no sweeps).

A modulus m is alive for the k=3 configuration at threshold X iff the least
prime P1(m) == 1 (mod m) exceeds X, and for the k=2 configuration iff
P1(K(m)) > X with K(m) = prod q^ceil(a/2).  The pool floor

    phi_min(X) = min { phi(m) : m alive at X }

is therefore pure arithmetic.  Under the random-span model a record at a
floor modulus has density ~ c / sqrt(phi_min(X)), which the measured k=3
ladder matches at its top rungs (winner phi / floor phi = 1.0-1.3 from
X = 10^4 on; the k=2 winners never pin, hovering 1.5-5x the floor).
Fitting c on the top two measured rungs gives falsifiable predictions for
unmeasured thresholds, printed below.

usage: python pool_floor.py [--xcap 10000000] [--mcap 250000]
"""
import argparse
import math

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xcap", type=int, default=10 ** 7)
    ap.add_argument("--mcap", type=int, default=250000)
    args = ap.parse_args()
    XCAP, MCAP = args.xcap, args.mcap

    s = np.ones(XCAP + 1, dtype=bool); s[:2] = False
    for i in range(2, int(XCAP ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    phi = np.arange(MCAP + 1, dtype=np.int64)
    spf = np.zeros(MCAP + 1, dtype=np.int64)
    for p in range(2, MCAP + 1):
        if phi[p] == p:
            phi[p::p] -= phi[p::p] // p
            spf[p::p][spf[p::p] == 0] = p

    def P1(k):
        p = k + 1
        while p <= XCAP:
            if s[p]:
                return p
            p += k
        return XCAP + 1

    P1m = np.zeros(MCAP + 1, dtype=np.int64)
    for m in range(2, MCAP + 1):
        P1m[m] = P1(m)
    kcache = {}
    P1K = np.zeros(MCAP + 1, dtype=np.int64)
    for m in range(2, MCAP + 1):
        mm, K = m, 1
        while mm > 1:
            q = int(spf[mm]); a = 0
            while mm % q == 0:
                mm //= q; a += 1
            K *= q ** ((a + 1) // 2)
        if K not in kcache:
            kcache[K] = P1(K)
        P1K[m] = kcache[K]

    grid = [10, 30, 100, 300, 1000, 3000, 10 ** 4, 3 * 10 ** 4, 10 ** 5,
            3 * 10 ** 5, 10 ** 6, 3 * 10 ** 6, 10 ** 7]
    grid = [X for X in grid if X <= XCAP]
    print("  X          k3 floor (at m)      k2 floor (at m)")
    floors = {}
    for X in grid:
        a3 = np.nonzero(P1m[2:] > X)[0] + 2
        a2 = np.nonzero(P1K[2:] > X)[0] + 2
        j3 = a3[np.argmin(phi[a3])]; j2 = a2[np.argmin(phi[a2])]
        floors[X] = int(phi[j3])
        print(f"{X:>9}   {int(phi[j3]):>7} (m={int(j3):>6})"
              f"   {int(phi[j2]):>7} (m={int(j2):>6})")

    # fit on the top two certified rungs of the k=3 ladder
    c = np.mean([290 / 6578 * math.sqrt(floors.get(10 ** 5, 2344)),
                 322 / 10454 * math.sqrt(floors.get(3 * 10 ** 5, 5040))])
    print(f"\nfit c = delta3 * sqrt(floor) on the top measured rungs: {c:.3f}")
    for X in grid:
        if X > 3 * 10 ** 5:
            print(f"  prediction: delta3(X={X:>8}) ~ {c / math.sqrt(floors[X]):.5f}")


if __name__ == "__main__":
    main()
