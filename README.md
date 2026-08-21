# shifted-prime-patterns

[![verify-certificates](https://github.com/michaelmross/shifted-prime-patterns/actions/workflows/verify.yml/badge.svg)](https://github.com/michaelmross/shifted-prime-patterns/actions/workflows/verify.yml)

A **shifted prime pattern** is a polynomial progression whose shift parameter is
one less than a prime:

$$a,\ a + P_1(p-1),\ \dots,\ a + P_{k-1}(p-1), \qquad p \ \text{prime},$$

where $P_1,\dots,P_{k-1} \in \mathbb{Z}[y]$ have zero constant term. This
repository searches for these patterns exhaustively inside a set
$A \subseteq [0,N)$ and issues exact certificates for the primes that admit
none. By a theorem of Wooley and Ziegler, no set of positive density avoids
them for every prime, so the certificates measure how far — and at what
density — avoidance can be pushed.

## Method

$A$ is held as a single bitmask. For a prime $p$ with $d = p-1$, put
$\text{offsets} = \{0, P_1(d), \dots, P_{k-1}(d)\}$ and $\ell = \min$ offsets.
A witness exists if and only if

$$M \;=\; \bigwedge_{o \in \text{offsets}} \bigl(A \gg (o - \ell)\bigr) \;\neq\; 0,$$

and the lowest set bit of $M$ is the least witness. Normalizing by $\ell$ lets
polynomials with negative coefficients be handled without special cases. When
$M = 0$ the prime is certified pattern-free across the entire window, which is
the interesting direction.

Cost per prime is $O(kN/64)$ word operations. Measured: 1,664 primes at
$N = 2\times 10^8$, $k=2$ in 36 s, or about 21 ms per prime. The binding
constraint is the window, not time. For $\deg P_i = d$ you need
$c\,p^{d} < N$, so only $p \lesssim N^{1/d}$ is testable at all. Primes past
that ceiling are recorded as `skipped_window` rather than silently dropped.

## Usage

```bash
# three-term progression, spacing p-1, inside 6Z
python3 shifted_prime_patterns.py --N 1000000 --polys "y,2y" \
    --set congruence:6 --pmax 200 --out runs/cong6

# square differences in a digit-restricted set
python3 shifted_prime_patterns.py --N 20000000 --polys "y^2" \
    --set digits:8,0,1,3 --pmax 4000 --out runs/digits8

# re-run a stored certificate and compare
python3 shifted_prime_patterns.py --verify runs/cong6.json
```

Each run writes `<prefix>.json` (certificate) and `<prefix>.csv` (per-prime
table with offsets, status, least witness, and optionally the witness count).

## Set families

| spec | set | role |
|---|---|---|
| `all` | $[0,N)$ | sanity check |
| `congruence:q[,r...]` | union of residue classes mod $q$ | exact local obstruction |
| `digits:b,d1,d2,...` | all base-$b$ digits in $D$ | adversarial, Behrend/Ruzsa style |
| `bohr:alpha,c` | $\{n : \lVert \alpha n\rVert < c\}$ | control |
| `quad:alpha,c` | $\{n : \lVert \alpha n^2\rVert < c\}$ | control |
| `random:delta[,seed]` | independent bits | control |
| `file:path` | integers read from a file | custom constructions |

`bohr` and `quad` use the dyadic rational $\text{round}(\alpha \cdot 2^{48})/2^{48}$
evaluated exactly in `uint64` arithmetic, so the set searched is the one named in
the certificate rather than a floating-point approximation of it.

## Reproducibility

Every certificate records the SHA-256 of the packed bitmask of $A$, the SHA-256
of the script, and the full per-prime result table. `--verify` rebuilds the set
from the recorded parameters, re-runs the search, and compares. A changed set, a
changed result, or an edited summary all fail loudly and name the section that
moved.

## The campaign (first data)

`cg_campaign.py` computes, per threshold $X$, the densest known periodic set
avoiding $S_X = \{(p-1)^2 : p \le X\}$ — Motzkin's difference-avoidance
problem for these shifts. Two families: the trivial $m^2\mathbb{Z}$, and
Cantor–Gordon interval sets $\{n : cn \bmod m < L\}$ swept over
$m \le 3000$. Winners are certified end to end by the harness (`cg:` set kind).

| $X$ | $\pi(X)$ | trivial | best CG (reduced) | exact record |
|---|---|---|---|---|
| 10 | 4 | $1/25$ | $2/5$ at $(5,2,2)$ | $2/5$, optimal for $m\le 80$ |
| 30 | 10 | $1/169$ | $36/156 = 3/13$ | $\mathbf{16/69}$ at $m=69$, beats CG |
| 100 | 25 | $1/289$ | $3/17$ | open |
| 300 | 62 | $1/961$ | $32/336 = 2/21$ | open |
| 1000 | 168 | $1/7225$ | $40/580 = 2/29$ | open |

For periodic sets the certificate is $N$-free: pattern existence at a prime is
equivalent to residue admissibility mod the period, which the harness now
computes for `cg:` sets and records as `periodic_no_pattern_for_all_p_upto`.
A bitmask pass is an independent code path and yields the archival core digest,
but is not required for the mathematical claim. The sweep cap `Mmax` truncates
the family: raising it from 3000 to 8000 improved the $X=10^4$ record by 12%.

Empirically $\delta(X) \gtrsim X^{-0.4}$ on this range, against $\sim X^{-2}$
for the trivial family: the quadratic structure of the shifts, not their count,
controls the decay. Already at $X=30$ an exact circulant independence
computation strictly beats the interval family, so exact $\alpha$ over growing
$m$ is the real instrument for the record table. Certificates for all six
winners are in `certificates/`.

## What identifies a run

The digest of the JSON file is **not** a stable identifier. It moves whenever the
schema gains a field or the timestamp changes, even though the mathematics is
untouched. Every certificate therefore carries a `core_digest`: a SHA-256 over a
frozen scope (`core/1`) consisting of the parameters that determine the
computation, the mask digest, and the per-prime table reduced to
`(p, status, witness_a)`. Timestamps, the script hash, timings, and every
diagnostic field are excluded by construction, so adding output to a later
version of this script cannot move it.

Cite and archive the core digest. Use the file digest only for the file.

`--verify` reports whether the core digest matches, and falls back to
recomputing it for certificates written before the field existed.

## Campaign design notes

**Random sets carry no signal.** For random $A$ of density $\delta$ the expected
witness count at one prime is $\sim \delta^k N$, so $p=2$ succeeds essentially
always and $p_{\min}$ is constant. Everything interesting comes from structured
$A$.

**The residue scan runs on every set.** For each modulus $q \le$ `--local-scan`
(default 64) the script computes the exact image of $A$ bmod $q$ and asks whether
any residue $r$ admits $r + P_i(d)$ for all $i$. If none does, no prime with
$p-1 \equiv d$ can carry a pattern, at any $N$. This turns a share of the
no-pattern certificates into congruence statements and reports what fraction is
explained that way. It is the first thing to read before treating a null result
as evidence of anything. A blocked prime that reports a pattern is logged as a
contradiction, which is a hard bug signal.

**The local obstruction is exact and cheap.** For a periodic set the script
computes, for every $d \bmod q$, whether some residue $r$ has $r + P_i(d) \in A$
for all $i$, and predicts $p_{\min}$ from the least admissible prime. The
prediction applies the same degeneracy and window filters as the search, so the
two numbers are directly comparable and a mismatch signals a bug. This is the
built-in consistency check.

**Congruence sets give certified lower bounds.** With $P_i(y) = iy$ and
$A = q\mathbb{Z}$, admissibility forces $p \equiv 1 \pmod q$, so
$p_{\min} = P(q,1)$, the least prime in that progression, conjecturally
$\asymp \varphi(q)\log^2 q$. Degree softens this: for $P(y) = y^d$ and
$A = m^d\mathbb{Z}$ only $m \mid p-1$ is needed, giving
$p_{\min} \gtrsim \delta^{-1/d}$. Verified for $P=y^2$ at $q=4,9,25$, where
$p_{\min} = 3, 7, 11$ — the least primes $\equiv 1$ mod $2, 3, 5$.

**Worked example of the trap.** A base-8 digit set $D=\{0,1,3\}$ is AP-free and
carry-free, so $A$ contains no three-term progression at all. Run it against
$P(y)=y^2$ instead and 435 of 550 primes report no pattern, which looks like
resistance. It is not. Since $A \bmod 8 = \{0,1,3\}$ and no $r$ there has
$r+4 \in \{0,1,3\}$, no difference is $\equiv 4 \bmod 8$. And
$(p-1)^2 \equiv 4 \bmod 8$ exactly when $p \equiv 3 \bmod 4$, so 280 of those
435 are a single residue class. Restricted to $p \equiv 1 \bmod 4$ the hit rate
is 0.424 against a length-matched control of 0.378, so the surviving primes carry
no deficit at all.

**No-pattern verdicts are window-limited unless you check.** A prime reports no
pattern when the witness fails to appear *inside $[0,N)$*, which is weaker than
no witness existing. The run reports the largest witness element it saw as a
percentage of $N$, and warns when that exceeds half the window. At 98% the
no-pattern certificates should not be trusted without enlarging $N$. For a
$q$-periodic $A$ the question does not arise, since $N \ge \text{span} + q$
already settles every prime. For a self-similar set it very much does.

**Where a contribution would sit.** Maximize the density of $A \subseteq [N]$
admitting no configuration for all $p \le X$. Verification is exact bitmask
intersection, so any such set is certifiable. The gap between the best known
constructions and the density upper bounds of Krause–Mousavi–Tao–Teräväinen
(arXiv:2608.19525) is enormous, and no computation can close it, but which
construction family wins at moderate $X$ is a question numerics can answer.

## Notes

- Degenerate primes, where the offsets collide or vanish, are skipped by default.
  `--allow-degenerate` keeps them.
- `--count-witnesses` reports every witness per prime instead of only the least.
  It costs a full popcount of $M$, so leave it off for large windows.
- `random` sets depend on `--chunk-bits`, which is therefore recorded in the
  certificate.
- `bohr` and `quad` require $N \le 2^{32}$.
