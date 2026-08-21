#!/usr/bin/env python3
"""
check_all.py -- re-derive and verify every certificate in certificates/.

Runs `shifted_prime_patterns.py --verify` on each JSON, prints one line per
certificate, and exits nonzero if any fails.  Suitable as a CI step: a passing
run means every archived result was reproduced byte-for-byte from its recorded
parameters by the current harness.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "shifted_prime_patterns.py")
CERTS = sorted(glob.glob(os.path.join(HERE, "certificates", "*.json")))

if not CERTS:
    print("no certificates found")
    sys.exit(1)

failed = []
for cert in CERTS:
    r = subprocess.run([sys.executable, HARNESS, "--verify", cert],
                       capture_output=True, text=True)
    verdict = next((ln for ln in r.stdout.splitlines()
                    if ln.startswith(("PASS", "FAIL"))), "FAIL  no verdict")
    print(f"{os.path.basename(cert):32s} {verdict}")
    if not verdict.startswith("PASS"):
        failed.append(os.path.basename(cert))
        for ln in r.stdout.splitlines():
            if ln.startswith(("FAIL", "      ")):
                print("    " + ln)

print()
if failed:
    print(f"{len(failed)}/{len(CERTS)} certificate(s) FAILED: {', '.join(failed)}")
    sys.exit(1)
print(f"all {len(CERTS)} certificates reproduced byte-for-byte")
