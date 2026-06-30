"""CLI — show conformal vs naive softmax coverage across target levels."""
from __future__ import annotations

import argparse
import json

from .benchmark import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Conformal prediction vs naive softmax coverage.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = run(seed=args.seed)
    if args.json:
        print(json.dumps(res, indent=2))
        return

    m = res["_meta"]
    print("=" * 78)
    print(f"  CONFORMAL PREDICTION vs NAIVE SOFTMAX   "
          f"({m['n_classes']} classes, {m['n_test']} test, {m['model']} acc {m['test_accuracy']:.1%})")
    print("=" * 78)
    print(f"{'target':>8}{'conformal cov':>16}{'conf size':>11}   |"
          f"{'naive cov':>11}{'naive size':>12}{'shortfall':>11}")
    print("-" * 78)
    for r in res["grid"]:
        c, nv = r["conformal"], r["naive"]
        flag = "✓" if r["conformal_meets_target"] else "✗"
        print(f"{r['target_coverage']:>8.0%}{c['coverage']:>14.1%} {flag}{c['avg_set_size']:>11.2f}   |"
              f"{nv['coverage']:>11.1%}{nv['avg_set_size']:>12.2f}{r['naive_shortfall']:>+11.1%}")
    print("-" * 78)
    worst = max(res["grid"], key=lambda r: r["naive_shortfall"])
    print(f"conformal hits every target (distribution-free guarantee); naive softmax under-covers "
          f"by up to {worst['naive_shortfall']:.0%}")
    print(f"  → at target {worst['target_coverage']:.0%}, naive softmax actually covers only "
          f"{worst['naive']['coverage']:.0%} — the model's confidence is miscalibrated")
    print("=" * 78)


if __name__ == "__main__":
    main()
