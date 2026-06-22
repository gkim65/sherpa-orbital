"""
MPC stationkeeping feasibility check (Decision D3, external review 2026-06-22).

Runs the deterministic Strategy-3 MPC (baselines/mpc.py) against each EXISTING truth
model over a multi-day horizon and tabulates, per truth model:
  - does the orbit survive (periapsis altitude stays > 5 km)?
  - for how long (survival time)?
  - how many burns and what total ΔV (m/s) did it cost?

This answers "can we even stationkeep this orbit?" at the two cheapest fidelity rungs
(CR3BP+EncJ2, CR3BP+SaturnJ2+EncJ2) before investing in higher-fidelity propagators.

Usage:
    python -m scripts.mpc_feasibility            # default 7-day horizon
    python -m scripts.mpc_feasibility --days 14

Note on fuel: ΔV is reported in m/s. Converting to propellant mass (kg) needs Isp and
spacecraft wet mass from MacKenzie et al. 2020 §3.5 (MR-106E) added to constants.py.
"""

import argparse

import numpy as np

from baselines.mpc import run_mpc
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.cr3bp_saturn_j2 import cr3bp_saturn_enc_j2_eom
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

# One "rev" between control crossings ≈ T/3 (3 periapsis passes per period-3 orbit).
ONE_REV_S = PERIOD3_PERIOD_S / 3.0

TRUTH_MODELS = [
    ("CR3BP + EncJ2",            cr3bp_j2_eom),
    ("CR3BP + SaturnJ2 + EncJ2", cr3bp_saturn_enc_j2_eom),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="MPC stationkeeping feasibility check")
    parser.add_argument("--days", type=float, default=7.0,
                        help="Mission horizon in days (default 7).")
    parser.add_argument("--n-revs", type=int, default=3,
                        help="Multiple-shooting horizon N_m (default 3).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-burn diagnostics.")
    args = parser.parse_args()

    ic = _nd_to_phys(PERIOD3_IC_ND)
    horizon_s = args.days * 86400.0

    print(f"\nMPC stationkeeping feasibility check")
    print(f"  IC: period-3 L1 halo (north-pole periapsis, symmetric degenerate)")
    print(f"  horizon: {args.days:.1f} days ({horizon_s/3600:.0f} hr)")
    print(f"  N_m (shooting revs): {args.n_revs}")
    print(f"  control shell: 600 km altitude; crash: 5 km altitude\n")

    rows = []
    for name, eom in TRUTH_MODELS:
        print(f"--- {name} ---")
        r = run_mpc(ic, truth_eom=eom, period_s=ONE_REV_S,
                    horizon_s=horizon_s, n_revs=args.n_revs, verbose=args.verbose)
        rows.append((name, r))
        print(f"  outcome={r['outcome'].upper()}: "
              f"loss/horizon t={r['survival_time_s']/3600:8.2f} hr "
              f"({r['survival_time_s']/86400:5.2f} d), "
              f"controller active until {r.get('controller_active_until_s', r['survival_time_s'])/3600:.2f} hr, "
              f"{r['n_burns']:3d} burns, ΔV={r['total_dv_ms']:8.2f} m/s, "
              f"min peri alt={r['min_peri_alt_km']:.1f} km\n")

    # Summary table.
    print("=" * 86)
    print(f"{'Truth model':<28}{'Outcome':<10}{'Loss/active t':<16}"
          f"{'Burns':<8}{'ΔV (m/s)':<10}")
    print("-" * 86)
    for name, r in rows:
        if r["outcome"] in ("crash", "escape"):
            t_report = f"{r['survival_time_s']/86400:.2f} d"
        else:
            t_report = f"{r.get('controller_active_until_s', r['survival_time_s'])/3600:.1f} hr act."
        print(f"{name:<28}{r['outcome']:<10}{t_report:<16}"
              f"{r['n_burns']:<8}{r['total_dv_ms']:<10.2f}")
    print("=" * 86)
    print("\nΔV→fuel(kg) deferred: needs ISP + wet mass from MacKenzie 2020 §3.5.\n")


if __name__ == "__main__":
    main()