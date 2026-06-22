"""
MPC planner-model ablation: is stationkeeping killed by the MODEL GAP or by the
ORBITAL INSTABILITY?

The production MPC (baselines/mpc.py) plans burns with the ONBOARD model (pure CR3BP),
deliberately blind to the truth perturbations — that model gap is what the POMDP is
meant to absorb. This script asks the prior question: would a controller with a PERFECT
model even survive? It re-runs the same MPC but lets the burn solver plan with the SAME
EOM as the truth (the split is collapsed HERE, on purpose, as a diagnostic only — never
in the committed controller).

Interpretation:
  - If matched-model planning SURVIVES where blind-CR3BP fails → the killer is the model
    gap, and a better belief model (the POMDP) is the right lever.
  - If matched-model planning ALSO fails (same lifetime) → the killer is the raw orbital
    instability / control cadence, NOT model knowledge. No belief model can save it; the
    fix is a different orbit, a faster control cadence, or manifold-aware targeting.

Result (2026-06-22, symmetric period-3 IC, 7-day horizon):
    TRUTH = CR3BP+EncJ2 : blind→escape@78hr ; matched→crash@77hr  (≈ same lifetime)
    TRUTH = CR3BP+SatJ2 : blind→escape@18.6hr; matched→escape@18.6hr (IDENTICAL)
  => Instability-dominated. A perfect onboard model does not extend survival. The
     period-3 bifurcation orbit is too unstable for once-per-pass apse-targeting,
     regardless of model fidelity.

Usage:
    python -m scripts.mpc_planner_ablation [--days 7]
"""

import argparse

import baselines.mpc as mpc
from src.dynamics.cr3bp import cr3bp_eom
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.cr3bp_saturn_j2 import cr3bp_saturn_enc_j2_eom
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

ONE_REV_S = PERIOD3_PERIOD_S / 3.0

# (label, truth EOM, planner EOM). planner == truth means "perfect onboard model".
CASES = [
    ("EncJ2  | plan=CR3BP   (blind, production)", cr3bp_j2_eom,             cr3bp_eom),
    ("EncJ2  | plan=EncJ2   (perfect model)",     cr3bp_j2_eom,             cr3bp_j2_eom),
    ("SatJ2  | plan=CR3BP   (blind, production)", cr3bp_saturn_enc_j2_eom,  cr3bp_eom),
    ("SatJ2  | plan=SatJ2   (perfect model)",     cr3bp_saturn_enc_j2_eom,  cr3bp_saturn_enc_j2_eom),
]


def _solve_burn_with(plan_eom):
    """Wrap the real solve_burn so its planning EOM is forced to ``plan_eom``."""
    real = mpc.solve_burn

    def wrapped(state0, period_s, **kwargs):
        kwargs["eom"] = plan_eom
        return real(state0, period_s, **kwargs)

    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description="MPC planner-model ablation")
    parser.add_argument("--days", type=float, default=7.0)
    args = parser.parse_args()

    ic = _nd_to_phys(PERIOD3_IC_ND)
    horizon_s = args.days * 86400.0
    real_solve = mpc.solve_burn

    print(f"\nMPC planner-model ablation  (horizon {args.days:.1f} d, "
          f"symmetric period-3 IC)\n")
    print(f"{'case':<44}{'outcome':<9}{'lifetime':<11}{'burns':<7}{'ΔV (m/s)':<10}")
    print("-" * 81)
    try:
        for label, truth_eom, plan_eom in CASES:
            mpc.solve_burn = _solve_burn_with(plan_eom)
            r = mpc.run_mpc(ic, truth_eom=truth_eom, period_s=ONE_REV_S,
                            horizon_s=horizon_s, n_revs=3)
            print(f"{label:<44}{r['outcome']:<9}"
                  f"{r['survival_time_s']/3600:>7.1f} hr  "
                  f"{r['n_burns']:<7}{r['total_dv_ms']:<10.2f}")
    finally:
        mpc.solve_burn = real_solve  # restore — never leave the split collapsed

    print("\nIf 'perfect model' rows match the 'blind' rows, the failure is "
          "instability-\ndominated, not model-gap-dominated (see module docstring).\n")


if __name__ == "__main__":
    main()