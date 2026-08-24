"""
SARSOP-policy rollout vs the real truth model — the honest test of the toy policy.

Rolls the SARSOP-generated stationkeeping policy (solved in Julia against the crude
drift+burn toy) out against each EXISTING Python truth model, over a 30-day horizon,
and tabulates the same metrics as the MPC feasibility check so the two are directly
comparable:
  - survival rate over N seeded rollouts (the policy + nav noise are stochastic)
  - min periapsis altitude (km)
  - total ΔV (m/s)

For reference, the deterministic MPC baseline (baselines/mpc.py) is run once per rung
alongside, so the report shows POMDP-policy vs MPC side by side at each fidelity rung.
The MPC baseline uses MacKenzie Strategy 3 (mode="position", apse position-vector
targeting) — the paper's WORKING controller, which HOLDS the period-3 halo for 30 days
on CR3BP+EncJ2 (60 burns, 76 m/s; see docs/session-log/2026-06-22b.md §11). Apse-
altitude targeting (Strategy 1/2) is the mode MacKenzie says fails, and it escapes in
~3.25 d — so it is the wrong baseline to compare the POMDP against.

Fidelity rungs are a CONFIG LIST (TRUTH_MODELS); adding the future SPICE truth is one
more entry, not a copy-paste.

Usage:
    python -m scripts.pomdp_rollout_feasibility                 # 30-day, 20 rollouts
    python -m scripts.pomdp_rollout_feasibility --days 30 --n 50
    python -m scripts.pomdp_rollout_feasibility --verbose       # per-step trace (n=1)

Honesty note: the toy policy's action tuning is hand-picked, not physics-fitted, and
its HIGH bin is a known dead-end artifact. A poor rollout result is EXPECTED and is a
valid, important finding (see docs/session-log/2026-07-06-julia-pomdp.md §5). Do not
tune the toy to make this look good.
"""

import argparse

import numpy as np

from baselines.mpc import run_mpc
from baselines.pomdp_rollout import SarsopPolicy, run_pomdp_rollout
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.cr3bp_saturn_j2 import cr3bp_saturn_enc_j2_eom
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

# One "rev" between periapsis approaches ≈ T/3 (3 periapsis passes per period-3 orbit).
ONE_REV_S = PERIOD3_PERIOD_S / 3.0

# Fidelity ladder — add the SPICE inertial truth here as one more (name, eom) entry.
TRUTH_MODELS = [
    ("CR3BP + EncJ2",            cr3bp_j2_eom),
    ("CR3BP + SaturnJ2 + EncJ2", cr3bp_saturn_enc_j2_eom),
]


def summarize(results: list[dict]) -> dict:
    """Aggregate a list of rollout result dicts into survival-rate + metric stats."""
    n = len(results)
    survived = sum(1 for r in results if r["outcome"] in ("held", "idle"))
    min_peris = [r["min_peri_alt_km"] for r in results if np.isfinite(r["min_peri_alt_km"])]
    dvs = [r["total_dv_ms"] for r in results]
    times = [r["survival_time_s"] for r in results]
    return {
        "n": n,
        "survival_rate": survived / n if n else np.nan,
        "min_peri_median": float(np.median(min_peris)) if min_peris else np.nan,
        "dv_median": float(np.median(dvs)) if dvs else np.nan,
        "surv_time_median_d": float(np.median(times)) / 86400.0 if times else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SARSOP-policy rollout feasibility check")
    parser.add_argument("--days", type=float, default=30.0,
                        help="Mission horizon in days (default 30).")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of seeded rollouts per rung (default 20).")
    parser.add_argument("--seed", type=int, default=20260706,
                        help="Base RNG seed.")
    parser.add_argument("--verbose", action="store_true",
                        help="Per-step trace (forces n=1).")
    args = parser.parse_args()

    ic = _nd_to_phys(PERIOD3_IC_ND)
    horizon_s = args.days * 86400.0
    n_roll = 1 if args.verbose else args.n
    policy = SarsopPolicy.load()

    print(f"\nSARSOP-policy rollout vs real truth model")
    print(f"  policy: {policy.meta.get('solver', '?')} toy "
          f"(bins {policy.bin_edges}, states {policy.states})")
    print(f"  IC: period-3 L1 halo (north-pole periapsis, symmetric degenerate)")
    print(f"  horizon: {args.days:.1f} days   rollouts/rung: {n_roll}")
    print(f"  control shell: periapsis approach; crash 5 km; "
          f"escape > {5.0 * 1110.0:.0f} km (MPC-style)\n")

    pomdp_rows = []
    mpc_rows = []
    for name, eom in TRUTH_MODELS:
        print(f"--- {name} ---")

        # Deterministic MPC baseline (one run — it is deterministic).
        # Use MacKenzie Strategy 3 (mode="position"): bound the full apse POSITION
        # vectors |r_apse − r_nom| < 1 km. This is the paper's WORKING strategy and
        # the correct baseline — apse-ALTITUDE targeting (Strategy 1/2) is the one
        # MacKenzie says fails, and it does (escapes ~3.25 d on the period-3 halo).
        # See docs/session-log/2026-06-22b.md §11.
        mpc = run_mpc(ic, truth_eom=eom, period_s=ONE_REV_S,
                      horizon_s=horizon_s, n_revs=3,
                      mode="position", ref_ic=ic)
        mpc_rows.append((name, mpc))
        print(f"  MPC-S3: outcome={mpc['outcome'].upper():6s} "
              f"loss/horizon t={mpc['survival_time_s']/86400:5.2f} d  "
              f"{mpc['n_burns']:3d} burns  ΔV={mpc['total_dv_ms']:8.2f} m/s  "
              f"min peri={mpc['min_peri_alt_km']:.1f} km")

        # SARSOP-policy rollouts (N seeded — stochastic nav + burns).
        results = []
        for k in range(n_roll):
            rng = np.random.default_rng(args.seed + k)
            r = run_pomdp_rollout(ic, truth_eom=eom, period_s=ONE_REV_S,
                                  horizon_s=horizon_s, policy=policy, rng=rng,
                                  verbose=args.verbose)
            results.append(r)
        s = summarize(results)
        pomdp_rows.append((name, s))
        print(f"  POMDP : survival={s['survival_rate']*100:5.1f}%  "
              f"surv t(med)={s['surv_time_median_d']:5.2f} d  "
              f"ΔV(med)={s['dv_median']:8.2f} m/s  "
              f"min peri(med)={s['min_peri_median']:.1f} km  (n={s['n']})\n")

    # Summary table.
    print("=" * 92)
    print(f"{'Truth model':<28}{'Method':<8}{'Survival':<12}{'Surv t':<12}"
          f"{'ΔV (m/s)':<12}{'Min peri':<10}")
    print("-" * 92)
    for (name, mpc), (_, s) in zip(mpc_rows, pomdp_rows):
        mpc_surv = "yes" if mpc["outcome"] in ("held", "idle") else "no"
        print(f"{name:<28}{'MPC':<8}{mpc_surv:<12}"
              f"{mpc['survival_time_s']/86400:<12.2f}"
              f"{mpc['total_dv_ms']:<12.2f}{mpc['min_peri_alt_km']:<10.1f}")
        pomdp_surv = f"{s['survival_rate']*100:.1f}%"
        print(f"{'':<28}{'POMDP':<8}{pomdp_surv:<12}"
              f"{s['surv_time_median_d']:<12.2f}"
              f"{s['dv_median']:<12.2f}{s['min_peri_median']:<10.1f}")
    print("=" * 92)
    print("\nToy policy tuning is hand-picked, NOT physics-fitted (HIGH bin is a known")
    print("dead-end artifact). A poor rollout is an expected, valid result.\n")


if __name__ == "__main__":
    main()