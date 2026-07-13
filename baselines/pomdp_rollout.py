"""
High-fidelity rollout simulator for the SARSOP-generated stationkeeping policy.

Purpose (closes a circularity)
------------------------------
The toy POMDP was solved in Julia against a CRUDE drift+burn model, and the Julia
simulation only re-tested the policy against that SAME model — proving
self-consistency, not that the policy holds a REAL orbit. This module rolls the
SARSOP policy out against the actual Python CR3BP(+perturbation) TRUTH model,
reusing the production integrator and the Phase-2 noise models. It is the honest
test of whether the toy policy is worth anything, and it is directly comparable to
the deterministic MPC baseline (`baselines/mpc.py`) — same crash/escape
definitions, same period-3 IC, same fidelity rungs.

Architecture (decided 2026-07-06)
---------------------------------
The policy crosses the Julia→Python boundary exactly once, as a JSON file written
by `pomdp-julia/src/export_policy.jl` (alpha vectors + action labels + the toy T/O
tables + bin edges). Python loads it and does the evaluation. No live PyCall bridge.

Control model (mirrors MPC's cadence, MacKenzie Strategy 3)
-----------------------------------------------------------
One policy DECISION per periapsis approach (MPC fires one burn per inbound descent).
Per control step:
  1. Coast under the TRUTH EOM to the next periapsis, watching CONTINUOUSLY for the
     terminal crash (peri alt < 5 km) and physical escape (alt > ESCAPE_ALT_KM)
     events — so escape during the apoapsis coast is caught by the integrator, not
     missed because we only sample at periapsis.
  2. At periapsis, map the true periapsis altitude → toy bin, draw a NOISY altitude
     observation (`nav.observe_altitude`, σ = SIGMA_NAV_POS) → observed bin.
  3. Update the discrete belief with the standard Bayes filter using the exported
     T/O tables (this is exactly POMDPTools' DiscreteUpdater).
  4. Query the SARSOP policy (argmax over alpha·belief) for an action.
  5. Convert the action to a physical burn and execute it via `thruster.apply_dv_noisy`
     (η_eff ~ Uniform(0.8, 1.0)); the burn is a periapsis-raise (see action_to_dv).

Terminal escape (empirically motivated — see docs/session-log/2026-07-06-pomdp-rollout.md)
------------------------------------------------------------------------------------------
The toy's ">120 km periapsis = ESCAPED" bin is a scalar-altitude artifact. For this
unstable orbit, escape is the whole ellipse ballooning out (apoapsis → thousands of
km); an uncontrolled run showed periapsis and apoapsis grow TOGETHER, so the toy bin
fires but LATE (and under Saturn J2 the first post-decay periapsis already reads
~241 km, skipping past HIGH). We therefore use the SAME physical escape as MPC
(ascend past ESCAPE_ALT_KM ≈ 5550 km) as the terminal condition — earlier and
comparable — and ALSO log the toy bin each step so the report shows the agreement/lag.

TRUTH/ONBOARD split (CLAUDE.md): the world is integrated ONLY under `truth_eom`.
The policy's "onboard model" is the discrete belief filter over the toy bins — it
never sees the truth perturbations. That model-gap is the uncertainty under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from src.constants import R_ENCELADUS, SIGMA_NAV_POS, PERIAPSIS_CRASH_ALT
from src.dynamics.cr3bp import X_ENCELADUS
from src.dynamics.integrator import (
    propagate,
    make_periapsis_event,
    make_crash_event,
    RTOL_TRUTH,
    ATOL_TRUTH,
)
from baselines.mpc import _altitude_km, _make_escape_event, ESCAPE_ALT_KM
from src.spacecraft import nav, thruster


# Default location of the exported SARSOP policy (written by the Julia exporter).
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "policy" / "sarsop_policy.json"


# ── Policy container ──────────────────────────────────────────────────────────
class SarsopPolicy:
    """
    The solved SARSOP policy + the toy T/O tables, loaded from the Julia export.

    Reproduces POMDPTools' AlphaVectorPolicy query and DiscreteUpdater belief
    update in Python, so the policy is exercised exactly as it was solved.
    """

    def __init__(self, data: dict) -> None:
        self.states: list[str] = list(data["states"])
        self.actions: list[str] = list(data["actions"])
        self.observations: list[str] = list(data["observations"])
        self.terminal_states: set[str] = set(data["terminal_states"])
        self.initial_state: str = data["initial_state"]
        self.discount: float = float(data["discount"])
        self.bin_edges: list[float] = list(data["bin_edges"])
        self.action_dv_cost: dict[str, float] = {
            k: float(v) for k, v in data["action_dv_cost"].items()
        }
        self.action_raise_km: dict[str, float] = {
            k: float(v) for k, v in data["action_raise_km"].items()
        }
        self.alphas: np.ndarray = np.asarray(data["alphas"], dtype=float)  # (n_alpha, |S|)
        # exported action indices are 1-based (Julia); store 0-based.
        self.alpha_actions: np.ndarray = np.asarray(data["alpha_actions"], dtype=int) - 1
        self.T: np.ndarray = np.asarray(data["T"], dtype=float)  # [s, a, s']
        self.O: np.ndarray = np.asarray(data["O"], dtype=float)  # [s', o]
        self.meta: dict = data.get("meta", {})

        self.n_states = len(self.states)
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.action_index = {a: i for i, a in enumerate(self.actions)}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY_PATH) -> "SarsopPolicy":
        """Load a SarsopPolicy from the JSON file written by export_policy.jl."""
        with open(path) as f:
            return cls(json.load(f))

    def initial_belief(self) -> np.ndarray:
        """Deterministic belief on the exported initial state (NOMINAL)."""
        b = np.zeros(self.n_states)
        b[self.state_index[self.initial_state]] = 1.0
        return b

    def action(self, belief: np.ndarray) -> str:
        """
        Greedy action for a belief: argmax over alpha vectors of (alpha · belief),
        returning the action attached to the dominating alpha (AlphaVectorPolicy).
        """
        values = self.alphas @ belief
        best_alpha = int(np.argmax(values))
        return self.actions[self.alpha_actions[best_alpha]]

    def update_belief(self, belief: np.ndarray, action: str, obs_bin: str) -> np.ndarray:
        """
        Discrete Bayes filter (POMDPTools DiscreteUpdater):
            b'(s') ∝ O[s', o] · Σ_s T[s, a, s'] · b(s)

        Parameters
        ----------
        belief : np.ndarray, shape (|S|,)
            Current belief over states.
        action : str
            Action just taken (indexes the transition table).
        obs_bin : str
            Observed bin label (indexes the observation table).

        Returns
        -------
        np.ndarray, shape (|S|,) — the normalized posterior belief.
        """
        ai = self.action_index[action]
        oi = self.state_index[obs_bin]  # observations share the state labels
        # Predict: bp(s') = Σ_s T[s,a,s'] b(s).
        bp = belief @ self.T[:, ai, :]
        # Correct: multiply by O[s', o].
        bp = bp * self.O[:, oi]
        total = bp.sum()
        if total <= 0.0:
            # Degenerate observation (should not happen with the toy's smooth O);
            # fall back to the prediction to avoid a divide-by-zero.
            bp = belief @ self.T[:, ai, :]
            total = bp.sum()
        return bp / total

    def bin_of(self, alt_km: float) -> str:
        """Map an absolute periapsis altitude (km) to its toy bin label.

        Replicates `bin_of` in pomdp-julia/src/dynamics.jl EXACTLY (same edges,
        same half-open [lo, hi) convention)."""
        e = self.bin_edges
        if alt_km < e[0]:
            return "CRASHED"
        elif alt_km < e[1]:
            return "LOW"
        elif alt_km < e[2]:
            return "NOMINAL"
        elif alt_km < e[3]:
            return "HIGH"
        else:
            return "ESCAPED"


# ── Action → physical burn ─────────────────────────────────────────────────────
def action_to_dv(
    action: str,
    state: np.ndarray,
    policy: SarsopPolicy,
) -> np.ndarray:
    """
    Convert a toy policy action into a physical 3-D commanded ΔV (km/s) at a
    periapsis state.

    The toy action is an abstract "raise periapsis altitude by ACTION_RAISE_KM".
    We realize it physically as a PROGRADE burn (ΔV along the current velocity
    direction) that raises the orbit — the standard way to raise the OPPOSITE apse.
    We size it from the vis-viva-like sensitivity of periapsis altitude to a
    prograde impulse at periapsis; because the toy raise (18 / 45 km) is a bin-scale
    quantity rather than a physically fitted number, this is a first-order mapping,
    not a calibrated burn. NO_BURN returns a zero vector.

    Sizing rationale
    ----------------
    At periapsis a small prograde ΔV mainly raises the apoapsis; to first order the
    change in the OPPOSITE-apse radius per unit ΔV is 2/n where n is roughly the
    local mean motion. We use a compact, dimensionally-consistent proxy: scale the
    commanded raise Δh (km) by v_peri via  |ΔV| ≈ Δh · (v_peri) / (2 r_peri)  — the
    linearized two-body apse-raise relation (Δr_apo ≈ (2 r² /μ) v Δv → invert). This
    keeps the burn magnitude on the right order (m/s) without pretending to be a
    fitted controller; the honest test is whether even a plausibly-sized burn from
    this hand-tuned policy can hold the orbit.

    Parameters
    ----------
    action : str
        One of the policy's action labels (NO_BURN / SMALL_BURN / LARGE_BURN).
    state : np.ndarray, shape (6,)
        Barycentre-frame periapsis state [x,y,z,vx,vy,vz] (km, km/s).
    policy : SarsopPolicy
        Provides the per-action commanded raise (km).

    Returns
    -------
    np.ndarray, shape (3,) — commanded ΔV (km/s), prograde.
    """
    raise_km = policy.action_raise_km[action]
    if raise_km <= 0.0:
        return np.zeros(3)

    v = state[3:]
    v_mag = float(np.linalg.norm(v))
    if v_mag == 0.0:
        return np.zeros(3)

    dx = state[0] - X_ENCELADUS
    r_peri = float(np.sqrt(dx**2 + state[1] ** 2 + state[2] ** 2))

    # Linearized apse-raise: |ΔV| ≈ Δh · v_peri / (2 r_peri).  (km/s)
    dv_mag = raise_km * v_mag / (2.0 * r_peri)
    return (v / v_mag) * dv_mag


# ── Rollout ─────────────────────────────────────────────────────────────────────
def run_pomdp_rollout(
    state0: np.ndarray,
    truth_eom: Callable,
    period_s: float,
    horizon_s: float,
    policy: SarsopPolicy,
    rng: np.random.Generator | None = None,
    sigma_nav_km: float = SIGMA_NAV_POS,
    rtol_truth: float = RTOL_TRUTH,
    atol_truth: float = ATOL_TRUTH,
    max_steps: int = 2000,
    verbose: bool = False,
) -> dict:
    """
    Roll the SARSOP policy out against a TRUTH model over a mission horizon.

    The world propagates under ``truth_eom`` (the ONLY place it is integrated). One
    policy decision is taken per periapsis approach. Crash and physical escape are
    watched continuously by terminal integrator events on every leg (including the
    apoapsis coast). Metrics mirror ``baselines.mpc.run_mpc`` for direct comparison.

    Parameters
    ----------
    state0 : np.ndarray, shape (6,)
        Initial barycentre-frame state (km, km/s), typically the period-3 IC.
    truth_eom : Callable
        World dynamics (cr3bp_j2_eom, cr3bp_saturn_enc_j2_eom, or future SPICE).
        Passed in so the SAME harness runs at every fidelity rung.
    period_s : float
        Single-revolution period estimate (s); sets the per-leg time cap.
    horizon_s : float
        Total mission horizon to simulate (s).
    policy : SarsopPolicy
        The loaded SARSOP policy + toy tables.
    rng : np.random.Generator, optional
        RNG for the nav-observation noise and burn η_eff (reproducibility).
    sigma_nav_km : float
        Nav 1-σ altitude noise (km).
    rtol_truth, atol_truth : float
        Truth-model integration tolerances.
    max_steps : int
        Hard cap on control steps (safety guard).
    verbose : bool
        Per-step diagnostics.

    Returns
    -------
    dict with keys (superset-compatible with run_mpc):
        'survived'        : bool
        'outcome'         : str — 'held' / 'idle' / 'crash' / 'escape' / 'max_steps'
        'survival_time_s' : float — time of loss, or horizon_s
        'n_burns'         : int — non-zero burns executed
        'n_steps'         : int — policy decisions taken
        'total_dv_ms'     : float — cumulative APPLIED ΔV (m/s)
        'min_peri_alt_km' : float — smallest periapsis altitude seen (km)
        'steps'           : list[dict] — per-step records
                            {t_s, peri_alt_km, true_bin, obs_bin, action, dv_ms, eta_eff}
    """
    if rng is None:
        rng = np.random.default_rng()

    state = state0.copy()
    t_now = 0.0
    total_dv_ms = 0.0
    n_burns = 0
    min_peri_alt = np.inf
    steps: list[dict] = []
    belief = policy.initial_belief()

    def _terminal_return(outcome: str, t_loss: float) -> dict:
        return {
            "survived": outcome not in ("crash", "escape", "max_steps"),
            "outcome": outcome,
            "survival_time_s": t_loss,
            "n_burns": n_burns,
            "n_steps": len(steps),
            "total_dv_ms": total_dv_ms,
            "min_peri_alt_km": float(min_peri_alt) if np.isfinite(min_peri_alt) else np.nan,
            "steps": steps,
        }

    while t_now < horizon_s:
        if len(steps) >= max_steps:
            return _terminal_return("max_steps", t_now)

        peri_ev = make_periapsis_event(terminal=True)
        crash_ev = make_crash_event(PERIAPSIS_CRASH_ALT)
        escape_ev = _make_escape_event()

        sol = propagate(
            truth_eom, state, (0.0, horizon_s - t_now),
            events=[peri_ev, crash_ev, escape_ev],
            rtol=rtol_truth, atol=atol_truth,
        )

        # Escape (event 2) — checked FIRST: if the orbit left the science region on
        # this leg (during the apoapsis coast), that is the loss, regardless of a
        # later periapsis event that scipy may also have queued.
        if len(sol.t_events[2]) > 0:
            t_esc = t_now + float(sol.t_events[2][0])
            if verbose:
                print(f"  ESCAPE at t={t_esc/3600:.2f} hr after {len(steps)} steps")
            return _terminal_return("escape", t_esc)

        # Crash (event 1).
        if len(sol.t_events[1]) > 0:
            t_crash = t_now + float(sol.t_events[1][0])
            min_peri_alt = min(min_peri_alt, PERIAPSIS_CRASH_ALT)
            if verbose:
                print(f"  CRASH at t={t_crash/3600:.2f} hr after {len(steps)} steps")
            return _terminal_return("crash", t_crash)

        # Periapsis (event 0)?
        if len(sol.t_events[0]) == 0:
            # No periapsis before the horizon and no terminal event: the orbit
            # drifted off without returning. Survived (no crash) but idle.
            break

        t_peri = float(sol.t_events[0][0])
        state_peri = sol.y_events[0][0].copy()
        t_now += t_peri

        # 1. True periapsis altitude → true bin.
        peri_alt = _altitude_km(state_peri)
        min_peri_alt = min(min_peri_alt, peri_alt)
        true_bin = policy.bin_of(peri_alt)

        # 2. Noisy observation → observed bin.
        obs_alt = nav.observe_altitude(peri_alt, sigma_r=sigma_nav_km, rng=rng)
        obs_bin = policy.bin_of(obs_alt)

        # 3. Query the policy from the CURRENT belief, then 4. update the belief with
        #    (action, obs) — the standard predict-then-query-then-correct order used
        #    by a POMDP agent: it acts on the belief it holds entering periapsis.
        action = policy.action(belief)

        # 5. Convert to a physical burn and execute noisily.
        dv_cmd = action_to_dv(action, state_peri, policy)
        dv_applied, eta_eff = thruster.apply_dv_noisy(dv_cmd, rng=rng)
        dv_ms = float(np.linalg.norm(dv_applied) * 1.0e3)
        state_post = state_peri.copy()
        state_post[3:] = state_post[3:] + dv_applied
        total_dv_ms += dv_ms
        if dv_ms > 0.0:
            n_burns += 1

        # Advance the belief for the NEXT decision (Bayes filter on this obs+action).
        belief = policy.update_belief(belief, action, obs_bin)

        steps.append({
            "t_s": t_now,
            "peri_alt_km": peri_alt,
            "true_bin": true_bin,
            "obs_bin": obs_bin,
            "action": action,
            "dv_ms": dv_ms,
            "eta_eff": eta_eff,
        })
        if verbose:
            print(f"  step {len(steps):3d} @ t={t_now/3600:7.2f} hr  "
                  f"peri={peri_alt:8.1f} km  true={true_bin:8s} obs={obs_bin:8s}  "
                  f"{action:11s} ΔV={dv_ms:7.3f} m/s")

        state = state_post

    # Reached the horizon with no crash/escape.
    idle = t_now < horizon_s
    return {
        "survived": True,
        "outcome": "idle" if idle else "held",
        "survival_time_s": horizon_s,
        "n_burns": n_burns,
        "n_steps": len(steps),
        "total_dv_ms": total_dv_ms,
        "min_peri_alt_km": float(min_peri_alt) if np.isfinite(min_peri_alt) else np.nan,
        "steps": steps,
    }