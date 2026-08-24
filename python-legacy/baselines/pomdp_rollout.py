"""
High-fidelity rollout of the science+safety SARSOP policy vs the real CR3BP truth.

Closes the same circularity as before (the 2026-07-06 version): the POMDP is solved in
Julia against MEASURED but discrete tables; this rolls the resulting policy out against
the actual Python CR3BP(+perturbation) truth model, so we test whether it holds the
orbit AND gathers the science coverage it planned. Directly comparable to the MPC
baseline (baselines/mpc.py) — same period-3 IC, same crash/escape definitions.

Policy (2026-07-15, science+safety)
-----------------------------------
State  = (dev, cov):
  dev = apse-POSITION deviation |r_peri - r_peri_nom| binned OK/DRIFT/FAR/LOST(+CRASHED)
        — the SAFETY variable (measured, exps 02/11b).
  cov = 3-bit mask over science altitude bands LOW/MID/HIGH — the SCIENCE variable.
Actions (INTENT; the burn VECTOR is solved live by mpc.solve_burn — exp 04 showed a
fixed-direction menu fails):
  OBSERVE      : no burn (nav reading only).
  CORRECT      : solve_burn(mode="position") toward the nominal apses (holds the orbit).
  EXCURSE_LOW  : one-pass single-burn excursion toward band k (position-target at the
  EXCURSE_MID    band's scaled periapsis), banking that band into cov if it survives.
  EXCURSE_HIGH

Belief is over the 26 discrete states; the observation informs only the dev bin (cov is
known exactly from the actions we commanded), which the standard discrete Bayes filter
handles because O is defined over dev-observations.

TRUTH/ONBOARD split (CLAUDE.md): the world is integrated ONLY under `truth_eom`; the
policy's onboard model is the discrete belief filter over dev bins. solve_burn plans on
the onboard CR3BP. The exact ΔV vector crosses no language boundary — only the policy
JSON does, once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from src.constants import R_ENCELADUS, SIGMA_NAV_POS, PERIAPSIS_CRASH_ALT
from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate,
    make_periapsis_event,
    make_crash_event,
    make_altitude_event,
    RTOL_TRUTH,
    ATOL_TRUTH,
)
from baselines.mpc import (
    _altitude_km,
    _make_escape_event,
    ESCAPE_ALT_KM,
    CONTROL_ALT_KM,
    solve_burn,
    nominal_apse_positions,
)
from src.spacecraft import nav, thruster
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys


DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "policy" / "sarsop_policy.json"

# One rev between periapsis approaches ≈ T/3 (period-3 orbit).
ONE_REV_S = PERIOD3_PERIOD_S / 3.0


# ── Policy container ──────────────────────────────────────────────────────────
class SarsopPolicy:
    """The solved science+safety SARSOP policy + measured tables, from the Julia export.

    Reproduces the AlphaVectorPolicy query and the DiscreteUpdater belief filter in
    Python, over the (dev, cov) state space.
    """

    def __init__(self, data: dict) -> None:
        self.states: list[str] = list(data["states"])            # "DEV|cov<mask>"
        self.state_dev: list[str] = list(data["state_dev"])      # per-state dev label
        self.state_cov: list[int] = [int(c) for c in data["state_cov"]]
        self.actions: list[str] = list(data["actions"])
        self.observations: list[str] = list(data["observations"])  # dev-bin obs labels
        self.terminal_states: set[str] = set(data["terminal_states"])
        self.initial_state: str = data["initial_state"]
        self.discount: float = float(data["discount"])
        self.dev_edges: list[float] = list(data["dev_edges"])      # (15, 60, 200) km
        self.band_names: list[str] = list(data["band_names"])      # LOW/MID/HIGH
        self.band_target_km: dict[str, float] = {
            k: float(v) for k, v in data["band_target_km"].items()
        }
        self.action_dv_cost: dict[str, float] = {
            k: float(v) for k, v in data["action_dv_cost"].items()
        }
        self.alphas: np.ndarray = np.asarray(data["alphas"], dtype=float)  # (n_alpha,|S|)
        self.alpha_actions: np.ndarray = np.asarray(data["alpha_actions"], dtype=int) - 1
        self.T: np.ndarray = np.asarray(data["T"], dtype=float)    # [s, a, s']
        self.O: np.ndarray = np.asarray(data["O"], dtype=float)    # [s, o] over dev obs
        self.meta: dict = data.get("meta", {})

        self.n_states = len(self.states)
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.action_index = {a: i for i, a in enumerate(self.actions)}
        self.obs_index = {o: i for i, o in enumerate(self.observations)}
        # Map an EXCURSE action to its band index (1-based, matching the cov bitmask).
        self.excurse_band = {
            f"EXCURSE_{b}": i + 1 for i, b in enumerate(self.band_names)
        }

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY_PATH) -> "SarsopPolicy":
        with open(path) as f:
            return cls(json.load(f))

    # -- state label helpers --------------------------------------------------
    @staticmethod
    def make_label(dev: str, cov: int) -> str:
        return f"{dev}|cov{cov}"

    def initial_belief(self) -> np.ndarray:
        b = np.zeros(self.n_states)
        b[self.state_index[self.initial_state]] = 1.0
        return b

    def action(self, belief: np.ndarray) -> str:
        values = self.alphas @ belief
        return self.actions[self.alpha_actions[int(np.argmax(values))]]

    def update_belief(self, belief: np.ndarray, action: str, obs_dev: str) -> np.ndarray:
        """Discrete Bayes filter b'(s') ∝ O[s',o] Σ_s T[s,a,s'] b(s), o over dev bins."""
        ai = self.action_index[action]
        oi = self.obs_index[obs_dev]
        bp = belief @ self.T[:, ai, :]
        bp = bp * self.O[:, oi]
        total = bp.sum()
        if total <= 0.0:
            bp = belief @ self.T[:, ai, :]
            total = bp.sum()
        return bp / total

    def dev_bin(self, dev_km: float) -> str:
        """Bin an apse-position deviation (km). Matches dynamics.jl dev_bin exactly."""
        e = self.dev_edges
        if not np.isfinite(dev_km):
            return "LOST"
        if dev_km < e[0]:
            return "OK"
        if dev_km < e[1]:
            return "DRIFT"
        if dev_km < e[2]:
            return "FAR"
        return "LOST"


# ── Geometry helpers (Enceladus-relative) ───────────────────────────────────────
def _enc_rel(r: np.ndarray) -> np.ndarray:
    r = r.copy()
    r[0] -= X_ENCELADUS
    return r


def _scale_to_alt(r_nom: np.ndarray, alt_km: float) -> np.ndarray:
    """Scale a nominal apse position vector radially (about Enceladus) to `alt_km`."""
    rr = _enc_rel(r_nom)
    rr2 = rr * ((R_ENCELADUS + alt_km) / np.linalg.norm(rr))
    out = rr2.copy()
    out[0] += X_ENCELADUS
    return out


# ── Rollout ─────────────────────────────────────────────────────────────────────
def run_pomdp_rollout(
    state0: np.ndarray,
    truth_eom: Callable,
    period_s: float,
    horizon_s: float,
    policy: SarsopPolicy,
    ref_ic: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    sigma_nav_km: float = SIGMA_NAV_POS,
    rtol_truth: float = RTOL_TRUTH,
    atol_truth: float = ATOL_TRUTH,
    max_steps: int = 2000,
    verbose: bool = False,
) -> dict:
    """
    Roll the science+safety SARSOP policy out against a TRUTH model.

    One decision per periapsis approach. Each control step:
      1. Coast under truth to the 600-km descending shell (watching crash/escape).
      2. Query the policy from the current belief → action.
      3. Realize the action: CORRECT → solve_burn(position) toward nominal;
         EXCURSE_k → solve_burn(position) toward band k's scaled periapsis (banking the
         band into cov if the pass survives); OBSERVE → no burn.
      4. Execute the burn noisily (thruster.apply_dv_noisy), coast to the next periapsis.
      5. Measure the true apse deviation → dev bin; draw a noisy obs; update belief.

    Metrics mirror run_mpc, plus science coverage. Returns a dict.
    """
    if rng is None:
        rng = np.random.default_rng()
    ref = state0 if ref_ic is None else ref_ic
    r_peri_nom, r_apo_nom = nominal_apse_positions(ref, period_s, eom=cr3bp_eom)
    apo_nom_alt = np.linalg.norm(_enc_rel(r_apo_nom)) - R_ENCELADUS

    def dev_of(peri_state: np.ndarray) -> float:
        return float(np.linalg.norm(peri_state[:3] - r_peri_nom))

    state = state0.copy()
    t_now = 0.0
    total_dv_ms = 0.0
    n_burns = 0
    min_peri_alt = np.inf
    cov = 0                      # science coverage bitmask
    belief = policy.initial_belief()
    steps: list[dict] = []

    def _terminal(outcome: str, t_loss: float) -> dict:
        return {
            "survived": outcome not in ("crash", "escape", "max_steps"),
            "outcome": outcome,
            "survival_time_s": t_loss,
            "n_burns": n_burns,
            "n_steps": len(steps),
            "total_dv_ms": total_dv_ms,
            "min_peri_alt_km": float(min_peri_alt) if np.isfinite(min_peri_alt) else np.nan,
            "science_cov": cov,
            "n_bands": bin(cov).count("1"),
            "steps": steps,
        }

    def _coast_to_shell(s: np.ndarray, horizon: float):
        ctrl = make_altitude_event(CONTROL_ALT_KM, terminal=True)
        crash = make_crash_event(PERIAPSIS_CRASH_ALT)
        esc = _make_escape_event()
        sol = propagate(truth_eom, s, (0.0, horizon), events=[ctrl, crash, esc],
                        rtol=rtol_truth, atol=atol_truth)
        if len(sol.t_events[1]):
            return "crash", float(sol.t_events[1][0])
        if len(sol.t_events[2]):
            return "escape", float(sol.t_events[2][0])
        if not len(sol.t_events[0]):
            return "none", horizon
        return sol.y_events[0][0].copy(), float(sol.t_events[0][0])

    def _coast_to_peri(s: np.ndarray, horizon: float):
        peri = make_periapsis_event(terminal=True)
        crash = make_crash_event(PERIAPSIS_CRASH_ALT)
        esc = _make_escape_event()
        sol = propagate(truth_eom, s, (0.0, horizon), events=[peri, crash, esc],
                        rtol=rtol_truth, atol=atol_truth)
        if len(sol.t_events[1]):
            return "crash", float(sol.t_events[1][0])
        if len(sol.t_events[2]):
            return "escape", float(sol.t_events[2][0])
        if not len(sol.t_events[0]):
            return "none", horizon
        return sol.y_events[0][0].copy(), float(sol.t_events[0][0])

    while t_now < horizon_s:
        if len(steps) >= max_steps:
            return _terminal("max_steps", t_now)

        # 1. Coast to the control shell.
        sc, dt = _coast_to_shell(state, horizon_s - t_now)
        if isinstance(sc, str):
            if sc in ("crash", "escape"):
                if sc == "crash":
                    min_peri_alt = min(min_peri_alt, PERIAPSIS_CRASH_ALT)
                return _terminal(sc, t_now + dt)
            break  # 'none': no shell before horizon → survived, idle
        t_now += dt

        # 2. Query the policy.
        action = policy.action(belief)

        # 3. Realize the action into a commanded ΔV.
        dv_cmd = np.zeros(3)
        band_attempt = 0
        if action == "CORRECT":
            b = solve_burn(sc, period_s, n_revs=3, eom=cr3bp_eom, mode="position",
                           r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
            dv_cmd = b["dv"]
        elif action.startswith("EXCURSE_"):
            band_attempt = policy.excurse_band[action]
            band_name = policy.band_names[band_attempt - 1]
            rp = _scale_to_alt(r_peri_nom, policy.band_target_km[band_name])
            ra = _scale_to_alt(r_apo_nom, apo_nom_alt)
            b = solve_burn(sc, period_s, n_revs=3, eom=cr3bp_eom, mode="position",
                           r_peri_nom=rp, r_apo_nom=ra)
            dv_cmd = b["dv"]
        # OBSERVE → zero ΔV.

        # 4. Execute noisily and coast to the next periapsis.
        dv_applied, eta_eff = thruster.apply_dv_noisy(dv_cmd, rng=rng)
        dv_ms = float(np.linalg.norm(dv_applied) * 1.0e3)
        total_dv_ms += dv_ms
        if dv_ms > 0.0:
            n_burns += 1
        s_post = sc.copy()
        s_post[3:] = s_post[3:] + dv_applied

        pr, dt2 = _coast_to_peri(s_post, horizon_s - t_now)
        if isinstance(pr, str):
            if pr in ("crash", "escape"):
                if pr == "crash":
                    min_peri_alt = min(min_peri_alt, PERIAPSIS_CRASH_ALT)
                # record the step's action before terminating
                steps.append({"t_s": t_now, "action": action, "dv_ms": dv_ms,
                              "true_dev_km": np.nan, "true_bin": pr.upper(),
                              "obs_bin": pr.upper(), "cov": cov, "eta_eff": eta_eff})
                return _terminal(pr, t_now + dt2)
            break
        t_now += dt2

        # 5. Measure true deviation → dev bin; bank science if the excursion survived.
        peri_alt = _altitude_km(pr)
        min_peri_alt = min(min_peri_alt, peri_alt)
        dev_km = dev_of(pr)
        true_dev = policy.dev_bin(dev_km)
        if band_attempt != 0:
            cov |= (1 << (band_attempt - 1))     # banked this band

        # Noisy observation of the deviation (nav noise on the apse deviation magnitude).
        obs_dev_km = dev_km + rng.normal(0.0, sigma_nav_km)
        obs_dev = policy.dev_bin(abs(obs_dev_km))

        # Belief update at the NEW cov: fold the observation in, then re-concentrate the
        # belief onto the known-cov block (cov is observed exactly from our actions).
        belief = policy.update_belief(belief, action, obs_dev)
        _reconcentrate_cov(belief, policy, cov)

        steps.append({"t_s": t_now, "action": action, "dv_ms": dv_ms,
                      "true_dev_km": dev_km, "true_bin": true_dev,
                      "obs_bin": obs_dev, "cov": cov, "eta_eff": eta_eff})
        if verbose:
            print(f"  step {len(steps):3d} t={t_now/3600:7.2f}h  dev={dev_km:7.1f}km "
                  f"({true_dev:5s})  {action:12s} ΔV={dv_ms:6.2f}  cov={cov:03b}")

        state = pr

    # Reached horizon with no crash/escape.
    idle = t_now < horizon_s
    return {
        "survived": True,
        "outcome": "idle" if idle else "held",
        "survival_time_s": horizon_s,
        "n_burns": n_burns,
        "n_steps": len(steps),
        "total_dv_ms": total_dv_ms,
        "min_peri_alt_km": float(min_peri_alt) if np.isfinite(min_peri_alt) else np.nan,
        "science_cov": cov,
        "n_bands": bin(cov).count("1"),
        "steps": steps,
    }


def _reconcentrate_cov(belief: np.ndarray, policy: SarsopPolicy, cov: int) -> None:
    """Zero out belief mass on states whose cov != the known cov (or terminal), then
    renormalize IN PLACE. cov is known exactly from the actions we commanded, so the
    belief should live only on the current-cov dev-block (+ terminal states).

    If that would zero everything (e.g. belief already fully on terminal), leave it."""
    keep = np.zeros(policy.n_states)
    for i, (dev, c) in enumerate(zip(policy.state_dev, policy.state_cov)):
        if dev in ("LOST", "CRASHED") or c == cov:
            keep[i] = 1.0
    masked = belief * keep
    tot = masked.sum()
    if tot > 0:
        belief[:] = masked / tot
