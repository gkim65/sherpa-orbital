"""
Experiment 12 — calibrate the EXCURSION actions for the science+safety POMDP.

The POMDP visits altitude BANDS spanning the SINGLE-BURN reachable range ~33-65 km to
gather science, while keeping the orbit safe. Three bands (placeholder until MacKenzie
plume numbers): LOW~35, MID~48, HIGH~62 km. Each EXCURSE_* action = a one-pass SINGLE-
BURN excursion (position-target, exp 06/07) toward that band, then recover to nominal.
(Single-burn only — we avoid the two-burn timing/fragility for now, per scope.) We need,
per band, MEASURED from EncJ2 truth:
  - achieved periapsis altitude (does the excursion reach the band?),
  - the dev-bin the orbit is in AFTER the excursion + one recovery burn (safety cost),
  - crash/escape rate (does excursing this band ever lose the orbit?).

Run: PYTHONPATH=. python scripts/pomdp_experiments/12_excursion_band_calibration.py
"""
import numpy as np
from collections import defaultdict
from baselines.mpc import (
    solve_burn, nominal_apse_positions, _altitude_km, _make_escape_event,
    CONTROL_ALT_KM,
)
from src.dynamics.cr3bp import cr3bp_eom, X_ENCELADUS
from src.dynamics.cr3bp_j2 import cr3bp_j2_eom
from src.dynamics.integrator import (
    propagate, make_altitude_event, make_periapsis_event, make_apoapsis_event,
    make_crash_event, RTOL_TRUTH, ATOL_TRUTH,
)
from src.constants import R_ENCELADUS, PERIAPSIS_CRASH_ALT
from src.utils.halo_ic import PERIOD3_IC_ND, PERIOD3_PERIOD_S, _nd_to_phys

ONE_REV_S = PERIOD3_PERIOD_S / 3.0
ic = _nd_to_phys(PERIOD3_IC_ND)
r_peri_nom, r_apo_nom = nominal_apse_positions(ic, ONE_REV_S, eom=cr3bp_eom)
APO_NOM = np.linalg.norm((r_apo_nom - np.array([X_ENCELADUS, 0, 0]))) - R_ENCELADUS

DEV_EDGES = (15.0, 60.0, 200.0)
def dev_bin(d):
    if not np.isfinite(d): return "LOST"
    return "OK" if d < DEV_EDGES[0] else "DRIFT" if d < DEV_EDGES[1] else "FAR" if d < DEV_EDGES[2] else "LOST"
def dev_of(p): return float(np.linalg.norm(p[:3] - r_peri_nom))

# Altitude BANDS in the single-burn reachable range ~33-65 km (band -> commanded target).
# From exp 07, single-burn commanded->achieved is compressed, so we command a bit high.
BANDS = {"LOW": (40.0, "single"), "MID": (70.0, "single"), "HIGH": (120.0, "single")}
def band_of(alt):
    """Classify achieved periapsis altitude into the science band it fell in."""
    if alt < 42: return "LOW"     # ~33-42 km
    if alt < 54: return "MID"     # ~42-54 km
    return "HIGH"                 # >=54 km

def enc_rel(r): r = r.copy(); r[0] -= X_ENCELADUS; return r
def scale_to_alt(rn, a):
    rr = enc_rel(rn); rr2 = rr * ((R_ENCELADUS + a) / np.linalg.norm(rr))
    o = rr2.copy(); o[0] += X_ENCELADUS; return o

def coast(maker, s, h=4*ONE_REV_S):
    # maker is a zero-arg thunk that builds the primary (terminal) event.
    ev = maker(); cr = make_crash_event(PERIAPSIS_CRASH_ALT); es = _make_escape_event()
    sol = propagate(cr3bp_j2_eom, s, (0.0, h), events=[ev, cr, es], rtol=RTOL_TRUTH, atol=ATOL_TRUTH)
    if len(sol.t_events[1]): return "CRASHED"
    if len(sol.t_events[2]): return "LOST"
    if not len(sol.t_events[0]): return "LOST"
    return sol.y_events[0][0].copy()

def _shell(): return make_altitude_event(CONTROL_ALT_KM, terminal=True)
def _peri():  return make_periapsis_event(terminal=True)
def _apo():   return make_apoapsis_event(terminal=True)

def nominal_correct(sc):
    b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                   r_peri_nom=r_peri_nom, r_apo_nom=r_apo_nom)
    o = sc.copy(); o[3:] += b["dv"]; return o, b["dv_mag_ms"]

def excurse(s, band):
    """One SINGLE-BURN science excursion to `band` from a periapsis state s. Returns
    (achieved_band_or_terminal, dev_after_recovery_bin, total_dv_ms)."""
    tgt, _ = BANDS[band]; total = 0.0
    sc = coast(_shell, s)   # to shell
    if isinstance(sc, str): return sc, sc, total
    rp = scale_to_alt(r_peri_nom, tgt); ra = scale_to_alt(r_apo_nom, APO_NOM)
    b = solve_burn(sc, ONE_REV_S, n_revs=3, eom=cr3bp_eom, mode="position",
                   r_peri_nom=rp, r_apo_nom=ra)
    total += b["dv_mag_ms"]; sp = sc.copy(); sp[3:] += b["dv"]
    pr = coast(_peri, sp)
    if isinstance(pr, str): return pr, pr, total
    achieved = band_of(_altitude_km(pr))
    # recover: one nominal CORRECT at next shell
    sc2 = coast(_shell, pr)
    if isinstance(sc2, str): return achieved, sc2, total
    sp2, dv2 = nominal_correct(sc2); total += dv2
    pr2 = coast(_peri, sp2)
    if isinstance(pr2, str): return achieved, pr2, total
    return achieved, dev_bin(dev_of(pr2)), total

# From the held nominal orbit, calibrate each band a few times.
st = coast(_peri, ic)
print("band   -> achieved   dev_after_recovery   ΔV(m/s)")
for band in ("LOW", "MID", "HIGH"):
    for trial in range(3):
        ach, devb, dv = excurse(st, band)
        print(f"  {band:4s} -> {str(ach):8s}  {str(devb):8s}  {dv:6.2f}")
