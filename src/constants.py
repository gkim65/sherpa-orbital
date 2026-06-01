"""
Physical constants for the Saturn-Enceladus CR3BP simulator.

All values are in km-km/s-s units unless noted.

The CR3BP unit system is set to match the JPL Three-Body Periodic Orbits Catalog
(DE440 ephemeris), which is the source of the MacKenzie 2020 science orbit design:
  μ   = 1.901109735892602e-7  (mass ratio)
  LU  = 238529.333 km         (Enceladus mean orbital semi-major axis)
  TU  = 18913.2799 s          (1 / mean-motion = 1 / OMEGA)

The CR3BP system is internally consistent: OMEGA² = GM_total / LU³ = 1 / TU².

Sources:
  - LU, TU, μ: JPL Three-Body Periodic Orbits Catalog (DE440 ephemeris)
  - R_ENCELADUS, J2_ENCELADUS: Iess et al. 2014 (Cassini gravity science)
  - THRUST_NOM: MacKenzie et al. 2020 §3.5 (MR-106E heritage thruster)
"""

# ── Orbital geometry ──────────────────────────────────────────────────────────
A_ENCELADUS: float = 238_529.333_386_019   # km  mean orbital semi-major axis (JPL DE440)

# ── CR3BP unit system (derived from JPL catalog values) ───────────────────────
# Fix TU_JPL = 1/OMEGA so the nondim system matches the periodic orbit catalog.
# Then GM_total = LU³/TU² and the mass ratio μ fixes GM_ENCELADUS and GM_SATURN.
_TU_JPL: float   = 18913.279_860_410_4        # s  (JPL catalog)
_MU_JPL: float   = 1.901109735892602e-7       # dimensionless (JPL catalog)
_GM_TOTAL: float = A_ENCELADUS**3 / _TU_JPL**2   # km³/s²

# Physical GMs consistent with the CR3BP system
GM_ENCELADUS: float = _MU_JPL * _GM_TOTAL    # km³/s²  (CR3BP-consistent)
GM_SATURN: float    = _GM_TOTAL - GM_ENCELADUS  # km³/s²  (effective Saturn-system)

# ── Enceladus physical properties ─────────────────────────────────────────────
R_ENCELADUS: float  = 252.1       # km  mean radius       (Iess et al. 2014)
J2_ENCELADUS: float = 5.435e-3    # dimensionless          (Iess et al. 2014)

# ── Propulsion ────────────────────────────────────────────────────────────────
THRUST_NOM: float = 22.0          # N   MR-106E monopropellant (MacKenzie 2020 §3.5)

# ── CR3BP derived parameters ──────────────────────────────────────────────────
MU: float = GM_ENCELADUS / _GM_TOTAL   # = _MU_JPL by construction

# Angular velocity of the rotating frame (rad/s)
# OMEGA = sqrt(GM_total / LU³) ≡ 1/TU_JPL by construction.
import math as _math
OMEGA: float = _math.sqrt(_GM_TOTAL / A_ENCELADUS**3)  # rad/s

# Normalisation (physical ↔ CR3BP non-dimensional)
L_STAR: float = A_ENCELADUS           # km
T_STAR: float = 1.0 / OMEGA           # s  (≡ TU_JPL)
V_STAR: float = A_ENCELADUS * OMEGA   # km/s

# ── Stationkeeping / mission design ───────────────────────────────────────────
# From MacKenzie et al. 2020, Exhibit 3-14
PERIAPSIS_ALT_MIN: float = 19.8    # km  minimum periapsis altitude
PERIAPSIS_ALT_MAX: float = 64.3    # km  maximum periapsis altitude
APOAPSIS_ALT_MIN:  float = 1000.0  # km  apoapsis altitude range
APOAPSIS_ALT_MAX:  float = 1110.0  # km

PERIAPSIS_CRASH_ALT: float = 5.0   # km  below this → terminal state

# Optical navigation noise (1-sigma), from MacKenzie 2020 §C.1
SIGMA_NAV_POS: float = 2.0         # km