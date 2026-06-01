"""
Physical constants for the Saturn-Enceladus CR3BP simulator.

All values are in km-km/s-s units unless noted.
Sources:
  - GM_SATURN, GM_ENCELADUS, A_ENCELADUS: Jacobson et al. 2006 / Cassini navigation
  - J2_ENCELADUS, R_ENCELADUS: Iess et al. 2014 (Cassini gravity science)
  - THRUST_NOM: MacKenzie et al. 2020 §3.5 (MR-106E heritage thruster)
"""

# ── Gravitational parameters ─────────────────────────────────────────────────
GM_SATURN: float = 37_931_207.0    # km³/s²
GM_ENCELADUS: float = 7.211        # km³/s²

# ── Orbital geometry ─────────────────────────────────────────────────────────
A_ENCELADUS: float = 238_020.0     # km  semi-major axis of Enceladus about Saturn

# ── Enceladus physical properties ────────────────────────────────────────────
R_ENCELADUS: float = 252.1         # km  mean radius
J2_ENCELADUS: float = 5.435e-3     # dimensionless  (Iess et al. 2014, Cassini)

# ── Propulsion ───────────────────────────────────────────────────────────────
THRUST_NOM: float = 22.0           # N   MR-106E monopropellant thruster

# ── Derived CR3BP parameters ─────────────────────────────────────────────────
MU: float = GM_ENCELADUS / (GM_SATURN + GM_ENCELADUS)   # mass ratio, dimensionless

# Angular velocity of the rotating frame, rad/s
import math as _math
OMEGA: float = _math.sqrt((GM_SATURN + GM_ENCELADUS) / A_ENCELADUS**3)  # rad/s

# Normalisation helpers (physical ↔ CR3BP non-dimensional)
# Length scale: L* = A_ENCELADUS
# Time scale:   T* = 1/OMEGA   (so that n = 1 in non-dim units)
# Velocity scale: V* = L* * OMEGA
L_STAR: float = A_ENCELADUS                      # km
T_STAR: float = 1.0 / OMEGA                      # s
V_STAR: float = A_ENCELADUS * OMEGA              # km/s

# ── Stationkeeping / mission design ──────────────────────────────────────────
# From MacKenzie et al. 2020, Exhibit 3-14
PERIAPSIS_ALT_MIN: float = 19.8    # km  minimum periapsis altitude (science orbit)
PERIAPSIS_ALT_MAX: float = 64.3    # km  maximum periapsis altitude
APOAPSIS_ALT_MIN: float = 1000.0   # km  apoapsis altitude range
APOAPSIS_ALT_MAX: float = 1110.0   # km

# Crash threshold for environment termination
PERIAPSIS_CRASH_ALT: float = 5.0   # km  below this → terminal

# Optical navigation noise (1-sigma), from §C.1
SIGMA_NAV_POS: float = 2.0         # km
