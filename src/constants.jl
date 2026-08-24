"""
Physical constants for the Saturn-Enceladus CR3BP simulator.

All values are in km-km/s-s units unless noted. This is the ONLY place a physical
constant is defined; no other file may hardcode one.

The CR3BP unit system is set to match the JPL Three-Body Periodic Orbits Catalog
(DE440 ephemeris), which is the source of the MacKenzie 2020 science orbit design:

    μ   = 1.901109735892602e-7  (mass ratio)
    LU  = 238529.333 km         (Enceladus mean orbital semi-major axis)
    TU  = 18913.2799 s          (1 / mean-motion = 1 / OMEGA)

The system is internally consistent: OMEGA² = GM_total / LU³ = 1 / TU².

Sources
  LU, TU, μ                    JPL Three-Body Periodic Orbits Catalog (DE440 ephemeris)
  R_ENCELADUS, J2_ENCELADUS    Iess et al. 2014 (Cassini gravity science)
  R_SATURN, J2_SATURN          Jacobson et al. 2006, AJ 132:2520 (SAT389 field)
  THRUST_NOM, ISP_MR106E       MacKenzie et al. 2020 §3.5, Exhibit 3-11
  apse bands                   MacKenzie et al. 2020, Exhibit 3-14
"""

# ── Orbital geometry ──────────────────────────────────────────────────────────
const A_ENCELADUS = 238_529.333_386_019   # km  mean orbital semi-major axis (JPL DE440)

# ── CR3BP unit system (derived from JPL catalog values) ───────────────────────
# Fix TU_JPL = 1/OMEGA so the nondim system matches the periodic orbit catalog.
# Then GM_total = LU³/TU² and the mass ratio μ fixes GM_ENCELADUS and GM_SATURN.
const _TU_JPL   = 18913.279_860_410_4         # s  (JPL catalog)
const _MU_JPL   = 1.901109735892602e-7        # dimensionless (JPL catalog)
const _GM_TOTAL = A_ENCELADUS^3 / _TU_JPL^2   # km³/s²

# Physical GMs consistent with the CR3BP system
const GM_ENCELADUS = _MU_JPL * _GM_TOTAL         # km³/s²  (CR3BP-consistent)
const GM_SATURN    = _GM_TOTAL - GM_ENCELADUS    # km³/s²  (effective Saturn-system)

# ── Enceladus physical properties ─────────────────────────────────────────────
const R_ENCELADUS  = 252.1      # km  mean radius  (Iess et al. 2014)
const J2_ENCELADUS = 5.435e-3   # dimensionless    (Iess et al. 2014)

# ── Saturn physical properties ────────────────────────────────────────────────
# Used by the high-fidelity truth model only (dynamics/cr3bp_saturn_j2.jl). Saturn's
# oblateness is the dominant non-point-mass perturbation in this system (external
# review 2026-06-22); at Enceladus' orbital distance its J2 acceleration exceeds
# Enceladus' own J2 across most of the science orbit. J2 is referenced to R_SATURN.
const R_SATURN  = 60_268.0      # km  equatorial radius (1 bar)  (Jacobson et al. 2006)
const J2_SATURN = 1.629071e-2   # dimensionless, normalized to R_SATURN (Jacobson et al. 2006)

# ── Propulsion ────────────────────────────────────────────────────────────────
const THRUST_NOM = 22.0    # N  MR-106E monopropellant (MacKenzie 2020 §3.5)

# Specific impulse of the MR-106E 22-N monoprop thruster (MacKenzie 2020, Exhibit 3-11
# propulsion table: "Number of thrusters (specific impulse, Isp): ... 8x - 22N (220 s)").
# Used for ΔV -> propellant-mass bookkeeping (rocket equation).
const ISP_MR106E = 220.0   # s  MR-106E specific impulse (MacKenzie 2020, Exhibit 3-11)

# Spacecraft wet mass at launch (MacKenzie 2020 §3.4 / Exhibit 3-6, "Total Wet Mass MPV").
# Includes the 30% CBE margin; an upper bound for the mass available during Enceladus-orbit
# stationkeeping (post-cruise, post-orbit-insertion mass will be lower).
const M_SPACECRAFT_WET = 6610.0   # kg  (MacKenzie 2020 §3.4)

# ── CR3BP derived parameters ──────────────────────────────────────────────────
const MU = GM_ENCELADUS / _GM_TOTAL   # = _MU_JPL by construction

# Angular velocity of the rotating frame: OMEGA = sqrt(GM_total / LU³) ≡ 1/TU_JPL.
const OMEGA = sqrt(_GM_TOTAL / A_ENCELADUS^3)   # rad/s

# Normalisation (physical ↔ CR3BP non-dimensional)
const L_STAR = A_ENCELADUS           # km
const T_STAR = 1.0 / OMEGA           # s  (≡ TU_JPL)
const V_STAR = A_ENCELADUS * OMEGA   # km/s

# ── Stationkeeping / mission design ───────────────────────────────────────────
# From MacKenzie et al. 2020, Exhibit 3-14
const PERIAPSIS_ALT_MIN = 19.8     # km  minimum periapsis altitude
const PERIAPSIS_ALT_MAX = 64.3     # km  maximum periapsis altitude
const APOAPSIS_ALT_MIN  = 1000.0   # km  apoapsis altitude range
const APOAPSIS_ALT_MAX  = 1110.0   # km

const PERIAPSIS_CRASH_ALT = 5.0    # km  below this → terminal state

# Optical navigation noise (1-sigma), from MacKenzie 2020 §C.1
const SIGMA_NAV_POS = 2.0          # km