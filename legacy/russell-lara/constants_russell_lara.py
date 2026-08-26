"""
Constants for the Russell & Lara (2009) Enceladus science-orbit model.

These are SEPARATE from src/constants.py on purpose: Russell-Lara use a slightly
different parameter set (Table 1) than our JPL DE440-based CR3BP. Keeping them
apart avoids silently contaminating our truth/onboard models, and makes the
comparison (their model vs. ours) explicit.

Source: Russell, R. P., & Lara, M. (2009). "On the design of an Enceladus
science orbit." Acta Astronautica 65, 27-39, Table 1 and Table 2.
"""

import numpy as np


def rl_ic_to_rotating(ic_table: np.ndarray) -> np.ndarray:
    """
    Convert a Russell-Lara Table 2 IC to a rotating-frame state for propagation.

    FRAME CONVENTION (established empirically, 2026-06-22):
    Table 2 POSITIONS are body-fixed/rotating, but the VELOCITY columns are
    INERTIAL. To propagate in our rotating-frame Hill EOM we must convert the
    velocity:  v_rot = v_inertial - omega x r,  omega = [0, 0, N_RL].

    Validation: with this conversion the halo IC closes to ~6 km after ~12.0 hr
    in the full Hill+J2/J3/C22 model (Jacobi conserved to 7e-5); without it the
    orbit impacts/diverges. (Note: Table 2 osculating a, e match the paper when
    computed from the rotating-frame velocity — a separate element convention.)

    Parameters
    ----------
    ic_table : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] exactly as listed in Table 2 (km, km/s).

    Returns
    -------
    np.ndarray, shape (6,)
        State ready for hill_nonspherical_eom (rotating-frame velocity).
    """
    s = ic_table.copy()
    s[3:] = s[3:] - np.cross(np.array([0.0, 0.0, N_RL]), s[:3])
    return s


# ── Russell-Lara Table 1 (Enceladus and Saturn parameters) ───────────────────
GM_ENCELADUS_RL: float = 7.209544428892310      # km³/s²
GM_SATURN_RL:    float = 37_940_000.0            # km³/s²
MASS_RATIO_RL:   float = 1.900248565867070e-07   # dimensionless (derived)
N_RL:            float = 5.303637005052082e-05   # rad/s  system rotation rate (mean motion)
R_ENCELADUS_RL:  float = 256.3                   # km  Enceladus mean radius
J2_RL:           float = 0.0025                  # normalized (their adopted value)
J3_RL:           float = -0.00001                # normalized
C22_RL:          float = 0.0025                  # normalized
DIST_SAT_ENC_RL: float = 238_040.0               # km  Saturn-Enceladus distance
LU_RL:           float = 1368.52713426300        # km  Hill length unit (their normalization)
TU_RL:           float = 18_854.9857210709       # s   Hill time unit
L1_DIST_RL:      float = 950.0                    # km  approx. Lagrange-point distance (derived)

# ── Russell-Lara Table 2: example stable orbit initial conditions ────────────
# State in the Enceladus-centred rotating (body-fixed) frame, km and km/s.
# Columns are given directly in physical units in the paper.

# 8:35 science orbit (near-polar, ~500 km SMA, stable)
IC_8_35_RL: np.ndarray = np.array([
    -0.2355272394516388e3,   # x0  (km)
    -0.4377725059013570e3,   # y0
     0.0,                    # z0
     0.5065296483365822e-1,  # u0  (km/s)
    -0.3764519699459160e-1,  # v0
     0.1028014839043201e0,   # w0
])
A0_8_35_RL: float = 0.4987636181497566e3   # km   initial osculating SMA
E0_8_35_RL: float = 0.7594742316109369e-1  #      initial osculating eccentricity

# Halo orbit (south-pole-grazing L1 halo; e≈0.756, a≈1196 km).
# This is the orbit conceptually closest to the MacKenzie Orbilander halo.
IC_HALO_RL: np.ndarray = np.array([
     0.9083149767107928e2,   # x0  (km)
     0.5008517273524893e3,   # y0
     0.0,                    # z0
     0.9689335750231572e-2,  # u0  (km/s)
     0.8939747017881099e-1,  # v0
     0.1192129026982496e0,   # w0
])
A0_HALO_RL: float = 0.1195698077647146e4   # km   initial osculating SMA (~1196 km)
E0_HALO_RL: float = 0.7560442959851756e0   #      initial osculating eccentricity (~0.756)

# Epoch used in their ephemeris propagations (for reference; not needed for the
# conservative-model reproduction): January 1, 2028, JD = 2461772.0.
EPHEMERIS_EPOCH_JD_RL: float = 2461772.0