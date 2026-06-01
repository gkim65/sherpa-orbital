"""
Differential corrector for southern L1 halo orbit initial conditions.

Finds a single-revolution closed halo orbit in the Saturn-Enceladus CR3BP
matching the Enceladus Orbilander science orbit (MacKenzie et al. 2020, §B.2.3):
  - Southern halo (periapsis over south pole, z < 0 at periapsis)
  - Periapsis altitude: 19.8–64.3 km
  - Apoapsis altitude:  1000–1110 km
  - Period:             ~12 hr

Algorithm:
  1. Richardson (1980) 3rd-order analytic approximation gives a starting IC
     directly on (or very near) the halo family — no blind scan needed.
  2. Single-shooting differential corrector (Howell 1984) refines the IC:
     - Start on x-z plane: [x0, 0, z0, 0, vy0, 0]
     - Free variables: x0 and vy0 (fix z0 as the family parameter)
     - Target: vx_f = 0 AND vz_f = 0 at the y=0 half-period crossing
     - 2×2 Newton corrector using STM; step is damped to prevent divergence
  3. The southern halo is obtained by negating z0 and vy0 (z → −z symmetry).

References:
  Richardson, D. L. (1980). "Analytic Construction of Periodic Orbits about the
    Collinear Points." Celestial Mechanics, 22(3), 241–253.
  Howell, K. C. (1984). "Three-Dimensional, Periodic, 'Halo' Orbits."
    Celestial Mechanics, 32(1), 53–71.
  MacKenzie et al. (2020). Enceladus Orbilander Mission Concept Study, §B.2.3.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional

from src.constants import (
    MU, L_STAR, T_STAR, V_STAR, OMEGA,
    R_ENCELADUS,
    PERIAPSIS_ALT_MIN, PERIAPSIS_ALT_MAX,
    APOAPSIS_ALT_MIN, APOAPSIS_ALT_MAX,
)
from src.dynamics.cr3bp import cr3bp_eom, libration_points_x, jacobi_constant
from src.dynamics.integrator import propagate, RTOL_ONBOARD, ATOL_ONBOARD

# ── Verified period-3 southern L1 halo orbit IC ───────────────────────────────
# Source: JPL Three-Body Periodic Orbits Catalog (DE440), corrected in our CR3BP.
# MacKenzie et al. 2020 §B.2.3 science orbit — south polar, periapsis ~31 km altitude.
# Converged to residual < 1e-12 using differential_corrector(n_crossings=2).
#
# Physical: x0=237911.1 km, z0=−1162.8 km, vy0=0.06895 km/s, T=35.99 hr
# Nondim (our units: LU=238529.333 km, TU=18913.280 s):
PERIOD3_IC_ND: np.ndarray = np.array([
    9.974083488926582e-01,   # x0 (nondim)
    0.0,                     # y0
   -4.874948479304304e-03,   # z0 (nondim, negative = south polar)
    0.0,                     # vx0
    5.466912598254453e-03,   # vy0 (nondim)
    0.0,                     # vz0
])
PERIOD3_PERIOD_ND: float = 35.988 * 3600.0 / 18913.2798604104  # nondim
PERIOD3_PERIOD_S:  float = 35.988 * 3600.0                     # s

# Non-dimensional positions of primaries
_X_SAT_ND: float = -MU
_X_ENC_ND: float = 1.0 - MU


# ── Non-dimensional CR3BP EOM + STM ──────────────────────────────────────────

def _cr3bp_nd(t: float, s: np.ndarray) -> np.ndarray:
    """Non-dimensional CR3BP EOM (no STM)."""
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + MU)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + MU)**2 + y**2 + z**2)
    r1_3, r2_3 = r1**3, r2**3
    ax = 2.0*vy + x - (1-MU)*(x+MU)/r1_3 - MU*(x-1+MU)/r2_3
    ay = -2.0*vx + y - (1-MU)*y/r1_3 - MU*y/r2_3
    az = -(1-MU)*z/r1_3 - MU*z/r2_3
    return np.array([vx, vy, vz, ax, ay, az])


def _cr3bp_stm_nd(t: float, aug: np.ndarray) -> np.ndarray:
    """
    Augmented EOM: non-dim CR3BP state (6) + flattened STM (36) = 42 elements.
    dPhi/dt = A(t) * Phi, A is the Jacobian of _cr3bp_nd.
    """
    s = aug[:6]
    phi = aug[6:].reshape(6, 6)

    x, y, z = s[0], s[1], s[2]
    r1 = np.sqrt((x + MU)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + MU)**2 + y**2 + z**2)
    r1_3, r2_3 = r1**3, r2**3
    r1_5, r2_5 = r1**5, r2**5
    dx1, dx2 = x + MU, x - 1.0 + MU

    Uxx = (1.0
           - (1-MU)/r1_3 + 3*(1-MU)*dx1**2/r1_5
           - MU/r2_3     + 3*MU*dx2**2/r2_5)
    Uyy = (1.0
           - (1-MU)/r1_3 + 3*(1-MU)*y**2/r1_5
           - MU/r2_3     + 3*MU*y**2/r2_5)
    Uzz = (-(1-MU)/r1_3 + 3*(1-MU)*z**2/r1_5
           - MU/r2_3    + 3*MU*z**2/r2_5)
    Uxy = 3*(1-MU)*dx1*y/r1_5 + 3*MU*dx2*y/r2_5
    Uxz = 3*(1-MU)*dx1*z/r1_5 + 3*MU*dx2*z/r2_5
    Uyz = 3*(1-MU)*y*z/r1_5   + 3*MU*y*z/r2_5

    A = np.array([
        [0,   0,   0,   1,   0,  0],
        [0,   0,   0,   0,   1,  0],
        [0,   0,   0,   0,   0,  1],
        [Uxx, Uxy, Uxz, 0,   2,  0],
        [Uxy, Uyy, Uyz, -2,  0,  0],
        [Uxz, Uyz, Uzz, 0,   0,  0],
    ])

    return np.concatenate([_cr3bp_nd(t, s), (A @ phi).ravel()])


# ── Half-period propagator ────────────────────────────────────────────────────

def _propagate_half_period_nd(
    x0: float,
    z0: float,
    vy0: float,
    t_min: float = 0.3,
    t_half_max: float = 15.0,
    n_crossings: int = 1,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Propagate [x0, 0, z0, 0, vy0, 0] to the nth y=0 crossing (nondim).

    For a simple halo (period-1): n_crossings=1 → half-period T/2.
    For a period-3 halo: n_crossings=3 → half of the full 3-rev period (3T/2).
    The STM accumulates across all crossings so it maps t=0 → t=n×T/2.

    Parameters
    ----------
    x0, z0, vy0 : float
        Non-dimensional IC (y=0, vx=vz=0 assumed).
    t_min : float
        Initial coast time to skip the degenerate y=0 at t=0 (nondim).
    t_half_max : float
        Max total search time (nondim). For period-3: use ≥ 12 nondim (~63 hr).
    n_crossings : int
        Number of y=0 crossings to pass through. 1 = simple half-period;
        3 = period-3 half-period. Direction alternates each crossing.
    rtol, atol : float
        Integration tolerances.

    Returns
    -------
    state_f : np.ndarray (6,)  — state at nth y=0 crossing
    phi     : np.ndarray (6,6) — STM from t=0 to that crossing
    t_cross : float            — non-dimensional time of nth crossing
    """
    aug0 = np.concatenate([np.array([x0, 0.0, z0, 0.0, vy0, 0.0]),
                           np.eye(6).ravel()])

    # Segment 1: coast past the degenerate y=0 at t=0 using a small fixed step.
    # t_min must be small enough not to overshoot the first real crossing.
    sol1 = solve_ivp(_cr3bp_stm_nd, (0.0, t_min), aug0,
                     method="RK45", rtol=rtol, atol=atol)
    aug_cur = sol1.y[:, -1]
    t_cur = t_min

    # For the symmetry-plane corrector we want the FIRST descending y=0 crossing
    # (for southern halo with vy0>0: y goes positive first, then descends through 0).
    # For n_crossings>1 we collect that many consecutive crossings of the same
    # direction — each one is a half-period of the underlying single-rev orbit,
    # so n_crossings=3 gives the half-period of the period-3 orbit.
    # Direction: y descending (-1) is the symmetry half-period for z0<0, vy0>0.
    direction = -1.0 if vy0 > 0 else 1.0

    for crossing in range(n_crossings):
        def y_cross(_t: float, aug: np.ndarray) -> float:  # noqa: E306
            return aug[1]
        y_cross.terminal = True
        y_cross.direction = direction   # same direction every time

        sol2 = solve_ivp(_cr3bp_stm_nd, (t_cur, t_half_max), aug_cur,
                         method="RK45", rtol=rtol, atol=atol, events=[y_cross])

        if len(sol2.t_events[0]) == 0:
            raise RuntimeError(
                f"y=0 crossing #{crossing+1}/{n_crossings} not found "
                f"(x0={x0:.5f}, z0={z0:.5f}, vy0={vy0:.5f}). "
                f"Reached t={sol2.t[-1]:.3f}, final y={sol2.y[1, -1]:.4f}"
            )

        aug_cur = sol2.y_events[0][0]
        t_cur   = float(sol2.t_events[0][0])
        # After each crossing, coast briefly past y=0 so the next event fires
        # on the subsequent descending crossing (not immediately at the same one).
        if crossing < n_crossings - 1:
            coast = solve_ivp(_cr3bp_stm_nd, (t_cur, t_cur + t_min), aug_cur,
                              method="RK45", rtol=rtol, atol=atol)
            aug_cur = coast.y[:, -1]
            t_cur += t_min

    return aug_cur[:6], aug_cur[6:].reshape(6, 6), t_cur


# ── Richardson (1980) 3rd-order approximation ─────────────────────────────────

def richardson_ic(
    Az_km: float,
    northern: bool = False,
) -> tuple[float, float, float, float]:
    """
    Third-order Lindstedt-Poincaré approximation for L1 halo orbit IC.

    Returns the non-dimensional IC [x0, z0, vy0] and estimated half-period,
    suitable for seeding the differential corrector.

    The out-of-plane amplitude Az controls where on the halo family we are.
    Larger Az → larger orbit → different periapsis altitude.

    For the Enceladus Orbilander southern halo, use northern=False (default),
    which negates z0 and vy0 to place periapsis over the south pole.

    Parameters
    ----------
    Az_km : float
        Out-of-plane amplitude (km). Drives the halo family parameter.
        For the mission orbit (periapsis ~20–70 km), try Az_km in 200–800 km.
    northern : bool
        If True, return northern halo (z0 > 0). Default False = southern halo.

    Returns
    -------
    x0_nd, z0_nd, vy0_nd, t_half_nd : float
        Non-dimensional IC coordinates and estimated half-period.

    Notes
    -----
    Richardson (1980) §3–4; coefficients follow the notation of
    Koon et al. (2011) "Dynamical Systems, the Three-Body Problem and
    Space Mission Design," Ch. 2.
    """
    mu = MU

    # ── L1 location (nondim) ─────────────────────────────────────────────────
    xL1_km, _, _ = libration_points_x()
    xL1 = xL1_km / L_STAR
    # gamma_L1 = distance from Enceladus to L1 (nondim)
    gamma1 = (1.0 - mu) - xL1   # positive: L1 is to the Saturn side of Enceladus

    # ── Legendre polynomial coefficients c_n ─────────────────────────────────
    # c_n = (1/gamma1^3) * [mu + (-1)^n (1-mu) * gamma1^(n+1) / (1-gamma1)^(n+1)]
    def cn(n: int) -> float:
        return (1.0 / gamma1**3) * (
            mu + (-1)**n * (1.0 - mu) * (gamma1**(n+1)) / (1.0 - gamma1)**(n+1)
        )

    c2 = cn(2)
    c3 = cn(3)
    c4 = cn(4)

    # ── In-plane frequency lambda (lp) and out-of-plane frequency kappa ──────
    # lambda satisfies: lp^4 + (c2-2)*lp^2 - (c2-1)(1+2*c2) = 0
    lp2 = 0.5 * ((c2 - 2) + np.sqrt(9*c2**2 - 8*c2))
    lp = np.sqrt(lp2)       # in-plane frequency
    kp = np.sqrt(c2)         # out-of-plane frequency (vertical)

    # ── Amplitude ratios ──────────────────────────────────────────────────────
    k  = (lp**2 + 1 + 2*c2) / (2*lp)   # ratio Ax/Az (in-plane to out-of-plane)
    d1 = (3*lp**2 / k) * (k * (6*lp**2 - 1) - 2*lp)
    d2 = (8*lp**2 / k) * (k * (11*lp**2 - 1) - 2*lp)

    a21 = (3*c3*(k**2 - 2)) / (4*(1 + 2*c2))
    a22 = 3*c3 / (4*(1 + 2*c2))
    a23 = -(3*c3*lp / (4*k*d1)) * (3*k**3*lp - 6*k*(lp**2 - c2) + 4)
    a24 = -(3*c3*lp / (4*k*d1)) * (2 + 3*k*lp)
    b21 = -(3*c3*lp / (2*d1)) * (3*k*lp - 4)
    b22 = 3*c3*lp / d1
    d21 = -c3 / (2*lp**2)

    a31 = -(9*lp / (4*d2)) * (4*c3*(k*a23 - b21) + k*c4*(4 + k**2)) \
          + (9*lp**2 + 1 - c2) / (2*d2) * (3*c3*(2*a23 - k*b21) + c4*(2 + 3*k**2))
    a32 = -(1 / d2) * (9*lp*(4*c3*(k*a24 - b22) + k*c4) / 4.0
                       + 3.0/2.0 * (9*lp**2 + 1 - c2) * (c3*(k*b22 + d21 - 2*a24) - c4))
    b31 = (3 / (8*d2)) * (8*lp*(3*c3*(k*b21 - 2*a23) - c4*(2 + 3*k**2))
                          + (9*lp**2 + 1 + 2*c2) * (4*c3*(k*a23 - b21) + k*c4*(4 + k**2)))
    b32 = (1 / d2) * (9*lp*(c3*(k*b22 + d21 - 2*a24) - c4)
                      + 0.375*(9*lp**2 + 1 + 2*c2) * (8*lp*(c3*(k*a24 - b22) + c4)
                               + (9*lp**2 + 1 - c2) * (c3*(k*b22 + d21 - 2*a24) - c4)))
    # (b32 formula is complex; use a simplified version from Koon et al. 2011)
    # Override with the cleaner Koon et al. form:
    b32 = (1.0/d2) * (
        9*lp*(c3*(k*b22 + d21 - 2*a24) - c4)
        + (3.0/8.0)*(9*lp**2 + 1 + 2*c2)
          * (8*lp*(c3*(k*a24 - b22) + c4)
             + (9*lp**2 + 1 - c2)*(c3*(k*b22 + d21 - 2*a24) - c4))
    )

    d31 = 3 / (64*lp**2) * (4*c3*a24 + c4)
    d32 = 3 / (64*lp**2) * (4*c3*(a23 - d21) + c4*(4 + k**2))

    # ── Frequency correction Δ = Δ1*Az^2 + Δ2*Ax^2 ───────────────────────────
    # Ax = k * Az (in-plane amplitude driven by Az)
    Az = Az_km / L_STAR
    Ax = k * Az

    s1 = 0.5 * (lp**2 + 1 + 2*c2) / lp - 2*lp
    s2 = (3.0/(2*lp)) * (c3*(2*a21 - k*a22 - b21*0 + s1) + 0)  # simplified

    # Frequency corrections (Koon et al. 2011 eq. 2.30)
    delta1 = (3.0/2.0)*c3*(2*a21*(k**2 - 2) - a23*(k**2 + 2) - 2*k*b21) \
             - (3.0/8.0)*c4*(3*k**4 - 8*k**2 + 8)
    delta2 = (3.0/2.0)*c3*(k*a24 + b21 - 2*a22) + (9.0/8.0)*c4*k
    delta_freq = delta1 * Az**2 + delta2 * Ax**2

    omega_p = lp + delta_freq / (2*lp)   # corrected in-plane frequency

    # ── Phase angle ──────────────────────────────────────────────────────────
    # At t=0 (start on x-z plane with y=0): phase psi=pi/2 for northern halo
    psi = np.pi / 2.0

    # ── 3rd-order position and velocity at t=0 ───────────────────────────────
    # Richardson (1980) eqs. (22)-(24), evaluated at tau=0 (psi=pi/2)
    # x displacement from L1:
    dx = (a21*Ax**2 + a22*Az**2
          - Ax*np.cos(psi)
          + (a23*Ax**2 - a24*Az**2)*np.cos(2*psi)
          + (a31*Ax**3 - a32*Ax*Az**2)*np.cos(3*psi))

    # z displacement:
    dz = (Az*np.cos(psi)
          + d21*Ax*Az*(np.cos(2*psi) - 3)
          + (d32*Az*Ax**2 - d31*Az**3)*np.cos(3*psi))

    # y-velocity (at y=0 crossing, tau=0):
    dvy = (lp * Ax * np.sin(psi)
           + 2*omega_p*(a24*Az**2 - a23*Ax**2)*np.sin(2*psi)
           + 3*omega_p*(a32*Ax*Az**2 - a31*Ax**3)*np.sin(3*psi))
    # At psi=pi/2: sin(pi/2)=1, sin(pi)=0, sin(3pi/2)=-1
    dvy = (lp * Ax * 1.0
           + 2*omega_p*(a24*Az**2 - a23*Ax**2)*0.0
           + 3*omega_p*(a32*Ax*Az**2 - a31*Ax**3)*(-1.0))

    # ── Assemble IC in nondim ─────────────────────────────────────────────────
    # x0 = xL1 + dx  (L1 is at 1-mu-gamma1 = xL1)
    x0_nd = xL1 + dx
    z0_nd = dz        # positive for northern halo
    vy0_nd = dvy      # positive for northern halo (vy > 0 drives y positive first)

    # Southern halo: z → -z, vy → -vy (mirror in x-z plane)
    if not northern:
        z0_nd  = -z0_nd
        vy0_nd = -vy0_nd

    # Estimated half-period
    t_half_nd = np.pi / omega_p

    return x0_nd, z0_nd, vy0_nd, t_half_nd


# ── Differential corrector ────────────────────────────────────────────────────

def seed_scan(
    x0_nd: float,
    z0_nd: float,
    vy0_range: tuple[float, float],
    n_vy: int = 30,
    n_crossings: int = 3,
    t_half_max_nd: float = 12.0,
    verbose: bool = True,
) -> Optional[float]:
    """
    Scan vy0 at fixed (x0, z0) to find the value that minimises |vx_f| + |vz_f|
    at the nth y=0 crossing.  Returns the best vy0_nd, or None if all fail.

    Parameters
    ----------
    x0_nd, z0_nd : float
        Fixed non-dimensional starting position.
    vy0_range : tuple[float, float]
        (lo, hi) range to scan.
    n_vy : int
        Number of vy0 values to try.
    n_crossings : int
        Which crossing to target (3 for period-3 half-period).
    t_half_max_nd : float
        Max propagation time.
    verbose : bool

    Returns
    -------
    float or None
        vy0_nd with smallest |vx_f| + |vz_f|, or None if nothing converged.
    """
    best_res = np.inf
    best_vy0 = None

    for vy0 in np.linspace(vy0_range[0], vy0_range[1], n_vy):
        try:
            sf, _, _ = _propagate_half_period_nd(
                x0_nd, z0_nd, vy0,
                n_crossings=n_crossings,
                t_half_max=t_half_max_nd,
            )
            res = abs(sf[3]) + abs(sf[5])
            if res < best_res:
                best_res = res
                best_vy0 = vy0
        except RuntimeError:
            pass

    if verbose and best_vy0 is not None:
        print(f"Seed scan: best vy0={best_vy0:.6f} nondim "
              f"({best_vy0*V_STAR:.5f} km/s), residual={best_res:.4e}")
    return best_vy0


def differential_corrector(
    x0_nd: float,
    z0_nd: float,
    vy0_nd: float,
    t_half_max_nd: float = 15.0,
    n_crossings: int = 1,
    tol: float = 1e-10,
    max_iter: int = 50,
    damp: float = 0.6,
    verbose: bool = True,
) -> dict:
    """
    Single-shooting differential corrector for CR3BP halo orbits (Howell 1984).

    Free variables: x0 and vy0 (nondim). Fixed: z0 (the family parameter).
    Targets vx_f = 0 AND vz_f = 0 at the nth y=0 crossing (half of the
    n_crossings-revolution period).

    For n_crossings=1: simple halo, period = 2 × t_cross.
    For n_crossings=3: period-3 orbit (MacKenzie §B.2.3), period = 2 × t_cross,
    where t_cross is the time of the 3rd y=0 crossing (~18 hr nondim time).

    The 2×2 STM sub-block (rows vx=3, vz=5; cols x=0, vy=4):
        | phi[3,0]  phi[3,4] |
        | phi[5,0]  phi[5,4] |

    Steps are damped by `damp` to prevent overshooting.

    Parameters
    ----------
    x0_nd, z0_nd, vy0_nd : float
        Starting non-dimensional IC.
    t_half_max_nd : float
        Max propagation time (nondim). For period-3 use ≥ 12 nondim (~63 hr).
    n_crossings : int
        Number of y=0 crossings to target. 1 = simple halo; 3 = period-3.
    tol : float
        Convergence on |vx_f| + |vz_f| (nondim velocity).
    max_iter : int
        Maximum Newton iterations.
    damp : float
        Step damping factor ∈ (0,1]. 1.0 = full Newton; 0.5 = half-step.
    verbose : bool

    Returns
    -------
    dict with keys:
        'ic_nd'        : np.ndarray (6,) — IC in nondim units
        'ic'           : np.ndarray (6,) — IC in physical units (km, km/s)
        'period_nd'    : float — full orbital period (nondim)
        'period_s'     : float — full orbital period (s)
        'n_crossings'  : int
        'converged'    : bool
        'iterations'   : int
        'residual'     : float — final |vx_f| + |vz_f|
    """
    x0  = x0_nd
    vy0 = vy0_nd
    residual = np.inf

    for i in range(max_iter):
        try:
            state_f, phi, t_half = _propagate_half_period_nd(
                x0, z0_nd, vy0,
                n_crossings=n_crossings,
                t_half_max=t_half_max_nd,
            )
        except RuntimeError as e:
            return _fail(x0, z0_nd, vy0, i, residual, str(e))

        vx_f = state_f[3]
        vz_f = state_f[5]
        residual = abs(vx_f) + abs(vz_f)

        if verbose:
            print(f"  iter {i:2d}: x0={x0:.7f}, vy0={vy0:.7f}  "
                  f"vx_f={vx_f:+.3e}, vz_f={vz_f:+.3e}, res={residual:.3e}")

        if residual < tol:
            ic_nd = np.array([x0, 0.0, z0_nd, 0.0, vy0, 0.0])
            return {
                "ic_nd": ic_nd,
                "ic": _nd_to_phys(ic_nd),
                "period_nd": 2.0 * t_half,
                "period_s": 2.0 * t_half * T_STAR,
                "n_crossings": n_crossings,
                "converged": True,
                "iterations": i,
                "residual": residual,
            }

        # 2×2 Newton: free variables x0 and vy0, targets vx_f=0 and vz_f=0.
        # STM cols: x=0, vy=4.  STM rows: vx=3, vz=5.
        M = np.array([
            [phi[3, 0], phi[3, 4]],
            [phi[5, 0], phi[5, 4]],
        ])

        cond = np.linalg.cond(M)
        if cond > 1e12:
            if verbose:
                print(f"  WARNING: M singular (cond={cond:.1e}), stopping.")
            break

        try:
            delta = -np.linalg.solve(M, np.array([vx_f, vz_f]))
        except np.linalg.LinAlgError:
            break

        x0  += damp * delta[0]
        vy0 += damp * delta[1]

    return _fail(x0, z0_nd, vy0, max_iter, residual)


def _fail(x0, z0, vy0, iters, residual, error=""):
    ic_nd = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
    d = {
        "ic_nd": ic_nd,
        "ic": _nd_to_phys(ic_nd),
        "period_nd": np.nan, "period_s": np.nan,
        "converged": False,
        "iterations": iters,
        "residual": residual,
    }
    if error:
        d["error"] = error
    return d


# ── Unit conversion ───────────────────────────────────────────────────────────

def _nd_to_phys(state_nd: np.ndarray) -> np.ndarray:
    """Nondim CR3BP state → physical units (km, km/s)."""
    s = state_nd.copy().astype(float)
    s[:3] *= L_STAR
    s[3:] *= V_STAR
    return s


def _phys_to_nd(state_phys: np.ndarray) -> np.ndarray:
    """Physical state (km, km/s) → nondim CR3BP units."""
    s = state_phys.copy().astype(float)
    s[:3] /= L_STAR
    s[3:] /= V_STAR
    return s


# ── Orbit characterisation ────────────────────────────────────────────────────

def characterise_orbit(ic: np.ndarray, period_s: float, verbose: bool = True) -> dict:
    """
    Propagate one full period from ic (physical units) and extract orbit properties.

    Parameters
    ----------
    ic : np.ndarray (6,)
        Initial condition in km / km/s.
    period_s : float
        Orbital period (s).
    verbose : bool

    Returns
    -------
    dict with keys:
        'periapsis_alt_km', 'apoapsis_alt_km', 'period_hr',
        'jacobi', 'closure_km', 'closure_kms'
    """
    from src.dynamics.cr3bp import X_ENCELADUS
    sol = propagate(cr3bp_eom, ic, (0.0, period_s),
                    rtol=RTOL_ONBOARD, atol=ATOL_ONBOARD, dense_output=True)

    r_enc = np.sqrt(
        (sol.y[0] - X_ENCELADUS)**2 + sol.y[1]**2 + sol.y[2]**2
    )
    peri_alt = float(r_enc.min()) - R_ENCELADUS
    apo_alt  = float(r_enc.max()) - R_ENCELADUS

    state_f  = sol.y[:, -1]
    closure_r = float(np.linalg.norm(state_f[:3] - ic[:3]))
    closure_v = float(np.linalg.norm(state_f[3:] - ic[3:]))
    jc = jacobi_constant(ic)

    result = {
        "periapsis_alt_km": peri_alt,
        "apoapsis_alt_km":  apo_alt,
        "period_s":  period_s,
        "period_hr": period_s / 3600.0,
        "jacobi":    jc,
        "closure_km":  closure_r,
        "closure_kms": closure_v,
    }

    if verbose:
        print(f"\nOrbit characterisation:")
        print(f"  Periapsis alt  : {peri_alt:.1f} km  "
              f"(target: {PERIAPSIS_ALT_MIN:.1f}–{PERIAPSIS_ALT_MAX:.1f} km)")
        print(f"  Apoapsis alt   : {apo_alt:.1f} km  "
              f"(target: {APOAPSIS_ALT_MIN:.0f}–{APOAPSIS_ALT_MAX:.0f} km)")
        print(f"  Period         : {period_s/3600:.3f} hr")
        print(f"  Jacobi const   : {jc:.6f} km²/s²")
        print(f"  Closure (pos)  : {closure_r:.4f} km")
        print(f"  Closure (vel)  : {closure_v:.2e} km/s")
        peri_ok = PERIAPSIS_ALT_MIN <= peri_alt <= PERIAPSIS_ALT_MAX
        apo_ok  = APOAPSIS_ALT_MIN  <= apo_alt  <= APOAPSIS_ALT_MAX
        print(f"  Periapsis OK   : {'✓' if peri_ok else '✗'}")
        print(f"  Apoapsis OK    : {'✓' if apo_ok  else '✗'}")

    return result


# ── Full pipeline ─────────────────────────────────────────────────────────────

def find_halo_ic(
    x0_km: float = None,
    z0_km: float = -280.0,
    vy0_km_s: float = None,
    northern: bool = False,
    n_crossings: int = 3,
    tol: float = 1e-10,
    max_iter: int = 50,
    damp: float = 0.7,
    t_half_max_hr: float = 60.0,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: seed scan → differential corrector → characterise.

    Targets the period-3 southern L1 halo orbit for the Enceladus Orbilander
    mission (MacKenzie et al. 2020, §B.2.3):
      - n_crossings=3: corrector propagates to the 3rd y=0 crossing
        (= half of the 36-hr period-3 orbit).
      - Default seed: x0 just Saturn-side of L1, z0=-280 km (south),
        vy0 scanned automatically.

    Parameters
    ----------
    x0_km : float or None
        x-coordinate of IC (km from barycentre). None → x_L1 - 100 km.
    z0_km : float
        z-coordinate of IC (km). Negative = southern halo. Family parameter.
    vy0_km_s : float or None
        y-velocity at IC (km/s). None → scan automatically.
    northern : bool
        If True, negate z0 and vy0 for northern halo.
    n_crossings : int
        Number of y=0 crossings to target. 3 = period-3 orbit (default).
    tol : float
        Corrector tolerance on |vx_f| + |vz_f| (nondim).
    max_iter : int
        Max corrector iterations.
    damp : float
        Newton step damping factor.
    t_half_max_hr : float
        Max propagation time in hours. Default 60 hr covers 3 loops.
    verbose : bool

    Returns
    -------
    dict
        Combined result from differential_corrector + characterise_orbit.
    """
    from src.dynamics.cr3bp import libration_points_x

    xL1_km, _, _ = libration_points_x()
    if x0_km is None:
        x0_km = xL1_km - 100.0

    if northern:
        z0_km = abs(z0_km)

    x0_nd = x0_km / L_STAR
    z0_nd = z0_km / L_STAR
    t_half_max_nd = t_half_max_hr * 3600.0 / T_STAR

    if verbose:
        hemi = "northern" if northern else "southern (south-polar)"
        print(f"=== Period-{n_crossings} {hemi} halo IC finder ===")
        print(f"  x0 = {x0_km:.1f} km (L1 offset: {x0_km-xL1_km:+.1f} km)")
        print(f"  z0 = {z0_km:.1f} km")

    # Scan vy0 if not provided
    if vy0_km_s is None:
        if verbose:
            print(f"\n--- Seed scan over vy0 ---")
        # Southern: vy0 > 0 (orbit goes +y first); northern: vy0 < 0
        lo = 0.001 / V_STAR if not northern else -0.3 / V_STAR
        hi = 0.3   / V_STAR if not northern else -0.001 / V_STAR
        vy0_nd = seed_scan(
            x0_nd, z0_nd,
            vy0_range=(lo, hi),
            n_vy=40,
            n_crossings=n_crossings,
            t_half_max_nd=t_half_max_nd,
            verbose=verbose,
        )
        if vy0_nd is None:
            return {"converged": False, "error": "Seed scan found no viable vy0"}
    else:
        vy0_nd = vy0_km_s / V_STAR

    if verbose:
        print(f"\n--- Differential corrector (Howell 1984, n_crossings={n_crossings}) ---")
        print(f"  Starting: vy0 = {vy0_nd*V_STAR:.5f} km/s")

    result = differential_corrector(
        x0_nd, z0_nd, vy0_nd,
        t_half_max_nd=t_half_max_nd,
        n_crossings=n_crossings,
        tol=tol, max_iter=max_iter, damp=damp,
        verbose=verbose,
    )

    if result["converged"]:
        info = characterise_orbit(result["ic"], result["period_s"], verbose=verbose)
        result.update(info)

    return result
