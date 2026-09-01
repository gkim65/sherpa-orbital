"""
CR3BP equations of motion in the Saturn-Enceladus rotating frame.

This is the ONBOARD model — no J2, no solar perturbation.
State vector: [x, y, z, vx, vy, vz] in km and km/s.

Frame origin: Saturn-Enceladus barycentre.
x-axis points from Saturn toward Enceladus.
z-axis is normal to the orbital plane (aligned with angular momentum).

    Saturn    is at x = -mu * L_STAR        (left of barycentre)
    Enceladus is at x = (1-mu) * L_STAR     (right of barycentre)

Reference: Szebehely (1967), "Theory of Orbits"; Koon et al. (2011) "Dynamical Systems".
Physical-unit formulation: Scheeres (2012), Ch. 3.
"""

# Positions of the primaries in the rotating frame (km)
const X_SATURN    = -MU * L_STAR          # Saturn is at negative x
const X_ENCELADUS = (1.0 - MU) * L_STAR   # Enceladus is at positive x

"""
    cr3bp_eom!(du, u, p, t)

CR3BP equations of motion (onboard model, no J2), in place.

  - `du` — receives `[vx, vy, vz, ax, ay, az]` (km/s, km/s²)
  - `u` — barycentre-frame state `[x, y, z, vx, vy, vz]` (km, km/s)
  - `p`, `t` — unused; present for the `OrdinaryDiffEq` signature
"""
function cr3bp_eom!(du, u, p, t)
    x, y, z, vx, vy, vz = u

    # Distance from Saturn and Enceladus (km)
    r1 = sqrt((x - X_SATURN)^2 + y^2 + z^2)
    r2 = sqrt((x - X_ENCELADUS)^2 + y^2 + z^2)

    r1_3 = r1^3
    r2_3 = r2^3

    # Gravitational accelerations
    gx = -GM_SATURN * (x - X_SATURN) / r1_3 - GM_ENCELADUS * (x - X_ENCELADUS) / r2_3
    gy = -GM_SATURN * y / r1_3 - GM_ENCELADUS * y / r2_3
    gz = -GM_SATURN * z / r1_3 - GM_ENCELADUS * z / r2_3

    du[1] = vx
    du[2] = vy
    du[3] = vz
    # Coriolis and centrifugal accelerations (physical-unit rotating frame)
    du[4] = gx + 2.0 * OMEGA * vy + OMEGA^2 * x
    du[5] = gy - 2.0 * OMEGA * vx + OMEGA^2 * y
    du[6] = gz
    return nothing
end

"""
    cr3bp_eom(u) -> Vector{Float64}

Allocating form of [`cr3bp_eom!`](@ref).

  - `u` — barycentre-frame state (km, km/s)

Returns the 6-element derivative (km/s, km/s²).
"""
function cr3bp_eom(u::AbstractVector{<:Real})
    du = similar(u, 6)
    cr3bp_eom!(du, u, nothing, 0.0)
    return du
end

"""
    jacobi_constant(u) -> Float64

Jacobi constant, the CR3BP energy integral.

  - `u` — barycentre-frame state (km, km/s)

Returns the constant in km²/s². Conserved along a CR3BP trajectory, so drift in it
measures integration error.

    C = 2*Omega(x,y,z) - v²,   Omega = (omega²/2)*(x²+y²) + GM_S/r1 + GM_E/r2
"""
function jacobi_constant(u::AbstractVector{<:Real})
    x, y, z, vx, vy, vz = u

    r1 = sqrt((x - X_SATURN)^2 + y^2 + z^2)
    r2 = sqrt((x - X_ENCELADUS)^2 + y^2 + z^2)

    omega_eff = 0.5 * OMEGA^2 * (x^2 + y^2) + GM_SATURN / r1 + GM_ENCELADUS / r2
    v_sq = vx^2 + vy^2 + vz^2

    return 2.0 * omega_eff - v_sq
end

"""
    libration_points_x() -> (x_L1, x_L2, x_L3)

x-coordinates of the L1, L2, L3 collinear libration points, from the quintic polynomial
formulation in the rotating frame (Murray & Dermott 1999).

Returns `(x_L1, x_L2, x_L3)` in km, barycentre frame.

NOTE: L1 and L2 are solved to tolerance; L3 is a low-order series estimate. Halo orbit
design here uses L1 only.
"""
function libration_points_x()
    mu = MU

    # gamma = distance from Enceladus to the libration point (non-dim).
    # L1: gamma^5 - (3-mu)gamma^4 + (3-2mu)gamma^3 - mu*gamma^2 + 2mu*gamma - mu = 0
    # L2: gamma^5 + (3-mu)gamma^4 + (3-2mu)gamma^3 - mu*gamma^2 - 2mu*gamma - mu = 0
    # Both have a single small positive root; Newton from the Hill-radius estimate
    # converges to it.
    p_L1(g) = g^5 - (3 - mu) * g^4 + (3 - 2mu) * g^3 - mu * g^2 + 2mu * g - mu
    d_L1(g) = 5g^4 - 4 * (3 - mu) * g^3 + 3 * (3 - 2mu) * g^2 - 2mu * g + 2mu
    p_L2(g) = g^5 + (3 - mu) * g^4 + (3 - 2mu) * g^3 - mu * g^2 - 2mu * g - mu
    d_L2(g) = 5g^4 + 4 * (3 - mu) * g^3 + 3 * (3 - 2mu) * g^2 - 2mu * g - 2mu

    gamma_L1 = _newton(p_L1, d_L1, cbrt(mu / 3))
    gamma_L2 = _newton(p_L2, d_L2, cbrt(mu / 3))

    x_L1 = (1.0 - mu - gamma_L1) * L_STAR
    x_L2 = (1.0 - mu + gamma_L2) * L_STAR

    # L3 (beyond Saturn): low-order series.
    gamma_L3_guess = 1.0 - 7mu / 12
    x_L3 = -(mu + gamma_L3_guess) * L_STAR

    return x_L1, x_L2, x_L3
end

# Scalar Newton solve used only by libration_points_x.
function _newton(f, df, g0; tol = 1e-15, maxiter = 100)
    g = g0
    for _ in 1:maxiter
        step = f(g) / df(g)
        g -= step
        abs(step) < tol * max(abs(g), 1.0) && break
    end
    return g
end