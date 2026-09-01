"""
CR3BP + Saturn J2 + Enceladus J2 — the high-fidelity truth model.

Saturn's oblateness is the dominant non-point-mass perturbation in this system, larger
than Enceladus' own J2 across most of the science orbit: Enceladus J2 falls off as 1/r⁴
from a tiny moon, while Saturn J2 is referenced to a far larger body and acts at the
comparatively short Enceladus orbital distance.

This module adds the Saturn J2 zonal-harmonic term on top of the CR3BP + Enceladus J2
truth model. The onboard model (`cr3bp.jl`) is not touched.

    a_J2 = (3/2) * J2 * GM_sat * R_sat² / r1⁵
           * [ (5*(z/r1)² - 1)*dx_sat,
               (5*(z/r1)² - 1)*dy_sat,
               (5*(z/r1)² - 3)*z        ]

where (dx_sat, dy_sat, z) = position relative to Saturn.

Symmetry-axis assumption
------------------------
The formula takes the rotating-frame z-axis as Saturn's spin/oblateness axis, which is
exact only if Saturn's pole is parallel to the Saturn-Enceladus orbit normal. Enceladus
orbits very nearly in Saturn's equatorial plane, so the misalignment is sub-degree and its
effect on a ~1e-6 km/s² perturbation is negligible here. The approximation is local to
this rotating-frame model; an inertial SPICE model would take Saturn's true pole from a
PCK kernel instead.

Reference: Balmino (1994), eq. 4; Schaub & Junkins (2018) §9.2.
Saturn J2, R_sat: Jacobson et al. (2006), AJ 132:2520.
"""

"""
    saturn_j2_acceleration(u) -> 3-tuple

J2 perturbation acceleration from Saturn's oblateness.

  - `u` — barycentre-frame state (km, km/s); only the position is read

Returns `(ax, ay, az)` in km/s². Position is taken relative to Saturn at `X_SATURN`, with
the rotating-frame z-axis as Saturn's symmetry axis (see the module docstring).
"""
function saturn_j2_acceleration(u::AbstractVector{<:Real})
    x, y, z = u[1], u[2], u[3]

    dx = x - X_SATURN
    dy = y
    r1 = sqrt(dx^2 + dy^2 + z^2)

    # Common factor: (3/2) * J2 * GM_sat * R_sat² / r1⁵
    factor = 1.5 * J2_SATURN * GM_SATURN * R_SATURN^2 / r1^5

    z_ratio_sq = (z / r1)^2

    ax = factor * (5.0 * z_ratio_sq - 1.0) * dx
    ay = factor * (5.0 * z_ratio_sq - 1.0) * dy
    az = factor * (5.0 * z_ratio_sq - 3.0) * z

    return (ax, ay, az)
end

"""
    cr3bp_saturn_enc_j2_eom!(du, u, p, t)

CR3BP + Saturn J2 + Enceladus J2 equations of motion, in place.

  - `du` — derivative vector, written in place
  - `u` — barycentre-frame state (km, km/s)
  - `p`, `t` — unused; present for the `ODEProblem` signature

Builds on [`cr3bp_j2_eom!`](@ref) by adding Saturn's J2. Integrate at the truth tolerances
`RTOL_TRUTH` / `ATOL_TRUTH`.
"""
function cr3bp_saturn_enc_j2_eom!(du, u, p, t)
    cr3bp_j2_eom!(du, u, p, t)

    ax, ay, az = saturn_j2_acceleration(u)
    du[4] += ax
    du[5] += ay
    du[6] += az
    return nothing
end

"""
    cr3bp_saturn_enc_j2_eom(u) -> Vector{Float64}

Allocating form of [`cr3bp_saturn_enc_j2_eom!`](@ref).

  - `u` — barycentre-frame state (km, km/s)

Returns the 6-element derivative.
"""
function cr3bp_saturn_enc_j2_eom(u::AbstractVector{<:Real})
    du = similar(u, 6)
    cr3bp_saturn_enc_j2_eom!(du, u, nothing, 0.0)
    return du
end