"""
CR3BP + Enceladus J2 perturbation — the TRUTH model.

The J2 term accounts for Enceladus' oblateness as measured by Cassini. It adds an
axisymmetric gravitational perturbation to Enceladus' monopole term:

    a_J2 = (3/2) * J2 * GM_enc * R_enc² / r2⁵
           * [ (5*(z/r2)² - 1)*dx_enc,
               (5*(z/r2)² - 1)*dy_enc,
               (5*(z/r2)² - 3)*z        ]

where (dx_enc, dy_enc, z) = position relative to Enceladus.

Note the z-component has a different coefficient (−3 vs −1). This is the standard
zonal-harmonic form for an oblate body whose symmetry axis is aligned with z
(Enceladus spin axis ≈ orbit normal).

NOTE: kept strictly separate from the onboard model in `cr3bp.jl`. The gap between them is
the model uncertainty under study, so the two must not be collapsed.

Reference: Balmino (1994), eq. 4; Schaub & Junkins (2018) §9.2.
"""

"""
    j2_acceleration(u) -> NTuple{3,Float64}

J2 perturbation acceleration from Enceladus' oblateness.

  - `u` — barycentre-frame state (km, km/s); only the position is read

Returns `(ax, ay, az)` in km/s².
"""
function j2_acceleration(u::AbstractVector{<:Real})
    x, y, z = u[1], u[2], u[3]

    dx = x - X_ENCELADUS
    dy = y
    r2 = sqrt(dx^2 + dy^2 + z^2)

    # Common factor: (3/2) * J2 * GM_enc * R_enc² / r2⁵
    factor = 1.5 * J2_ENCELADUS * GM_ENCELADUS * R_ENCELADUS^2 / r2^5

    z_ratio_sq = (z / r2)^2

    ax = factor * (5.0 * z_ratio_sq - 1.0) * dx
    ay = factor * (5.0 * z_ratio_sq - 1.0) * dy
    az = factor * (5.0 * z_ratio_sq - 3.0) * z

    return (ax, ay, az)
end

"""
    cr3bp_j2_eom!(du, u, p, t)

CR3BP + Enceladus J2 equations of motion, in place.

  - `du` — derivative vector, written in place
  - `u` — barycentre-frame state (km, km/s)
  - `p`, `t` — unused; present for the `ODEProblem` signature

Same units as [`cr3bp_eom!`](@ref). Integrate at the truth tolerances `RTOL_TRUTH` /
`ATOL_TRUTH`.
"""
function cr3bp_j2_eom!(du, u, p, t)
    cr3bp_eom!(du, u, p, t)

    ax, ay, az = j2_acceleration(u)
    du[4] += ax
    du[5] += ay
    du[6] += az
    return nothing
end

"""
    cr3bp_j2_eom(u) -> Vector{Float64}

Allocating form of [`cr3bp_j2_eom!`](@ref).

  - `u` — barycentre-frame state (km, km/s)

Returns the 6-element derivative.
"""
function cr3bp_j2_eom(u::AbstractVector{<:Real})
    du = similar(u, 6)
    cr3bp_j2_eom!(du, u, nothing, 0.0)
    return du
end