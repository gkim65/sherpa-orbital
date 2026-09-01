"""
Orbital-element conversions and CR3BP unit normalisation.

Provides:
  - [`cr3bp_to_nondim`](@ref) / [`nondim_to_cr3bp`](@ref)  physical ↔ non-dimensional state
  - [`ic_nondim_to_physical`](@ref)                        literature (non-dim) IC → km, km/s
  - [`r_from_enceladus`](@ref) / [`altitude_from_enceladus`](@ref)  distance helpers
  - [`cartesian_to_keplerian`](@ref)                      two-body diagnostic elements

The normalisation uses the Saturn-Enceladus system:

    L* = A_ENCELADUS = L_STAR   km
    T* = 1 / OMEGA              s
    V* = L* · OMEGA             km/s

In non-dimensional units the mass ratio is `MU` and the frame rate is 1. Note that
`src/dynamics/` works in PHYSICAL units; the non-dimensional form is used by the halo
differential corrector ([`halo_ic.jl`](@ref)) and when quoting literature ICs.

Reference for the normalisation: Koon et al. (2011), "Dynamical Systems, the Three-Body
Problem and Space Mission Design", Ch. 2.
"""

"""
    cr3bp_to_nondim(state_phys) -> Vector{Float64}

Convert a physical CR3BP state to non-dimensional units.

  - `state_phys` — `[x, y, z, vx, vy, vz]` (km, km/s)

Returns a fresh non-dimensional state: position / `L_STAR`, velocity / `V_STAR`.
"""
function cr3bp_to_nondim(state_phys::AbstractVector{<:Real})
    s = collect(float.(state_phys))
    s[1:3] ./= L_STAR
    s[4:6] ./= V_STAR
    return s
end

"""
    nondim_to_cr3bp(state_nd) -> Vector{Float64}

Convert a non-dimensional CR3BP state to physical units.

  - `state_nd` — `[x, y, z, vx, vy, vz]`, non-dimensional

Returns a fresh physical state (km, km/s). Inverse of [`cr3bp_to_nondim`](@ref).
"""
function nondim_to_cr3bp(state_nd::AbstractVector{<:Real})
    s = collect(float.(state_nd))
    s[1:3] .*= L_STAR
    s[4:6] .*= V_STAR
    return s
end

"""
    ic_nondim_to_physical(x_nd, y_nd, z_nd, vx_nd, vy_nd, vz_nd) -> Vector{Float64}

Convert literature-reported non-dimensional CR3BP initial conditions to physical units.

  - `x_nd, y_nd, z_nd` — non-dimensional position
  - `vx_nd, vy_nd, vz_nd` — non-dimensional velocity

Returns the physical state (km, km/s). In the standard non-dimensional convention the
barycentre is at the origin, Saturn at `x = -MU` and Enceladus at `x = 1 - MU`.
"""
ic_nondim_to_physical(x_nd::Real, y_nd::Real, z_nd::Real,
                      vx_nd::Real, vy_nd::Real, vz_nd::Real) =
    nondim_to_cr3bp([x_nd, y_nd, z_nd, vx_nd, vy_nd, vz_nd])

# ── Distance helpers in the CR3BP frame ──────────────────────────────────────
# Same quantities as `r_enceladus`/`altitude` in dynamics/integrator.jl, under the names
# used by the orbit-generation layer.

"""
    r_from_enceladus(state) -> Float64

Distance from the spacecraft to the Enceladus centre.

  - `state` — physical barycentre-frame state (km, km/s)

Returns the distance in km.
"""
r_from_enceladus(state::AbstractVector{<:Real}) =
    sqrt((state[1] - X_ENCELADUS)^2 + state[2]^2 + state[3]^2)

"""
    altitude_from_enceladus(state) -> Float64

Altitude above the Enceladus reference surface.

  - `state` — physical barycentre-frame state (km, km/s)

Returns the altitude in km.
"""
altitude_from_enceladus(state::AbstractVector{<:Real}) =
    r_from_enceladus(state) - R_ENCELADUS

# ── Keplerian elements (two-body, Enceladus-centred) ─────────────────────────

"""
    cartesian_to_keplerian(state) -> NamedTuple

Osculating Keplerian elements about Enceladus.

  - `state` — Cartesian state RELATIVE TO ENCELADUS (km, km/s)

Returns `(a, e, i, raan, aop, ta, period)`: semi-major axis in km, angles in rad, period in
s (`Inf` for a non-elliptic orbit).

NOTE: the input is Enceladus-relative, not barycentre-frame — subtract `X_ENCELADUS` from
`x` before calling. This is a two-body diagnostic and does not describe the full CR3BP
motion.
"""
function cartesian_to_keplerian(state::AbstractVector{<:Real})
    r_vec = float.(state[1:3])
    v_vec = float.(state[4:6])
    GM = GM_ENCELADUS

    r = norm(r_vec)
    v = norm(v_vec)

    # Specific angular momentum
    h_vec = cross(r_vec, v_vec)
    h = norm(h_vec)

    # Eccentricity vector
    e_vec = cross(v_vec, h_vec) ./ GM .- r_vec ./ r
    e = norm(e_vec)

    # Semi-major axis from vis-viva
    energy = v^2 / 2.0 - GM / r
    a = -GM / (2.0 * energy)

    # Inclination
    i = acos(clamp(h_vec[3] / h, -1.0, 1.0))

    # Node vector
    k_hat = [0.0, 0.0, 1.0]
    n_vec = cross(k_hat, h_vec)
    n = norm(n_vec)

    # RAAN
    raan = 0.0
    if n > 1e-10
        raan = acos(clamp(n_vec[1] / n, -1.0, 1.0))
        if n_vec[2] < 0
            raan = 2.0 * pi - raan
        end
    end

    # Argument of periapsis
    aop = 0.0
    if n > 1e-10 && e > 1e-10
        aop = acos(clamp(dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[3] < 0
            aop = 2.0 * pi - aop
        end
    end

    # True anomaly
    ta = 0.0
    if e > 1e-10
        ta = acos(clamp(dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if dot(r_vec, v_vec) < 0
            ta = 2.0 * pi - ta
        end
    end

    period = a > 0 ? 2.0 * pi * sqrt(a^3 / GM) : Inf

    return (a = a, e = e, i = i, raan = raan, aop = aop, ta = ta, period = period)
end