"""
nav.jl — navigation observation model: Gaussian position/altitude noise.

The Enceladus Orbilander navigates autonomously by limb localization and landmark tracking
(MacKenzie et al. 2020, §C.1). Noise magnitude is `SIGMA_NAV_POS` in `constants.jl`, which
documents how it compares to the mission concept's predicted performance.

NOTE: upstream of the dynamics. This module maps a true position or altitude to a noisy
observation and never calls a CR3BP integrator.

Every observer takes an explicit `rng` and never touches the global RNG, so a rollout's
stochastic stream is a function of its own seed alone.
"""

"""
    observe_position(true_position, rng; sigma_r = SIGMA_NAV_POS) -> Vector{Float64}

Noisy position observation: independent isotropic Gaussian noise added to each Cartesian
component.

  - `true_position` — true position vector (km), length 3
  - `rng` — random stream for the noise draws
  - `sigma_r` — 1σ position noise per axis (km); default `SIGMA_NAV_POS`

Returns the noisy position (km), same length as the input.

NOTE: no caller in-tree — possibly stale.
"""
function observe_position(
    true_position::AbstractVector{<:Real},
    rng::AbstractRNG;
    sigma_r::Real = SIGMA_NAV_POS,
)
    r = collect(float.(true_position))
    return r .+ sigma_r .* randn(rng, length(r))
end

"""
    observe_altitude(true_altitude_km, rng; sigma_r = SIGMA_NAV_POS) -> Float64

Noisy scalar altitude observation: the true altitude plus zero-mean Gaussian noise.

  - `true_altitude_km` — true altitude above the Enceladus surface (km)
  - `rng` — random stream for the noise draw
  - `sigma_r` — 1σ altitude noise (km); default `SIGMA_NAV_POS`

Returns the noisy altitude (km).

WARNING: unbounded below — a small true altitude can be observed as negative. Callers that
bin the result must handle that; `alt_bin` maps it to the lowest bin.
"""
observe_altitude(
    true_altitude_km::Real,
    rng::AbstractRNG;
    sigma_r::Real = SIGMA_NAV_POS,
) = float(true_altitude_km) + sigma_r * randn(rng)

"""
    observe_deviation(true_dev_km, rng; sigma_r = SIGMA_NAV_POS) -> Float64

Noisy observation of an apse-position deviation magnitude.

  - `true_dev_km` — true deviation magnitude (km)
  - `rng` — random stream for the noise draw
  - `sigma_r` — 1σ noise (km); default `SIGMA_NAV_POS`

Returns the noisy deviation (km), folded with `abs` so a negative draw reflects rather
than passing through — a deviation is a non-negative magnitude.

NOTE: no caller in-tree, and the rollout harness bins altitudes rather than deviation
norms. Possibly stale.
"""
observe_deviation(
    true_dev_km::Real,
    rng::AbstractRNG;
    sigma_r::Real = SIGMA_NAV_POS,
) = abs(float(true_dev_km) + sigma_r * randn(rng))

"""
    observe_state(true_state, rng; sigma_r = SIGMA_NAV_POS, sigma_v = nothing) -> Vector{Float64}

Noisy full-state observation: independent Gaussian noise on each position and velocity
component.

  - `true_state` — true 6-state `[x, y, z, vx, vy, vz]` (km, km/s)
  - `rng` — random stream for the noise draws
  - `sigma_r` — 1σ PER-AXIS position noise (km)
  - `sigma_v` — 1σ PER-AXIS velocity noise (km/s); `nothing` scales from `sigma_r` via
    [`nav_sigma_vel_for`](@ref)

Returns the noisy 6-state.

This is what a controller PLANS from when it is not given truth. Both halves matter:
`solve_burn` propagates forward to predict the next apses, so velocity error accumulates
into apse-position error over an orbit rather than staying small.

NOTE: noise is PER AXIS, so the position-error magnitude is about `sigma_r * sqrt(3)`.

NOTE: isotropic and uncorrelated, which a real OD covariance is not.
"""
function observe_state(
    true_state::AbstractVector{<:Real},
    rng::AbstractRNG;
    sigma_r::Real = SIGMA_NAV_POS,
    sigma_v::Union{Nothing,Real} = nothing,
)
    sv = sigma_v === nothing ? nav_sigma_vel_for(sigma_r) : float(sigma_v)
    out = collect(float.(true_state))
    out[1:3] .+= sigma_r .* randn(rng, 3)
    out[4:6] .+= sv .* randn(rng, 3)
    return out
end
