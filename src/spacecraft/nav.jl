"""
nav.jl — navigation observation model: Gaussian position/altitude noise.

The Enceladus Orbilander performs autonomous optical navigation using limb localization and
landmark tracking (MacKenzie et al. 2020, §C.1 "Autonomous Optical Navigation"). The
steady-state position error reported there is ~300 m (notional scenario, Exhibit C-8),
degrading 2–3× (~1 km) under the conservative parameter set (Exhibit C-9). The project's
onboard-belief navigation noise is the round, slightly conservative `SIGMA_NAV_POS = 2 km`,
cited to §C.1 in `constants.jl`.

⚠️ OPEN ITEM (`docs/todo.md`, standing cautions): σ_nav = 2 km is conservative vs MacKenzie
§C.1.1.2 (~0.3–1 km). Left as-is here — the port must reproduce the reference, not
re-tune it.

⚠️ DELIBERATELY UPSTREAM OF THE DYNAMICS. This module maps a true position (or altitude) to
a noisy observation and never calls a CR3BP integrator (CLAUDE.md rule).

RNG DISCIPLINE. Every observer takes an explicit `rng::AbstractRNG` and never touches the
global RNG. As with `thruster.jl`, this port cannot be bit-compared against the Python
reference (PCG64 vs Xoshiro); it is validated distributionally — see `scratch/compare/`.
"""

"""
    observe_position(true_position, rng; sigma_r = SIGMA_NAV_POS) -> Vector{Float64}

Noisy position observation: independent isotropic Gaussian noise of standard deviation
`sigma_r` added to each Cartesian component.

  - `true_position` — true position vector (km), length 3.
  - `rng` — random stream for the noise draws.
  - `sigma_r` — 1σ position noise PER AXIS (km); default `SIGMA_NAV_POS`
    (MacKenzie 2020 §C.1).

Returns the noisy position (km), same length as the input.
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

Noisy scalar altitude observation: the true altitude plus zero-mean Gaussian noise of
standard deviation `sigma_r`.

  - `true_altitude_km` — true altitude above the Enceladus SURFACE (km).
  - `rng` — random stream for the noise draw.
  - `sigma_r` — 1σ altitude noise (km); default `SIGMA_NAV_POS` (MacKenzie 2020 §C.1).

Returns the noisy altitude (km). Note the observation is unbounded below: a small true
altitude can be observed as negative. That is the caller's business — the rollout harness
bins `abs(·)` of a deviation, not an altitude.
"""
observe_altitude(
    true_altitude_km::Real,
    rng::AbstractRNG;
    sigma_r::Real = SIGMA_NAV_POS,
) = float(true_altitude_km) + sigma_r * randn(rng)

"""
    observe_deviation(true_dev_km, rng; sigma_r = SIGMA_NAV_POS) -> Float64

Noisy observation of an apse-position DEVIATION magnitude (km), as consumed by the
belief filter in the rollout harness.

A deviation is a non-negative magnitude, so the noisy read is folded with `abs`: the
underlying scalar Gaussian model is the same one `observe_altitude` uses, but a negative
draw is reflected rather than passed through to the binner. The fold lives here rather
than at the call site so every consumer gets the same convention.
"""
observe_deviation(
    true_dev_km::Real,
    rng::AbstractRNG;
    sigma_r::Real = SIGMA_NAV_POS,
) = abs(float(true_dev_km) + sigma_r * randn(rng))