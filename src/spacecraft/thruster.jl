"""
thruster.jl — deterministic ΔV application and noisy burn execution.

The Enceladus Orbilander uses Aerojet Rocketdyne MR-106E 22-N monopropellant thrusters for
stationkeeping burns (MacKenzie et al. 2020, Exhibit 3-11: "Number of thrusters (specific
impulse, Isp): ... 8x - 22N (220 s)"). The associated specific impulse and spacecraft wet
mass are cited constants in `constants.jl` (`ISP_MR106E`, `M_SPACECRAFT_WET`) and are used
here only for documentation / future fuel-mass bookkeeping — this module is ΔV-only,
matching `baselines/mpc.jl`'s existing ΔV-only convention.

NOTE: upstream of the dynamics. This module only maps a commanded ΔV vector to an applied
ΔV vector. It never calls a CR3BP integrator and must not be used to merge the truth and
onboard models.

Burn execution errors (thruster throughput variability, valve response, attitude-control
coupling) are modelled as a single scalar efficiency factor:

    ΔV_applied = η_eff · ΔV_commanded

A minimal stand-in for burn execution error, not a thruster degradation model.

Every sampler takes an explicit `rng` and never touches the global RNG, so a rollout's
stochastic stream is a function of its own seed alone.
"""

"""
    apply_dv(dv_commanded) -> Vector{Float64}

Deterministic ΔV application: the applied ΔV exactly equals the commanded ΔV.

  - `dv_commanded` — commanded ΔV vector (km/s), length 3

Returns a fresh applied ΔV vector (km/s); the caller's input is never aliased.
"""
apply_dv(dv_commanded::AbstractVector{<:Real}) = collect(float.(dv_commanded))

# Cassini in-flight maneuver magnitude error, MacKenzie et al. 2020 Exhibit B-24: 1σ of
# 0.7% (Model 1) and 2.0% (Model 2), SYMMETRIC about the commanded magnitude.
const THRUSTER_SIGMA_PCT_B24_MODEL1 = 0.7
const THRUSTER_SIGMA_PCT_B24_MODEL2 = 2.0

"""
    sample_eta_eff(rng; sigma_pct) -> Float64
    sample_eta_eff(rng, n; kwargs...) -> Vector{Float64}

Draw one (or `n`) burn-efficiency samples from a symmetric `η ~ N(1, sigma_pct/100)`.

  - `rng` — random stream to draw from
  - `n` — number of samples; the scalar method returns a single draw
  - `sigma_pct` — 1σ magnitude error in percent; defaults to B-24 Model 2 (2.0)

Returns a dimensionless efficiency, clamped at 0 so a >5σ draw cannot reverse the burn
direction.

The law is Cassini in-flight maneuver error (MacKenzie et al. 2020, Exhibit B-24) and is
the only one available, so an unqualified `noisy_thruster = true` is always physical.
`sigma_pct` is set from `StationkeepingPOMDP.thruster_sigma_pct`, which both
[`calibrate_tables`](@ref) and [`run_rollout`](@ref) forward here.
"""
function sample_eta_eff(
    rng::AbstractRNG;
    sigma_pct::Real = THRUSTER_SIGMA_PCT_B24_MODEL2,
)
    return max(0.0, 1.0 + (float(sigma_pct) / 100) * randn(rng))
end

sample_eta_eff(rng::AbstractRNG, n::Integer; kwargs...) =
    [sample_eta_eff(rng; kwargs...) for _ in 1:n]

"""
    apply_dv_noisy(dv_commanded, rng; eta_eff = nothing) -> (dv_applied, eta_eff)

Noisy-execution ΔV application: `ΔV_applied = ΔV_commanded · η_eff`.

  - `dv_commanded` — commanded ΔV vector (km/s), length 3
  - `rng` — random stream for the η_eff draw; ignored when `eta_eff` is given
  - `eta_eff` — optional fixed efficiency (dimensionless) instead of a random draw, for
    deterministic tests
  - remaining keywords are forwarded to [`sample_eta_eff`](@ref) — in practice `sigma_pct`,
    the 1σ magnitude error in percent

Returns `(dv_applied, eta_eff)` — the applied ΔV (km/s) and the dimensionless efficiency
that produced it.
"""
function apply_dv_noisy(
    dv_commanded::AbstractVector{<:Real},
    rng::AbstractRNG;
    eta_eff::Union{Real,Nothing} = nothing,
    kwargs...,
)
    dv = collect(float.(dv_commanded))
    η = eta_eff === nothing ? sample_eta_eff(rng; kwargs...) : float(eta_eff)
    return dv .* η, η
end