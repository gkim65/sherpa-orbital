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

# Uniform burn-efficiency bounds (original spec: "sample halo IC +
# eta_eff ~ Uniform(0.8, 1.0)").
const ETA_EFF_MIN = 0.8
const ETA_EFF_MAX = 1.0

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
    sample_eta_eff(rng; model, sigma_pct, eta_min, eta_max) -> Float64
    sample_eta_eff(rng, n; kwargs...) -> Vector{Float64}

Draw one (or `n`) burn-efficiency samples.

  - `rng` — random stream to draw from
  - `n` — number of samples; the scalar method returns a single draw
  - `model` — `:uniform` for `η ~ U(eta_min, eta_max)`, or `:gaussian_pct` for a symmetric
    `η ~ N(1, sigma_pct/100)`
  - `sigma_pct` — 1σ magnitude error in percent, for `:gaussian_pct`
  - `eta_min`, `eta_max` — bounds for `:uniform` (dimensionless)

Returns a dimensionless efficiency. `:gaussian_pct` is clamped at 0 so a >5σ draw cannot
reverse the burn direction.

NOTE: `:uniform` is the default and every noisy burn in-tree uses it. It is not physical —
`U(0.8, 1.0)` is 0-20% underburn, mean 10% short and never over, against Exhibit B-24's
symmetric 0.7%/2.0% 1σ. `:gaussian_pct` implements B-24 but nothing in-tree selects it:
`thruster_kwargs` is the intended route and `run_rollout` does not forward it.
"""
function sample_eta_eff(
    rng::AbstractRNG;
    model::Symbol = :uniform,
    sigma_pct::Real = THRUSTER_SIGMA_PCT_B24_MODEL2,
    eta_min::Real = ETA_EFF_MIN,
    eta_max::Real = ETA_EFF_MAX,
)
    if model === :uniform
        return eta_min + (eta_max - eta_min) * rand(rng)
    elseif model === :gaussian_pct
        return max(0.0, 1.0 + (float(sigma_pct) / 100) * randn(rng))
    end
    throw(ArgumentError("unknown thruster model $model; expected :uniform or :gaussian_pct"))
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
  - remaining keywords are forwarded to [`sample_eta_eff`](@ref) to select the noise law
    (`model`, `sigma_pct`, `eta_min`, `eta_max`)

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