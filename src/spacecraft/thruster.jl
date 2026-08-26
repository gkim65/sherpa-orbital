"""
thruster.jl — deterministic ΔV application and noisy burn execution.

The Enceladus Orbilander uses Aerojet Rocketdyne MR-106E 22-N monopropellant thrusters for
stationkeeping burns (MacKenzie et al. 2020, Exhibit 3-11: "Number of thrusters (specific
impulse, Isp): ... 8x - 22N (220 s)"). The associated specific impulse and spacecraft wet
mass are cited constants in `constants.jl` (`ISP_MR106E`, `M_SPACECRAFT_WET`) and are used
here only for documentation / future fuel-mass bookkeeping — this module is ΔV-only,
matching `baselines/mpc.jl`'s existing ΔV-only convention.

⚠️ DELIBERATELY UPSTREAM OF THE DYNAMICS. This module only maps a commanded ΔV vector to an
applied ΔV vector. It never calls a CR3BP integrator and must not be used to merge the
truth and onboard models (CLAUDE.md rule).

Burn execution errors (thruster throughput variability, valve response, attitude-control
coupling) are modelled as a single scalar efficiency factor η_eff:

    η_eff ~ Uniform(ETA_EFF_MIN, ETA_EFF_MAX) = Uniform(0.8, 1.0)
    ΔV_applied = η_eff · ΔV_commanded

A minimal stand-in for burn execution error, not a full thruster degradation model — the
random-walk degradation model is explicitly deferred (`docs/todo.md` Phase 2).

RNG DISCIPLINE. Every sampler takes an explicit `rng::AbstractRNG` and never touches the
global RNG, so a rollout's stochastic stream is a function of its own seed alone. Note this
port CANNOT be bit-compared against the Python reference: `numpy.random.default_rng`
(PCG64) and Julia's `Xoshiro` are different generators, so identical seeds give different
streams. The port is validated DISTRIBUTIONALLY instead — see `scratch/compare/`.
"""

# Uniform burn-efficiency bounds (Phase 3 spec, docs/todo.md: "sample halo IC +
# eta_eff ~ Uniform(0.8, 1.0)").
const ETA_EFF_MIN = 0.8
const ETA_EFF_MAX = 1.0

"""
    apply_dv(dv_commanded) -> Vector{Float64}

Deterministic ΔV application: the applied ΔV exactly equals the commanded ΔV.

  - `dv_commanded` — commanded ΔV vector (km/s), length 3.

Returns a fresh applied ΔV vector (km/s), so the caller's input is never aliased.
"""
apply_dv(dv_commanded::AbstractVector{<:Real}) = collect(float.(dv_commanded))

"""
    sample_eta_eff(rng) -> Float64
    sample_eta_eff(rng, n) -> Vector{Float64}

Draw one (or `n`) burn-efficiency samples from `Uniform(ETA_EFF_MIN, ETA_EFF_MAX)`.

  - `rng` — the random stream to draw from. Explicit and required: a rollout's
    reproducibility must not depend on global RNG state.

Returns a dimensionless efficiency in `[ETA_EFF_MIN, ETA_EFF_MAX]`.
"""
sample_eta_eff(rng::AbstractRNG) =
    ETA_EFF_MIN + (ETA_EFF_MAX - ETA_EFF_MIN) * rand(rng)

sample_eta_eff(rng::AbstractRNG, n::Integer) =
    [sample_eta_eff(rng) for _ in 1:n]

"""
    apply_dv_noisy(dv_commanded, rng; eta_eff = nothing) -> (dv_applied, eta_eff)

Noisy-execution ΔV application: `ΔV_applied = ΔV_commanded · η_eff`, with
`η_eff ~ Uniform(ETA_EFF_MIN, ETA_EFF_MAX)` unless supplied explicitly.

  - `dv_commanded` — commanded ΔV vector (km/s), length 3.
  - `rng` — random stream for the η_eff draw. Ignored when `eta_eff` is given.
  - `eta_eff` — optional fixed efficiency (dimensionless) instead of a random draw,
    for deterministic tests.

Returns `(dv_applied::Vector{Float64} [km/s], eta_eff::Float64)`.
"""
function apply_dv_noisy(
    dv_commanded::AbstractVector{<:Real},
    rng::AbstractRNG;
    eta_eff::Union{Real,Nothing} = nothing,
)
    dv = collect(float.(dv_commanded))
    η = eta_eff === nothing ? sample_eta_eff(rng) : float(eta_eff)
    return dv .* η, η
end