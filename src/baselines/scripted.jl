"""
baselines/scripted.jl — scripted high-level controllers sharing the POMDP's low-level layer.

Each baseline answers the same question the POMDP does — which maneuver objective to attempt
this pass — with a rule instead of a policy. They fly the identical hierarchy otherwise: the
same `solve_burn` planner, the same persistent excursion reference, the same commanded band
altitudes, and the same OBSERVED-altitude coverage banking under nav noise. That makes the
comparison single-variable: only the top layer changes.

  - [`CyclicController`](@ref)    — a fixed LOW/MID/HIGH rotation with `k` corrections between
  - [`GreedyController`](@ref)    — always excurse to the least-sampled band
  - [`ThresholdController`](@ref) — excurse when the orbit damage is below a cutoff, else correct

NOTE: coverage is banked from the OBSERVED altitude bin for every baseline, exactly as the
SARSOP controller does. Crediting a baseline on the true bin would hand it a better sensor
than the policy it is being compared against.

NOTE: THE ARMS DIFFER IN HOW MUCH THEY CONSUME AN OBSERVATION, which matters when reading a
nav sweep. `GreedyController` and `ThresholdController` choose actions from the observed
coverage and the damage bin, so nav noise changes what they DO. `CyclicController` follows
a fixed rotation and reads the observation only to bank coverage and to decide when every
band has saturated, so it is close to open-loop; `MPCController` implements no
`controller_observe!` at all and is fully open-loop. A flat line across a nav sweep is
therefore not evidence of robustness for those two — they barely consume the swept
variable. Compare the policy against the closed-loop baselines.

NOTE: `controller_command` receives the TRUE state, so every controller plans its burn
against truth and nav noise never corrupts a maneuver, only a decision. In flight the
planner would target from the navigation solution instead. This understates what navigation
error costs, uniformly across arms.

NOTE: excursion commands are PERSISTENT here too — an `EXCURSE_*` sets the active reference
and it stays set until a `CORRECT` clears it. Single-impulse authority is poor, so a band is
reached by settling over several passes.
"""

# ── Shared scripted-controller state ──────────────────────────────────────────
"""
    ScriptedCore(; band_names, band_bins, band_target_km, alt_edges, residual_edges,
                 visit_cap, sigma_nav_km, ref_ic, retarget_bands, family_table)

The machinery every scripted baseline shares: band targeting, coverage banking, and damage
binning. Holds exactly the fields a [`SARSOPController`](@ref) uses for those jobs, so the
baselines and the policy resolve band altitudes and bin observations identically.

  - `band_names` — science band labels, in visit-tuple order
  - `band_bins` — the altitude bin each band corresponds to
  - `band_target_km` — commanded periapsis altitude per band (km)
  - `alt_edges` — altitude bin boundaries (km)
  - `residual_edges` — orbit-damage bin boundaries (km)
  - `visit_cap` — saturation ceiling for the per-band sample count
  - `sigma_nav_km` — 1-sigma nav noise on the measured altitude (km)
  - `ref_ic` — reference orbit IC; `nothing` uses the rollout's initial state
"""
Base.@kwdef mutable struct ScriptedCore
    band_names::Vector{String}
    band_bins::Vector{String}
    band_target_km::Dict{String,Float64}
    alt_edges::Vector{Float64}
    residual_edges::Vector{Float64}
    visit_cap::Int
    sigma_nav_km::Float64
    # 1-sigma PER-AXIS position noise (km) on the state the controller PLANS from, set by
    # `scripted_core` to the config's own `sigma_nav_km` so planning and deciding share one
    # navigation quality. 0.0 plans from truth, which is an oracle — see
    # `controller_nav_sigma`.
    nav_sigma_km::Float64 = 0.0
    ref_ic::Union{Nothing,Vector{Float64}} = nothing
    retarget_bands::Bool = false
    family_table::Union{Nothing,Vector{NamedTuple}} = nothing
    # Live state, mirroring SARSOPController's.
    visits::Vector{Int} = Int[]
    residual::String = "R_OK"
    active_band::String = ""
    pass::Int = 0
    r_peri_nom::Union{Nothing,Vector{Float64}} = nothing
    r_apo_nom::Union{Nothing,Vector{Float64}} = nothing
    apo_nom_alt_km::Float64 = NaN
    band_targets::Dict{String,Tuple{Vector{Float64},Vector{Float64}}} =
        Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
end

"""
    scripted_core(config) -> ScriptedCore

Build a [`ScriptedCore`](@ref) from a [`StationkeepingPOMDP`](@ref), so a baseline is
configured by the same struct that defines the POMDP and cannot drift from it.

  - `config` — the scenario

Returns a `ScriptedCore` with zeroed coverage and an undamaged residual.
"""
function scripted_core(config::StationkeepingPOMDP; kwargs...)
    names = [String(b) for b in config.band_names]
    return ScriptedCore(;
        band_names     = names,
        band_bins      = [String(b) for b in config.band_bins],
        band_target_km = Dict(String(b) => config.band_target_km[b]
                              for b in config.band_names),
        alt_edges      = collect(Float64.(config.alt_edges)),
        residual_edges = collect(Float64.(RESIDUAL_EDGES)),
        visit_cap      = config.visit_cap,
        sigma_nav_km   = config.sigma_nav_km,
        # Planning noise defaults to the same sigma the decision layer observes with; a
        # caller can still override it to isolate one from the other.
        nav_sigma_km   = config.sigma_nav_km,
        visits         = zeros(Int, length(names)),
        kwargs...)
end

"""_scripted_alt_bin(core, alt_km) -> String. Bin an altitude with the core's edges."""
function _scripted_alt_bin(core::ScriptedCore, alt_km::Real)
    isfinite(alt_km) || return "LOST"
    e = core.alt_edges
    alt_km < e[1] && return "BELOW_20"
    alt_km < e[2] && return "A20_27"
    alt_km < e[3] && return "A27_34"
    alt_km < e[4] && return "A34_44"
    return "ABOVE_44"
end

"""_scripted_residual_bin(core, residual_km) -> String. Non-finite bins as `R_CRITICAL`."""
function _scripted_residual_bin(core::ScriptedCore, residual_km::Real)
    e = core.residual_edges
    isfinite(residual_km) || return "R_CRITICAL"
    residual_km < e[1] && return "R_OK"
    residual_km < e[2] && return "R_DEGRADED"
    return "R_CRITICAL"
end

"""
    _scripted_band_targets(core, band_name) -> (r_peri, r_apo)

Apse position targets for a band, phase-matched to the vehicle's own nominal orbit.

  - `core` — the shared scripted state
  - `band_name` — science band label

Returns the periapsis and apoapsis target position vectors (km).

NOTE: takes the RADIUS from the retargeted family member and the DIRECTION from our own
nominal periapsis. Importing the member's own apse positions mixes two orbital phases and
loses the vehicle — the same trap documented on `_sarsop_band_targets`.
"""
function _scripted_band_targets(core::ScriptedCore, band_name::AbstractString)
    alt = core.band_target_km[String(band_name)]
    if !core.retarget_bands
        return (_scale_to_altitude(core.r_peri_nom, alt),
                _scale_to_altitude(core.r_apo_nom, core.apo_nom_alt_km))
    end
    return get!(core.band_targets, String(band_name)) do
        table = core.family_table === nothing ? halo_family_table_cached() : core.family_table
        member = retarget_to_altitude(table, alt)
        member === nothing && error(
            "scripted baseline: band $band_name targets $(alt) km, but the continued L1 " *
            "halo family contains no member at that periapsis altitude.")
        (_scale_to_altitude(core.r_peri_nom, member.info.periapsis_alt_km),
         collect(core.r_apo_nom))
    end
end

"""_scripted_setup!(core, state0) -> nothing. Resolve the nominal apse targets."""
function _scripted_setup!(core::ScriptedCore, state0::AbstractVector)
    ref = core.ref_ic === nothing ? state0 : core.ref_ic
    core.r_peri_nom, core.r_apo_nom = next_apse_positions(ref; eom! = cr3bp_eom!)
    core.apo_nom_alt_km = norm(_enc_relative(core.r_apo_nom)) - R_ENCELADUS
    return nothing
end

"""
    _scripted_command(core, action, shell_state, period_s) -> (dv, label, extra)

Execute a maneuver objective through the shared low-level planner.

  - `core` — the shared scripted state
  - `action` — `"CORRECT"` or `"EXCURSE_<BAND>"`
  - `shell_state` — the state at the control shell crossing
  - `period_s` — control cadence (s)

Returns the commanded ΔV (km/s), the action label, and the solve diagnostics including the
targeting residual that becomes the damage bin.
"""
function _scripted_command(core::ScriptedCore, action::AbstractString,
                           shell_state::AbstractVector, period_s::Real)
    if action == "CORRECT"
        core.active_band = ""
        b = solve_burn(shell_state, period_s; eom! = cr3bp_eom!, mode = :position,
                       r_peri_nom = core.r_peri_nom, r_apo_nom = core.r_apo_nom)
        return b.dv, :CORRECT, (band = 0, converged = b.converged,
                                residual_km = b.residual_km, peri_err_km = b.peri_err_km)
    end
    band_name = replace(action, "EXCURSE_" => "")
    band_idx  = findfirst(==(band_name), core.band_names)
    core.active_band = band_name
    _, ra = _scripted_band_targets(core, band_name)
    b = solve_burn(shell_state, period_s; eom! = cr3bp_eom!, mode = :altitude_position,
                   peri_target_km = core.band_target_km[band_name], r_apo_nom = ra)
    return b.dv, Symbol(action), (band = band_idx, converged = b.converged,
                                  residual_km = b.residual_km, peri_err_km = b.peri_err_km)
end

"""
    _scripted_observe!(core, peri_state, extra, rng) -> NamedTuple

Post-pass update: bank coverage from the OBSERVED altitude bin and re-bin orbit damage.

  - `core` — the shared scripted state
  - `peri_state` — the achieved periapsis state
  - `extra` — the command's diagnostics, supplying `residual_km`
  - `rng` — random stream for the nav draw

Returns the trace fields `(true_bin, obs_bin, true_alt_km, obs_alt_km, visits, residual)`.
"""
function _scripted_observe!(core::ScriptedCore, peri_state::AbstractVector,
                            extra::NamedTuple, rng::AbstractRNG)
    true_alt = norm(_enc_relative(peri_state[1:3])) - R_ENCELADUS
    obs_alt  = observe_altitude(true_alt, rng; sigma_r = core.sigma_nav_km)
    true_bin = _scripted_alt_bin(core, true_alt)
    obs_bin  = _scripted_alt_bin(core, obs_alt)

    bi = findfirst(==(obs_bin), core.band_bins)
    bi === nothing || (core.visits[bi] = min(core.visits[bi] + 1, core.visit_cap))

    core.residual = _scripted_residual_bin(core, get(extra, :residual_km, NaN))
    core.pass += 1

    return (true_bin = true_bin, obs_bin = obs_bin, true_alt_km = true_alt,
            obs_alt_km = obs_alt, visits = copy(core.visits), residual = core.residual)
end

# ── Cyclic schedule ───────────────────────────────────────────────────────────
"""
    CyclicController(core, k)

A fixed open-loop rotation: excurse to each science band in turn, with `k` `CORRECT` passes
between consecutive excursions. The schedule ignores every observation, so it is the
"scripted sequence a mission would validate on the ground" baseline.

  - `core` — shared scripted state, from [`scripted_core`](@ref)
  - `k` — recovery passes between excursions. `k = 0` is a bare LOW/MID/HIGH round robin

At `k = 2` the cycle is LOW, CORRECT, CORRECT, MID, CORRECT, CORRECT, HIGH, ... repeating.
The rotation stops and holds once every band has saturated.

NOTE: the only thing separating this from the POMDP is WHAT PICKS THE ACTION — the planner,
band targets, coverage banking and nav noise are shared. A performance gap is therefore
attributable to the decision layer rather than to the maneuver layer.
"""
mutable struct CyclicController <: AbstractController
    core::ScriptedCore
    k::Int
    schedule::Vector{String}
end

function CyclicController(core::ScriptedCore, k::Int)
    k >= 0 || throw(ArgumentError("k must be >= 0, got $k"))
    sched = String[]
    for b in core.band_names
        push!(sched, "EXCURSE_$b")
        append!(sched, fill("CORRECT", k))
    end
    return CyclicController(core, k, sched)
end

controller_type(c::CyclicController) = "CYCLIC_k$(c.k)"
controller_nav_sigma(c::CyclicController) = c.core.nav_sigma_km

controller_setup!(c::CyclicController, state0::AbstractVector, ::Real) =
    _scripted_setup!(c.core, state0)

function controller_command(c::CyclicController, shell_state::AbstractVector, period_s::Real)
    # Stop the rotation once every band has saturated. Without this the schedule keeps
    # excursing for science that can no longer be earned, and its accumulated orbit damage
    # measures "cannot read the coverage state" rather than how it sequences maneuvers.
    # Every scripted arm and the policy share this rule, so the comparison is on sequencing.
    action = all(>=(c.core.visit_cap), c.core.visits) ? "CORRECT" :
             c.schedule[mod(c.core.pass, length(c.schedule)) + 1]
    return _scripted_command(c.core, action, shell_state, period_s)
end

controller_observe!(c::CyclicController, peri_state::AbstractVector, ::Real, ::Symbol,
                    extra::NamedTuple, rng::AbstractRNG) =
    _scripted_observe!(c.core, peri_state, extra, rng)

# ── Science-greedy ────────────────────────────────────────────────────────────
"""
    GreedyController(core)

Always excurse to the least-sampled science band, ties broken by band order. Never corrects
unless every band has saturated, at which point it holds.

  - `core` — shared scripted state, from [`scripted_core`](@ref)

An upper bound on science appetite and a lower bound on caution: it is the policy that
ignores orbit damage entirely.
"""
mutable struct GreedyController <: AbstractController
    core::ScriptedCore
end

controller_type(::GreedyController) = "GREEDY"
controller_nav_sigma(c::GreedyController) = c.core.nav_sigma_km

controller_setup!(c::GreedyController, state0::AbstractVector, ::Real) =
    _scripted_setup!(c.core, state0)

function controller_command(c::GreedyController, shell_state::AbstractVector, period_s::Real)
    v = c.core.visits
    action = all(>=(c.core.visit_cap), v) ? "CORRECT" :
             "EXCURSE_$(c.core.band_names[argmin(v)])"
    return _scripted_command(c.core, action, shell_state, period_s)
end

controller_observe!(c::GreedyController, peri_state::AbstractVector, ::Real, ::Symbol,
                    extra::NamedTuple, rng::AbstractRNG) =
    _scripted_observe!(c.core, peri_state, extra, rng)

# ── Damage threshold ──────────────────────────────────────────────────────────
"""
    ThresholdController(core; max_residual = "R_OK")

Excurse to the least-sampled band while the orbit damage is at or below `max_residual`;
otherwise correct. The reactive use of the damage variable, against the POMDP's predictive
use of it.

  - `core` — shared scripted state, from [`scripted_core`](@ref)
  - `max_residual` — the worst damage bin at which an excursion is still attempted.
    `"R_OK"` excurses only from a clean orbit; `"R_DEGRADED"` also excurses from a degraded
    one

This baseline sees exactly what the policy sees. What it cannot see is what an action will
DO to the damage: the measured kernels say a LOW excursion needs several corrections before
the next one is safe, and no rule conditioned on the CURRENT bin can express that.
"""
mutable struct ThresholdController <: AbstractController
    core::ScriptedCore
    max_residual_idx::Int
    max_residual::String
end

function ThresholdController(core::ScriptedCore; max_residual::AbstractString = "R_OK")
    idx = findfirst(==(String(max_residual)), String.(collect(RESIDUAL_BINS)))
    idx === nothing && throw(ArgumentError(
        "max_residual must be one of $(RESIDUAL_BINS), got $max_residual"))
    return ThresholdController(core, idx, String(max_residual))
end

controller_type(c::ThresholdController) = "THRESHOLD_$(c.max_residual)"
controller_nav_sigma(c::ThresholdController) = c.core.nav_sigma_km

controller_setup!(c::ThresholdController, state0::AbstractVector, ::Real) =
    _scripted_setup!(c.core, state0)

function controller_command(c::ThresholdController, shell_state::AbstractVector,
                            period_s::Real)
    bins = String.(collect(RESIDUAL_BINS))
    cur  = findfirst(==(c.core.residual), bins)
    v    = c.core.visits
    healthy = cur !== nothing && cur <= c.max_residual_idx
    action = (healthy && !all(>=(c.core.visit_cap), v)) ?
             "EXCURSE_$(c.core.band_names[argmin(v)])" : "CORRECT"
    return _scripted_command(c.core, action, shell_state, period_s)
end

controller_observe!(c::ThresholdController, peri_state::AbstractVector, ::Real, ::Symbol,
                    extra::NamedTuple, rng::AbstractRNG) =
    _scripted_observe!(c.core, peri_state, extra, rng)

# ── Coverage accessor ─────────────────────────────────────────────────────────
# Every scripted baseline banks coverage in its shared core, from the OBSERVED bin exactly
# as `SARSOPController` does, so `run_rollout` reports `science_visits` for them too.
# Without these the generic `AbstractController` fallback returns an empty vector and a
# baseline silently reports zero science.
_controller_visits(c::CyclicController)    = c.core.visits
_controller_visits(c::GreedyController)    = c.core.visits
_controller_visits(c::ThresholdController) = c.core.visits