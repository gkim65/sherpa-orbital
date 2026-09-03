"""
common/simulate.jl — ONE rollout harness for every stationkeeping baseline.

The world is written once and only the CONTROLLER is swapped, so a POMDP-vs-MPC comparison
cannot pick up a harness difference — a different escape shell or min-periapsis bookkeeping
would otherwise read as a performance difference.

    result = run_rollout(MPCController(ref_ic = ic), ic, cr3bp_j2_eom!, period_s, horizon_s)
    result = run_rollout(SARSOPController(policy, ref_ic = ic), ic, cr3bp_j2_eom!, …; rng)

Control concept (MacKenzie §B.2.3 Strategy 3), one decision per periapsis approach:

  1. Coast under TRUTH to the next descending crossing of the `CONTROL_ALT_KM` = 600 km
     shell, watching for a crash (`PERIAPSIS_CRASH_ALT`) or escape (`ESCAPE_ALT_KM`).
  2. Ask the controller for a commanded ΔV at that shell state.
  3. Execute it (optionally through the noisy thruster model) and coast under TRUTH past
     the shell to the next periapsis, so the descending shell event re-arms rather than
     re-firing in place.
  4. Hand the controller the achieved periapsis state so it can update its own internal
     state — a belief for SARSOP, nothing for MPC.

NOTE: truth/onboard split. `truth_eom!` is a positional argument, integrated only inside
this file's coast helpers. Controllers plan exclusively with `cr3bp_eom!` via `solve_burn`
and never see it, which is what lets one controller run unchanged against CR3BP+EncJ2,
+Saturn J2, and later SPICE.

[`run_mpc`](@ref) in `baselines/mpc.jl` is a separately-validated implementation of the
same concept. The two are not expected to agree bit-for-bit: `run_mpc` records every
periapsis it passes en route to the shell for its min-periapsis metric, while this records
the one it stops at.
"""

# ── Controllers ───────────────────────────────────────────────────────────────
"""
    AbstractController

A stationkeeping controller. Implement three methods to plug a baseline into
[`run_rollout`](@ref):

  - `controller_setup!(c, state0, period_s)` — one-time initialization (e.g. computing the
    nominal apse targets). Called before the first control step.
  - `controller_command(c, shell_state, period_s)` — return
    `(dv_cmd::Vector{Float64} [km/s], label::Symbol, extra::NamedTuple)` at a shell state.
    `label` names the action for the trace; `extra` is controller-specific per-step data.
  - `controller_observe!(c, peri_state, dev_km, label, extra, rng)` — post-step update from
    the achieved periapsis. Return a NamedTuple of extra trace fields (may be empty).

Only `controller_command` may plan, and it may only use the ONBOARD model.
"""
abstract type AbstractController end

controller_setup!(::AbstractController, ::AbstractVector, ::Real) = nothing
controller_observe!(::AbstractController, ::AbstractVector, ::Real, ::Symbol,
                    ::NamedTuple, ::AbstractRNG) = NamedTuple()

"""
    controller_type(c) -> String

The baseline's name for reporting: `"MPC"` or `"SARSOP"`. A method on the controller rather
than a string compared inside the loop, so adding a baseline cannot silently fall through
to a default.
"""
controller_type(::AbstractController) = "UNKNOWN"

# ── MPC baseline ──────────────────────────────────────────────────────────────
"""
    MPCController(; ref_ic, mode, peri_target_km, apo_target_km, target_alt_km,
                  family_table)

The deterministic Strategy-3 MPC baseline as a controller: at every shell crossing, solve
for the ΔV that re-targets the next apses. Stateless apart from the nominal targets it
caches in `controller_setup!`.

  - `ref_ic` — reference orbit for the `:position`-mode nominal apse targets; `nothing`
    uses the rollout's own initial state
  - `mode` — `:position` (Strategy 3 proper, apse position-vector bounding) or `:altitude`
    (Strategy 1/2, which MacKenzie reports as failing)
  - `peri_target_km`, `apo_target_km` — apse altitude targets (km) for `:altitude`
  - `target_alt_km` — commanded periapsis altitude (km), or `nothing` to hold whatever
    orbit `ref_ic` is. This is the retargeting toggle
  - `family_table` — prebuilt family to retarget against; `nothing` uses
    [`halo_family_table_cached`](@ref), which spans roughly 19-63 km and costs ~80 s to
    continue cold. Pass one explicitly for a band outside that span, which otherwise throws

With `target_alt_km` set, `controller_setup!` looks up a genuine L1-halo family member at
that altitude ([`retarget_to_altitude`](@ref)) and takes the nominal apse targets from it,
so `CORRECT` defends the commanded orbit rather than pulling back to `ref_ic`. Left
`nothing`, the targets are computed once from `ref_ic` and never updated.

NOTE: achieved periapsis settles ~6 km ABOVE the commanded altitude. That is the
single-impulse `:position` residual floor, not the truth/onboard model gap — the truth
model tracks the onboard prediction to ~0.2 km. At a drifted operating point the six
position constraints are not simultaneously satisfiable by one impulse, so least squares
splits the error and much of it lands on periapsis radius. The bias is stable and
invariant under gain and horizon.

NOTE: that residual is LOAD-BEARING — do not add control authority to null it. Two-impulse
and solve-two-execute-one variants both deliver the commanded altitude far more accurately
and then lose the vehicle within days. Crushing six constraints into three controls is what
pins the orbit's orientation and buys a stable limit cycle.

NOTE: do not pre-shift the commanded altitude to compensate either. It converges
numerically, but it fights any controller that also reasons about altitude and pushes the
commanded value below the range the family contains, manufacturing a fake altitude floor.
The productive direction is a supervisor that knows about the bias and commands
accordingly.

NOTE: a transfer ceiling exists well above the science range — an altitude holdable when
started on its own IC can still escape when transferred to from the nominal orbit. Far
above it the onboard prediction loses an apse, `solve_burn` returns ΔV = 0, and the run is
an uncontrolled coast; check `n_failed_solves` before reading any outcome there.
"""
Base.@kwdef mutable struct MPCController <: AbstractController
    ref_ic::Union{Nothing,Vector{Float64}} = nothing
    mode::Symbol                           = :position
    peri_target_km::Float64                = PERIAPSIS_ALT_TARGET
    apo_target_km::Float64                 = APOAPSIS_ALT_TARGET
    # Retargeting toggle: nothing = pinned reference.
    target_alt_km::Union{Nothing,Float64}  = nothing
    # Prebuilt family table to retarget against. `nothing` uses `halo_family_table_cached()`.
    # Pass one to control the span/resolution, or to keep a test fast — a cold continuation
    # of the default family costs minutes.
    family_table::Union{Nothing,Vector{NamedTuple}} = nothing
    # Filled by controller_setup!.
    r_peri_nom::Union{Nothing,Vector{Float64}} = nothing
    r_apo_nom::Union{Nothing,Vector{Float64}}  = nothing
    # The family member the targets came from, when retargeting. Recorded so a run can be
    # audited: `nothing` here means the run was pinned, whatever was requested.
    ref_member::Union{Nothing,NamedTuple}      = nothing
end

controller_type(::MPCController) = "MPC"

function controller_setup!(c::MPCController, state0::AbstractVector, period_s::Real)
    if c.mode in (:position, :altitude_position)
        ref = c.ref_ic === nothing ? state0 : c.ref_ic

        # Retargeting path (opt-in). The reference must be a REAL family member: a radially
        # scaled apse vector is not a solution of the dynamics and does not produce a
        # holdable orbit (measured 2026-08-26 — every scaled attempt escaped, two with
        # negative periapsis). Throw rather than silently fall back to the pinned reference,
        # which would report a plausible run that ignored the command.
        # `:altitude_position` commands the periapsis ALTITUDE directly, so it needs no
        # family member for the periapsis half — the scalar target IS the command, and
        # `peri_target_km` carries it. Only the apoapsis position target is looked up, and
        # that stays the nominal orbit's (see the mode's docstring in `apse_residual`).
        if c.mode === :altitude_position
            c.peri_target_km = c.target_alt_km === nothing ?
                (norm(_enc_relative(next_apse_positions(ref; eom! = cr3bp_eom!)[1])) -
                 R_ENCELADUS) : c.target_alt_km
            _, c.r_apo_nom = next_apse_positions(ref; eom! = cr3bp_eom!)
            c.r_peri_nom = nothing
            return nothing
        end

        if c.target_alt_km !== nothing
            table = c.family_table === nothing ? halo_family_table_cached() : c.family_table
            member = retarget_to_altitude(table, c.target_alt_km)
            member === nothing && error(
                "MPCController: the continued L1 halo family contains no member at " *
                "periapsis altitude $(c.target_alt_km) km, so there is no orbit to hold " *
                "there. Widen halo_family_table's span, or command an altitude inside it.")
            c.ref_member = member
            ref = member.ic
        end

        # `period_s` is the control cadence, not a valid apse-search window: it contains
        # one periapsis and NO apoapsis, so a window search returns empty. Hence the
        # count-based `next_apse_positions`.
        c.r_peri_nom, c.r_apo_nom = next_apse_positions(ref; eom! = cr3bp_eom!)
    end
    return nothing
end

function controller_command(c::MPCController, shell_state::AbstractVector, period_s::Real)
    b = solve_burn(shell_state, period_s;
                   eom! = cr3bp_eom!,
                   peri_target_km = c.peri_target_km,
                   apo_target_km  = c.apo_target_km,
                   mode = c.mode,
                   r_peri_nom = c.r_peri_nom, r_apo_nom = c.r_apo_nom)
    return b.dv, :CORRECT, (converged = b.converged, residual_km = b.residual_km,
                            peri_err_km = b.peri_err_km)
end

# ── SARSOP baseline ───────────────────────────────────────────────────────────
"""
    SARSOPController(policy_data; config, ref_ic, sigma_nav_km)

The solved POMDP policy as a controller. `policy_data` is the parsed
`artifacts/policy.json` payload written by [`export_policy`](@ref) — it carries the state /
action / observation labels, the discretization, the alpha vectors, and the T/O tables, so
the greedy query and the discrete Bayes filter are reproduced here without re-deriving the
model or depending on a solver.

Per step: query the greedy action from the current belief, realize it as a commanded ΔV
(`CORRECT` → `solve_burn` toward the nominal apses; `EXCURSE_<BAND>` → `solve_burn` toward
that band's COMMANDED ALTITUDE via `mode = :altitude_position`), then fold a noisy altitude
observation into the belief. Every action burns — there is no `OBSERVE` (removed
2026-08-30; see [`actions`](@ref)).

NOTE: both onboard halves — the discrete belief filter and `solve_burn`'s CR3BP
prediction — are onboard-only. Neither sees `truth_eom!`.

## `retarget_bands` — aim an excursion at a REAL ORBIT rather than a scaled waypoint

Affects `EXCURSE_<BAND>` ONLY. `CORRECT` always returns to the original reference orbit, which
is what the action means.

`false` (DEFAULT, so existing runs reproduce): `EXCURSE_<BAND>` targets the nominal periapsis
vector radially SCALED to the band altitude. A scaled vector is not a solution of the dynamics,
so the aim is poor — measured noise-free over 30 days, the three bands miss by
**LOW −6.8 / MID −22.2 / HIGH −56.3 km** while `n_bands = 3` is banked regardless.

`true`: `EXCURSE_<BAND>` targets the apses of the genuine halo-family member at that band
altitude ([`retarget_to_altitude`](@ref)). Measured, aim improves substantially — MID goes from
−22.2 km to −7.1 km — and the residual error is then the pre-existing ~6 km single-impulse
`:position` floor rather than a bad target.

An excursion is still a ONE-PASS visit: the next `CORRECT` returns to the nominal orbit by
design. Holding a band for multiple passes would need the policy to keep choosing that band,
which is a reward/state question, not a targeting one.

NOTE: a policy solved against waypoint dynamics and rolled out with `retarget_bands = true`
tests the targeting mechanism, not a matched policy — the kernels describe the behaviour the
policy was solved against. Re-solve rather than mixing them.

Coverage is banked from the OBSERVED periapsis altitude (2026-08-30), not from which action
was commanded — so `CORRECT` banks its own band passively and a missed excursion banks
nothing. Counts saturate at `visit_cap`, so a revisit keeps paying until the cap.
"""
mutable struct SARSOPController <: AbstractController
    # Model description, straight from the exported policy.
    states::Vector{String}
    state_alt::Vector{String}
    state_visits::Vector{Vector{Int}}
    # ORBIT-DAMAGE bin per state, plus the edges to bin a live residual the same way the
    # model was calibrated. Exactly observed (the onboard solver computes it), so the belief
    # is projected onto the matching block just as it is for the visit counts.
    state_residual::Vector{String}
    residual_edges::Vector{Float64}
    actions::Vector{String}
    observations::Vector{String}
    alt_edges::Vector{Float64}
    band_names::Vector{String}
    band_bins::Vector{String}
    visit_cap::Int
    band_target_km::Dict{String,Float64}
    alphas::Matrix{Float64}         # (n_alpha, |S|)
    alpha_actions::Vector{Int}      # 1-based action index per alpha
    T::Array{Float64,3}             # [s, a, s']
    O::Matrix{Float64}              # [s, o]
    # Configuration.
    ref_ic::Union{Nothing,Vector{Float64}}
    sigma_nav_km::Float64
    # Band-retargeting toggle. `false` (default) = the pre-2026-08-29 waypoint behaviour.
    retarget_bands::Bool
    family_table::Union{Nothing,Vector{NamedTuple}}
    # Live state.
    belief::Vector{Float64}
    visits::Vector{Int}
    # The damage bin the LAST pass's onboard solve produced. Starts `:R_OK` — the vehicle
    # begins on the reference orbit, where the solve is clean.
    residual::String
    # PERSISTENT excursion reference: the band currently being aimed at, or "" for the
    # nominal orbit. This is what makes EXCURSE_* a multi-pass command rather than a
    # one-pass dip — see `controller_command`.
    active_band::String
    r_peri_nom::Union{Nothing,Vector{Float64}}
    r_apo_nom::Union{Nothing,Vector{Float64}}
    apo_nom_alt_km::Float64
    # Cache of band name -> (r_peri, r_apo) from that band's real family member, so the
    # family is continued once rather than per decision step.
    band_targets::Dict{String,Tuple{Vector{Float64},Vector{Float64}}}
end

controller_type(::SARSOPController) = "SARSOP"

"""
    SARSOPController(policy_data::AbstractDict; ref_ic = nothing,
                     sigma_nav_km = SIGMA_NAV_POS)

Build a controller from a parsed policy payload. Use
[`load_policy`](@ref) to read one from disk.
"""
function SARSOPController(policy_data::AbstractDict;
                          ref_ic::Union{Nothing,AbstractVector{<:Real}} = nothing,
                          sigma_nav_km::Real = SIGMA_NAV_POS,
                          retarget_bands::Bool = false,
                          family_table::Union{Nothing,Vector{NamedTuple}} = nothing)
    d = policy_data
    S  = String.(d["states"])
    A  = String.(d["actions"])
    Ω  = String.(d["observations"])

    alphas_v = [Float64.(α) for α in d["alphas"]]
    alphas   = reduce(vcat, (reshape(α, 1, :) for α in alphas_v))

    # T is exported nested as T[s][a][s'] and O as O[s][o]; rebuild the dense arrays.
    T_nested = d["T"]
    T = Array{Float64,3}(undef, length(S), length(A), length(S))
    for si in eachindex(S), ai in eachindex(A), spi in eachindex(S)
        T[si, ai, spi] = Float64(T_nested[si][ai][spi])
    end
    O = Matrix{Float64}(undef, length(S), length(Ω))
    for si in eachindex(S), oi in eachindex(Ω)
        O[si, oi] = Float64(d["O"][si][oi])
    end

    # Initial belief: concentrated on the declared initial state.
    belief = zeros(Float64, length(S))
    belief[findfirst(==(String(d["initial_state"])), S)] = 1.0

    band_names = String.(d["band_names"])

    # The residual dimension is REQUIRED (2026-08-31). A policy artifact without it was
    # solved against a state space that cannot represent orbit damage, so rolling it out
    # here would silently index the wrong states. Reject rather than default, for the same
    # reason `load_policy` rejects a stale action set.
    haskey(d, "state_residual") || error(
        "this policy artifact has no \"state_residual\" — it was solved against the " *
        "PRE-2026-08-31 state space, which carries no orbit-damage dimension. Re-solve " *
        "(experiments/calibrate.jl, then experiments/example.jl) rather than rolling it out.")

    return SARSOPController(
        S, String.(d["state_alt"]),
        [Int.(v) for v in d["state_visits"]],
        String.(d["state_residual"]), Float64.(d["residual_edges"]), A, Ω,
        Float64.(d["alt_edges"]), band_names,
        String.(d["band_bins"]), Int(d["visit_cap"]),
        Dict{String,Float64}(string(k) => Float64(v) for (k, v) in d["band_target_km"]),
        alphas, Int.(d["alpha_actions"]), T, O,
        ref_ic === nothing ? nothing : collect(float.(ref_ic)),
        float(sigma_nav_km),
        retarget_bands, family_table,
        belief, zeros(Int, length(band_names)), "R_OK", "", nothing, nothing, NaN,
        Dict{String,Tuple{Vector{Float64},Vector{Float64}}}(),
    )
end

"""
    SARSOPController(policy, config; ref_ic = nothing, sigma_nav_km = config.sigma_nav_km,
                     retarget_bands = false, family_table = nothing, tables = nothing)

Build a controller from a SOLVED POLICY OBJECT plus the config describing the environment
to fly it in. No JSON.

  - `policy` — an `AlphaVectorPolicy`, either straight from `solve` or read back by
    `SARSOP.load_policy(pomdp, "<stem>.out")`
  - `config` — the environment to FLY IN. `T`/`O` and every label come from here, so this
    need NOT be the config the policy was solved against
  - `tables` — measured kernels; `nothing` resolves them from `config`
  - `sigma_nav_km` — defaults to `config.sigma_nav_km`, unlike the Dict method whose
    default is the `SIGMA_NAV_POS` constant regardless of config
  - remaining keywords as for the Dict method

Returns a [`SARSOPController`](@ref) ready for [`run_rollout`](@ref).

This is the cross-θ primitive: pass a policy solved at θ_i with a config at θ_j and the
belief filter runs on θ_j's dynamics, which is what a regret matrix cell is. Prefer it over
[`export_policy`](@ref) + [`load_policy`](@ref), which serializes the DENSE `T[s][a][s']`
(|S|^2 |A| floats — gigabytes at |S| = 5627), parses it back, and then has its `T`/`O`
overwritten anyway.

NOTE: only `|S|` is checked. Alpha vectors are indexed by state ENUMERATION ORDER, so a
`config` whose states enumerate differently at the same `|S|` misindexes silently and
yields a plausible, meaningless result. Sharing `S`/`A`/`O` across a θ family — which the
regret formulation requires anyway — is what makes this safe.
"""
function SARSOPController(policy, config::StationkeepingPOMDP;
                          ref_ic::Union{Nothing,AbstractVector{<:Real}} = nothing,
                          sigma_nav_km::Real = config.sigma_nav_km,
                          retarget_bands::Bool = false,
                          family_table::Union{Nothing,Vector{NamedTuple}} = nothing,
                          tables::Union{Nothing,AltTables} = nothing)
    tbl   = tables === nothing ? load_tables(config) : tables
    S     = states(config)
    A     = actions(config)
    alphas_v, alpha_acts = alpha_vectors(policy, collect(A))

    length(first(alphas_v)) == length(S) || error(
        "policy has length-$(length(first(alphas_v))) alpha vectors but this config has " *
        "|S| = $(length(S)) — it was solved against a different state space")

    belief = zeros(Float64, length(S))
    belief[state_index(config)[SKState(config.correct_bin,
                                       _zero_visits(config), 1, :R_OK)]] = 1.0
    band_names = String.(collect(config.band_names))

    return SARSOPController(
        state_label.(S), [string(s.alt) for s in S], [collect(s.visits) for s in S],
        [string(s.residual) for s in S], collect(RESIDUAL_EDGES),
        String.(collect(A)), String.(observations(config)),
        collect(config.alt_edges), band_names, String.(collect(config.band_bins)),
        config.visit_cap,
        Dict{String,Float64}(string(k) => Float64(v) for (k, v) in config.band_target_km),
        reduce(vcat, (reshape(α, 1, :) for α in alphas_v)), alpha_acts,
        transition_matrix(config, tbl), observation_matrix(config),
        ref_ic === nothing ? nothing : collect(float.(ref_ic)),
        float(sigma_nav_km), retarget_bands, family_table,
        belief, zeros(Int, length(band_names)), "R_OK", "", nothing, nothing, NaN,
        Dict{String,Tuple{Vector{Float64},Vector{Float64}}}(),
    )
end

"""
    policy_residual_bin(c, residual_km) -> String

Bin a live onboard `solve_burn` residual (km) using the EXPORTED edges.

  - `c` — the controller, supplying the exported edges
  - `residual_km` — the live `solve_burn` residual (km)

Returns one of the residual bin labels.

NOTE: must stay identical to [`residual_bin`](@ref) in `states.jl`. It reads the edges from
the policy artifact rather than the live config, so a policy solved against one damage
discretization cannot be silently rolled out against another. A non-finite residual is the
lost-apse-pair case and bins as the most degraded live bin.
"""
function policy_residual_bin(c::SARSOPController, residual_km::Real)
    e = c.residual_edges
    isfinite(residual_km) || return "R_CRITICAL"
    residual_km < e[1] && return "R_OK"
    residual_km < e[2] && return "R_DEGRADED"
    return "R_CRITICAL"
end

"""
    load_policy(path = DEFAULT_POLICY_PATH) -> Dict

Read an exported policy JSON from disk and check its action list against this model's.

The artifact carries its OWN action labels (as it must — a policy is only valid for the
action set it was solved against), so a stale one stays loadable and its actions flow
straight into `controller_command`. Rejecting the mismatch here mirrors
[`load_tables`](@ref), which refuses a kernel artifact whose bin ordering has moved rather
than remapping it. Re-solve with `experiments/calibrate.jl` then `experiments/example.jl`.
"""
function load_policy(path::AbstractString = DEFAULT_POLICY_PATH)
    isfile(path) || error("no exported policy at $path — solve one and call export_policy")
    d = JSON.parsefile(path)

    got = sort(String.(d["actions"]))
    want = sort(String.(actions(StationkeepingPOMDP())))
    got == want || error(
        "$path was solved against actions $got, but this model has $want. A policy is only " *
        "valid for the action set it was solved against — re-solve rather than rolling this " *
        "one out (experiments/calibrate.jl, then experiments/example.jl).")

    return d
end

"""
    policy_alt_bin(c, alt_km) -> String

Bin an achieved periapsis altitude (km) using the EXPORTED edges, half-open [lo, hi).

  - `c` — the controller, supplying the exported edges
  - `alt_km` — achieved periapsis altitude (km)

Returns an altitude bin label, or `"LOST"` for a non-finite altitude.

NOTE: must stay identical to [`alt_bin`](@ref) in `states.jl`. It reads the edges from the
policy artifact rather than the live config, so a policy solved against one discretization
cannot be silently rolled out against another.
"""
function policy_alt_bin(c::SARSOPController, alt_km::Real)
    e = c.alt_edges
    isfinite(alt_km) || return "LOST"
    alt_km < e[1] && return "BELOW_20"
    alt_km < e[2] && return "A20_27"
    alt_km < e[3] && return "A27_34"
    alt_km < e[4] && return "A34_44"
    return "ABOVE_44"
end

"""Greedy action from the current belief: argmax over α·b."""
policy_action(c::SARSOPController) =
    c.actions[c.alpha_actions[argmax(c.alphas * c.belief)]]

"""
Discrete Bayes filter, b'(s') ∝ O[s', o] Σ_s T[s, a, s'] b(s), with `o` over altitude bins.
Falls back to the prediction alone if the observation has zero likelihood everywhere.
"""
function policy_update_belief!(c::SARSOPController, action::String, obs::String)
    ai = findfirst(==(action), c.actions)
    oi = findfirst(==(obs), c.observations)
    bp = vec(c.belief' * c.T[:, ai, :])
    bp .*= c.O[:, oi]
    tot = sum(bp)
    if tot <= 0.0
        bp = vec(c.belief' * c.T[:, ai, :])
        tot = sum(bp)
    end
    c.belief = bp ./ tot
    return c.belief
end

"""
Re-concentrate the belief onto the known-`visits` AND known-`residual` block.

The visit counts are a DETERMINISTIC function of the observed altitude bin, so the
controller always knows them exactly and belief mass on any other visit tuple is spurious.
Terminal states are always kept. A no-op if masking would zero everything.

The residual is projected the same way and for a stronger reason: the visit counts are
exactly known because they derive from the observation, while the residual is exactly known
because the onboard solver computed it. Leaving mass on the other damage bins would model
uncertainty the spacecraft does not have.

NOTE: this stays licensed only because coverage gates on the OBSERVED altitude. Gating on
the true altitude would make coverage stochastic and partially observed, breaking the
exact-observation assumption the projection rests on. The cost is accuracy, not soundness —
a misbinned pass banks the wrong band and the belief then confidently carries that wrong
count.
"""
function policy_reconcentrate_cov!(c::SARSOPController)
    keep = [(c.state_alt[i] in ("LOST", "CRASHED") ||
             (c.state_visits[i] == c.visits && c.state_residual[i] == c.residual)) ?
            1.0 : 0.0 for i in eachindex(c.states)]
    masked = c.belief .* keep
    tot = sum(masked)
    tot > 0 && (c.belief = masked ./ tot)
    return c.belief
end

function controller_setup!(c::SARSOPController, state0::AbstractVector, period_s::Real)
    ref = c.ref_ic === nothing ? state0 : c.ref_ic
    # Count-based — see the MPCController setup above for why period_s is not a valid
    # apse-search window.
    c.r_peri_nom, c.r_apo_nom = next_apse_positions(ref; eom! = cr3bp_eom!)
    c.apo_nom_alt_km = norm(_enc_relative(c.r_apo_nom)) - R_ENCELADUS
    return nothing
end

"""
    _sarsop_band_targets(c, band_name) -> (r_peri, r_apo)

Apse POSITION targets for an `EXCURSE_<BAND>` action.

`retarget_bands = false` (default): the historical behaviour — the nominal periapsis vector
radially scaled to the band altitude, which is a WAYPOINT, not an orbit (see
[`_scale_to_altitude`](@ref)).

`retarget_bands = true`: the apses of the genuine halo-family member at that band altitude
([`retarget_to_altitude`](@ref)), so the band is an orbit the vehicle can settle onto. Cached
per band, because continuing the family is expensive relative to a decision step.

Throws if the family has no member at a band's altitude — a band that cannot be realised as an
orbit is a configuration error, and silently substituting a scaled waypoint would hide it.
"""
function _sarsop_band_targets(c::SARSOPController, band_name::AbstractString)
    alt = c.band_target_km[band_name]
    if !c.retarget_bands
        return (_scale_to_altitude(c.r_peri_nom, alt),
                _scale_to_altitude(c.r_apo_nom, c.apo_nom_alt_km))
    end
    # PHASE-MATCHED retarget. The family member supplies the periapsis RADIUS; the
    # DIRECTION comes from our own nominal periapsis, and apoapsis stays pinned to nominal.
    #
    # NOTE: do not import the member's apse POSITIONS. `next_apse_positions(member.ic)`
    # gives where THAT member's periapsis sits at ITS OWN epoch, which is unrelated to our
    # phase — two orbits ~20 km apart in altitude can be hundreds of km apart in position,
    # so the solver burns enormously and loses the vehicle. Take the RADIUS from the member
    # and the DIRECTION from our own nominal.
    #
    # `MPCController.target_alt_km` avoids this by resolving the member at setup from the
    # IC, where vehicle and reference are phase-aligned; the bug only appears mid-flight.
    key = String(band_name)
    return get!(c.band_targets, key) do
        table = c.family_table === nothing ? halo_family_table_cached() : c.family_table
        member = retarget_to_altitude(table, alt)
        member === nothing && error(
            "SARSOPController: band $key targets $(alt) km, but the continued L1 halo " *
            "family contains no member at that periapsis altitude, so it is not an orbit " *
            "that can be held. Widen the family table or change band_target_km.")
        (_scale_to_altitude(c.r_peri_nom, member.info.periapsis_alt_km),
         collect(c.r_apo_nom))
    end
end

function controller_command(c::SARSOPController, shell_state::AbstractVector, period_s::Real)
    action = policy_action(c)

    if action == "CORRECT"
        # CORRECT CLEARS the active excursion and re-aims at the ORIGINAL reference orbit.
        # It is the only way back to nominal, which is what makes the excursion reference
        # persistent rather than one-pass.
        c.active_band = ""
        b = solve_burn(shell_state, period_s; eom! = cr3bp_eom!,
                       mode = :position,
                       r_peri_nom = c.r_peri_nom, r_apo_nom = c.r_apo_nom)
        return b.dv, :CORRECT, (band = 0, converged = b.converged,
                                residual_km = b.residual_km,
                                peri_err_km = b.peri_err_km)
    end

    # EXCURSE_<BAND>: SET the active reference to that band and aim at it.
    #
    # A PERSISTENT command, not a one-pass dip: the band stays the reference until CORRECT
    # clears it, so choosing the same EXCURSE again continues the approach. Single-impulse
    # authority is poor, so settling over several passes is what actually reaches a
    # commanded altitude.
    #
    # `:altitude_position` constrains the periapsis by ALTITUDE — the quantity actually
    # commanded — while keeping the apoapsis-position constraints that pin the orientation.
    # A pure `:position` aim cannot deliver a band: every commanded altitude lands in
    # roughly the same bin, leaving the other EXCURSE rows unmeasured.
    band_name = replace(action, "EXCURSE_" => "")
    band_idx  = findfirst(==(band_name), c.band_names)
    c.active_band = band_name
    _, ra = _sarsop_band_targets(c, band_name)
    b = solve_burn(shell_state, period_s; eom! = cr3bp_eom!,
                   mode = :altitude_position,
                   peri_target_km = c.band_target_km[band_name], r_apo_nom = ra)
    return b.dv, Symbol(action), (band = band_idx, converged = b.converged,
                                  residual_km = b.residual_km,
                                  peri_err_km = b.peri_err_km)
end

function controller_observe!(c::SARSOPController, peri_state::AbstractVector,
                             dev_km::Real, label::Symbol, extra::NamedTuple,
                             rng::AbstractRNG)
    true_alt = norm(_enc_relative(peri_state[1:3])) - R_ENCELADUS
    obs_alt  = observe_altitude(true_alt, rng; sigma_r = c.sigma_nav_km)

    true_bin = policy_alt_bin(c, true_alt)
    obs_bin  = policy_alt_bin(c, obs_alt)

    # Bank on the OBSERVED bin, indifferent to which action ran — so CORRECT banks its own
    # band passively and a missed excursion banks nothing. Saturates at `visit_cap`.
    bi = findfirst(==(obs_bin), c.band_bins)
    bi === nothing || (c.visits[bi] = min(c.visits[bi] + 1, c.visit_cap))

    # Update the damage bin BEFORE re-concentrating: the residual is what the onboard
    # solver reported for the burn this pass just flew, so it is known exactly and becomes
    # the conditioning the next decision is made under.
    c.residual = policy_residual_bin(c, get(extra, :residual_km, NaN))

    policy_update_belief!(c, String(label), obs_bin)
    policy_reconcentrate_cov!(c)

    return (true_bin = true_bin, obs_bin = obs_bin,
            true_alt_km = true_alt, obs_alt_km = obs_alt,
            visits = copy(c.visits), residual = c.residual)
end

# ── Geometry helpers ──────────────────────────────────────────────────────────
"""Enceladus-relative position from a barycentre-frame position vector (km)."""
function _enc_relative(r::AbstractVector{<:Real})
    out = collect(float.(r[1:3]))
    out[1] -= X_ENCELADUS
    return out
end

"""
    _scale_to_altitude(r_nom, alt_km) -> Vector{Float64}

Scale a nominal apse POSITION vector radially about Enceladus to altitude `alt_km`,
returning a barycentre-frame position. Used to place an excursion's periapsis target
without changing the orbit's geometry, only its radius.

  - `r_nom` — nominal apse position vector (km, barycentre frame)
  - `alt_km` — target altitude above the surface (km)

Returns the scaled position (km, barycentre frame).

NOTE: a scaled vector is NOT a reference orbit — do not use this to build one. It is not a
solution of the dynamics, so nothing can settle onto it; scaled-apoapsis references escape.
It works here only because an `EXCURSE_*` target is a waypoint the next `CORRECT` abandons.
To hold a commanded altitude use [`retarget_to_altitude`](@ref), which returns a genuine
family member, or `MPCController`'s `target_alt_km` toggle, which wraps it.
"""
function _scale_to_altitude(r_nom::AbstractVector{<:Real}, alt_km::Real)
    rr  = _enc_relative(r_nom)
    out = rr .* ((R_ENCELADUS + alt_km) / norm(rr))
    out[1] += X_ENCELADUS
    return out
end

# ── The rollout ───────────────────────────────────────────────────────────────
"""
    run_rollout(controller, state0, truth_eom!, period_s, horizon_s; kwargs...) -> NamedTuple

Roll `controller` out against a TRUTH model over `horizon_s` seconds. One decision per
periapsis approach, triggered at the descending `CONTROL_ALT_KM` shell.

  - `controller` — an [`AbstractController`](@ref): [`MPCController`](@ref) or
    [`SARSOPController`](@ref). This is the `type` switch; the world is identical.
  - `state0` — initial barycentre-frame state (km, km/s), typically the halo IC.
  - `truth_eom!` — world dynamics ([`cr3bp_j2_eom!`](@ref),
    [`cr3bp_saturn_enc_j2_eom!`](@ref), later SPICE). An argument, never closed over.
  - `period_s` — the inter-periapsis interval (s).
  - `rng` — random stream for the thruster and nav noise. Required whenever
    `noisy_thruster = true` or the controller draws observations; defaults to a fresh
    `Xoshiro(0)` so a caller cannot accidentally consume global RNG state.
  - `noisy_thruster` — execute ΔV through [`apply_dv_noisy`](@ref) rather than
    [`apply_dv`](@ref). **Default `false`** — see below.

    NOTE: the default is `false` to MATCH THE DEFAULT KERNELS. `artifacts/tables.json` is
    calibrated noise-free (`meta.theta.noisy_thruster`), so a noisy rollout of a policy
    solved against THOSE is mismatched — the policy acts on a transition model that
    understates execution error. Set this `true` deliberately, and against kernels
    calibrated the same way (`artifacts/tables_noisy_gaussian2.0.json` is the B-24 Model 2
    pair). The severe mismatch symptom on record — missing the science bands badly and
    sitting deep in `R_CRITICAL` — was measured under a since-removed unphysical uniform
    law that overstated execution error by roughly 5x.

  - `thruster_sigma_pct` — 1σ burn-magnitude error in percent, forwarded to
    [`apply_dv_noisy`](@ref); ignored when `noisy_thruster = false`. Must MATCH the value
    the kernels were calibrated at (`meta.theta.thruster_sigma_pct`), or the policy is
    flown under a law it was not solved against.
  - `max_steps` — hard cap on control steps (safety guard; outcome `:max_steps`).

Returns a NamedTuple with the SAME fields for every baseline, so a comparison table needs
no per-baseline special-casing:

  - `type` — `"MPC"` / `"SARSOP"`
  - `survived`, `outcome` — `:held`, `:idle`, `:crash`, `:escape`, `:max_steps`
  - `survival_time_s`, `n_steps`, `n_burns`, `total_dv_ms`
  - `n_solves`, `n_failed_solves` — how many control steps ran a burn solve, and how many of
    those did not converge. CHECK THESE before reading any other number: a failed solve
    returns ΔV = 0, so a run in which every solve failed had no control at all, yet reports
    a plausible `n_steps` and `survived = true`. `n_failed_solves == n_solves > 0` means the
    result describes an uncontrolled coast whatever the outcome field says.
  - `min_peri_alt_km` — smallest periapsis altitude visited (km)
  - `science_visits`, `n_bands`, `n_samples` — per-band visit counts, how many distinct
    bands were sampled at least once, and the total sample count. Empty / 0 for MPC.
    Banked on the OBSERVED altitude, not ground truth: a band is credited when the observed
    periapsis lands in its bin, so nav noise misbins passes near an edge. Use
    `peri_alts_km` for where the passes truly went. Gating on the true altitude was
    rejected deliberately — it makes coverage stochastic and partially observed, breaking
    the exact-observation assumption behind `policy_reconcentrate_cov!`.
  - `peri_alts_km`, `peri_lats_deg` — achieved periapsis altitude (km) and latitude (deg)
    per control step. Latitude is what the south-polar science case turns on and is
    invisible in an altitude or a deviation norm.
  - `final_state` — the full final 6-state (km, km/s). Lets a rollout be CHAINED, which
    `steps[].peri_pos` (position only) cannot support — needed for a staged transfer that
    advances the commanded altitude one rung at a time.
  - `max_dev_trans_km` — largest TRANSVERSE (along-track + out-of-plane) periapsis
    deviation. Altitude errors are radial; this is the part of a position miss that a
    radius cannot see, and it is where most of the `:position` residual lives.
  - `steps` — per-step trace records
"""
function run_rollout(
    controller::AbstractController,
    state0::AbstractVector{<:Real},
    truth_eom!,
    period_s::Real,
    horizon_s::Real;
    rng::AbstractRNG = Xoshiro(0),
    # Default false to match the DEFAULT kernels (`artifacts/tables.json`, measured
    # noise-free). Set it true together with the matching `thruster_sigma_pct` and the
    # kernels calibrated at that sigma; noisy-here-but-noise-free-kernels tests a mismatch.
    noisy_thruster::Bool = false,
    thruster_sigma_pct::Real = THRUSTER_SIGMA_PCT_B24_MODEL2,
    rtol_truth::Real = RTOL_TRUTH,
    atol_truth::Real = ATOL_TRUTH,
    max_steps::Integer = 2000,
    verbose::Bool = false,
)
    state     = collect(float.(state0))
    horizon_s = float(horizon_s)
    period_s  = float(period_s)
    t_now     = 0.0

    total_dv_ms  = 0.0
    n_burns      = 0
    min_peri_alt = Inf
    steps        = NamedTuple[]

    # Failed-solve bookkeeping. A non-converged solve_burn returns ΔV = 0, which is
    # INDISTINGUISHABLE from a deliberate decision not to burn — so a run whose every burn
    # failed looks, in the summary, exactly like a run that chose to coast. These counters
    # are what make those two outcomes different at a glance. `n_solves` is the denominator:
    # `n_failed_solves` alone cannot tell "0 of 0" from "0 of 40".
    n_solves        = 0
    n_failed_solves = 0

    controller_setup!(controller, state, period_s)

    # The deviation metric is the same one the POMDP's dev bins are defined on: the
    # apse-POSITION deviation from the nominal periapsis. Computed here (not in the
    # controller) so both baselines report the identical quantity.
    #
    # NOTE: this is the ORIGINAL reference, always. A controller deliberately holding a
    # DIFFERENT orbit (`MPCController.target_alt_km`) therefore reports a large deviation
    # even when its hold is perfect, so anything reading this signal will think the vehicle
    # is off-course while it does exactly what it was commanded. Distinct from the ~6 km
    # periapsis bias, which is the single-impulse residual floor — the two were conflated
    # once already.
    #
    # Not fixed here on purpose: `dev_of` is defined once, outside the controllers, so both
    # baselines report the identical quantity and the comparison stays apples-to-apples.
    r_peri_nom, _ = next_apse_positions(
        controller isa MPCController && controller.ref_ic !== nothing ? controller.ref_ic :
        controller isa SARSOPController && controller.ref_ic !== nothing ? controller.ref_ic :
        state0;
        eom! = cr3bp_eom!)
    dev_of(u) = norm(u[1:3] .- r_peri_nom)

    finish(outcome, t_loss, survived) = (
        type            = controller_type(controller),
        survived        = survived,
        outcome         = outcome,
        survival_time_s = t_loss,
        n_steps         = length(steps),
        n_burns         = n_burns,
        n_solves        = n_solves,
        n_failed_solves = n_failed_solves,
        total_dv_ms     = total_dv_ms,
        min_peri_alt_km = isfinite(min_peri_alt) ? min_peri_alt : NaN,
        # Achieved-geometry summary. `peri_alts_km` is the per-pass altitude sequence, so a
        # commanded science excursion can be checked against where it actually went rather
        # than trusted from the noisy read the policy banked on (see `science_visits`).
        peri_alts_km    = [s.peri_alt_km for s in steps],
        peri_lats_deg   = [s.peri_lat_deg for s in steps],
        max_dev_trans_km = isempty(steps) ? NaN :
            maximum(s -> isfinite(s.dev_transverse_km) ? s.dev_transverse_km : -Inf, steps),
        science_visits  = _controller_visits(controller),
        n_bands         = count(>(0), _controller_visits(controller)),
        n_samples       = sum(_controller_visits(controller)),
        # Full final 6-state (km, km/s), so a rollout can be CHAINED — e.g. a staged transfer
        # that advances the reference one rung at a time and hands the achieved state forward.
        # `steps[].peri_pos` is position only, which is not enough to resume propagation.
        # Empty when a leg ended with no usable state (`:none`).
        final_state     = copy(state),
        steps           = steps,
    )

    while t_now < horizon_s
        length(steps) >= max_steps && return finish(:max_steps, t_now, false)

        # 1. Coast under TRUTH to the control shell.
        shell = _coast_to(truth_eom!, state, horizon_s - t_now,
                          _terminal_shell_callback, CONTROL_ALT_KM;
                          rtol = rtol_truth, atol = atol_truth)
        if shell.outcome !== :ok
            shell.outcome === :crash && (min_peri_alt = min(min_peri_alt, PERIAPSIS_CRASH_ALT))
            # `:none` — never came back inbound before the horizon: survived, but idle.
            shell.outcome === :none && break
            return finish(shell.outcome, t_now + shell.t, false)
        end
        t_now += shell.t

        # 2. Ask the controller for a command (ONBOARD planning only).
        dv_cmd, label, extra = controller_command(controller, shell.u, period_s)

        # Every action burns now that OBSERVE is gone, so every step ran a solve.
        n_solves += 1
        extra.converged || (n_failed_solves += 1)

        # 3. Execute it.
        dv_applied, eta_eff = noisy_thruster ?
            apply_dv_noisy(dv_cmd, rng; sigma_pct = thruster_sigma_pct) :
            (apply_dv(dv_cmd), 1.0)
        dv_ms = norm(dv_applied) * 1.0e3
        total_dv_ms += dv_ms
        dv_ms > 0.0 && (n_burns += 1)

        state_post = copy(shell.u)
        state_post[4:6] .+= dv_applied

        # 4. Coast under TRUTH past the shell to the next periapsis (re-arms the trigger).
        peri = _coast_to(truth_eom!, state_post, horizon_s - t_now,
                          _terminal_periapsis_callback_arg, nothing;
                          rtol = rtol_truth, atol = atol_truth)
        if peri.outcome !== :ok
            peri.outcome === :crash && (min_peri_alt = min(min_peri_alt, PERIAPSIS_CRASH_ALT))
            # Record the step that led to the loss before returning.
            # Same field schema as the nominal push below — a lost leg has no achieved
            # periapsis, so the position fields are NaN rather than absent. Keeping the
            # schema uniform means a trace can be tabulated without per-row branching.
            push!(steps, (t_s = t_now, action = label, dv_ms = dv_ms, eta_eff = eta_eff,
                          true_dev_km = NaN, converged = extra.converged,
                          residual_km = extra.residual_km,
                          peri_err_km = get(extra, :peri_err_km, NaN),
                          peri_alt_km = NaN, peri_lat_deg = NaN, peri_lon_deg = NaN,
                          peri_pos = fill(NaN, 3),
                          dev_radial_km = NaN, dev_transverse_km = NaN,
                          extra = (true_bin = uppercase(String(peri.outcome)),
                                   obs_bin  = uppercase(String(peri.outcome)),
                                   true_alt_km = NaN, obs_alt_km = NaN,
                                   visits = copy(_controller_visits(controller)))))
            peri.outcome === :none && break
            return finish(peri.outcome, t_now + peri.t, false)
        end
        t_now += peri.t
        state  = copy(peri.u)

        # 5. Measure, and let the controller update its internal state.
        min_peri_alt = min(min_peri_alt, altitude(state))
        dev_km = dev_of(state)
        obs_extra = controller_observe!(controller, state, dev_km, label, extra, rng)

        # Record WHERE the periapsis actually landed, not just how far off it was.
        # `true_dev_km` is a single norm, and altitude is a single radius — between them
        # they cannot distinguish "right place" from "right distance, wrong direction".
        # Two errors are invisible in altitude alone and both matter here:
        #   * along-track (phase) error — correct altitude at the wrong point on the orbit,
        #     which is most of the ~10 km position residual the :position mode cannot null;
        #   * out-of-plane (z) error — the latitude of the periapsis pass, i.e. exactly the
        #     quantity the south-polar science case turns on (the IC's periapsis is at
        #     +87 deg NORTH, a known open defect). A z-error cannot show up in a radius.
        # Logged per step so a run can be audited after the fact rather than re-run.
        rel_peri  = _enc_relative(state)
        peri_alt  = altitude(state)
        peri_lat  = asind(clamp(rel_peri[3] / norm(rel_peri), -1.0, 1.0))
        peri_lon  = atand(rel_peri[2], rel_peri[1])
        dev_vec   = state[1:3] .- r_peri_nom
        # Radial / transverse split of the deviation: how much of the miss is altitude
        # (radial) versus along-track+out-of-plane (transverse, invisible to altitude).
        r_hat     = rel_peri ./ norm(rel_peri)
        dev_rad   = dot(dev_vec, r_hat)
        dev_trans = norm(dev_vec .- dev_rad .* r_hat)

        push!(steps, (t_s = t_now, action = label, dv_ms = dv_ms, eta_eff = eta_eff,
                      true_dev_km = dev_km, converged = extra.converged,
                      residual_km = extra.residual_km,
                      peri_err_km = get(extra, :peri_err_km, NaN),
                      peri_alt_km = peri_alt, peri_lat_deg = peri_lat,
                      peri_lon_deg = peri_lon, peri_pos = copy(state[1:3]),
                      dev_radial_km = dev_rad, dev_transverse_km = dev_trans,
                      extra = obs_extra))

        verbose && @printf("  step %3d t=%7.2f hr  dev=%8.2f km  %-13s ΔV=%6.2f m/s  η=%.3f\n",
                           length(steps), t_now / 3600, dev_km, String(label), dv_ms, eta_eff)
    end

    # Horizon reached with no crash and no escape. `:idle` flags that the controller
    # stopped triggering (the orbit drifted off the shell) before the horizon.
    return finish(t_now < horizon_s ? :idle : :held, horizon_s, true)
end

"""Per-band visit counts of a controller (empty for baselines with no science state)."""
_controller_visits(::AbstractController) = Int[]
_controller_visits(c::SARSOPController)  = c.visits

"""
    _coast_to(eom!, state, horizon, make_primary, arg; rtol, atol) -> NamedTuple

Propagate under TRUTH until the primary event fires, or a crash or an escape ends the leg
first. `make_primary` is one of the terminal-callback constructors from `baselines/mpc.jl`;
`arg` is its altitude argument (`nothing` for the periapsis event).

Returns `(outcome, t, u)` with `outcome ∈ (:ok, :crash, :escape, :none)`. This is the ONE
place the three-way "which condition ended this leg?" dispatch is written, so both
baselines inherit identical crash and escape semantics.
"""
function _coast_to(eom!, state::AbstractVector{<:Real}, horizon::Real,
                   make_primary, arg; rtol::Real, atol::Real)
    primary = _EventRecord()
    crash   = _EventRecord()
    esc     = _EventRecord()

    primary_cb = arg === nothing ? make_primary(primary) : make_primary(arg, primary)

    propagate(
        eom!, state, (0.0, float(horizon));
        callback = CallbackSet(
            primary_cb,
            _terminal_shell_callback(PERIAPSIS_CRASH_ALT, crash),
            _terminal_escape_callback(ESCAPE_ALT_KM, esc),
        ),
        rtol = rtol, atol = atol,
    )

    # Ordering matters: escape and crash are terminal failures and take precedence over a
    # primary event that may have fired at a very similar time.
    esc.fired   && return (outcome = :escape, t = esc.t,   u = esc.u)
    crash.fired && return (outcome = :crash,  t = crash.t, u = crash.u)
    primary.fired && return (outcome = :ok,   t = primary.t, u = primary.u)
    return (outcome = :none, t = float(horizon), u = Float64[])
end

"""Adapter so the periapsis callback has the same `(arg, record)` shape as the shell one."""
_terminal_periapsis_callback_arg(rec::_EventRecord) = _terminal_periapsis_callback(rec)

# ── Reporting ─────────────────────────────────────────────────────────────────
"""
    summarize_rollouts(results) -> NamedTuple

Aggregate repeated stochastic rollouts into a survival rate and medians.

  - `results` — a vector of [`run_rollout`](@ref) results

Returns a NamedTuple of `n`, `survival_rate`, medians of min-periapsis, ΔV, survival time
and band count, plus the solve counters. `survived` counts `:held` and `:idle` — no crash,
no escape.

`failed_solve_rate` is the fraction of all attempted burn solves across all rollouts that did
not converge. Read it FIRST: a survival rate computed over rollouts whose burns all failed
is measuring an uncontrolled coast, not a controller. A value near 1.0 invalidates every
other field here rather than qualifying it.
"""
function summarize_rollouts(results::AbstractVector)
    n = length(results)
    n == 0 && return (n = 0, survival_rate = NaN, min_peri_median = NaN,
                      dv_median = NaN, surv_time_median_d = NaN, bands_median = NaN,
                      n_solves = 0, n_failed_solves = 0, failed_solve_rate = NaN)
    survived = count(r -> r.outcome in (:held, :idle), results)
    peris = [r.min_peri_alt_km for r in results if isfinite(r.min_peri_alt_km)]
    # `get` so a caller can still summarize result NamedTuples from an older run.
    tot_solves = sum(r -> get(r, :n_solves, 0), results)
    tot_failed = sum(r -> get(r, :n_failed_solves, 0), results)
    return (
        n                  = n,
        survival_rate      = survived / n,
        min_peri_median    = isempty(peris) ? NaN : median(peris),
        dv_median          = median([r.total_dv_ms for r in results]),
        surv_time_median_d = median([r.survival_time_s for r in results]) / 86400.0,
        bands_median       = median([r.n_bands for r in results]),
        n_solves           = tot_solves,
        n_failed_solves    = tot_failed,
        failed_solve_rate  = tot_solves == 0 ? NaN : tot_failed / tot_solves,
    )
end

"""
    discounted_return(result, pomdp; gamma = pomdp.discount, score_on = :true_bin) -> Float64

The discounted return `Σ γᵗ r(sₜ, aₜ)` of a [`run_rollout`](@ref) result, scored under
`pomdp`'s reward function.

**This is the quantity offline-policy-reuse regret is defined on**
(`R_ij = V^{π_i}_{θ_i}(b₀) − V^{π_j}_{θ_i}(b₀)`, where `V^π_θ(b₀) = E[Σ γᵗ R(sₜ,aₜ)]`).
`run_rollout` otherwise reports survival, ΔV, band visits and achieved altitudes — none of
which is a return — so an evaluator comparing policies across environments needs this.

Post-hoc over `result.steps` rather than accumulated inside the rollout, deliberately: the
reward belongs to a POMDP (a `θ`), the trajectory belongs to the physics, and the same
trajectory is legitimately scored under several `θ` when filling a regret matrix. Folding the
reward into the rollout would tie one trajectory to one reward and make the off-diagonal
entries impossible without re-flying.

NOTE: this is the TRUTH-MODEL return, not the discrete-model one. `POMDPs.simulate` with a
`RolloutSimulator` samples states from `T` and takes milliseconds; this scores states the
CR3BP dynamics actually produced and takes seconds. Both are valid, they are not
interchangeable, and a regret matrix must use one or the other throughout.

NOTE: intensity is drawn from `T`, not measured. The physics produces the altitude bin and
visit counts but has no notion of plume intensity, so that term enters through the reward's
expectation over `T`, exactly as it does in the model. Sampling it here as well would count
science twice.

`r(s,a)` is charged at the state the decision was made FROM, matching the `r(sₜ, aₜ)`
convention and the kernel's own accounting (`calibrate.jl` charges a lost apse pair to the
bin the decision was made in, not the bin the vehicle fled to). A terminal step contributes
the mission-loss cost through the same reward, since the reward already carries it as an
expectation over `T`; `t` still advances so the discount is right.

  - `result` — a [`run_rollout`](@ref) result
  - `pomdp` — the `θ` whose reward the trajectory is scored under
  - `gamma` — discount per step (dimensionless)
  - `score_on` — which altitude bin drives the scored trajectory, `:true_bin` or `:obs_bin`

Returns the discounted return (reward units).

# Scoring on truth versus on the observation

`:true_bin` (the DEFAULT) advances the scored trajectory on the altitude the vehicle
ACTUALLY reached: the successor bin, the banked visit counts, and the terminal test all
come from truth. `:obs_bin` instead replays what the controller believed, binning exactly
as the policy did.

NOTE: the default is `:true_bin`, so returns reported before this keyword existed were
`:obs_bin` numbers and do not compare against current ones.

Truth is the right default because a return is a property of the trajectory, not of the
observer. Under `:obs_bin` nav noise leaks into the score through three channels at once —
the successor bin the next reward is charged at, the visit counts the science term reads,
and the terminal test — so a noisy sensor can credit the vehicle with a science band it
never reached and end the scored episode on a misread. That makes measurement error a
source of reward, and returns then RISE with `sigma_nav_km`, which inverts the sweep it is
meant to measure.

`:obs_bin` stays reachable because "score it the way the policy saw it" is a legitimate
question, and it is the like-for-like setting when comparing against a controller whose
own banking is observation-driven.

Only the action and the solve residual come from the controller under either setting.
`MPCController` records an EMPTY `extra` (only `SARSOPController` observes), so both
settings fall back to binning `peri_alt_km` for it — which is already the truth, leaving
an MPC rollout scored identically either way.
"""
function discounted_return(result, pomdp::StationkeepingPOMDP;
                           gamma::Real = pomdp.discount,
                           score_on::Symbol = :true_bin)
    score_on in (:true_bin, :obs_bin) || throw(ArgumentError(
        "score_on must be :true_bin or :obs_bin, got :$score_on"))
    T, _ = model_tables(pomdp)
    r    = reward_function(pomdp, T)
    idx  = state_index(pomdp)

    total = 0.0
    visits = _zero_visits(pomdp)          # the run starts with nothing banked
    alt = :A34_44                         # the limit cycle the rollout is initialised on
    res = :R_OK                           # starts on the reference orbit: a clean solve

    for (t, step) in enumerate(result.steps)
        s = SKState(alt, visits, 1, res)
        # A state the model does not enumerate cannot be scored; skip rather than guess.
        haskey(idx, s) || break
        total += gamma^(t - 1) * r(s, Symbol(step.action))

        # Advance to where the pass landed. Under `:true_bin` that is the altitude the
        # vehicle ACTUALLY reached; under `:obs_bin` it is what the controller believed.
        # Both fall back to binning the achieved altitude, because `MPCController` records
        # an EMPTY `extra` (only `SARSOPController` observes) and the baseline must still
        # be scoreable — comparing a POMDP policy against MPC is the point.
        nxt = if hasproperty(step.extra, score_on)
            Symbol(getproperty(step.extra, score_on))
        elseif isfinite(step.peri_alt_km)
            alt_bin(pomdp, step.peri_alt_km)
        else
            :LOST
        end
        # Terminating on `nxt` means `:true_bin` ends the episode on a real loss, while
        # `:obs_bin` can end it on a misread. That is the point of the distinction.
        isterminal_alt(nxt) && break      # cost already charged above

        # Visit counts. Under `:true_bin`, bank from the altitude actually achieved —
        # never from `extra.visits`, which `controller_observe!` banks from the OBSERVED
        # bin and which would otherwise re-introduce phantom coverage the reward pays for.
        # Saturates at the cap exactly as `transition.jl` does.
        visits = if score_on === :obs_bin && hasproperty(step.extra, :visits)
            # `SARSOPController` stores visits as a Vector; `SKState` needs an NTuple.
            v = step.extra.visits
            v isa Tuple ? v : ntuple(i -> Int(v[i]), n_bands(pomdp))
        else
            b = band_of_alt(pomdp, step.peri_alt_km)
            b === nothing ? visits : visit_inc(visits, b, pomdp.visit_cap)
        end
        # Advance the damage bin from the residual the onboard solve actually reported.
        # Recorded on every step by both controllers, so this needs no controller-specific
        # fallback the way the visit counts do.
        res = residual_bin(step.residual_km)
        alt = nxt
    end
    return total
end

"""
    simulate(args...; kwargs...)

Deprecated alias for [`run_rollout`](@ref). Renamed 2026-08-31 because `POMDPs.simulate`
exists and the collision breaks any consumer doing `using POMDPs` alongside
`using SherpaOrbital`. Not exported; use [`run_rollout`](@ref) in new code.
"""
simulate(args...; kwargs...) = run_rollout(args...; kwargs...)

