"""
common/simulate.jl — ONE rollout harness for every stationkeeping baseline.

WHY THIS FILE EXISTS
The Python reference had two independently-written rollout loops: `baselines/mpc.py`'s
`run_mpc` and `baselines/pomdp_rollout.py`'s `run_pomdp_rollout`. They shared a control
concept (one decision per periapsis approach, triggered at the 600-km descending shell) and
the same crash/escape definitions, but they implemented it twice. Any drift between them —
a different escape shell, a different min-periapsis bookkeeping, a burn counted in one and
not the other — would show up as a POMDP-vs-MPC performance difference that is really a
harness difference. Here the WORLD is written once and only the CONTROLLER is swapped, so
the comparison is apples-to-apples by construction rather than by careful maintenance.

    result = simulate(MPCController(ref_ic = ic), ic, cr3bp_j2_eom!, ONE_REV_S, horizon_s)
    result = simulate(SARSOPController(policy, ref_ic = ic), ic, cr3bp_j2_eom!, …; rng)

⚠️ TRUTH/ONBOARD SPLIT (CLAUDE.md rule). `truth_eom!` is a positional ARGUMENT and is
integrated ONLY inside this file's coast helpers. Every controller plans exclusively with
the onboard model (`cr3bp_eom!`, via `solve_burn`) and never sees `truth_eom!`. That is what
lets one controller run unchanged against CR3BP+EncJ2, +Saturn J2, and later SPICE.

CONTROL CONCEPT (identical for both baselines, from MacKenzie §B.2.3 Strategy 3)
Each control step:
  1. Coast under TRUTH to the next descending crossing of the `CONTROL_ALT_KM` = 600 km
     shell, watching for a crash (`PERIAPSIS_CRASH_ALT`) or escape (`ESCAPE_ALT_KM`).
  2. Ask the controller for a commanded ΔV at that shell state.
  3. Execute it (optionally through the noisy thruster model) and coast under TRUTH past
     the shell to the next periapsis, so the descending shell event re-arms rather than
     re-firing in place. That sets the cadence to one burn per periapsis approach.
  4. Hand the controller the achieved periapsis state so it can update its own internal
     state (a belief, for SARSOP; nothing, for MPC).

RELATIONSHIP TO `run_mpc`
`run_mpc` in `baselines/mpc.jl` is retained UNCHANGED as the validated Session-3 reference
— it is what reproduces the documented "escape @ 77.96 hr, 5 burns, 37.579 m/s" signature.
`simulate` with an `MPCController` is the harness-unified path; the two are cross-checked
numerically in `scratch/compare/compare_simulate.jl`. They are not expected to agree
bit-for-bit: `run_mpc` records every periapsis it passes en route to the shell for its
min-periapsis metric, whereas `simulate` records the periapsis it actually stops at, and
`simulate` routes the ΔV through `apply_dv` (identity when `noisy_thruster = false`).
"""

# ── Controllers ───────────────────────────────────────────────────────────────
"""
    AbstractController

A stationkeeping controller. Implement three methods to plug a baseline into `simulate`:

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

The baseline's name for reporting: `"MPC"` or `"SARSOP"`. This is the `type` the brief
asks the harness to switch on; it is a method on the controller rather than a string
compared inside the loop, so adding a baseline cannot silently fall through to a default.
"""
controller_type(::AbstractController) = "UNKNOWN"

# ── MPC baseline ──────────────────────────────────────────────────────────────
"""
    MPCController(; ref_ic, mode, n_revs, peri_target_km, apo_target_km)

The deterministic Strategy-3 MPC baseline as a controller: at every shell crossing, solve
for the ΔV that re-targets the next apses. Stateless apart from the nominal targets it
caches in `controller_setup!`.

  - `ref_ic` — reference orbit for the `:position`-mode nominal apse targets. `nothing`
    uses the rollout's own initial state.
  - `mode` — `:position` (Strategy 3 proper, apse POSITION-vector bounding; MacKenzie's
    working strategy) or `:altitude` (Strategy 1/2, which MacKenzie says fails).
  - `n_revs` — multiple-shooting horizon `N_m` for `solve_burn`.
"""
Base.@kwdef mutable struct MPCController <: AbstractController
    ref_ic::Union{Nothing,Vector{Float64}} = nothing
    mode::Symbol                           = :position
    n_revs::Int                            = 3
    peri_target_km::Float64                = PERIAPSIS_ALT_TARGET
    apo_target_km::Float64                 = APOAPSIS_ALT_TARGET
    # Filled by controller_setup!.
    r_peri_nom::Union{Nothing,Vector{Float64}} = nothing
    r_apo_nom::Union{Nothing,Vector{Float64}}  = nothing
end

controller_type(::MPCController) = "MPC"

function controller_setup!(c::MPCController, state0::AbstractVector, period_s::Real)
    if c.mode === :position
        ref = c.ref_ic === nothing ? state0 : c.ref_ic
        # Count-based, NOT the period-shaped nominal_apse_positions: callers pass
        # period_s = T/3, which is the inter-periapsis interval and falls 0.136 s short of
        # the first apoapsis, yielding a NaN target. `period_s` is still the CONTROL cadence
        # and is passed to solve_burn below; it is just not a valid apse-search window.
        c.r_peri_nom, c.r_apo_nom = next_apse_positions(ref; eom! = cr3bp_eom!)
    end
    return nothing
end

function controller_command(c::MPCController, shell_state::AbstractVector, period_s::Real)
    b = solve_burn(shell_state, period_s;
                   n_revs = c.n_revs, eom! = cr3bp_eom!,
                   peri_target_km = c.peri_target_km,
                   apo_target_km  = c.apo_target_km,
                   mode = c.mode,
                   r_peri_nom = c.r_peri_nom, r_apo_nom = c.r_apo_nom)
    return b.dv, :CORRECT, (converged = b.converged, residual_km = b.residual_km)
end

# ── SARSOP baseline ───────────────────────────────────────────────────────────
"""
    SARSOPController(policy_data; config, ref_ic, n_revs, sigma_nav_km)

The solved POMDP policy as a controller. `policy_data` is the parsed
`artifacts/policy.json` payload written by [`export_policy`](@ref) — it carries the state /
action / observation labels, the discretization, the alpha vectors, and the T/O tables, so
the greedy query and the discrete Bayes filter are reproduced here without re-deriving the
model or depending on a solver.

Per step: query the greedy action from the current belief, realize it as a commanded ΔV
(`OBSERVE` → none; `CORRECT` → `solve_burn` toward the nominal apses; `EXCURSE_<BAND>` →
`solve_burn` toward that band's radially-scaled periapsis target), then fold a noisy
deviation observation into the belief.

⚠️ The ONBOARD model here is twofold and both halves are onboard-only: the discrete belief
filter over dev bins, and `solve_burn`'s CR3BP prediction. Neither sees `truth_eom!`.
"""
mutable struct SARSOPController <: AbstractController
    # Model description, straight from the exported policy.
    states::Vector{String}
    state_dev::Vector{String}
    state_cov::Vector{Int}
    actions::Vector{String}
    observations::Vector{String}
    dev_edges::Vector{Float64}
    band_names::Vector{String}
    band_target_km::Dict{String,Float64}
    alphas::Matrix{Float64}         # (n_alpha, |S|)
    alpha_actions::Vector{Int}      # 1-based action index per alpha
    T::Array{Float64,3}             # [s, a, s']
    O::Matrix{Float64}              # [s, o]
    # Configuration.
    ref_ic::Union{Nothing,Vector{Float64}}
    n_revs::Int
    sigma_nav_km::Float64
    # Live state.
    belief::Vector{Float64}
    cov::Int
    r_peri_nom::Union{Nothing,Vector{Float64}}
    r_apo_nom::Union{Nothing,Vector{Float64}}
    apo_nom_alt_km::Float64
end

controller_type(::SARSOPController) = "SARSOP"

"""
    SARSOPController(policy_data::AbstractDict; ref_ic = nothing, n_revs = 3,
                     sigma_nav_km = SIGMA_NAV_POS)

Build a controller from a parsed policy payload. Use
[`load_policy`](@ref) to read one from disk.
"""
function SARSOPController(policy_data::AbstractDict;
                          ref_ic::Union{Nothing,AbstractVector{<:Real}} = nothing,
                          n_revs::Integer = 3,
                          sigma_nav_km::Real = SIGMA_NAV_POS)
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

    return SARSOPController(
        S, String.(d["state_dev"]), Int.(d["state_cov"]), A, Ω,
        Float64.(d["dev_edges"]), String.(d["band_names"]),
        Dict{String,Float64}(string(k) => Float64(v) for (k, v) in d["band_target_km"]),
        alphas, Int.(d["alpha_actions"]), T, O,
        ref_ic === nothing ? nothing : collect(float.(ref_ic)),
        Int(n_revs), float(sigma_nav_km),
        belief, 0, nothing, nothing, NaN,
    )
end

"""
    load_policy(path = DEFAULT_POLICY_PATH) -> Dict

Read an exported policy JSON from disk. Thin wrapper so a caller does not need `JSON`.
"""
function load_policy(path::AbstractString = DEFAULT_POLICY_PATH)
    isfile(path) || error("no exported policy at $path — solve one and call export_policy")
    return JSON.parsefile(path)
end

"""
    policy_dev_bin(c, dev_km) -> String

Bin a deviation (km) using the EXPORTED edges, half-open [lo, hi).

⚠️ This must stay identical to [`dev_bin`](@ref) in `states.jl`. It deliberately reads the
edges out of the policy artifact rather than the live config, so a policy solved against
one discretization cannot be silently rolled out against another.
"""
function policy_dev_bin(c::SARSOPController, dev_km::Real)
    e = c.dev_edges
    isfinite(dev_km) || return "LOST"
    dev_km < e[1] && return "OK"
    dev_km < e[2] && return "DRIFT"
    dev_km < e[3] && return "FAR"
    return "LOST"
end

"""Greedy action from the current belief: argmax over α·b."""
policy_action(c::SARSOPController) =
    c.actions[c.alpha_actions[argmax(c.alphas * c.belief)]]

"""
Discrete Bayes filter, b'(s') ∝ O[s', o] Σ_s T[s, a, s'] b(s), with `o` over dev bins.
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
Re-concentrate the belief onto the known-`cov` block. `cov` is observed EXACTLY (we know
which excursions we commanded), so belief mass on any other coverage mask is spurious.
Terminal states are always kept. A no-op if masking would zero everything.
"""
function policy_reconcentrate_cov!(c::SARSOPController)
    keep = [(c.state_dev[i] in ("LOST", "CRASHED") || c.state_cov[i] == c.cov) ? 1.0 : 0.0
            for i in eachindex(c.states)]
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

function controller_command(c::SARSOPController, shell_state::AbstractVector, period_s::Real)
    action = policy_action(c)

    if action == "OBSERVE"
        return zeros(3), :OBSERVE, (band = 0, converged = true, residual_km = 0.0)
    end

    if action == "CORRECT"
        b = solve_burn(shell_state, period_s; n_revs = c.n_revs, eom! = cr3bp_eom!,
                       mode = :position,
                       r_peri_nom = c.r_peri_nom, r_apo_nom = c.r_apo_nom)
        return b.dv, :CORRECT, (band = 0, converged = b.converged,
                                residual_km = b.residual_km)
    end

    # EXCURSE_<BAND>: target that band's altitude at periapsis, holding apoapsis nominal.
    band_name = replace(action, "EXCURSE_" => "")
    band_idx  = findfirst(==(band_name), c.band_names)
    rp = _scale_to_altitude(c.r_peri_nom, c.band_target_km[band_name])
    ra = _scale_to_altitude(c.r_apo_nom, c.apo_nom_alt_km)
    b = solve_burn(shell_state, period_s; n_revs = c.n_revs, eom! = cr3bp_eom!,
                   mode = :position, r_peri_nom = rp, r_apo_nom = ra)
    return b.dv, Symbol(action), (band = band_idx, converged = b.converged,
                                  residual_km = b.residual_km)
end

function controller_observe!(c::SARSOPController, peri_state::AbstractVector,
                             dev_km::Real, label::Symbol, extra::NamedTuple,
                             rng::AbstractRNG)
    # Bank the science band if the excursion pass survived to a periapsis.
    extra.band != 0 && (c.cov = cov_set(c.cov, extra.band))

    true_bin = policy_dev_bin(c, dev_km)
    obs_bin  = policy_dev_bin(c, observe_deviation(dev_km, rng; sigma_r = c.sigma_nav_km))
    policy_update_belief!(c, String(label), obs_bin)
    policy_reconcentrate_cov!(c)

    return (true_bin = true_bin, obs_bin = obs_bin, cov = c.cov)
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
"""
function _scale_to_altitude(r_nom::AbstractVector{<:Real}, alt_km::Real)
    rr  = _enc_relative(r_nom)
    out = rr .* ((R_ENCELADUS + alt_km) / norm(rr))
    out[1] += X_ENCELADUS
    return out
end

# ── The rollout ───────────────────────────────────────────────────────────────
"""
    simulate(controller, state0, truth_eom!, period_s, horizon_s; kwargs...) -> NamedTuple

Roll `controller` out against a TRUTH model over `horizon_s` seconds. One decision per
periapsis approach, triggered at the descending `CONTROL_ALT_KM` shell.

  - `controller` — an [`AbstractController`](@ref): [`MPCController`](@ref) or
    [`SARSOPController`](@ref). This is the `type` switch: the world below is identical.
  - `state0` — initial barycentre-frame state (km, km/s), typically the period-3 IC.
  - `truth_eom!` — world dynamics ([`cr3bp_j2_eom!`](@ref),
    [`cr3bp_saturn_enc_j2_eom!`](@ref), later SPICE). An argument, never closed over.
  - `period_s` — the inter-periapsis interval (for the period-3 orbit, `T/3`).
  - `rng` — random stream for the thruster and nav noise. Required whenever
    `noisy_thruster = true` or the controller draws observations; defaults to a fresh
    `Xoshiro(0)` so a caller cannot accidentally consume global RNG state.
  - `noisy_thruster` — execute ΔV through [`apply_dv_noisy`](@ref) (η_eff ~ U(0.8, 1.0))
    rather than [`apply_dv`](@ref). Default `true`; set `false` for a deterministic run.
  - `max_steps` — hard cap on control steps (safety guard; outcome `:max_steps`).

Returns a NamedTuple with the SAME fields for every baseline, so a comparison table needs
no per-baseline special-casing:

  - `type` — `"MPC"` / `"SARSOP"`
  - `survived`, `outcome` — `:held`, `:idle`, `:crash`, `:escape`, `:max_steps`
  - `survival_time_s`, `n_steps`, `n_burns`, `total_dv_ms`
  - `n_solves`, `n_failed_solves` — how many control steps ran a burn solve, and how many of
    those did not converge. ⚠️ ALWAYS CHECK THESE before reading any other number. A failed
    solve returns ΔV = 0, so a run in which every solve failed is a run with no control at
    all, yet it reports a plausible `n_steps` and a `survived = true` outcome. This is not
    hypothetical: it is how the `T/3` NaN-apoapsis defect produced a full calibration table
    of quietly uncontrolled rollouts. `n_failed_solves == n_solves > 0` means the result
    describes an uncontrolled coast, whatever the outcome field says.
  - `min_peri_alt_km` — smallest periapsis altitude visited (km)
  - `science_cov`, `n_bands` — coverage bitmask and its popcount (0 for MPC).
    ⚠️ AN UPPER BOUND, NOT A MEASUREMENT. A band is banked because the excursion was
    COMMANDED and the pass survived to a periapsis — the achieved altitude is never checked
    against the band. With the `:position` targeting floor at ~10 km (see the planner notes)
    a commanded excursion can easily land outside its band and still be counted. Use
    `peri_alts_km` to check where the passes actually went. Gating the bitmask on achieved
    altitude is deliberately NOT done here: it would make `cov` stochastic and only
    partially observed, which breaks the exact-observation assumption behind
    `policy_reconcentrate_cov!` and would require re-measuring every transition kernel.
  - `peri_alts_km`, `peri_lats_deg` — achieved periapsis altitude (km) and latitude (deg)
    per control step. Latitude is what the south-polar science case turns on and is
    invisible in an altitude or a deviation norm.
  - `max_dev_trans_km` — largest TRANSVERSE (along-track + out-of-plane) periapsis
    deviation. Altitude errors are radial; this is the part of a position miss that a
    radius cannot see, and it is where most of the `:position` residual lives.
  - `steps` — per-step trace records
"""
function simulate(
    controller::AbstractController,
    state0::AbstractVector{<:Real},
    truth_eom!,
    period_s::Real,
    horizon_s::Real;
    rng::AbstractRNG = Xoshiro(0),
    noisy_thruster::Bool = true,
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
        # than trusted because it was commanded (see the science_cov caveat below).
        peri_alts_km    = [s.peri_alt_km for s in steps],
        peri_lats_deg   = [s.peri_lat_deg for s in steps],
        max_dev_trans_km = isempty(steps) ? NaN :
            maximum(s -> isfinite(s.dev_transverse_km) ? s.dev_transverse_km : -Inf, steps),
        science_cov     = _controller_cov(controller),
        n_bands         = cov_count(_controller_cov(controller)),
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

        # Count only steps that actually ran a solve. OBSERVE is a genuine no-burn DECISION,
        # not a solve, and folding it in would dilute the failure rate that these counters
        # exist to expose.
        if label !== :OBSERVE
            n_solves += 1
            extra.converged || (n_failed_solves += 1)
        end

        # 3. Execute it.
        dv_applied, eta_eff = noisy_thruster ? apply_dv_noisy(dv_cmd, rng) :
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
                          peri_alt_km = NaN, peri_lat_deg = NaN, peri_lon_deg = NaN,
                          peri_pos = fill(NaN, 3),
                          dev_radial_km = NaN, dev_transverse_km = NaN,
                          extra = (true_bin = uppercase(String(peri.outcome)),
                                   obs_bin  = uppercase(String(peri.outcome)),
                                   cov = _controller_cov(controller))))
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

"""Coverage mask of a controller (0 for baselines with no science state)."""
_controller_cov(::AbstractController) = 0
_controller_cov(c::SARSOPController)  = c.cov

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

Aggregate repeated stochastic rollouts into survival rate + medians, mirroring the Python
reference's `summarize`. `survived` counts `:held` and `:idle` (no crash, no escape).

`failed_solve_rate` is the fraction of all attempted burn solves across all rollouts that did
not converge. ⚠️ Read it FIRST: a survival rate computed over rollouts whose burns all failed
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