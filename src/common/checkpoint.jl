"""
common/checkpoint.jl — per-rollout checkpoints, written as each rollout completes.

A sweep writes one file per rollout the moment it finishes, so a run that is killed keeps
everything already flown and resumes by counting what is on disk. The per-step TRACE is
stored, not just the return: a scalar cannot be un-aggregated, so "what did the controller
actually do?" is unanswerable later from summaries alone. Nor do a policy's alpha vectors
answer it — they describe the value function, and a policy can carry vectors for actions it
never takes.

    dir = cell_dir("artifacts/sweeps/nav", (sigma_nav_km = 4.0,), "POMDP")
    n   = n_checkpoints(dir)                       # resume point
    save_rollout(dir, k, res, config; seed = s)    # after each rollout
    rows = load_cell(dir)                          # analysis, later

Stored per rollout: the action sequence, achieved periapsis altitude, solve residual, and
the observed/true bin pair, plus cheap summary fields so scanning many cells does not
require reading every trace.

NOTE: the bulky `steps` fields are deliberately dropped — position 3-vectors and apse
targets multiply file size for data no downstream analysis reads. A checkpoint is a few KB
against the ~10 s of compute that produced it.
"""

"""
    cell_dir(root, theta, arm) -> String

Path for one sweep cell's checkpoints, keyed by the parameters that produced it.

  - `root` — sweep output root
  - `theta` — the swept parameters as a NamedTuple, e.g. `(sigma_nav_km = 4.0,)`
  - `arm` — controller label

Returns the directory path. Self-describing, so a resume needs no external spec and an
orphaned directory can still be identified.
"""
function cell_dir(root::AbstractString, theta::NamedTuple, arm::AbstractString)
    key = isempty(theta) ? "nominal" :
          join(["$k=$(theta[k])" for k in keys(theta)], ",")
    return joinpath(root, replace(arm, " " => "_"), key)
end

"""
    rollout_filename(k) -> String

Checkpoint filename for rollout `k`, zero-padded so `readdir` sorts in flight order.
"""
rollout_filename(k::Integer) = @sprintf("rollout_%05d.jld2", k)

"""
    n_checkpoints(dir) -> Int

How many rollouts of a cell are already banked, i.e. where a resume should start.

  - `dir` — a cell directory from [`cell_dir`](@ref)

Returns the count, or 0 if the directory does not exist.
"""
function n_checkpoints(dir::AbstractString)
    isdir(dir) || return 0
    return count(f -> startswith(f, "rollout_") && endswith(f, ".jld2"), readdir(dir))
end

"""
    save_rollout(dir, k, result, config; seed, arm, theta, extra...) -> String

Write one rollout's checkpoint immediately, so it survives the process dying next second.

  - `dir` — cell directory from [`cell_dir`](@ref)
  - `k` — 0-based rollout index within the cell
  - `result` — the [`run_rollout`](@ref) result
  - `config` — the scenario the rollout was scored under
  - `seed` — the RNG seed used, so the rollout can be reproduced exactly
  - `arm` — controller label
  - `theta` — the swept parameters for this cell
  - `extra` — any additional scalars to store alongside

Returns the path written.
"""
function save_rollout(dir::AbstractString, k::Integer, result,
                      config::StationkeepingPOMDP;
                      seed::Integer = 0, arm::AbstractString = "",
                      theta::NamedTuple = NamedTuple(), extra...)
    mkpath(dir)
    steps = result.steps
    sci   = science_trace(result, config)
    dmg   = damage_trace(result)
    dlv   = delivery_trace(result, config)

    path = joinpath(dir, rollout_filename(k))
    JLD2.jldsave(path;
        # provenance — enough to re-fly this exact rollout
        k = k, seed = seed, arm = String(arm),
        theta = NamedTuple(pairs(theta)),
        plume_gradient = config.plume_gradient,
        sigma_nav_km = config.sigma_nav_km,
        noisy_thruster = config.noisy_thruster,
        thruster_sigma_pct = config.thruster_sigma_pct,
        # summary — cheap to read when scanning many cells
        outcome = String(result.outcome),
        survived = !(result.outcome in (:crash, :escape)),
        survival_days = result.survival_time_s / 86400,
        n_steps = length(steps),
        n_bands = result.n_bands,
        n_samples = isempty(result.science_visits) ? 0 : sum(result.science_visits),
        science_visits = collect(result.science_visits),
        total_dv_ms = result.total_dv_ms,
        min_peri_alt_km = result.min_peri_alt_km,
        science = isempty(sci.cumulative) ? 0.0 : last(sci.cumulative),
        frac_degraded = isempty(dmg.frac_degraded) ? 0.0 : last(dmg.frac_degraded),
        discounted_return = try discounted_return(result, config) catch; NaN end,
        # per-step trace — what the vehicle did and where it ended up
        t_days = [s.t_s / 86400 for s in steps],
        actions = [String(s.action) for s in steps],
        peri_alts_km = [s.peri_alt_km for s in steps],
        residuals_km = [s.residual_km for s in steps],
        dv_ms = [s.dv_ms for s in steps],
        eta_eff = [s.eta_eff for s in steps],
        obs_bins = [hasproperty(s.extra, :obs_bin) ? String(s.extra.obs_bin) : ""
                    for s in steps],
        true_bins = [hasproperty(s.extra, :true_bin) ? String(s.extra.true_bin) : ""
                     for s in steps],
        science_cum = sci.cumulative,
        # delivery accuracy, per excursion
        dlv_t_days = dlv.t_days, dlv_commanded_km = dlv.commanded_km,
        dlv_achieved_km = dlv.achieved_km, dlv_err_km = dlv.err_km,
        extra...)
    return path
end

"""
    load_cell(dir) -> Vector{Dict}

Read every checkpoint in a cell back, in flight order.

  - `dir` — a cell directory from [`cell_dir`](@ref)

Returns a Vector of Dicts keyed by the names [`save_rollout`](@ref) wrote. An unreadable
file is skipped rather than raising, so a checkpoint truncated by a kill does not block
analysis of the rest.
"""
function load_cell(dir::AbstractString)
    isdir(dir) || return Dict{String,Any}[]
    out = Dict{String,Any}[]
    for f in sort(filter(f -> startswith(f, "rollout_") && endswith(f, ".jld2"),
                         readdir(dir)))
        d = try JLD2.load(joinpath(dir, f)) catch; continue end
        push!(out, d)
    end
    return out
end

"""
    load_sweep(root) -> Vector{Dict}

Every checkpoint under a sweep root, across all arms and cells.

  - `root` — the sweep root passed to [`cell_dir`](@ref)

Returns a flat Vector of Dicts; each carries its own `arm` and `theta`, so the cell
structure is recoverable without walking the tree again.
"""
function load_sweep(root::AbstractString)
    isdir(root) || return Dict{String,Any}[]
    out = Dict{String,Any}[]
    for (dir, _, files) in walkdir(root)
        any(f -> startswith(f, "rollout_") && endswith(f, ".jld2"), files) || continue
        append!(out, load_cell(dir))
    end
    return out
end