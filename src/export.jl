"""
export.jl — serialize a solved policy so a rollout harness can consume it.

The exported JSON is self-describing: it carries the state/action/observation labels, the
discretization, the alpha vectors and the T/O tables, so a consumer can reproduce both the
greedy policy query and the discrete belief filter without re-deriving the model or
knowing the enumeration order. [`SARSOPController`](@ref) is the in-tree consumer.

NOTE: exported policies are gitignored — they are large and derived. Only the measured
kernels are committed: `artifacts/tables.json` (noise-free) and one
`artifacts/tables_noisy_gaussian<sigma>.json` per calibrated thruster-noise level.

NOTE: this JSON carries the DENSE `T[s][a][s']` — |S|^2 |A| floats, gigabytes at
|S| = 5627 — so it is an archive format, not a hot path. To solve and fly in one process,
take the alpha vectors off the policy object; to reload a solve, read SARSOP.jl's own
`<stem>.out` via `SARSOP.load_policy(pomdp, path)`, which is ~6x smaller and needs no
parsing of T/O that the config already determines.
"""

const DEFAULT_POLICY_PATH =
    normpath(joinpath(@__DIR__, "..", "artifacts", "policy.json"))

"""
    theta_slug(θ) -> String

A filesystem-safe slug for a θ NamedTuple.

  - `θ` — the environment parameters, e.g. `(sigma_nav_km = 2.0, plume_gradient = 4.0)`

Returns e.g. `"sigma_nav_km=2.0_plume_gradient=4.0"`. Field order follows the NamedTuple,
so the same θ always yields the same slug.

NOTE: `DEFAULT_TABLES_PATH` and `DEFAULT_POLICY_PATH` are single-slot. A sweep that does
not key its output by θ overwrites itself and leaves one artifact claiming to describe the
whole family.
"""
theta_slug(θ::NamedTuple) =
    join(["$(k)=$(v)" for (k, v) in pairs(θ)], "_")

"""
    theta_path(base, θ; ext = ".json") -> String

θ-keyed artifact path, `artifacts/<base>_<slug>.<ext>`.

  - `base` — filename stem, conventionally `"tables"` or `"policy"`
  - `θ` — the environment parameters, slugged by [`theta_slug`](@ref)
  - `ext` — file extension including the dot

Returns the absolute path. Use for both tables and policies so a sweep's outputs sit
beside each other and are self-identifying.

    write_tables(tbl; path = theta_path("tables", (plume_gradient = 4.0,)))
    export_policy(pol, cfg; path = theta_path("policy", (plume_gradient = 4.0,)))
"""
function theta_path(base::AbstractString, θ::NamedTuple; ext::AbstractString = ".json")
    dir = normpath(joinpath(@__DIR__, "..", "artifacts"))
    return joinpath(dir, string(base, "_", theta_slug(θ), ext))
end

"""
    alpha_vectors(policy, action_list) -> (alphas, alpha_actions)

Pull the alpha vectors and their action indices out of a solved `AlphaVectorPolicy`.

  - `policy` — the solved policy
  - `action_list` — actions in model order, from [`actions`](@ref)

Returns `(alphas, alpha_actions)`: a vector of |S|-length alpha vectors, and the 1-based
index into `action_list` each one votes for. Kept separate from `export_policy` so a
caller can inspect them.
"""
function alpha_vectors(policy, action_list::Vector{Symbol})
    aidx = Dict(a => i for (i, a) in enumerate(action_list))
    alphas = [collect(Float64.(α)) for α in policy.alphas]
    acts   = [aidx[a] for a in policy.action_map]
    return alphas, acts
end

"""
    export_policy(policy, config = StationkeepingPOMDP();
                  path = DEFAULT_POLICY_PATH, tables = nothing, meta = Dict())

Write a solved policy plus the model tables it was solved against to JSON.

  - `policy` — the solved policy
  - `config` — the scenario it was solved against
  - `path` — destination; defaults to the single-slot `DEFAULT_POLICY_PATH`, so use
    [`theta_path`](@ref) for a sweep
  - `tables` — measured kernels; `nothing` loads the artifact at `config.tables_path`
  - `meta` — extra provenance merged into the payload's `meta` block

Returns the path written.

    policy = solve(SARSOP.SARSOPSolver(; precision = 1e-3), build_pomdp(cfg))
    export_policy(policy, cfg)
"""
function export_policy(policy, config::StationkeepingPOMDP = StationkeepingPOMDP();
                       path::AbstractString = DEFAULT_POLICY_PATH,
                       tables::Union{Nothing,AltTables} = nothing,
                       meta::AbstractDict = Dict{String,Any}())
    tbl = tables === nothing ? load_tables(config) : tables

    S = states(config)
    A = actions(config)
    Ω = observations(config)
    T = transition_matrix(config, tbl)
    O = observation_matrix(config)

    alphas, alpha_acts = alpha_vectors(policy, A)

    # Nested lists: T[s][a][s'], O[s][o]. Explicit loops keep the index order unambiguous.
    T_nested = [[[T[si, ai, spi] for spi in eachindex(S)] for ai in eachindex(A)]
                for si in eachindex(S)]
    O_nested = [[O[si, oi] for oi in eachindex(Ω)] for si in eachindex(S)]

    payload = Dict(
        "states"          => state_label.(S),
        "state_alt"       => [string(s.alt) for s in S],
        "state_visits"    => [collect(s.visits) for s in S],
        # Orbit-damage bin per state, plus the edges defining it. A consumer needs the
        # labels to project its belief onto the known-residual block, and the edges to bin
        # its own live `solve_burn` residual the way the model was calibrated.
        "state_residual"  => [string(s.residual) for s in S],
        "residual_bins"   => string.(collect(RESIDUAL_BINS)),
        "residual_edges"  => collect(RESIDUAL_EDGES),
        "actions"         => string.(A),
        "observations"    => string.(Ω),
        "terminal_states" => [state_label(SKState(:LOST, _zero_visits(config), 1, :R_OK)),
                              state_label(SKState(:CRASHED, _zero_visits(config), 1, :R_OK))],
        "initial_state"   => state_label(SKState(config.correct_bin,
                                                 _zero_visits(config), 1, :R_OK)),
        "discount"        => config.discount,
        "alt_edges"       => collect(config.alt_edges),
        "band_names"      => string.(collect(config.band_names)),
        "band_bins"       => string.(collect(config.band_bins)),
        "visit_cap"       => config.visit_cap,
        "correct_bin"     => string(config.correct_bin),
        "band_target_km"  => Dict(string(k) => v for (k, v) in config.band_target_km),
        "action_dv_cost"  => Dict(string(k) => v for (k, v) in config.action_dv_cost),
        "sigma_nav_km"    => config.sigma_nav_km,
        "alphas"          => alphas,
        "alpha_actions"   => alpha_acts,
        "T"               => T_nested,
        "O"               => O_nested,
        "meta"            => merge(Dict{String,Any}(
            "n_states"       => length(S),
            "n_actions"      => length(A),
            "n_observations" => length(Ω),
            "n_alphas"       => length(alphas),
            "tables_meta"    => tbl.meta,
        ), Dict{String,Any}(meta)),
    )

    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, payload, 2)
    end
    return path
end