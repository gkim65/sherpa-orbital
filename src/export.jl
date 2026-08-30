"""
export.jl — serialize a solved policy so a rollout harness can consume it.

The exported JSON is self-describing: it carries the state/action/observation labels, the
discretization, the alpha vectors, AND the T/O tables. A consumer can therefore reproduce
both the greedy policy query and the discrete belief filter without re-deriving the model
or knowing the enumeration order.

While the rollout still lives in Python this file is the language seam. Once the rollout
is Julia-native it stays useful as a way to freeze a solved policy for reuse.
"""

const DEFAULT_POLICY_PATH =
    normpath(joinpath(@__DIR__, "..", "artifacts", "policy.json"))

"""
    alpha_vectors(policy, action_list) -> (alphas, alpha_actions)

Pull the alpha vectors and their 1-based action indices out of a solved
`AlphaVectorPolicy`. Kept separate from `export_policy` so a caller can inspect them.
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

Write a solved policy plus the model tables it was solved against to JSON. Returns the
path written.

    policy = solve(SARSOPSolver(), build_pomdp(cfg))
    export_policy(policy, cfg)
"""
function export_policy(policy, config::StationkeepingPOMDP = StationkeepingPOMDP();
                       path::AbstractString = DEFAULT_POLICY_PATH,
                       tables::Union{Nothing,AltTables} = nothing,
                       meta::AbstractDict = Dict{String,Any}())
    tbl = tables === nothing ?
        load_tables(something(config.tables_path, DEFAULT_TABLES_PATH)) : tables

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
        "actions"         => string.(A),
        "observations"    => string.(Ω),
        "terminal_states" => [state_label(SKState(:LOST, _zero_visits(config))),
                              state_label(SKState(:CRASHED, _zero_visits(config)))],
        "initial_state"   => state_label(SKState(config.correct_bin,
                                                 _zero_visits(config))),
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