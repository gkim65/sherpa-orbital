"""
export_policy.jl — export the solved SARSOP policy (+ the toy T/O tables and all
labels) to a single JSON file that the Python rollout harness reads.

Why this file exists
--------------------
The SARSOP policy is solved in Julia (NativeSARSOP), but the high-fidelity
rollout that TESTS it against the real CR3BP truth model lives in Python (that is
where the accurate integrator + Phase-2 noise models are). We keep a NARROW,
FILE-BASED seam between the two: the policy crosses the language boundary exactly
once, as this JSON artifact — no live PyCall bridge.

What gets written
-----------------
An AlphaVectorPolicy is a set of |S|-dimensional alpha vectors, each tagged with
the action that is greedy where that vector dominates. The greedy action for a
belief b is argmax_i (alpha_i · b). We export:
  - states / actions / observations       : the label ordering (Symbols -> strings)
  - alphas       : Vector of |S|-vectors   (the alpha vectors)
  - alpha_actions: the action index (1-based, into `actions`) attached to each alpha
  - T[s,a,s']    : transition table         (for the Python discrete belief update)
  - O[s',o]      : observation table        (same)
  - terminal_states / initial_state / discount / bin_edges / representative_alt
  - meta         : solver settings + reward params, for provenance in the report

Python then reproduces `action(policy, b)` as argmax over (alpha · b), and runs the
standard discrete Bayes filter b'(s') ∝ O[s',o]·Σ_s T[s,a,s']·b(s) using the SAME
tables the policy was solved from (this mirrors POMDPTools' DiscreteUpdater).

No new package dependency: the payload is small and flat (strings + nested
number lists), so we write JSON with a tiny hand-rolled serializer rather than
adding JSON.jl to the toy's Project.toml.

Run from the pomdp-julia/ directory:
    julia --project=. src/export_policy.jl [output_path]
Default output: ../policy/sarsop_policy.json (repo-level `policy/` dir).
"""

using POMDPs
using POMDPTools
using NativeSARSOP
using Random

include("stationkeeping_pomdp.jl")

# ── Minimal JSON writer (avoids a JSON.jl dependency for this tiny payload) ────
_jstr(s::AbstractString) = '"' * replace(String(s), '\\' => "\\\\", '"' => "\\\"") * '"'
_json(x::AbstractString) = _jstr(x)
_json(x::Bool) = x ? "true" : "false"
_json(x::Integer) = string(x)
function _json(x::AbstractFloat)
    isfinite(x) || error("cannot JSON-encode non-finite float $x")
    return string(Float64(x))
end
_json(x::AbstractVector) = "[" * join((_json(v) for v in x), ",") * "]"
function _json(x::AbstractDict)
    # Deterministic key order for reproducible, diff-friendly output.
    ks = sort(collect(keys(x)); by = string)
    return "{" * join((_jstr(string(k)) * ":" * _json(x[k]) for k in ks), ",") * "}"
end

"""
    extract_alphas(policy) -> (alphas, alpha_action_indices)

Pull the alpha vectors and their associated action indices out of a solved
AlphaVectorPolicy in a version-robust way. NativeSARSOP returns an
`AlphaVectorPolicy` whose fields are `alphas` (Vector of Vector{Float64}) and
`action_map` (the action attached to each alpha). We map each attached action
back to its 1-based index in ACTION_NAMES so Python needs only integer indices.
"""
function extract_alphas(policy, actions)
    aidx = Dict(a => i for (i, a) in enumerate(actions))
    alphas = policy.alphas                     # Vector{Vector{Float64}}
    acts = policy.action_map                   # Vector of actions (one per alpha)
    alpha_action_indices = [aidx[a] for a in acts]
    return alphas, alpha_action_indices
end

function main()
    out_path = length(ARGS) >= 1 ? ARGS[1] :
        normpath(joinpath(@__DIR__, "..", "..", "policy", "sarsop_policy.json"))
    mkpath(dirname(out_path))

    # Fixed seed so the exported tables match a reproducible solve (mirrors solve.jl).
    seed = 20260706
    rng = MersenneTwister(seed)
    discount = 0.95
    n_mc = 40_000
    drift_mean, drift_sigma = 12.0, 6.0

    # Build the toy POMDP and grab the exact T/O tables handed to SARSOP.
    T = transition_matrix(; n_mc = n_mc, rng = MersenneTwister(seed),
                          drift_mean = drift_mean, drift_sigma = drift_sigma)
    O = observation_matrix()

    pomdp = build_stationkeeping_pomdp(; discount = discount, n_mc = n_mc,
                                       rng = rng, drift_mean = drift_mean,
                                       drift_sigma = drift_sigma)

    solver = SARSOPSolver(; max_time = 20.0, precision = 1e-3, verbose = false)
    policy = solve(solver, pomdp)

    states = collect(STATE_NAMES)
    actions = collect(ACTION_NAMES)
    observations = collect(STATE_NAMES)
    alphas, alpha_actions = extract_alphas(policy, actions)

    # Convert the 3-D / 2-D Julia arrays to nested JSON-friendly lists.
    # T: [s][a][s'] ; O: [s'][o].
    T_nested = [[[T[si, ai, spi] for spi in 1:N_STATES]
                 for ai in 1:N_ACTIONS] for si in 1:N_STATES]
    O_nested = [[O[si, oi] for oi in 1:N_STATES] for si in 1:N_STATES]

    payload = Dict(
        "states"        => string.(states),
        "actions"       => string.(actions),
        "observations"  => string.(observations),
        "terminal_states" => string.(collect(TERMINAL_STATES)),
        "initial_state" => "NOMINAL",
        "discount"      => discount,
        "bin_edges"     => collect(BIN_EDGES),
        "representative_alt" => Dict(string(k) => v for (k, v) in BIN_REPRESENTATIVE_ALT),
        "action_dv_cost" => Dict(string(k) => v for (k, v) in ACTION_DV_COST),
        "action_raise_km" => Dict(string(k) => v for (k, v) in ACTION_RAISE_KM),
        "alphas"        => alphas,               # Vector of |S|-vectors
        "alpha_actions" => alpha_actions,        # 1-based action index per alpha
        "T"             => T_nested,
        "O"             => O_nested,
        "meta"          => Dict(
            "solver" => "NativeSARSOP",
            "seed" => seed,
            "n_mc" => n_mc,
            "drift_mean" => drift_mean,
            "drift_sigma" => drift_sigma,
            "precision" => 1e-3,
            "max_time_s" => 20.0,
            "note" => "Toy drift+burn policy. Bins are ABSOLUTE periapsis altitude (km). " *
                      "HIGH is a known dead-end artifact; tuning is hand-picked, not physics-fitted.",
        ),
    )

    open(out_path, "w") do io
        write(io, _json(payload))
    end
    println("Wrote SARSOP policy + tables -> $out_path")
    println("  states=$(states)  actions=$(actions)")
    println("  n_alphas=$(length(alphas))  discount=$discount")
end

main()
