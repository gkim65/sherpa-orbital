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

    discount = 0.95

    # Build the science+safety POMDP and grab the exact T/O tables handed to SARSOP.
    T = transition_matrix()
    O = observation_matrix()
    pomdp = build_stationkeeping_pomdp(; discount = discount)

    solver = SARSOPSolver(; max_time = 20.0, precision = 1e-3, verbose = false)
    policy = solve(solver, pomdp)

    actions = collect(ACTION_NAMES)
    observations = collect(OBS_NAMES)
    alphas, alpha_actions = extract_alphas(policy, actions)

    # Serializable state labels "DEV|cov<mask>" so Python can reconstruct (dev,cov).
    state_labels = ["$(s.dev)|cov$(s.cov)" for s in STATES]

    T_nested = [[[T[si, ai, spi] for spi in 1:N_STATES]
                 for ai in 1:N_ACTIONS] for si in 1:N_STATES]
    O_nested = [[O[si, oi] for oi in 1:N_OBS] for si in 1:N_STATES]

    payload = Dict(
        "states"        => state_labels,
        "state_dev"     => [string(s.dev) for s in STATES],
        "state_cov"     => [s.cov for s in STATES],
        "actions"       => string.(actions),
        "observations"  => string.(observations),
        "terminal_states" => ["LOST|cov0", "CRASHED|cov0"],
        "initial_state" => "OK|cov0",
        "discount"      => discount,
        "dev_edges"     => collect(DEV_EDGES),
        "band_names"    => string.(collect(BAND_NAMES)),
        "band_target_km" => Dict(string(k) => v for (k, v) in BAND_TARGET_KM),
        "action_dv_cost" => Dict(string(k) => v for (k, v) in ACTION_DV_COST),
        "alphas"        => alphas,               # Vector of |S|-vectors
        "alpha_actions" => alpha_actions,        # 1-based action index per alpha
        "T"             => T_nested,
        "O"             => O_nested,
        "meta"          => Dict(
            "solver" => "NativeSARSOP",
            "precision" => 1e-3,
            "max_time_s" => 20.0,
            "note" => "Science+safety POMDP (2026-07-15). State=(dev,cov); actions " *
                      "OBSERVE/CORRECT/EXCURSE_{LOW,MID,HIGH}. Tables measured from " *
                      "CR3BP+EncJ2 (exps 11b/12). Direction solved by mpc.solve_burn.",
        ),
    )

    open(out_path, "w") do io
        write(io, _json(payload))
    end
    println("Wrote SARSOP policy + tables -> $out_path")
    println("  |S|=$(N_STATES)  |A|=$(N_ACTIONS)  |O|=$(N_OBS)")
    println("  n_alphas=$(length(alphas))  discount=$discount")
end

main()
