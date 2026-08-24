#=
example.jl — the copy-paste entry point.

    julia --project=experiments experiments/example.jl

Solve the baseline scenario, show the greedy policy, and export it. Edit the
StationkeepingPOMDP keywords below to change the scenario; every hyperparameter is a
field with a literal default (see src/StationkeepingPOMDP.jl).
=#

using SherpaOrbital
using NativeSARSOP
using POMDPs

# ── Scenario ─────────────────────────────────────────────────────────────────
# Baseline. Try r_science = 40.0 to value science more, or tighten the safety bins
# with dev_edges = (10.0, 50.0, 150.0).
config = StationkeepingPOMDP()

print_model_summary(config)

# ── Solve ────────────────────────────────────────────────────────────────────
pomdp  = build_pomdp(config)
solver = SARSOPSolver(; precision = 1e-3, max_time = 30.0, verbose = false)

println("\nSolving with NativeSARSOP ...")
t_solve = @elapsed (policy = solve(solver, pomdp))
println("  solved in $(round(t_solve, digits = 3)) s")

print_policy_table(policy, config)

# ── Export ───────────────────────────────────────────────────────────────────
path = export_policy(policy, config)
println("\nWrote policy -> $path")