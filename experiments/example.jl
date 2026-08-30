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

# ⚠️ `use_binning = false` IS REQUIRED, not a tuning choice. With the default `true`,
# NativeSARSOP throws `InexactError: Int64(NaN)` on this model. Diagnosed 2026-08-30: its
# `entropy(::SparseVector)` method omits the `p > 0` guard that the dense method has, and
# the belief update leaves observation-zeroed entries in the sparsity pattern at exactly
# 0.0, so `0.0 * log(0.0) = NaN` flows into `get_interval_idx`. Belief binning is the only
# caller of `entropy`, so disabling it avoids the bug entirely. Binning is a search
# heuristic for grouping near-duplicate beliefs — switching it off costs some speed and
# changes neither the POMDP nor the solution. Full write-up in `StationkeepingPOMDP.jl`
# beside `visit_cap`.
#
# `max_time = 120` because the unbinned search is slower; measured, it converges in ~63 s
# with 1991 alpha vectors, so this is headroom rather than a truncation.
solver = SARSOPSolver(; precision = 1e-3, max_time = 120.0, verbose = false,
                      use_binning = false)

println("\nSolving with NativeSARSOP ...")
t_solve = @elapsed (policy = solve(solver, pomdp))
println("  solved in $(round(t_solve, digits = 3)) s")

print_policy_table(policy, config)

# ── Export ───────────────────────────────────────────────────────────────────
path = export_policy(policy, config)
println("\nWrote policy -> $path")