#=
example.jl — the copy-paste entry point.

    julia --project=experiments experiments/example.jl

Solve the baseline scenario, show the greedy policy, and export it. Edit the
StationkeepingPOMDP keywords below to change the scenario; every hyperparameter is a
field with a literal default (see src/StationkeepingPOMDP.jl).
=#

using SherpaOrbital
using SARSOP
using POMDPs

# ── Scenario ─────────────────────────────────────────────────────────────────
# Baseline. Try r_science = 40.0 to value science more, or plume_gradient = 4.0 for a
# steep plume altitude gradient.
config = StationkeepingPOMDP()

print_model_summary(config)

# ── Solve ────────────────────────────────────────────────────────────────────
pomdp  = build_pomdp(config)

# NOTE: use SARSOP.jl (the C++ wrapper), NOT NativeSARSOP, on this model. NativeSARSOP
# fails SILENTLY — no error is raised. Its Fast Informed Bound initialises the upper bound
# at 133.66, BELOW the true optimum of ~144.48, so the precision gap is negative from the
# first iteration and the search stops after 4 iterations with 22 alpha vectors. The
# result is identical at every θ, which is what makes it easy to mistake for a converged
# solve in a sweep.
#
# NativeSARSOP also needs `use_binning = false` on this model or it throws
# `InexactError: Int64(NaN)`: its `entropy(::SparseVector)` omits the `p > 0` guard the
# dense method has, and observation-zeroed entries left in the sparsity pattern make
# `0.0 * log(0.0)`. Moot here, but it is a second reason the wrapper is the path.
#
# SARSOP.jl's keyword is `timeout`, not `max_time`. Most of the wall clock is pomdpx
# serialisation of a 5627-state model, not search.
solver = SARSOP.SARSOPSolver(; precision = 1e-3, timeout = 900.0, verbose = false)

println("\nSolving with SARSOP ...")
t_solve = @elapsed (policy = solve(solver, pomdp))
println("  solved in $(round(t_solve, digits = 3)) s")

print_policy_table(policy, config)

# ── Export ───────────────────────────────────────────────────────────────────
path = export_policy(policy, config)
println("\nWrote policy -> $path")