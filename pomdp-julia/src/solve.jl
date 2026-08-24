"""
solve.jl — solve the science+safety stationkeeping POMDP with NativeSARSOP and show
that the policy sensibly trades SCIENCE (sample altitude bands) against SAFETY (hold
the orbit). See dynamics.jl / stationkeeping_pomdp.jl and
docs/pomdp-proposal-2026-07-13.md.

Run from pomdp-julia/:  julia --project=. src/solve.jl
"""

using POMDPs
using POMDPTools
using NativeSARSOP
using Random
using Printf

include("stationkeeping_pomdp.jl")

cov_label(mask::Int) = mask == 0 ? "none" :
    join([string(BAND_NAMES[b]) for b in 1:N_BANDS if cov_has(mask, b)], "+")

function main()
    println("="^72)
    println("Science+safety Enceladus stationkeeping POMDP — NativeSARSOP")
    println("="^72)
    println("\nState = (dev safety bin, science coverage mask)")
    println("  dev bins   : ", DEV_NAMES, "  (+ CRASHED)")
    println("  bands      : ", BAND_NAMES, "  targets(km)=",
            [BAND_TARGET_KM[b] for b in BAND_NAMES])
    println("  actions    : ", ACTION_NAMES)
    @printf("  |S|=%d  |A|=%d  |O|=%d\n", N_STATES, N_ACTIONS, N_OBS)

    pomdp = build_stationkeeping_pomdp(; discount = 0.95)

    println("\nSolving with NativeSARSOP ...")
    solver = SARSOPSolver(; max_time = 30.0, precision = 1e-3, verbose = false)
    t_solve = @elapsed (policy = solve(solver, pomdp))
    @printf("  solved in %.3f s -> alpha-vector policy\n", t_solve)

    # ---- Policy table: greedy action from a belief concentrated on each (dev,cov).
    # Grouped by coverage so the science-vs-safety tradeoff is readable.
    println("\nGreedy policy  a*(dev, coverage)  (belief concentrated on the state):")
    @printf("  %-6s", "dev\\cov")
    covs = 0:(N_COV - 1)
    for c in covs
        @printf(" %-14s", cov_label(c))
    end
    println()
    for dev in (:OK, :DRIFT, :FAR)
        @printf("  %-6s", dev)
        for c in covs
            s = SKState(dev, c)
            b = SparseCat(STATES, [x == s ? 1.0 : 0.0 for x in STATES])
            a = action(policy, b)
            @printf(" %-14s", a)
        end
        println()
    end

    # ---- Simulation from the start state (dev OK, no science yet).
    println("\nSimulation from (OK, none) under the SARSOP policy:")
    up = DiscreteUpdater(pomdp)
    hist = simulate(HistoryRecorder(; max_steps = 30, rng = MersenneTwister(7)),
                    pomdp, policy, up)
    @printf("  %-4s  %-16s  %-13s  %-8s  %8s\n", "t", "state(dev,cov)", "action", "obs", "reward")
    for (t, step) in enumerate(hist)
        @printf("  %-4d  (%-3s,%-9s)  %-13s  %-8s  %8.2f\n",
                t, step.s.dev, cov_label(step.s.cov), step.a, step.o, step.r)
    end
    @printf("\n  discounted return over %d steps: %.2f\n",
            length(hist), discounted_reward(hist))
    println("="^72)
    return policy, hist
end

main()
