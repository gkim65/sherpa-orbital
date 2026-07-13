"""
solve.jl — solve the toy stationkeeping POMDP with NativeSARSOP and simulate.

Run from the pomdp-julia/ directory:
    julia --project=. src/solve.jl

What it does:
  1. Builds the toy POMDP (crude drift+burn tables).
  2. Solves it OFFLINE with NativeSARSOP -> an alpha-vector policy.
  3. Prints the greedy action the policy takes from a belief concentrated on
     each state (a readable "policy table").
  4. Runs a short simulation from NOMINAL and reports the discounted return.

NativeSARSOP is pure Julia (no C++ pomdpsol binary) -> avoids the Apple-silicon
APPL build issue flagged in the toolchain scout note.
"""

using POMDPs
using POMDPTools
using NativeSARSOP
using Random
using Printf

include("stationkeeping_pomdp.jl")

function main()
    rng = MersenneTwister(20260706)   # fixed seed: reproducible tables + sim

    println("="^70)
    println("Toy Enceladus stationkeeping POMDP — NativeSARSOP")
    println("="^70)

    pomdp = build_stationkeeping_pomdp(; discount = 0.95, n_mc = 40_000, rng = rng)

    # ---- Sanity: show the transition + observation structure we handed SARSOP.
    println("\nStates      : ", collect(STATE_NAMES))
    println("Actions     : ", collect(ACTION_NAMES))
    println("Terminal    : ", collect(TERMINAL_STATES))

    println("\nTransition table P(s' | s, a)  (Monte-Carlo, 40k samples/pair):")
    T = transition_matrix(; n_mc = 40_000, rng = MersenneTwister(1),
                          drift_mean = 12.0, drift_sigma = 6.0)
    for (si, s) in enumerate(STATE_NAMES)
        s in TERMINAL_STATES && continue
        for (ai, a) in enumerate(ACTION_NAMES)
            probs = join((@sprintf("%s=%.2f", STATE_NAMES[k], T[si, ai, k])
                          for k in 1:N_STATES if T[si, ai, k] > 0.005), "  ")
            @printf("  %-8s + %-11s -> %s\n", s, a, probs)
        end
    end

    # ---- Solve offline with SARSOP.
    println("\nSolving with NativeSARSOP ...")
    solver = SARSOPSolver(; max_time = 20.0, precision = 1e-3, verbose = false)
    policy = solve(solver, pomdp)
    println("  solved -> alpha-vector policy.")

    # ---- Policy table: greedy action from a belief concentrated on each state.
    println("\nGreedy policy (action chosen when SURE we are in each state):")
    for s in STATE_NAMES
        s in TERMINAL_STATES && (println(@sprintf("  %-8s -> (terminal)", s)); continue)
        b = SparseCat(collect(STATE_NAMES),
                      [x == s ? 1.0 : 0.0 for x in STATE_NAMES])
        a = action(policy, b)
        @printf("  %-8s -> %s\n", s, a)
    end

    # ---- Short simulation from NOMINAL under the SARSOP policy + belief updater.
    println("\nSimulation from NOMINAL (belief-tracked, seeded):")
    up = DiscreteUpdater(pomdp)
    hist = simulate(HistoryRecorder(; max_steps = 25, rng = MersenneTwister(7)),
                    pomdp, policy, up)

    @printf("  %-4s  %-8s  %-11s  %-8s  %8s\n", "t", "state", "action", "obs", "reward")
    for (t, step) in enumerate(hist)
        @printf("  %-4d  %-8s  %-11s  %-8s  %8.2f\n",
                t, step.s, step.a, step.o, step.r)
    end
    ret = discounted_reward(hist)
    @printf("\n  discounted return over %d steps: %.2f\n", length(hist), ret)
    println("  (survived = never entered CRASHED/ESCAPED)")

    println("\n" * "="^70)
    return policy, hist
end

main()