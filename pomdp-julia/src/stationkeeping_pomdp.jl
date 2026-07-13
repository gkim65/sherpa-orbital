"""
stationkeeping_pomdp.jl — the toy Enceladus stationkeeping POMDP.

A tiny, self-contained POMDP that proves the POMDPs.jl + NativeSARSOP toolchain
and the stationkeeping formulation. NOT a fidelity model — the dynamics are the
crude drift+burn placeholder in `dynamics.jl`.

Formulation
  State   : periapsis-altitude bin  (CRASHED / LOW / NOMINAL / HIGH / ESCAPED)
            CRASHED and ESCAPED are terminal.
  Action  : NO_BURN / SMALL_BURN / LARGE_BURN  (fixed delta-V raises, applied
            with random burn efficiency eta_eff ~ Uniform(0.8, 1.0)).
  Obs     : noisy altitude bin (same 5 bins), from 2 km Gaussian nav noise ->
            partial observability.
  Reward  : reward for holding NOMINAL; penalise LOW, big penalty for CRASH /
            ESCAPE; small fuel penalty proportional to delta-V spent.

The T and O tables are Monte-Carlo / analytically precomputed in `dynamics.jl`;
SARSOP consumes them offline and never touches the dynamics.
"""

using POMDPs
using QuickPOMDPs
using POMDPTools

include("dynamics.jl")

# ---------------------------------------------------------------------------
# Reward parameters.
# ---------------------------------------------------------------------------
const R_NOMINAL   =  10.0     # holding the target band
const R_HIGH      =  -1.0     # safe but wasteful (too high / drifting out)
const R_LOW       =  -5.0     # dangerous: one step from crash
const R_CRASHED   = -100.0    # mission loss
const R_ESCAPED   = -100.0    # mission loss
const FUEL_WEIGHT =   1.0     # multiplies ACTION_DV_COST (m/s) as a penalty

const STATE_REWARD = Dict(
    :CRASHED => R_CRASHED,
    :LOW     => R_LOW,
    :NOMINAL => R_NOMINAL,
    :HIGH    => R_HIGH,
    :ESCAPED => R_ESCAPED,
)

"""
    build_stationkeeping_pomdp(; discount, n_mc, rng, drift_mean, drift_sigma)

Construct the toy stationkeeping POMDP as a DiscreteExplicitPOMDP-style
QuickPOMDP with precomputed transition/observation tables.
"""
function build_stationkeeping_pomdp(; discount::Float64 = 0.95,
                                    n_mc::Int = 20_000,
                                    rng = Random.default_rng(),
                                    drift_mean::Float64 = 12.0,
                                    drift_sigma::Float64 = 6.0)

    T = transition_matrix(; n_mc = n_mc, rng = rng,
                          drift_mean = drift_mean, drift_sigma = drift_sigma)
    O = observation_matrix()

    states = collect(STATE_NAMES)
    actions = collect(ACTION_NAMES)
    observations = collect(STATE_NAMES)

    sidx = Dict(s => i for (i, s) in enumerate(states))
    aidx = Dict(a => i for (i, a) in enumerate(actions))

    return QuickPOMDP(
        states = states,
        actions = actions,
        observations = observations,
        discount = discount,
        isterminal = s -> s in TERMINAL_STATES,

        # Start every episode holding NOMINAL (the injected orbit).
        initialstate = Deterministic(:NOMINAL),

        # T[s,a,:] over next states.
        transition = function (s, a)
            row = T[sidx[s], aidx[a], :]
            return SparseCat(states, row)
        end,

        # O depends only on the landed state s' here (nav noise on altitude).
        observation = function (a, sp)
            row = O[sidx[sp], :]
            return SparseCat(observations, row)
        end,

        # Reward = state-holding reward - fuel penalty. No reward once terminal.
        reward = function (s, a)
            s in TERMINAL_STATES && return 0.0
            return STATE_REWARD[s] - FUEL_WEIGHT * ACTION_DV_COST[a]
        end,
    )
end