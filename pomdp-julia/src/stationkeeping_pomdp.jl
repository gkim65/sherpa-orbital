"""
stationkeeping_pomdp.jl — the science+safety Enceladus stationkeeping POMDP.

Trades SCIENCE (sample a range of periapsis altitudes) against SAFETY (don't crash or
escape the unstable period-3 halo). Physics-grounded: state/action/transition come from
measured CR3BP+EncJ2 experiments (see dynamics.jl + docs/pomdp-proposal-2026-07-13.md).

Formulation
  State   : (dev, cov)  — dev = apse-deviation safety bin (OK/DRIFT/FAR/LOST/CRASHED),
            cov = 3-bit coverage mask over science altitude bands (LOW/MID/HIGH).
            LOST and CRASHED are terminal.
  Action  : OBSERVE / CORRECT / EXCURSE_LOW / EXCURSE_MID / EXCURSE_HIGH.
            Direction of any burn is solved by mpc.solve_burn in the rollout (exp 04);
            the action only picks INTENT.
  Obs     : noisy dev bin (nav noise on the apse deviation); cov known exactly.
  Reward  : + R_SCIENCE for each NEWLY sampled band, − fuel (ACTION_DV_COST),
            large − on CRASHED / LOST. The agent must sequence excursions while staying
            safe, and stop excursing once all bands are covered.
"""

using POMDPs
using QuickPOMDPs
using POMDPTools

include("dynamics.jl")

# ---------------------------------------------------------------------------
# Reward parameters.
# ---------------------------------------------------------------------------
const R_SCIENCE   =  20.0     # reward for sampling a band we had NOT sampled yet
const R_STEP_OK   =   0.5     # small living reward for staying safe (non-terminal)
const R_CRASHED   = -200.0    # mission loss
const R_LOST      = -200.0    # mission loss (escape)
const FUEL_WEIGHT =   1.0     # multiplies ACTION_DV_COST (m/s) as a penalty

"""
    build_stationkeeping_pomdp(; discount)

Construct the science+safety POMDP as a QuickPOMDP over the enumerated (dev,cov) states
with the measured transition/observation tables from dynamics.jl.
"""
function build_stationkeeping_pomdp(; discount::Float64 = 0.95, kwargs...)
    T = transition_matrix(; kwargs...)
    O = observation_matrix()

    states = STATES                      # Vector{SKState}
    actions = collect(ACTION_NAMES)
    observations = collect(OBS_NAMES)

    sidx = STATE_INDEX
    aidx = Dict(a => i for (i, a) in enumerate(actions))

    isterm(s::SKState) = isterminal_dev(s.dev)

    # Indices of the terminal states, for the expected terminal-entry penalty.
    i_crashed = sidx[SKState(:CRASHED, 0)]
    i_lost    = sidx[SKState(:LOST, 0)]

    # r(s,a): living reward + fuel penalty + science for a first-time band + the
    # EXPECTED penalty for transitioning INTO a terminal state this step. Encoding the
    # crash/escape cost as its expectation over T[s,a,·] is exact for expected-reward
    # planning and is the standard QuickPOMDP-compatible way to penalize terminal entry.
    function reward(s::SKState, a::Symbol)
        isterm(s) && return 0.0
        r = R_STEP_OK - FUEL_WEIGHT * ACTION_DV_COST[a]
        banded = get(EXCURSE_BAND, a, 0)
        if banded != 0 && !cov_has(s.cov, banded)
            r += R_SCIENCE                 # first time sampling this band
        end
        row = T[sidx[s], aidx[a], :]
        r += row[i_crashed] * R_CRASHED + row[i_lost] * R_LOST
        return r
    end

    return QuickPOMDP(
        states = states,
        actions = actions,
        observations = observations,
        discount = discount,
        isterminal = isterm,

        # Start: on the nominal orbit (dev OK), no bands sampled yet.
        initialstate = Deterministic(SKState(:OK, 0)),

        transition = function (s, a)
            row = T[sidx[s], aidx[a], :]
            return SparseCat(states, row)
        end,

        # Observation depends on the landed state's dev bin (nav noise); cov exact.
        observation = function (a, sp)
            row = O[sidx[sp], :]
            return SparseCat(observations, row)
        end,

        reward = reward,
    )
end
