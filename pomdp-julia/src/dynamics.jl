"""
dynamics.jl — the crude drift+burn model for the toy stationkeeping POMDP.

THIS IS A PLACEHOLDER FOR THE REAL CR3BP DYNAMICS.

The whole point of isolating this file is to give a clean seam: everything the
POMDP needs about the physics flows through `step_altitude` (one-step continuous
dynamics) and the two table builders (`transition_matrix`, `observation_matrix`).
When we are ready to bridge to the real integrator, `step_altitude` gets swapped
for a call into the Python CR3BP (via PyCall) or a native Julia CR3BP; nothing
else in the model has to change.

Physical reference numbers (mirrored from the Python `src/constants.py`, all km):
  - Crash altitude          : < 5 km      (PERIAPSIS_CRASH_ALT)
  - MacKenzie periapsis band: 19.8-64.3   (PERIAPSIS_ALT_MIN/MAX) — our NOMINAL
                              25-60 band sits inside this
  - Nav 1-sigma             : 2 km        (SIGMA_NAV_POS)

The stochastic pieces (burn efficiency eta_eff ~ Uniform(0.8, 1.0); Gaussian nav
noise) are re-implemented natively here as one-liners rather than imported from
the Python `src/spacecraft/` modules, per the toolchain plan.

Units: periapsis-altitude ERROR is tracked in km, measured relative to the
centre of the NOMINAL band. Positive = higher than nominal, negative = lower
(toward crash). This scalar is the toy's entire continuous state.
"""

using Random
using Distributions

# ---------------------------------------------------------------------------
# State discretisation — periapsis-altitude-error bins (km of ACTUAL altitude).
# These are absolute periapsis altitudes; the bin edges come straight from the
# MacKenzie bands + the crash/escape thresholds in the prompt.
# ---------------------------------------------------------------------------
# bin name      altitude range (km)     index
#   CRASHED       (-inf, 5)               1   terminal
#   LOW           [5, 25)                 2
#   NOMINAL       [25, 60)                3
#   HIGH          [60, 120)               4
#   ESCAPED       [120, +inf)             5   terminal
const BIN_EDGES = (5.0, 25.0, 60.0, 120.0)      # 4 edges -> 5 bins
const STATE_NAMES = (:CRASHED, :LOW, :NOMINAL, :HIGH, :ESCAPED)
const N_STATES = length(STATE_NAMES)
const TERMINAL_STATES = (:CRASHED, :ESCAPED)

# Representative (bin-centre-ish) altitude used when we need a continuous value
# to seed a Monte-Carlo rollout from a discrete state. For the open-ended
# terminal bins we pick a point just inside.
const BIN_REPRESENTATIVE_ALT = Dict(
    :CRASHED  => 2.5,     # already crashed; only used defensively
    :LOW      => 15.0,
    :NOMINAL  => 42.5,    # centre of 25-60
    :HIGH     => 90.0,
    :ESCAPED  => 150.0,   # already escaped; only used defensively
)

"""
    bin_of(alt_km) -> Symbol

Map an absolute periapsis altitude (km) to its discrete state bin.
"""
function bin_of(alt_km::Real)
    if alt_km < BIN_EDGES[1]
        return :CRASHED
    elseif alt_km < BIN_EDGES[2]
        return :LOW
    elseif alt_km < BIN_EDGES[3]
        return :NOMINAL
    elseif alt_km < BIN_EDGES[4]
        return :HIGH
    else
        return :ESCAPED
    end
end

# ---------------------------------------------------------------------------
# Actions — each a fixed commanded delta-V that raises periapsis altitude.
# In this crude model a burn's *effect* is a raise in periapsis altitude
# (km per step) proportional to the commanded delta-V, scaled by the random
# burn efficiency eta_eff. Real CR3BP burns are 3-D and couple both apses;
# the placeholder collapses that to a scalar altitude raise.
# ---------------------------------------------------------------------------
const ACTION_NAMES = (:NO_BURN, :SMALL_BURN, :LARGE_BURN)
const N_ACTIONS = length(ACTION_NAMES)

# Commanded altitude raise per action (km), before eta_eff. Tuned so a SMALL
# burn roughly counters one step of nominal drift and a LARGE burn climbs a bin.
const ACTION_RAISE_KM = Dict(
    :NO_BURN    => 0.0,
    :SMALL_BURN => 18.0,
    :LARGE_BURN => 45.0,
)

# Rough delta-V cost proxy (m/s) per action, only used to shape the reward's
# fuel penalty. Proportional to the altitude raise.
const ACTION_DV_COST = Dict(
    :NO_BURN    => 0.0,
    :SMALL_BURN => 1.0,
    :LARGE_BURN => 2.5,
)

# ---------------------------------------------------------------------------
# Stochastic pieces (native re-implementation of the Phase-2 Python models).
# ---------------------------------------------------------------------------
const ETA_EFF_MIN = 0.8      # Uniform(0.8, 1.0) burn efficiency (thruster.py)
const ETA_EFF_MAX = 1.0
const SIGMA_NAV_KM = 2.0     # Gaussian nav 1-sigma, km (SIGMA_NAV_POS / nav.py)

# ---------------------------------------------------------------------------
# One-step continuous dynamics — THE SWAPPABLE SEAM.
# ---------------------------------------------------------------------------
"""
    step_altitude(alt_km, action; rng, drift_mean, drift_sigma) -> Float64

Advance the periapsis altitude by one control step under the crude drift+burn
model. Returns the new absolute periapsis altitude (km).

Model (placeholder — NOT the real CR3BP):
  new_alt = alt + burn_raise - drift
where
  burn_raise = eta_eff * ACTION_RAISE_KM[action],  eta_eff ~ Uniform(0.8, 1.0)
  drift      ~ Normal(drift_mean, drift_sigma), clamped >= 0

The positive `drift_mean` encodes the empirical finding (see docs/todo.md Phase 4)
that this period-3 orbit is unstable and periapsis DECAYS toward the body without
control. The Normal spread is process noise standing in for the unmodelled
higher-fidelity perturbations (Saturn J2, moons).

When bridged to the real dynamics, replace the body of this function with a call
that propagates one control period of the CR3BP(+perturbation) truth model and
returns the resulting periapsis altitude — the rest of the POMDP is unchanged.
"""
function step_altitude(alt_km::Real, action::Symbol;
                       rng::AbstractRNG = Random.default_rng(),
                       drift_mean::Float64 = 12.0,
                       drift_sigma::Float64 = 6.0)
    eta_eff = ETA_EFF_MIN + (ETA_EFF_MAX - ETA_EFF_MIN) * rand(rng)
    burn_raise = eta_eff * ACTION_RAISE_KM[action]
    drift = max(0.0, drift_mean + drift_sigma * randn(rng))
    return alt_km + burn_raise - drift
end

# ---------------------------------------------------------------------------
# Monte-Carlo table builders. These turn the continuous crude model into the
# discrete T and O tables SARSOP consumes. SARSOP itself never calls the
# dynamics — it only sees these precomputed tables (offline / tabular solve).
# ---------------------------------------------------------------------------
"""
    transition_matrix(; n_mc, rng, kwargs...) -> Array{Float64,3}

Estimate T[s, a, s'] = P(next bin s' | current bin s, action a) by Monte-Carlo
rollout of `step_altitude` from each (state, action) pair. Terminal states
(CRASHED, ESCAPED) are absorbing: they transition to themselves w.p. 1.

Returns an (N_STATES x N_ACTIONS x N_STATES) array.
"""
function transition_matrix(; n_mc::Int = 20_000,
                           rng::AbstractRNG = Random.default_rng(),
                           kwargs...)
    T = zeros(Float64, N_STATES, N_ACTIONS, N_STATES)
    for (si, s) in enumerate(STATE_NAMES)
        if s in TERMINAL_STATES
            T[si, :, si] .= 1.0          # absorbing
            continue
        end
        alt0 = BIN_REPRESENTATIVE_ALT[s]
        for (ai, a) in enumerate(ACTION_NAMES)
            counts = zeros(Int, N_STATES)
            for _ in 1:n_mc
                alt1 = step_altitude(alt0, a; rng = rng, kwargs...)
                counts[findfirst(==(bin_of(alt1)), STATE_NAMES)] += 1
            end
            T[si, ai, :] .= counts ./ n_mc
        end
    end
    return T
end

"""
    observation_matrix() -> Matrix{Float64}

Estimate O[s', o] = P(observe bin o | landed in bin s'), from Gaussian nav noise
of 1-sigma SIGMA_NAV_KM on the altitude measurement. The observation space is
the same 5 bins as the state (a noisy read of which altitude band we're in).

Computed analytically per state using the bin representative altitude and the
Normal CDF over the bin edges (no Monte-Carlo needed for a 1-D Gaussian).
Terminal states are observed perfectly (the mission knows if it crashed/escaped).

Returns an (N_STATES x N_STATES) row-stochastic matrix.
"""
function observation_matrix()
    O = zeros(Float64, N_STATES, N_STATES)
    edges = (-Inf, BIN_EDGES..., Inf)      # 6 edges -> 5 bins
    for (si, s) in enumerate(STATE_NAMES)
        if s in TERMINAL_STATES
            O[si, si] = 1.0                # observed perfectly
            continue
        end
        mu = BIN_REPRESENTATIVE_ALT[s]
        d = Normal(mu, SIGMA_NAV_KM)
        for oi in 1:N_STATES
            lo, hi = edges[oi], edges[oi + 1]
            O[si, oi] = cdf(d, hi) - cdf(d, lo)
        end
        O[si, :] ./= sum(O[si, :])         # guard against tiny CDF round-off
    end
    return O
end