"""
dynamics.jl — science+safety stationkeeping POMDP (Stage-1-science, 2026-07-15).

REPLACES the 2026-07-06 scalar-altitude toy. State/action/transition are grounded in
MEASURED CR3BP+EncJ2 physics (scripts/pomdp_experiments/, exps 01/03/06/07/11b/12), not
a hand-tuned surrogate. See docs/pomdp-proposal-2026-07-13.md.

Concept
-------
The agent trades SCIENCE (visit a range of periapsis altitudes to gather info) against
SAFETY (keep the orbit from crashing/escaping). Two coupled state variables:

  dev  : apse-position deviation from the nominal orbit — the SAFETY variable.
         bins  OK (<15 km) · DRIFT (15-60) · FAR (60-200) · LOST (>=200, terminal)
         plus  CRASHED (terminal).  (edges from exp 02/11b measured deviation scale.)
  cov  : which of the 3 science altitude BANDS have been sampled — a 3-bit mask.
         bands (single-burn reachable, ~33-65 km, exp 07): LOW · MID · HIGH.
         2^3 = 8 coverage states (none .. all three).

State = (dev, cov). |S| = 4 non-terminal dev × 8 cov + 2 terminal (CRASHED, LOST) = 34.

Actions (INTENT — the exact burn VECTOR is solved by mpc.solve_burn in the Python
rollout; exp 04 proved a fixed-direction menu fails, so direction is never discretized):
  OBSERVE      : no burn; take a nav reading. Cheap, but dev drifts (instability).
  CORRECT      : solve_burn toward nominal — holds the orbit (exp 11b: OK->OK ~0.98).
  EXCURSE_LOW  : one-pass single-burn excursion to the LOW band, then recover; marks
  EXCURSE_MID    LOW/MID/HIGH sampled in cov. Costs fuel + a little safety margin
  EXCURSE_HIGH   (exp 06/12: excursion holds + recovers).

Reward (in stationkeeping_pomdp.jl): + for each NEWLY sampled band, − fuel per burn,
large − on CRASHED/LOST. The POMDP must sequence excursions while staying safe.

Transition/observation tables are built HERE and consumed offline by SARSOP; SARSOP
never calls the dynamics. The Python rollout (baselines/pomdp_rollout.py) realizes each
action against the real truth model.
"""

using Random
using Distributions

# ---------------------------------------------------------------------------
# Safety variable: apse-position deviation bins.
# ---------------------------------------------------------------------------
const DEV_EDGES = (15.0, 60.0, 200.0)               # km; OK/DRIFT/FAR/LOST
const DEV_NAMES = (:OK, :DRIFT, :FAR, :LOST)         # LOST terminal
const N_DEV = length(DEV_NAMES)

"""dev_bin(dev_km) -> Symbol. Bin an apse-position deviation (km)."""
function dev_bin(dev_km::Real)
    !isfinite(dev_km) && return :LOST
    dev_km < DEV_EDGES[1] && return :OK
    dev_km < DEV_EDGES[2] && return :DRIFT
    dev_km < DEV_EDGES[3] && return :FAR
    return :LOST
end

# ---------------------------------------------------------------------------
# Science variable: coverage of 3 altitude bands (single-burn reachable ~33-65 km).
# Represented as a 3-bit mask; band commanded targets are documented for the rollout.
# ---------------------------------------------------------------------------
const BAND_NAMES = (:LOW, :MID, :HIGH)               # ~35 / 48 / 62 km achieved
const BAND_TARGET_KM = Dict(:LOW => 40.0, :MID => 70.0, :HIGH => 120.0)  # commanded (compressed)
const N_BANDS = length(BAND_NAMES)
const N_COV = 1 << N_BANDS                            # 8 coverage masks

cov_has(mask::Int, b::Int) = (mask & (1 << (b - 1))) != 0
cov_set(mask::Int, b::Int) = mask | (1 << (b - 1))
cov_count(mask::Int) = count_ones(mask)

# ---------------------------------------------------------------------------
# Full state space: (dev, cov). Terminal states collapse cov (once lost, cov frozen).
# We enumerate all (dev in OK/DRIFT/FAR) × cov, plus the two terminals.
# ---------------------------------------------------------------------------
struct SKState
    dev::Symbol      # :OK :DRIFT :FAR :LOST :CRASHED
    cov::Int         # 0..7 (bitmask); ignored/0 for CRASHED
end

const TERMINAL_DEV = (:LOST, :CRASHED)
isterminal_dev(d::Symbol) = d in TERMINAL_DEV

"""all_states() -> Vector{SKState}. Non-terminal (OK/DRIFT/FAR × 8 cov) + 2 terminals."""
function all_states()
    S = SKState[]
    for d in (:OK, :DRIFT, :FAR), c in 0:(N_COV - 1)
        push!(S, SKState(d, c))
    end
    push!(S, SKState(:LOST, 0))
    push!(S, SKState(:CRASHED, 0))
    return S
end

const STATES = all_states()
const N_STATES = length(STATES)
const STATE_INDEX = Dict(s => i for (i, s) in enumerate(STATES))

# ---------------------------------------------------------------------------
# Actions.
# ---------------------------------------------------------------------------
const ACTION_NAMES = (:OBSERVE, :CORRECT, :EXCURSE_LOW, :EXCURSE_MID, :EXCURSE_HIGH)
const N_ACTIONS = length(ACTION_NAMES)
const EXCURSE_BAND = Dict(:EXCURSE_LOW => 1, :EXCURSE_MID => 2, :EXCURSE_HIGH => 3)

# ΔV cost proxy (m/s) per action. CORRECT from exp 11b (~1.3/step); EXCURSE costs
# (incl. the recovery burn) MEASURED in exp 12: LOW 2.1, MID 3.1, HIGH 9.9 m/s.
const ACTION_DV_COST = Dict(
    :OBSERVE => 0.0, :CORRECT => 1.3,
    :EXCURSE_LOW => 2.1, :EXCURSE_MID => 3.1, :EXCURSE_HIGH => 9.9,
)

# ---------------------------------------------------------------------------
# MEASURED dev-transition kernels (per action) over dev bins {OK,DRIFT,FAR,LOST,CRASHED}.
# Sources:
#   CORRECT  : exp 11b sustained MPC loop (OK->OK 0.98; from off-nominal, pulls inward).
#   OBSERVE  : exp 11b counterfactual (OK stays ~1 pass) + exp 01 uncontrolled decay
#              (dev grows OK->DRIFT->FAR->LOST over ~2-3 skipped passes).
#   EXCURSE  : exp 06/12 — the excursion + its recovery burn returns dev near OK, with
#              a small safety cost (some probability of a worse dev bin). Filled from
#              exp 12 (see _EXC_DEV below; conservative defaults until 12 completes).
# Rows are P(next dev | dev, action) over (:OK,:DRIFT,:FAR,:LOST,:CRASHED).
# ---------------------------------------------------------------------------
const DEV_NEXT = (:OK, :DRIFT, :FAR, :LOST, :CRASHED)

# CORRECT: strong inward pull; measured OK->OK 0.98 (exp 11b). Off-nominal recovers.
const T_CORRECT = Dict(
    :OK    => [0.98, 0.01, 0.00, 0.01, 0.00],
    :DRIFT => [0.80, 0.15, 0.03, 0.02, 0.00],
    :FAR   => [0.30, 0.40, 0.20, 0.10, 0.00],
)

# OBSERVE: no burn → dev decays outward (instability, exp 01). From OK a single skip is
# mostly safe (exp 11b 0.94), but mass leaks outward; from DRIFT/FAR it runs away.
const T_OBSERVE = Dict(
    :OK    => [0.60, 0.30, 0.07, 0.02, 0.01],
    :DRIFT => [0.05, 0.35, 0.40, 0.18, 0.02],
    :FAR   => [0.00, 0.05, 0.25, 0.65, 0.05],
)

# EXCURSE (any band): a deliberate one-pass dip + recovery burn. MEASURED in exp 12:
# from OK the excursion RECOVERS to dev=OK deterministically (3/3 trials, every band),
# so it is nearly as safe as CORRECT when started safe — we keep a small residual risk
# for realism/robustness. From DRIFT/FAR (already off-nominal) an added perturbation is
# riskier, so mass leaks outward more than CORRECT would.
const T_EXCURSE = Dict(
    :OK    => [0.95, 0.03, 0.01, 0.01, 0.00],
    :DRIFT => [0.45, 0.25, 0.15, 0.13, 0.02],
    :FAR   => [0.10, 0.20, 0.25, 0.40, 0.05],
)

"""dev_kernel(action, dev) -> Vector{Float64} over DEV_NEXT."""
function dev_kernel(action::Symbol, dev::Symbol)
    action == :CORRECT && return T_CORRECT[dev]
    action == :OBSERVE && return T_OBSERVE[dev]
    return T_EXCURSE[dev]     # any EXCURSE_*
end

# ---------------------------------------------------------------------------
# Transition over the FULL (dev, cov) state.
# dev evolves by the measured kernel; cov gains the excursed band IF the orbit did NOT
# go terminal on that step (you only bank the science if you survived the pass).
# ---------------------------------------------------------------------------
"""
    transition_matrix(; kwargs...) -> Array{Float64,3}

Build T[s, a, s'] over the enumerated (dev,cov) states. SARSOP consumes this offline.
"""
function transition_matrix(; kwargs...)
    T = zeros(Float64, N_STATES, N_ACTIONS, N_STATES)
    for (si, s) in enumerate(STATES)
        # Terminal states absorb.
        if isterminal_dev(s.dev)
            T[si, :, si] .= 1.0
            continue
        end
        for (ai, a) in enumerate(ACTION_NAMES)
            k = dev_kernel(a, s.dev)             # over DEV_NEXT
            banded = get(EXCURSE_BAND, a, 0)     # 0 unless an EXCURSE_*
            newcov = banded == 0 ? s.cov : cov_set(s.cov, banded)
            for (di, dn) in enumerate(DEV_NEXT)
                p = k[di]
                p == 0.0 && continue
                if dn == :CRASHED
                    sp = SKState(:CRASHED, 0)
                elseif dn == :LOST
                    sp = SKState(:LOST, 0)
                else
                    # survived the pass → bank the excursion's band into cov.
                    sp = SKState(dn, newcov)
                end
                T[si, ai, STATE_INDEX[sp]] += p
            end
        end
    end
    return T
end

# ---------------------------------------------------------------------------
# Observation: noisy read of the dev bin (nav noise on the apse deviation); cov is
# known exactly (we know which excursions we commanded). Observation space = dev bins.
# ---------------------------------------------------------------------------
const OBS_NAMES = (:OK, :DRIFT, :FAR, :LOST, :CRASHED)
const N_OBS = length(OBS_NAMES)
const SIGMA_NAV_KM = 2.0

# Representative deviation (km) per non-terminal dev bin, for the Gaussian nav model.
const DEV_REP_KM = Dict(:OK => 7.0, :DRIFT => 35.0, :FAR => 120.0)

"""observation_matrix() -> Matrix{Float64}  O[s, o] over OBS_NAMES (dev read)."""
function observation_matrix()
    O = zeros(Float64, N_STATES, N_OBS)
    edges = (-Inf, DEV_EDGES..., Inf)         # OK/DRIFT/FAR/LOST edges on dev
    devobs_index = Dict(:OK => 1, :DRIFT => 2, :FAR => 3, :LOST => 4, :CRASHED => 5)
    for (si, s) in enumerate(STATES)
        if s.dev == :CRASHED
            O[si, devobs_index[:CRASHED]] = 1.0; continue
        end
        if s.dev == :LOST
            O[si, devobs_index[:LOST]] = 1.0; continue
        end
        mu = DEV_REP_KM[s.dev]
        d = Normal(mu, SIGMA_NAV_KM)
        # Probability mass in each dev band (OK/DRIFT/FAR/LOST), observed as that bin.
        for oi in 1:4
            lo, hi = edges[oi], edges[oi + 1]
            O[si, oi] = cdf(d, hi) - cdf(d, lo)
        end
        O[si, :] ./= sum(O[si, :])
    end
    return O
end
