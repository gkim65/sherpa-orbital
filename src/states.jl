"""
states.jl — the factored (dev, cov) state space.

`dev` is the SAFETY variable (apse-position deviation bin); `cov` is the SCIENCE
variable (a bitmask of which altitude bands have been sampled). Terminal states collapse
`cov` to 0: once the orbit is lost, banked science no longer matters to the decision.
"""

# ── Coverage bitmask helpers ──────────────────────────────────────────────────
"""cov_has(mask, b) -> Bool. Has band `b` (1-based) been sampled?"""
cov_has(mask::Int, b::Int) = (mask & (1 << (b - 1))) != 0

"""cov_set(mask, b) -> Int. Bank band `b` (1-based) into the coverage mask."""
cov_set(mask::Int, b::Int) = mask | (1 << (b - 1))

"""cov_count(mask) -> Int. How many bands have been sampled."""
cov_count(mask::Int) = count_ones(mask)

# ── State type ────────────────────────────────────────────────────────────────
"""
    SKState(dev, cov)

A stationkeeping state. `dev ∈ (:OK, :DRIFT, :FAR, :LOST, :CRASHED)`; `cov` is a
3-bit coverage mask (0..7), and is 0 for the terminal states.
"""
struct SKState
    dev::Symbol
    cov::Int
end

const DEV_NAMES     = (:OK, :DRIFT, :FAR, :LOST)   # :LOST is terminal
const TERMINAL_DEV  = (:LOST, :CRASHED)
const NONTERM_DEV   = (:OK, :DRIFT, :FAR)
"""Next-dev support of the transition kernels (adds :CRASHED to DEV_NAMES)."""
const DEV_NEXT      = (:OK, :DRIFT, :FAR, :LOST, :CRASHED)

"""isterminal_dev(dev) -> Bool. Is this dev bin an absorbing mission-loss state?"""
isterminal_dev(d::Symbol) = d in TERMINAL_DEV

# ── State-space enumeration ───────────────────────────────────────────────────
"""
    n_bands(pomdp) -> Int
    n_cov(pomdp)   -> Int

Number of science bands, and the number of coverage masks (2^n_bands).
"""
n_bands(pomdp::StationkeepingPOMDP) = length(pomdp.band_names)
n_cov(pomdp::StationkeepingPOMDP)   = 1 << n_bands(pomdp)

"""
    states(pomdp) -> Vector{SKState}

Enumerate the state space: (OK/DRIFT/FAR) × every coverage mask, then the two terminal
states. Order is stable and defines the index convention for T, O, and the alpha vectors.
"""
function states(pomdp::StationkeepingPOMDP)
    S = SKState[]
    for d in NONTERM_DEV, c in 0:(n_cov(pomdp) - 1)
        push!(S, SKState(d, c))
    end
    push!(S, SKState(:LOST, 0))
    push!(S, SKState(:CRASHED, 0))
    return S
end

"""state_index(pomdp) -> Dict{SKState,Int}. Inverse of `states`."""
state_index(pomdp::StationkeepingPOMDP) =
    Dict(s => i for (i, s) in enumerate(states(pomdp)))

"""n_states(pomdp) -> Int. |S| = 3 * 2^n_bands + 2."""
n_states(pomdp::StationkeepingPOMDP) = 3 * n_cov(pomdp) + 2

# ── Binning ───────────────────────────────────────────────────────────────────
"""
    dev_bin(pomdp, dev_km) -> Symbol

Bin an apse-position deviation (km) into a safety bin, using the half-open [lo, hi)
convention. A non-finite deviation is :LOST. This must stay bit-identical to the
consumer-side binning in any rollout harness, or the belief filter is fed wrong labels.
"""
function dev_bin(pomdp::StationkeepingPOMDP, dev_km::Real)
    e = pomdp.dev_edges
    !isfinite(dev_km) && return :LOST
    dev_km < e[1] && return :OK
    dev_km < e[2] && return :DRIFT
    dev_km < e[3] && return :FAR
    return :LOST
end

"""
    state_label(s) -> String

Serialization label "DEV|cov<mask>", used in the exported policy JSON so a consumer can
reconstruct (dev, cov) without knowing the enumeration order.
"""
state_label(s::SKState) = "$(s.dev)|cov$(s.cov)"

"""cov_label(pomdp, mask) -> String. Human-readable coverage, e.g. "LOW+HIGH"."""
function cov_label(pomdp::StationkeepingPOMDP, mask::Int)
    mask == 0 && return "none"
    join([string(pomdp.band_names[b]) for b in 1:n_bands(pomdp) if cov_has(mask, b)], "+")
end