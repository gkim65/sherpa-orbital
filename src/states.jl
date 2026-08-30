"""
states.jl — the factored (alt, visits) state space.

`alt` is the achieved periapsis-ALTITUDE bin; `visits` counts how many times each science
band has been sampled. Both the live altitude bins and the two terminal outcomes live in
the SAME variable, so there is no separate safety dimension.

⚠️ REPLACES THE (dev, cov) STATE SPACE (2026-08-30). The old state was
`(dev_bin, cov_mask)` with a binned deviation NORM as the observation, which cannot
distinguish "6 km high" from "6 km low" from "6 km sideways" — so the policy could not
reason about altitude at all. Two consequences of the redesign, both deliberate:

  1. **There is no `dev` dimension.** The policy cannot see `converged = false`, so a
     controller that is quietly losing the orbit looks the same as a healthy one at the
     same altitude. Accepted (user, 2026-08-30) because `solve_burn`'s `:position` mode
     enforces the orientation lock itself, every pass, whatever the policy commands — the
     Session 6 escape that motivated a safety dimension was `:position_altitude`, a
     formulation that had DROPPED the apoapsis direction constraints. Under the shipped
     mode that failure path is closed by the controller, not by the policy.
  2. **Coverage is banked on the OBSERVED altitude**, not the commanded one and not the
     true one. See [`alt_bin`](@ref) for the accuracy this actually has.

`BELOW_20` and `ABOVE_50` are ordinary low-reward bins, NOT terminal: the crash boundary is
5 km and the vehicle holds out to ~148 km, so leaving the 20–50 km science range is a
science failure, not a mission failure. Only `CRASHED` and `LOST` absorb.
"""

# ── Altitude bins ─────────────────────────────────────────────────────────────

"""Live (non-absorbing) altitude bins, in ascending altitude order."""
const ALT_BINS = (:BELOW_20, :A20_30, :A30_40, :A40_50, :ABOVE_50)

"""Absorbing outcomes. `CRASHED` = below the crash altitude, `LOST` = escaped."""
const TERMINAL_ALT = (:CRASHED, :LOST)

"""Full support of the altitude variable: live bins plus the two terminal outcomes."""
const ALT_ALL = (ALT_BINS..., TERMINAL_ALT...)

"""isterminal_alt(a) -> Bool. Is this an absorbing outcome?"""
isterminal_alt(a::Symbol) = a in TERMINAL_ALT

# ── Visit counts ──────────────────────────────────────────────────────────────
"""
    visit_inc(visits, b, cap) -> NTuple

Increment band `b`'s visit count, saturating at `cap`. Saturation is what keeps the state
space finite: counts are `(cap + 1)^n_bands`, so this grows EXPONENTIALLY in the number of
bands, not linearly in `cap`. At 3 bands, `cap = 3` is 64 combinations and `cap = 20` is
9261 — the latter needs a different encoding (a total count, or a count only for the band
being worked), not a bigger tuple.
"""
function visit_inc(visits::NTuple{N,Int}, b::Int, cap::Int) where {N}
    return ntuple(i -> i == b ? min(visits[i] + 1, cap) : visits[i], N)
end

"""visit_total(visits) -> Int. Total banked samples across all bands."""
visit_total(visits::NTuple{N,Int}) where {N} = sum(visits)

"""_zero_visits(pomdp) -> NTuple. The all-zero visit tuple, as carried by terminal states."""
_zero_visits(pomdp::StationkeepingPOMDP) = ntuple(_ -> 0, n_bands(pomdp))

# ── State type ────────────────────────────────────────────────────────────────
"""
    SKState(alt, visits)

A stationkeeping state. `alt ∈ ALT_ALL`; `visits` is a per-band visit count, saturating at
the POMDP's `visit_cap`. Terminal states carry all-zero visits: once the orbit is lost,
banked science no longer affects the decision.
"""
struct SKState{N}
    alt::Symbol
    visits::NTuple{N,Int}
end

"""isterminal_state(s) -> Bool."""
isterminal_state(s::SKState) = isterminal_alt(s.alt)

# ── State-space enumeration ───────────────────────────────────────────────────
"""
    n_bands(pomdp) -> Int
    n_visit_combos(pomdp) -> Int

Number of science bands, and the number of distinct visit-count tuples,
`(visit_cap + 1)^n_bands`.
"""
n_bands(pomdp::StationkeepingPOMDP) = length(pomdp.band_names)
n_visit_combos(pomdp::StationkeepingPOMDP) =
    (pomdp.visit_cap + 1)^n_bands(pomdp)

"""
    visit_tuples(pomdp) -> Vector{NTuple}

Every visit-count tuple, in odometer order with band 1 varying fastest. Stable, and it
defines the index convention shared by `states`, T, O and the alpha vectors.
"""
function visit_tuples(pomdp::StationkeepingPOMDP)
    nb, cap = n_bands(pomdp), pomdp.visit_cap
    out = NTuple{nb,Int}[]
    for idx in 0:(n_visit_combos(pomdp) - 1)
        rem = idx
        t = ntuple(nb) do _
            d = rem % (cap + 1)
            rem ÷= (cap + 1)
            d
        end
        push!(out, t)
    end
    return out
end

"""
    states(pomdp) -> Vector{SKState}

Enumerate the state space: every live altitude bin × every visit tuple, then the two
terminal states. Order is stable and defines the index convention for T, O and the alpha
vectors.
"""
function states(pomdp::StationkeepingPOMDP)
    nb = n_bands(pomdp)
    S = SKState{nb}[]
    for a in ALT_BINS, v in visit_tuples(pomdp)
        push!(S, SKState(a, v))
    end
    zero_v = ntuple(_ -> 0, nb)
    for a in TERMINAL_ALT
        push!(S, SKState(a, zero_v))
    end
    return S
end

"""state_index(pomdp) -> Dict{SKState,Int}. Inverse of `states`."""
state_index(pomdp::StationkeepingPOMDP) =
    Dict(s => i for (i, s) in enumerate(states(pomdp)))

"""n_states(pomdp) -> Int. |S| = 5 * (visit_cap + 1)^n_bands + 2."""
n_states(pomdp::StationkeepingPOMDP) =
    length(ALT_BINS) * n_visit_combos(pomdp) + length(TERMINAL_ALT)

# ── Binning ───────────────────────────────────────────────────────────────────
"""
    alt_bin(pomdp, alt_km) -> Symbol

Bin an achieved periapsis altitude (km) into a live altitude bin, half-open [lo, hi)
against `pomdp.alt_edges`. A non-finite altitude is `:LOST`.

⚠️ BIN MEMBERSHIP ONLY — there is no tolerance margin, deliberately. A ±5 km margin was
specified and dropped: on 10 km bins it accepts a 20 km window, which overlaps BOTH
neighbours completely, so every observation would qualify for at least two bands and the
gate would dissolve rather than tighten.

⚠️ THE ACCURACY THIS HAS. Coverage is banked on the OBSERVED altitude, so binning is
deterministic given the observation (which is what keeps
`policy_reconcentrate_cov!`'s hard belief projection valid) but NOT correct: a true 32 km
pass read as 27 km banks the wrong band. With `sigma_nav_km = 2` and 10 km bins the error
is an edge effect — interior truth is essentially never misbinned at 2σ = 4 km, truth on an
edge is ~50/50 — giving roughly a 15–20% per-pass misbin rate, concentrated near
boundaries and SYMMETRIC (as likely to lose a visited band as to bank an unvisited one).
Report the science product with that rate attached; it is not clean coverage.

Finer bins do NOT help: σ_nav is set by the sensor, so narrower bins put MORE passes near a
boundary. Bins below ~2σ ≈ 4 km mostly encode noise. The levers are `sigma_nav_km` (ours is
a deliberately conservative 2 km vs MacKenzie §C.1.1.2's ~0.3–1 km) or the bin edges.

This must stay bit-identical to the consumer-side binning in any rollout harness, or the
belief filter is fed wrong labels.
"""
function alt_bin(pomdp::StationkeepingPOMDP, alt_km::Real)
    !isfinite(alt_km) && return :LOST
    e = pomdp.alt_edges
    alt_km < e[1] && return :BELOW_20
    alt_km < e[2] && return :A20_30
    alt_km < e[3] && return :A30_40
    alt_km < e[4] && return :A40_50
    return :ABOVE_50
end

"""
    band_of_alt(pomdp, alt_km) -> Union{Int,Nothing}

Which science band (1-based) an altitude falls in, or `nothing` if it is outside every
band. Bin membership only — see [`alt_bin`](@ref). This is the coverage gate: it is fed
the OBSERVED altitude, never the true one, and it is indifferent to which action ran, so
`CORRECT` banks its band passively just as an `EXCURSE_*` does.
"""
function band_of_alt(pomdp::StationkeepingPOMDP, alt_km::Real)
    b = alt_bin(pomdp, alt_km)
    idx = findfirst(==(b), pomdp.band_bins)
    return idx === nothing ? nothing : idx
end

"""
    state_label(s) -> String

Serialization label `"ALT|v1-v2-v3"`, used in the exported policy JSON so a consumer can
reconstruct `(alt, visits)` without knowing the enumeration order.
"""
state_label(s::SKState) = "$(s.alt)|" * join(s.visits, "-")

"""visit_label(pomdp, visits) -> String. Human-readable coverage, e.g. "LOW×2+MID×1"."""
function visit_label(pomdp::StationkeepingPOMDP, visits::NTuple{N,Int}) where {N}
    all(==(0), visits) && return "none"
    join([ "$(pomdp.band_names[b])×$(visits[b])"
           for b in 1:N if visits[b] > 0 ], "+")
end