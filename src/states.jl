"""
states.jl — the factored (alt, visits, intensity, residual) state space.

`alt` is the achieved periapsis-altitude bin and carries the two absorbing outcomes, so
there is no separate safety dimension. `visits` counts samples per science band. Only
`CRASHED` and `LOST` absorb; the outer altitude bins are ordinary low-reward bins.

NOTE: coverage is banked on the OBSERVED altitude — see [`alt_bin`](@ref) for the misbin
rate that carries.
"""

# ── Altitude bins ─────────────────────────────────────────────────────────────

"""Live (non-absorbing) altitude bins, in ascending altitude order."""
const ALT_BINS = (:BELOW_20, :A20_27, :A27_34, :A34_44, :ABOVE_44)

"""Absorbing outcomes. `CRASHED` = below the crash altitude, `LOST` = escaped."""
const TERMINAL_ALT = (:CRASHED, :LOST)

"""Full support of the altitude variable: live bins plus the two terminal outcomes."""
const ALT_ALL = (ALT_BINS..., TERMINAL_ALT...)

"""isterminal_alt(a) -> Bool. Is altitude label `a` one of the absorbing outcomes?"""
isterminal_alt(a::Symbol) = a in TERMINAL_ALT

# ── Residual (orbit-damage) bins ──────────────────────────────────────────────
"""
Orbit-damage bins, in ascending order of degradation. Bins the onboard `solve_burn`
residual: how badly the apse constraints failed to be satisfiable by one impulse.
"""
const RESIDUAL_BINS = (:R_OK, :R_DEGRADED, :R_CRITICAL)

"""
Residual bin edges (km), half-open [lo, hi). Measured against P(losing the apse pair on
the next pass); regenerate with `experiments/calibrate.jl`.

NOTE: part of the state-space definition, not a sweep axis. Every θ in a family shares
them, and `load_tables` rejects an artifact measured with different edges.
"""
const RESIDUAL_EDGES = (15.0, 25.0)

"""
    residual_bin(residual_km) -> Symbol

Bin an onboard `solve_burn` residual into an orbit-damage bin, half-open [lo, hi).

  - `residual_km` — the solve residual for the pass just flown (km)

Returns a member of `RESIDUAL_BINS`.

NOTE: a non-finite residual bins as `:R_CRITICAL`. That is the lost-apse-pair case — the
onboard model found no apse pair, `solve_burn` returned ΔV = 0, and the pass flew
uncontrolled. It is the worst live state rather than a bin of its own, because the loss it
usually causes is carried by the altitude variable's `:LOST` outcome on the next pass.
"""
function residual_bin(residual_km::Real)
    isfinite(residual_km) || return :R_CRITICAL
    residual_km < RESIDUAL_EDGES[1] && return :R_OK
    residual_km < RESIDUAL_EDGES[2] && return :R_DEGRADED
    return :R_CRITICAL
end

"""residual_index(r) -> Int. 1-based position of residual bin `r` in `RESIDUAL_BINS`."""
residual_index(r::Symbol) = findfirst(==(r), RESIDUAL_BINS)

# ── Visit counts ──────────────────────────────────────────────────────────────
"""
    visit_inc(visits, b, cap) -> NTuple

Increment band `b`'s visit count, saturating at `cap`.

  - `visits` — current per-band visit-count tuple
  - `b` — 1-based band index to increment
  - `cap` — saturation ceiling

Returns a new tuple; the input is not mutated.

NOTE: saturation is what keeps the state space finite. The count of tuples is
`(cap + 1)^n_bands`, exponential in bands rather than linear in `cap` — a large target
needs a different encoding (a total count, or a count only for the band being worked),
not a bigger cap.
"""
function visit_inc(visits::NTuple{N,Int}, b::Int, cap::Int) where {N}
    return ntuple(i -> i == b ? min(visits[i] + 1, cap) : visits[i], N)
end

"""visit_total(visits) -> Int. Total banked samples summed across all bands."""
visit_total(visits::NTuple{N,Int}) where {N} = sum(visits)

"""_zero_visits(pomdp) -> NTuple. The all-zero visit tuple, as carried by terminal states."""
_zero_visits(pomdp::StationkeepingPOMDP) = ntuple(_ -> 0, n_bands(pomdp))

# ── State type ────────────────────────────────────────────────────────────────
"""
    SKState(alt, visits)
    SKState(alt, visits, intensity)
    SKState(alt, visits, intensity, residual)

A stationkeeping state.

  - `alt` — altitude bin, a member of `ALT_ALL`
  - `visits` — per-band sample count, saturating at the POMDP's `visit_cap`. Terminal
    states carry all-zero visits
  - `intensity` — plume sample intensity the last pass yielded, `1:plume_levels`, drawn by
    the transition from `P_θ(i | band)`. Defaults to 1
  - `residual` — orbit-damage bin of that pass, a member of `RESIDUAL_BINS`. Defaults to
    `:R_OK`

`intensity` and `residual` describe the pass just flown, not the vehicle.

NOTE: a pass outside every science band carries `intensity = 1` as a "no sample" marker.
The band gate in `rewards.jl` authorizes payment, not the intensity field alone.

NOTE: `visits` and `residual` are observed exactly; only `alt` is noisy. The belief filter
projects onto the known block rather than carrying mass over them
(`policy_reconcentrate_cov!`).
"""
struct SKState{N}
    alt::Symbol
    visits::NTuple{N,Int}
    intensity::Int
    residual::Symbol
end

SKState(alt::Symbol, visits::NTuple{N,Int}) where {N} =
    SKState(alt, visits, 1, :R_OK)
SKState(alt::Symbol, visits::NTuple{N,Int}, intensity::Int) where {N} =
    SKState(alt, visits, intensity, :R_OK)

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

Every visit-count tuple, in odometer order with band 1 varying fastest.

  - `pomdp` — the configuration

Returns `(visit_cap + 1)^n_bands` tuples. The order is stable and defines the index
convention shared by `states`, T, O and the alpha vectors.
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

Enumerate the state space: every live altitude bin x visit tuple x intensity level x
residual bin, then the two terminal states.

  - `pomdp` — the configuration

Returns a Vector{SKState} of length `n_states(pomdp)`. Order is stable and defines the
index convention for T, O and the alpha vectors.
"""
function states(pomdp::StationkeepingPOMDP)
    nb = n_bands(pomdp)
    S = SKState{nb}[]
    # Residual varies fastest, then intensity, inside (alt, visits). Terminal states are
    # appended last and carry intensity 1 / residual :R_OK only — duplicating them across
    # the inner dimensions would add unreachable states and spurious absorbing sinks.
    for a in ALT_BINS, v in visit_tuples(pomdp), i in plume_levels_range(pomdp),
        r in RESIDUAL_BINS
        push!(S, SKState(a, v, i, r))
    end
    zero_v = ntuple(_ -> 0, nb)
    for a in TERMINAL_ALT
        push!(S, SKState(a, zero_v, 1, :R_OK))
    end
    return S
end

"""state_index(pomdp) -> Dict{SKState,Int}. Maps each state to its 1-based index."""
state_index(pomdp::StationkeepingPOMDP) =
    Dict(s => i for (i, s) in enumerate(states(pomdp)))

"""
n_states(pomdp) -> Int.
|S| = 5 * (visit_cap + 1)^n_bands * plume_levels * |RESIDUAL_BINS| + 2.

NOTE: linear in `plume_levels` and `|RESIDUAL_BINS|`, exponential in `n_bands`. The
shipped config (cap 4, 3 bands, k = 3, 3 residual bins) gives |S| = 5627.
"""
n_states(pomdp::StationkeepingPOMDP) =
    length(ALT_BINS) * n_visit_combos(pomdp) * pomdp.plume_levels *
    length(RESIDUAL_BINS) + length(TERMINAL_ALT)

# ── Binning ───────────────────────────────────────────────────────────────────
"""
    alt_bin(pomdp, alt_km) -> Symbol

Bin an achieved periapsis altitude (km) into a live altitude bin, half-open [lo, hi)
against `pomdp.alt_edges`. A non-finite altitude is `:LOST`.

  - `pomdp` — the configuration, supplying `alt_edges`
  - `alt_km` — achieved periapsis altitude (km)

Returns a member of `ALT_BINS`, or `:LOST` for a non-finite altitude.

Bin membership only — no tolerance margin. A margin wide enough to matter on these bins
overlaps both neighbours, so every observation would qualify for two bands and the gate
would dissolve rather than tighten.

NOTE: binning is deterministic given the observation, which is what keeps
`policy_reconcentrate_cov!`'s hard belief projection valid — but it is not correct. A true
32 km pass read as 27 km banks the wrong band. At `sigma_nav_km = 2` this is an edge
effect worth roughly 15-20% of passes, symmetric. Report the science product with that
rate attached.

NOTE: finer bins do not help — σ_nav is set by the sensor, so narrower bins put more
passes near a boundary. The levers are `sigma_nav_km` and the edges themselves.

NOTE: must stay identical to the consumer-side binning in any rollout harness
(`policy_alt_bin`), or the belief filter is fed wrong labels.
"""
function alt_bin(pomdp::StationkeepingPOMDP, alt_km::Real)
    return alt_bin(pomdp.alt_edges, alt_km)
end

"""
    alt_bin(alt_edges, alt_km) -> Symbol

Edges-only form, for callers that have `alt_edges` but no full config — notably
`calibrate_tables`.

  - `alt_edges` — the four bin boundaries (km)
  - `alt_km` — achieved periapsis altitude (km)

Returns a member of `ALT_BINS`, or `:LOST` for a non-finite altitude.
"""
function alt_bin(alt_edges::NTuple{4,<:Real}, alt_km::Real)
    !isfinite(alt_km) && return :LOST
    alt_km < alt_edges[1] && return :BELOW_20
    alt_km < alt_edges[2] && return :A20_27
    alt_km < alt_edges[3] && return :A27_34
    alt_km < alt_edges[4] && return :A34_44
    return :ABOVE_44
end

"""
    band_of_alt(pomdp, alt_km) -> Union{Int,Nothing}

Which science band an altitude falls in.

  - `pomdp` — the configuration
  - `alt_km` — periapsis altitude (km)

Returns a 1-based band index, or `nothing` if the altitude is outside every band.

NOTE: this is the coverage gate. It is fed the OBSERVED altitude, never the true one, and
is indifferent to which action ran — so `CORRECT` banks its band passively just as an
`EXCURSE_*` does.
"""
function band_of_alt(pomdp::StationkeepingPOMDP, alt_km::Real)
    b = alt_bin(pomdp, alt_km)
    idx = findfirst(==(b), pomdp.band_bins)
    return idx === nothing ? nothing : idx
end

"""
    state_label(s) -> String

Serialization label `"ALT|v1-v2-v3|i|RESIDUAL"`.

  - `s` — the state

Returns a String. Used in the exported policy JSON so a consumer can reconstruct
`(alt, visits, intensity, residual)` without knowing the enumeration order.
"""
state_label(s::SKState) = "$(s.alt)|" * join(s.visits, "-") * "|" *
                          string(s.intensity) * "|" * string(s.residual)

"""visit_label(pomdp, visits) -> String. Human-readable coverage, e.g. "LOW×2+MID×1";
"none" when nothing is banked."""
function visit_label(pomdp::StationkeepingPOMDP, visits::NTuple{N,Int}) where {N}
    all(==(0), visits) && return "none"
    join([ "$(pomdp.band_names[b])×$(visits[b])"
           for b in 1:N if visits[b] > 0 ], "+")
end