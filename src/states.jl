"""
states.jl — the factored (alt, visits, intensity, residual) state space.

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
const ALT_BINS = (:BELOW_20, :A20_27, :A27_34, :A34_44, :ABOVE_44)

"""Absorbing outcomes. `CRASHED` = below the crash altitude, `LOST` = escaped."""
const TERMINAL_ALT = (:CRASHED, :LOST)

"""Full support of the altitude variable: live bins plus the two terminal outcomes."""
const ALT_ALL = (ALT_BINS..., TERMINAL_ALT...)

"""isterminal_alt(a) -> Bool. Is this an absorbing outcome?"""
isterminal_alt(a::Symbol) = a in TERMINAL_ALT

# ── Residual (orbit-damage) bins ──────────────────────────────────────────────
"""
Residual/damage bins, in ascending order of degradation.

`residual_km` is the onboard `solve_burn` residual on the pass just flown: how badly the
six `:position` (or `:altitude_position`) constraints failed to be simultaneously
satisfiable by one impulse. It is the model's ORBIT-DAMAGE variable.

⚠️ WHY THIS DIMENSION EXISTS (added 2026-08-31). Without it the state is
`(alt, visits, intensity)` and the measured kernel reports `P(loss) = 0.0` for the
transition that actually kills the vehicle — correctly, for a FRESH excursion. The danger
is conditional on accumulated damage, which the old state could not represent, so the
policy could not learn that a LOW excursion needs TWO corrections before the next one.
No amount of extra calibration fixes that (more trials just average "fresh" and "degraded"
into one number), and raising `r_crashed`/`r_lost` does not either — they multiply
P(loss), and anything × 0.0 is 0.0.
"""
const RESIDUAL_BINS = (:R_OK, :R_DEGRADED, :R_CRITICAL)

"""
Residual bin edges (km), half-open [lo, hi).

⚠️ MEASURED, NOT GUESSED (2026-08-31). Derived from 3671 pooled pass-to-pass transitions
across 17 action patterns × 3 seeds × {noise-free, noisy thruster}, scoring each transition
by whether the NEXT pass lost the apse pair (non-finite residual / ΔV = 0, i.e. the
uncontrolled pass that loses the vehicle):

  | residual bin      |    n | n lost | P(lose apse pair next pass) |
  |-------------------|------|--------|-----------------------------|
  | `R_OK`       < 15 | 2363 |      0 |                       0.000 |
  | `R_DEGRADED` 15-25 |  661 |      6 |                       0.009 |
  | `R_CRITICAL` >= 25 |  647 |     40 |                       0.062 |

The load-bearing property is the HARD ZERO on `R_OK`: 0 losses in 2363 transitions. That
is what lets the policy distinguish a safe excursion from a dangerous one, which is
precisely the distinction the un-binned state could not express.

⚠️ THESE ARE NOT THE HANDOFF'S PROPOSED EDGES, and the difference is a real finding. The
prior reading was "~15-25 km = degraded, > ~30 km = one pass from losing the apse pair".
Measurement says 15-25 is only 0.9% hazard and, more pointedly, the `LOW`/`CORRECT`
pattern settles into a STABLE 23/17 km two-cycle that survives the full 30 days. A
residual of 23 km is a sustainable limit cycle, not a near-death state. The genuine
cliff is at 25 km.
"""
const RESIDUAL_EDGES = (15.0, 25.0)

"""
    residual_bin(residual_km) -> Symbol

Bin an onboard `solve_burn` residual (km) into a damage bin, half-open [lo, hi).

A NON-FINITE residual is `:R_CRITICAL`: that is the lost-apse-pair case, where the onboard
model found no apse pair to aim at, `solve_burn` returned ΔV = 0, and the pass flies
UNCONTROLLED. It is the most degraded state the vehicle can be in and still be alive, so it
belongs in the worst live bin rather than in a bin of its own — the LOSS it usually causes
is already carried by the altitude variable's `:LOST` outcome on the following pass.
"""
function residual_bin(residual_km::Real)
    isfinite(residual_km) || return :R_CRITICAL
    residual_km < RESIDUAL_EDGES[1] && return :R_OK
    residual_km < RESIDUAL_EDGES[2] && return :R_DEGRADED
    return :R_CRITICAL
end

"""residual_index(r) -> Int. 1-based position of a residual bin in `RESIDUAL_BINS`."""
residual_index(r::Symbol) = findfirst(==(r), RESIDUAL_BINS)

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
    SKState(alt, visits, intensity = 1)

A stationkeeping state. `alt ∈ ALT_ALL`; `visits` is a per-band visit count, saturating at
the POMDP's `visit_cap`. Terminal states carry all-zero visits: once the orbit is lost,
banked science no longer affects the decision.

`intensity ∈ 1:plume_levels` is the plume sample intensity the LAST pass yielded — an
OBSERVED dimension (it is what the instrument measured), drawn by the transition from
`P_θ(i | band)` and paid for by the reward. See [`plume_intensity_dist`](@ref).

⚠️ INTENSITY IS A PROPERTY OF THE PASS JUST FLOWN, not of the vehicle. A pass that lands
OUTSIDE every science band collects nothing, and carries `intensity = 1` (the lowest
level) as a canonical "no sample" marker rather than a separate value — so the state space
stays a clean product and no reward is paid, because the band gate in `rewards.jl` is what
authorizes payment, not the intensity field alone.

`intensity` defaults to 1 so every pre-intensity construction site (`plume_levels = 1`,
tests, terminal states) keeps working unchanged.

`residual ∈ RESIDUAL_BINS` is the ORBIT-DAMAGE bin of the pass just flown — how badly the
onboard solver's constraints failed, binned by [`residual_bin`](@ref). Like `intensity` it
is a property of the pass just flown, not of the vehicle.

⚠️ RESIDUAL IS OBSERVED EXACTLY, AND THAT IS A DELIBERATE MODELLING CHOICE (2026-08-31).
Altitude is observed through nav noise because a radius has to be MEASURED by a sensor.
The residual does not: it is computed by the ONBOARD `solve_burn` from the onboard CR3BP
model, on quantities the flight software already holds. The spacecraft genuinely knows it
to machine precision, so modelling it as noisy would be inventing uncertainty that does
not exist.

The consequence is that the state now mixes a partially-observed dimension (altitude) with
two exactly-observed ones (visits, residual). That is sound — it is the same structure the
visit counts already have (see `observations.jl`) — but it does mean the residual
dimension adds no INFERENCE problem, only a control-relevant distinction. It is carried
through the observation channel by the same deterministic projection the visit counts use
(`policy_reconcentrate_cov!`).

`residual` defaults to `:R_OK` so every pre-residual construction site keeps working.
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
    # Residual varies FASTEST, then intensity, inside (alt, visits). Terminal states are
    # appended last and carry intensity 1 / residual :R_OK only — a lost orbit measured
    # nothing and has no onboard solve, so duplicating the two absorbing states across the
    # inner dimensions would add unreachable states and spurious self-absorbing sinks.
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

"""state_index(pomdp) -> Dict{SKState,Int}. Inverse of `states`."""
state_index(pomdp::StationkeepingPOMDP) =
    Dict(s => i for (i, s) in enumerate(states(pomdp)))

"""
n_states(pomdp) -> Int.
|S| = 5 * (visit_cap + 1)^n_bands * plume_levels * |RESIDUAL_BINS| + 2.

⚠️ LINEAR in `plume_levels` AND in `|RESIDUAL_BINS|`, EXPONENTIAL in `n_bands`. At cap 4 /
3 bands / k=3: 1877 without the residual dimension, **5627** with it at 3 residual bins.
"""
n_states(pomdp::StationkeepingPOMDP) =
    length(ALT_BINS) * n_visit_combos(pomdp) * pomdp.plume_levels *
    length(RESIDUAL_BINS) + length(TERMINAL_ALT)

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
    return alt_bin(pomdp.alt_edges, alt_km)
end

"""
    alt_bin(alt_edges, alt_km) -> Symbol

Edges-only form, for callers that have `alt_edges` but no full config — notably
`calibrate_tables`, which used to carry its own hand-copied `bin_of` closure.

⚠️ THIS EXISTS SO THERE IS EXACTLY ONE COPY OF THE BINNING RULE. The calibration module
previously re-implemented this inline with a comment reading "must stay bit-identical to
`alt_bin` in states.jl, or the kernels are labelled with bins the model does not use" — two
copies of a rule that must match, which is a drift bug waiting to happen. One
implementation, two entry points.
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

Serialization label `"ALT|v1-v2-v3|i|RESIDUAL"`, used in the exported policy JSON so a
consumer can reconstruct `(alt, visits, intensity, residual)` without knowing the
enumeration order.
"""
state_label(s::SKState) = "$(s.alt)|" * join(s.visits, "-") * "|" *
                          string(s.intensity) * "|" * string(s.residual)

"""visit_label(pomdp, visits) -> String. Human-readable coverage, e.g. "LOW×2+MID×1"."""
function visit_label(pomdp::StationkeepingPOMDP, visits::NTuple{N,Int}) where {N}
    all(==(0), visits) && return "none"
    join([ "$(pomdp.band_names[b])×$(visits[b])"
           for b in 1:N if visits[b] > 0 ], "+")
end