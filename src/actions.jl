"""
actions.jl — the action set.

Actions encode INTENT, not a burn vector. A fixed menu of burn directions was measured
and shown to fail (experiments/studies exp 04); the direction is solved live by the
planner against the onboard model, so the POMDP only chooses what to attempt.
"""

"""
    excursion_bands(pomdp) -> Vector{Int}

Band indices that get their own EXCURSE action: every band EXCEPT the one `CORRECT`
already holds.

⚠️ CORRECT IS THE MID EXCURSION. Nominal periapsis is 30.98 km and the controller's limit
cycle settles ~37 km, both inside the MID band (30–40 km), so an `EXCURSE_MID` would
command what stationkeeping already does — a duplicate action costing policy space and
kernel measurements for nothing. MID coverage is still banked; it is banked passively,
because the coverage gate keys off the OBSERVED altitude and is indifferent to which
action produced it (see [`band_of_alt`](@ref)).
"""
excursion_bands(pomdp::StationkeepingPOMDP) =
    [b for b in 1:n_bands(pomdp) if pomdp.band_bins[b] !== pomdp.correct_bin]

"""
    actions(pomdp) -> Vector{Symbol}

OBSERVE (coast, nav read only), CORRECT (burn toward the nominal apses, which also holds
the MID band), and one EXCURSE_<BAND> per band CORRECT does not already cover.
"""
actions(pomdp::StationkeepingPOMDP) =
    [:OBSERVE, :CORRECT,
     (Symbol("EXCURSE_", pomdp.band_names[b]) for b in excursion_bands(pomdp))...]

"""n_actions(pomdp) -> Int."""
n_actions(pomdp::StationkeepingPOMDP) = 2 + length(excursion_bands(pomdp))

"""action_index(pomdp) -> Dict{Symbol,Int}. Inverse of `actions`."""
action_index(pomdp::StationkeepingPOMDP) =
    Dict(a => i for (i, a) in enumerate(actions(pomdp)))

"""
    excurse_band(pomdp) -> Dict{Symbol,Int}

Map each EXCURSE action to its 1-based band index (matching the visit-tuple order).
Non-excursion actions are absent; use `get(..., a, 0)` for "not an excursion".
"""
excurse_band(pomdp::StationkeepingPOMDP) =
    Dict(Symbol("EXCURSE_", pomdp.band_names[b]) => b
         for b in excursion_bands(pomdp))

"""is_excursion(pomdp, a) -> Bool."""
is_excursion(pomdp::StationkeepingPOMDP, a::Symbol) = haskey(excurse_band(pomdp), a)