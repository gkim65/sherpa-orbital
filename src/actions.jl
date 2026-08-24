"""
actions.jl — the action set.

Actions encode INTENT, not a burn vector. A fixed menu of burn directions was measured
and shown to fail (experiments/studies exp 04); the direction is solved live by the
planner against the onboard model, so the POMDP only chooses what to attempt.
"""

"""
    actions(pomdp) -> Vector{Symbol}

OBSERVE (coast, nav read only), CORRECT (burn toward the nominal apses), and one
EXCURSE_<BAND> per science band (dip toward that band's target altitude, banking the
band if the pass is survived).
"""
actions(pomdp::StationkeepingPOMDP) =
    [:OBSERVE, :CORRECT, (Symbol("EXCURSE_", b) for b in pomdp.band_names)...]

"""n_actions(pomdp) -> Int."""
n_actions(pomdp::StationkeepingPOMDP) = 2 + n_bands(pomdp)

"""action_index(pomdp) -> Dict{Symbol,Int}. Inverse of `actions`."""
action_index(pomdp::StationkeepingPOMDP) =
    Dict(a => i for (i, a) in enumerate(actions(pomdp)))

"""
    excurse_band(pomdp) -> Dict{Symbol,Int}

Map each EXCURSE action to its 1-based band index (matching the `cov` bit order).
Non-excursion actions are absent; use `get(..., a, 0)` for "not an excursion".
"""
excurse_band(pomdp::StationkeepingPOMDP) =
    Dict(Symbol("EXCURSE_", b) => i for (i, b) in enumerate(pomdp.band_names))

"""is_excursion(pomdp, a) -> Bool."""
is_excursion(pomdp::StationkeepingPOMDP, a::Symbol) = haskey(excurse_band(pomdp), a)