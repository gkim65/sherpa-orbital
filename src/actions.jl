"""
actions.jl — the action set.

Actions encode intent, not a burn vector: the direction is solved live by the planner
against the onboard model, so the POMDP only chooses what to attempt. A fixed menu of
burn directions was measured and shown to fail.
"""

"""
    excursion_bands(pomdp) -> Vector{Int}

Band indices that get their own EXCURSE action: every band whose bin is not
`pomdp.correct_bin`.

  - `pomdp` — the configuration

Returns 1-based band indices into `pomdp.band_names`.

With the shipped config `correct_bin` is `:A34_44`, which is not a science band, so all
three bands qualify and |A| = 4. A band sharing the CORRECT bin would be excluded: an
EXCURSE there would command what stationkeeping already does, and its coverage is banked
passively anyway because the gate keys off the observed altitude rather than the action
(see [`band_of_alt`](@ref)).
"""
excursion_bands(pomdp::StationkeepingPOMDP) =
    [b for b in 1:n_bands(pomdp) if pomdp.band_bins[b] !== pomdp.correct_bin]

"""
    actions(pomdp) -> Vector{Symbol}

`CORRECT` (burn toward the nominal apses) plus one `EXCURSE_<BAND>` per band that CORRECT
does not already hold. Every action burns.

  - `pomdp` — the configuration

Returns the action symbols, `CORRECT` first, in `band_names` order thereafter.

NOTE: there is no `OBSERVE` action. A no-burn coast is not a decision on this orbit — it
is a step toward losing the vehicle — and it was never information-gathering anyway:
`controller_observe!` runs after every pass, so the nav read and the coverage banking
happen whatever action was chosen. Sampling in the mission concept is passive.

Removing it leaves the partial observability untouched: that lives in the noisy altitude
read ([`observation_matrix`](@ref)), not in the existence of a no-op action.
"""
actions(pomdp::StationkeepingPOMDP) =
    [:CORRECT,
     (Symbol("EXCURSE_", pomdp.band_names[b]) for b in excursion_bands(pomdp))...]

"""n_actions(pomdp) -> Int. Number of actions: `CORRECT` plus one EXCURSE per band."""
n_actions(pomdp::StationkeepingPOMDP) = 1 + length(excursion_bands(pomdp))

"""action_index(pomdp) -> Dict{Symbol,Int}. Action symbol to its 1-based index."""
action_index(pomdp::StationkeepingPOMDP) =
    Dict(a => i for (i, a) in enumerate(actions(pomdp)))

"""
    excurse_band(pomdp) -> Dict{Symbol,Int}

Map each EXCURSE action to its 1-based band index, matching the visit-tuple order.

  - `pomdp` — the configuration

Returns a Dict; non-excursion actions are absent, so use `get(..., a, 0)` for
"not an excursion".
"""
excurse_band(pomdp::StationkeepingPOMDP) =
    Dict(Symbol("EXCURSE_", pomdp.band_names[b]) => b
         for b in excursion_bands(pomdp))

"""is_excursion(pomdp, a) -> Bool. Is action `a` an EXCURSE?"""
is_excursion(pomdp::StationkeepingPOMDP, a::Symbol) = haskey(excurse_band(pomdp), a)