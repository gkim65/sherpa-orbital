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

CORRECT (burn toward the nominal apses, which also holds the MID band), and one
EXCURSE_<BAND> per band CORRECT does not already cover. Every action burns.

⚠️ THERE IS NO `OBSERVE` ACTION (removed 2026-08-30). It was a no-burn coast, and on this
orbit that is not a decision — it is a slow loss of the vehicle.

Why it was never an information-gathering action. Nothing about the nav read was ever gated
on it: `controller_observe!` runs after EVERY pass, so the altitude measurement and the
coverage banking happen whatever action was chosen (coverage keys off the OBSERVED altitude
and is indifferent to the action — see [`band_of_alt`](@ref)). Sampling in the mission
concept is likewise passive: the spacecraft collects by FLYING THROUGH the region, not by
commanding a separate observation. So `OBSERVE` bought exactly one thing — not spending
fuel — and paid for it with a pass of uncorrected drift.

Why that trade is not available here. The period-3 halo escapes in ~2 revolutions
uncontrolled, so a coasted pass is a large step toward losing the orbit. Measured
2026-08-30: rolling out the solved policy, the vehicle held fine through three CORRECT
passes and then the policy chose `OBSERVE` twice at 44.54 km — the orbit ran to 108.06 km
and escaped at 3.78 d.

⚠️ AND THE POLICY WAS NOT AT FAULT — the kernel it was given lied to it. Four of the five
measured `OBSERVE` rows came back as 100% SELF-TRANSITIONS (`A40_50` → `A40_50` with
probability 1.0 over 9 trials), i.e. "coasting never changes your altitude bin." That is an
artifact of measuring the counterfactual from freshly-seeded PERIODIC family members, which
really do stay put for one pass, rather than from the drifted states the vehicle actually
occupies. The one row measured from genuinely drifted states (`A30_40`, from the sustained
loop) is the only one with realistic spread. A policy handed a free, risk-free action will
choose it, and it did.

Removing the action rather than re-measuring the row is the user's call (2026-08-30): the
orbit does not appear to permit any coasting, so the honest model is one where every pass
burns. Keeping `OBSERVE` with a corrected kernel is the alternative experiment — it would
test whether the policy LEARNS to avoid coasting rather than being denied the option. The
calibration still measures the OBSERVE counterfactual (`calibrate.jl`) even though no action
consumes it, because "what would have happened if we had not burned" stays the reference
quantity for judging whether a burn was worth its fuel.

This removes the information/fuel tradeoff that nominally motivates a POMDP over an MDP.
The partial observability is UNAFFECTED — it lives in the noisy altitude read
([`observation_matrix`](@ref)), not in the existence of a no-op action.
"""
actions(pomdp::StationkeepingPOMDP) =
    [:CORRECT,
     (Symbol("EXCURSE_", pomdp.band_names[b]) for b in excursion_bands(pomdp))...]

"""n_actions(pomdp) -> Int."""
n_actions(pomdp::StationkeepingPOMDP) = 1 + length(excursion_bands(pomdp))

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