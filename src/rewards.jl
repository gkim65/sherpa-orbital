"""
rewards.jl — r(s, a): the science-vs-safety tradeoff.

Four terms:
  + `r_step_ok`   for surviving a non-terminal step (a small living reward)
  + `r_science`   EXPECTED, for each band sample banked on this step while under the cap
  − fuel          `fuel_weight * action_dv_cost[a]`
  − terminal risk the EXPECTED cost of entering CRASHED/LOST on this step

Two of those terms are expectations over T, which is why the reward needs the transition
matrix. The mission-loss cost attaches to the transition, not to occupying the state; and
science is now earned by WHERE THE PASS LANDED, not by which action was chosen, so it is
only knowable through T as well. Encoding both as expectations over T[s,a,·] is exact for
expected-discounted-reward planning and keeps the model inside the state-action reward
interface QuickPOMDPs expects.

⚠️ CHANGED 2026-08-30, and this is a real semantic change, not a port. Previously science
paid out when an EXCURSE action was COMMANDED and the band was not yet in the `cov`
bitmask — the achieved altitude was never checked, which is why the shipped policy banked
all 3 bands while missing them by −6.8 / −22.2 / −56.3 km. Now a band pays when the
visit count actually increments, which happens only if the OBSERVED altitude landed in
that band's bin. Consequences:
  - `CORRECT` earns science whenever it lands in its band, because the gate is indifferent
    to which action ran.
  - An excursion that misses earns nothing.
  - Repeat visits keep paying until `visit_cap`, so a band is no longer worth zero on the
    second visit (the old `cov` bitmask was monotone, so it paid exactly once).
"""

"""
    reward_function(pomdp, T) -> (s, a) -> Float64

Build r(s, a) as a closure over the transition matrix. Terminal states earn nothing: once
the mission is lost the episode contributes no further value, and the loss itself was
already charged on the step that entered it (so it is not double-counted).
"""
function reward_function(pomdp::StationkeepingPOMDP, T::Array{Float64,3})
    S    = states(pomdp)
    sidx = state_index(pomdp)
    aidx = action_index(pomdp)
    nb   = n_bands(pomdp)

    zero_v    = _zero_visits(pomdp)
    i_crashed = sidx[SKState(:CRASHED, zero_v)]
    i_lost    = sidx[SKState(:LOST, zero_v)]

    function reward(s::SKState, a::Symbol)
        isterminal_state(s) && return 0.0

        r = pomdp.r_step_ok - pomdp.fuel_weight * pomdp.action_dv_cost[a]

        row = @view T[sidx[s], aidx[a], :]

        # Expected science: how many visit counts the successor banks over this state's,
        # weighted by transition probability. Counts saturate at `visit_cap`, so a band
        # at the cap contributes nothing and the sum is bounded.
        for (spi, sp) in enumerate(S)
            p = row[spi]
            p == 0.0 && continue
            isterminal_state(sp) && continue
            gained = visit_total(sp.visits) - visit_total(s.visits)
            gained > 0 && (r += p * pomdp.r_science * gained)
        end

        # Expected mission-loss cost for this (s, a).
        r += row[i_crashed] * pomdp.r_crashed + row[i_lost] * pomdp.r_lost
        return r
    end
    return reward
end