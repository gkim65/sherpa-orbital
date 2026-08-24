"""
rewards.jl — r(s, a): the science-vs-safety tradeoff.

Four terms:
  + `r_step_ok`   for surviving a non-terminal step (a small living reward)
  + `r_science`   when an excursion samples a band not already in `cov`
  − fuel          `fuel_weight * action_dv_cost[a]`
  − terminal risk the EXPECTED cost of entering CRASHED/LOST on this step

That last term is why the reward needs T: the mission-loss cost is attached to the
transition, not to occupying the state. Encoding it as its expectation over T[s,a,·] is
exact for expected-discounted-reward planning, and keeps the model inside the
state-action reward interface QuickPOMDPs expects.
"""

"""
    reward_function(pomdp, T) -> (s, a) -> Float64

Build r(s, a) as a closure over the transition matrix. Terminal states earn nothing:
once the mission is lost the episode contributes no further value, and the loss itself
was already charged on the step that entered it (so it is not double-counted).
"""
function reward_function(pomdp::StationkeepingPOMDP, T::Array{Float64,3})
    sidx = state_index(pomdp)
    aidx = action_index(pomdp)
    eb   = excurse_band(pomdp)

    i_crashed = sidx[SKState(:CRASHED, 0)]
    i_lost    = sidx[SKState(:LOST, 0)]

    function reward(s::SKState, a::Symbol)
        isterminal_dev(s.dev) && return 0.0

        r = pomdp.r_step_ok - pomdp.fuel_weight * pomdp.action_dv_cost[a]

        # Science pays out only the FIRST time a band is sampled.
        banded = get(eb, a, 0)
        if banded != 0 && !cov_has(s.cov, banded)
            r += pomdp.r_science
        end

        # Expected mission-loss cost for this (s, a).
        row = @view T[sidx[s], aidx[a], :]
        r += row[i_crashed] * pomdp.r_crashed + row[i_lost] * pomdp.r_lost
        return r
    end
    return reward
end