"""
rewards.jl — r(s, a): the science-vs-safety tradeoff.

Four terms:
  + `r_step_ok`   for surviving a non-terminal step
  + `r_science`   expected, for the sample this step collected
  − fuel          `fuel_weight * action_dv_cost[a]`
  − terminal risk expected cost of entering CRASHED/LOST on this step

The science term is
`r_science * visit_factor * value(intensity) * damage_yield(residual)`, where
`visit_factor` is the count increment under the cap and `repeat_factor` once saturated,
and the other two are unit-free multipliers running from a nonzero floor to 1.0.

The science and terminal-risk terms are expectations over T, which is why the reward needs
the transition matrix: the loss cost attaches to the transition rather than to occupying a
state, and science is earned by WHERE THE PASS LANDED rather than by which action was
chosen.

NOTE: every pass collects. Sampling is passive, so every altitude bin draws an intensity
and every pass is paid; only the visit COUNT is per science band.

NOTE: danger is not a reward coefficient. How risky an action is enters through T — the
measured P(LOST) multiplied by `r_lost` — not through the science term. `damage_yield`
prices the quality of a sample taken from a degraded orbit, not the safety penalty.
"""


"""
    reward_function(pomdp, T) -> (s, a) -> Float64

Build r(s, a) as a closure over the transition matrix.

  - `pomdp` — the configuration
  - `T` — the transition matrix from [`transition_matrix`](@ref)

Returns a function `(s::SKState, a::Symbol) -> Float64`.

NOTE: terminal states earn nothing. The loss was already charged on the step that entered
them, so paying again would double-count it.
"""
function reward_function(pomdp::StationkeepingPOMDP, T::Array{Float64,3})
    S    = states(pomdp)
    sidx = state_index(pomdp)
    aidx = action_index(pomdp)
    nb   = n_bands(pomdp)

    zero_v    = _zero_visits(pomdp)
    i_crashed = sidx[SKState(:CRASHED, zero_v, 1, :R_OK)]
    i_lost    = sidx[SKState(:LOST, zero_v, 1, :R_OK)]

    function reward(s::SKState, a::Symbol)
        isterminal_state(s) && return 0.0

        r = pomdp.r_step_ok - pomdp.fuel_weight * pomdp.action_dv_cost[a]

        row = @view T[sidx[s], aidx[a], :]

        # Expected science, paid for the realized intensity the successor records rather
        # than for the band label. Under the cap the count increments and the sample pays
        # in full; at the cap it saturates and the sample pays `repeat_factor`. Without the
        # second branch the policy goes indifferent once every band caps.
        for (spi, sp) in enumerate(S)
            p = row[spi]
            p == 0.0 && continue
            isterminal_state(sp) && continue
            # Two multipliers on the same footing so neither dominates: how strong a
            # sample the pass collected, and how degraded the orbit was when it did.
            # Damage keys on the SUCCESSOR — keying on the departing state would apply the
            # same factor to every action and cancel out of the comparison.
            val = plume_intensity_value(pomdp, sp.intensity) *
                  pomdp.damage_yield[residual_index(sp.residual)]
            gained = visit_total(sp.visits) - visit_total(s.visits)
            if gained > 0
                r += p * pomdp.r_science * gained * val
            else
                # No count increment — either the band saturated or the pass landed
                # outside every science band. Either way a sample was taken, so it pays at
                # the diminishing-returns rate; paying zero here makes stationkeeping
                # worthless in the objective and the policy excurses until it dies.
                #
                # `A34_44` is still not a science band: it never banks a visit and never
                # earns the full `r_science`, so a chosen excursion into an unsaturated
                # band is always worth strictly more.
                r += p * pomdp.r_science * pomdp.repeat_factor * val
            end
        end

        # Expected mission-loss cost for this (s, a).
        r += row[i_crashed] * pomdp.r_crashed + row[i_lost] * pomdp.r_lost
        return r
    end
    return reward
end