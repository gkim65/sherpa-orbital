"""
rewards.jl — r(s, a): the science-vs-safety tradeoff.

Four terms:
  + `r_step_ok`   for surviving a non-terminal step (a small living reward)
  + `r_science`   EXPECTED, for the sample this step collected
  − fuel          `fuel_weight * action_dv_cost[a]`
  − terminal risk the EXPECTED cost of entering CRASHED/LOST on this step

THE SCIENCE TERM, in full (2026-08-31):

    r_science * visit_factor * value(intensity) * damage_yield(residual)

with three independent factors, each answering a different question about the pass:

  | factor              | question                          | source                     |
  |---------------------|-----------------------------------|----------------------------|
  | `visit_factor`      | first visit to this band, or not? | `SKState.visits`           |
  | `value(intensity)`  | how strong was the sample?        | `SKState.intensity`        |
  | `damage_yield`      | how degraded was the orbit?       | `SKState.residual`         |

`visit_factor` is `gained` under the cap and `repeat_factor` once saturated. The other two
are unit-free multipliers on the same 0.3-to-1.0 scale, so one intensity level and one
damage level are comparable in magnitude and neither silently dominates.

⚠️ EVERY PASS COLLECTS. There is no "non-sampling" pass: sampling in the mission concept is
PASSIVE (the spacecraft collects by flying THROUGH the plume region), so every altitude bin
draws an intensity and every pass is paid. Only the VISIT COUNT is per science band. See
`transition.jl` and `plume_band_depth`.

⚠️ DANGER IS NOT A REWARD COEFFICIENT. How risky an action is enters through `T` — the
measured `P(LOST)` multiplied by `r_lost` below — not through the science term.
`damage_yield` prices the QUALITY of a sample taken from a degraded orbit; it is not the
safety penalty.

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
    i_crashed = sidx[SKState(:CRASHED, zero_v, 1, :R_OK)]
    i_lost    = sidx[SKState(:LOST, zero_v, 1, :R_OK)]

    function reward(s::SKState, a::Symbol)
        isterminal_state(s) && return 0.0

        r = pomdp.r_step_ok - pomdp.fuel_weight * pomdp.action_dv_cost[a]

        row = @view T[sidx[s], aidx[a], :]

        # Expected science, paid for the REALIZED intensity the successor records rather
        # than for the band label. Two regimes, and the second is new:
        #   under the cap  — the visit count increments, so `gained > 0` and the sample pays
        #                    `r_science * value(intensity)`.
        #   at the cap     — the count SATURATES, so `gained == 0` even though a real sample
        #                    was taken. Those pay `r_science * repeat_factor * value`.
        # ⚠️ Without the second branch this is a hard cliff: once every band caps, no action
        # earns science at all and the policy goes indifferent. See `repeat_factor`.
        for (spi, sp) in enumerate(S)
            p = row[spi]
            p == 0.0 && continue
            isterminal_state(sp) && continue
            # Two multipliers, on the same 0.3-to-1.0 footing so neither dominates:
            #   value(intensity) — how strong a sample the pass actually collected;
            #   damage_yield     — how degraded the orbit was when it collected it.
            # ⚠️ Damage keys on the SUCCESSOR `sp`, i.e. the damage the action LEAVES you
            # in. Keying on the departing state `s` would apply the same factor to every
            # action and cancel out of the comparison entirely (measured: 0 of 15 greedy
            # actions changed). This way `CORRECT`, which clears the damage, keeps full
            # value while an excursion that deepens it is discounted.
            val = plume_intensity_value(pomdp, sp.intensity) *
                  pomdp.damage_yield[residual_index(sp.residual)]
            gained = visit_total(sp.visits) - visit_total(s.visits)
            if gained > 0
                r += p * pomdp.r_science * gained * val
            else
                # No count increment — either the band saturated, or the pass landed outside
                # every science band. EITHER WAY A SAMPLE WAS TAKEN, so it pays at the
                # diminishing-returns rate.
                #
                # ⚠️ THIS USED TO `continue` FOR A NON-BAND PASS, paying it nothing
                # (2026-08-31). Sampling is PASSIVE — the spacecraft collects by flying
                # THROUGH the plume region, not by commanding an observation — so a
                # `CORRECT` pass at the ~37 km limit cycle does collect. Paying it zero made
                # stationkeeping worthless in the objective even after the residual
                # dimension made the danger visible: measured, `CORRECT` at
                # `A27_34/R_DEGRADED` repaired the orbit with P = 1.000 (n = 2412 at tree
                # depth 9) and still scored 0.5 against an excursion's 10.5, so the policy
                # excursed until it died.
                #
                # The 2026-08-30 edge rebase is PRESERVED: `A34_44` is still not a science
                # band, so it never banks a visit and never earns the full `r_science` — it
                # earns the repeat rate, scaled by the intensity its own depth draws. A
                # chosen excursion into an unsaturated band is still worth strictly more.
                r += p * pomdp.r_science * pomdp.repeat_factor * val
            end
        end

        # Expected mission-loss cost for this (s, a).
        r += row[i_crashed] * pomdp.r_crashed + row[i_lost] * pomdp.r_lost
        return r
    end
    return reward
end