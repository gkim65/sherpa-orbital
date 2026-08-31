"""
transition.jl — T[s, a, s'] over the factored (alt, visits, intensity, residual) state.

`alt` evolves by the measured kernel for the chosen action. `visits` increments the band
containing the SUCCESSOR altitude bin, but only if the pass did not go terminal: you bank
the science only if you survived it. That coupling is the point of the factored state — it
is what makes the agent sequence excursions rather than attempt them all at once.

⚠️ THE BAND IS DETERMINED BY WHERE THE PASS LANDED, NOT BY THE ACTION (changed
2026-08-30). Previously the excursed band was banked because it had been COMMANDED, so a
band could be credited while the pass missed it by tens of km. Now the successor altitude
bin decides, which is why `CORRECT` banks its own band passively and a missed excursion
banks nothing. Note the successor bin here is the TRUE one, while the runtime gate keys off
the OBSERVED altitude — the observation model carries that gap, not this kernel.
"""

"""
    transition_matrix(pomdp, tables) -> Array{Float64,3}

Build T[s, a, s'] for the enumerated state space. Terminal states self-absorb. The result
is consumed offline by the solver, which never calls the dynamics.
"""
function transition_matrix(pomdp::StationkeepingPOMDP, tables::AltTables)
    S    = states(pomdp)
    A    = actions(pomdp)
    sidx = state_index(pomdp)
    nb   = n_bands(pomdp)
    T    = zeros(Float64, length(S), length(A), length(S))

    zero_v = _zero_visits(pomdp)

    for (si, s) in enumerate(S)
        # Mission-loss states absorb: no action moves you out.
        if isterminal_state(s)
            T[si, :, si] .= 1.0
            continue
        end
        for (ai, a) in enumerate(A)
            # ⚠️ CONDITIONED ON THE RESIDUAL TOO (2026-08-31). The row is selected by
            # BOTH where the vehicle is and how degraded its orbit is, which is the whole
            # point of the dimension: keyed on altitude alone, the row is an average of a
            # fresh departure and a degraded one and reports P(loss) = 0.0 for the
            # transition that actually kills the vehicle.
            k = alt_kernel(tables, a, s.alt, s.residual)
            # The kernel's columns are the JOINT successor (alt, residual).
            for (di, (an, rn)) in enumerate(kernel_columns())
                p = k[di]
                p == 0.0 && continue
                if isterminal_alt(an)
                    T[si, ai, sidx[SKState(an, zero_v, 1, :R_OK)]] += p
                    continue
                end
                # Survived → bank the band this pass landed in, if it landed in one.
                #
                # ⚠️ VISIT COUNTING IS STILL PER SCIENCE BAND (3 slots), but SAMPLING IS NOT
                # (2026-08-31). These were conflated: a pass outside every band used to bank
                # nothing AND record no sample. Coverage is a per-band objective, so the
                # counter stays 3-wide — widening it to all 5 bins would take the visit
                # tuple from 5^3 = 125 combinations to 5^5 = 3125 and |S| from 5627 to
                # 140627, for a coverage question nobody asked.
                b = findfirst(==(an), pomdp.band_bins)
                v = b === nothing ? s.visits :
                    visit_inc(s.visits, b, pomdp.visit_cap)

                # ⚠️ THE GRADIENT ENTERS HERE, AND ONLY HERE, FOR EVERY ALTITUDE. The pass
                # landed in bin `an`, so the intensity it yielded is drawn from
                # P_θ(i | an) — the single point at which `plume_gradient` touches the
                # model. The reward then pays for the REALIZED level (see `rewards.jl`), so
                # a θ that tilts deep bins high genuinely makes them worth more.
                #
                # ⚠️ EVERY BIN DRAWS, INCLUDING THE NON-SCIENCE ONES. Non-band successors
                # used to be pinned to the canonical "no sample" level 1, which — together
                # with `plume_intensity_value(1) = 0.0` — meant a `CORRECT` pass at the
                # ~37 km limit cycle earned exactly zero science no matter how the reward
                # was tuned. Sampling in the mission concept is PASSIVE (you collect by
                # flying THROUGH the plume region), so every pass yields something and every
                # bin needs its own distribution. `plume_band_depth` is defined over all of
                # `ALT_BINS` for exactly this reason.
                #
                # |S| is UNCHANGED: the intensity dimension already existed and was already
                # enumerated for every altitude bin — those states were simply unreachable.
                for (lvl, pl) in enumerate(plume_intensity_dist(pomdp, an))
                    pl == 0.0 && continue
                    T[si, ai, sidx[SKState(an, v, lvl, rn)]] += p * pl
                end
            end
        end
    end
    return T
end