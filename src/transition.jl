"""
transition.jl — T[s, a, s'] over the factored (alt, visits, intensity, residual) state.

`alt` evolves by the measured kernel for the chosen action. `visits` increments the band
containing the successor altitude bin, but only if the pass did not go terminal — you bank
the science only if you survived it, which is what makes the agent sequence excursions
rather than attempt them all at once.

NOTE: the band is decided by where the pass LANDED, not by which action was commanded, so
`CORRECT` banks its own band passively and a missed excursion banks nothing.

NOTE: the successor bin here is the TRUE altitude, while the runtime coverage gate keys off
the OBSERVED one. The observation model carries that gap, not this kernel.
"""

"""
    transition_matrix(pomdp, tables) -> Array{Float64,3}

Build T[s, a, s'] for the enumerated state space.

  - `pomdp` — the configuration
  - `tables` — measured kernels from [`load_tables`](@ref)

Returns a dense `|S| x |A| x |S|` array. Terminal states self-absorb. Consumed offline by
the solver, which never calls the dynamics.
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
            # The row is selected by both where the vehicle is and how degraded its orbit
            # is. Keyed on altitude alone it would average fresh and degraded departures.
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
                # Visit counting is per SCIENCE BAND even though every pass samples.
                # Widening the counter to all five bins would take |S| from 5627 to
                # 140627, for a coverage question nobody asked.
                b = findfirst(==(an), pomdp.band_bins)
                v = b === nothing ? s.visits :
                    visit_inc(s.visits, b, pomdp.visit_cap)

                # The single point at which `plume_gradient` touches the model: the pass
                # landed in bin `an`, so its intensity is drawn from P_θ(i | an) and the
                # reward pays for the realized level.
                #
                # Every bin draws, including the non-science ones — sampling is passive, so
                # every pass yields something. `plume_band_depth` spans all of `ALT_BINS`
                # for this reason.
                for (lvl, pl) in enumerate(plume_intensity_dist(pomdp, an))
                    pl == 0.0 && continue
                    T[si, ai, sidx[SKState(an, v, lvl, rn)]] += p * pl
                end
            end
        end
    end
    return T
end