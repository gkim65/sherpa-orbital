"""
transition.jl — T[s, a, s'] over the factored (alt, visits) state.

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
            k = alt_kernel(tables, a, s.alt)
            for (di, an) in enumerate(ALT_ALL)
                p = k[di]
                p == 0.0 && continue
                sp = if isterminal_alt(an)
                    SKState(an, zero_v)
                else
                    # Survived → bank the band this pass actually landed in, if any.
                    b = findfirst(==(an), pomdp.band_bins)
                    SKState(an, b === nothing ? s.visits :
                                visit_inc(s.visits, b, pomdp.visit_cap))
                end
                T[si, ai, sidx[sp]] += p
            end
        end
    end
    return T
end