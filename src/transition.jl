"""
transition.jl — T[s, a, s'] over the factored (dev, cov) state.

`dev` evolves by the measured kernel for the chosen action. `cov` gains the excursed
band only if the pass did NOT go terminal: you bank the science only if you survived it.
That coupling is the entire point of the factored state — it is what makes the agent
sequence excursions rather than attempt them all at once.
"""

"""
    transition_matrix(pomdp, tables) -> Array{Float64,3}

Build T[s, a, s'] for the enumerated state space. Terminal states self-absorb. The result
is consumed offline by the solver, which never calls the dynamics.
"""
function transition_matrix(pomdp::StationkeepingPOMDP, tables::DevTables)
    S    = states(pomdp)
    A    = actions(pomdp)
    sidx = state_index(pomdp)
    eb   = excurse_band(pomdp)
    T    = zeros(Float64, length(S), length(A), length(S))

    for (si, s) in enumerate(S)
        # Mission-loss states absorb: no action moves you out.
        if isterminal_dev(s.dev)
            T[si, :, si] .= 1.0
            continue
        end
        for (ai, a) in enumerate(A)
            k      = dev_kernel(tables, a, s.dev)
            banded = get(eb, a, 0)
            newcov = banded == 0 ? s.cov : cov_set(s.cov, banded)
            for (di, dn) in enumerate(DEV_NEXT)
                p = k[di]
                p == 0.0 && continue
                sp = if dn === :CRASHED
                    SKState(:CRASHED, 0)
                elseif dn === :LOST
                    SKState(:LOST, 0)
                else
                    SKState(dn, newcov)   # survived → bank the band
                end
                T[si, ai, sidx[sp]] += p
            end
        end
    end
    return T
end