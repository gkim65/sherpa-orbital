"""
common/report.jl — human-readable views of a model and a solved policy.

Reporting is separated from the model so that inspecting a policy never risks mutating
it, and so the model files stay free of formatting code.
"""

"""
    print_model_summary(config = StationkeepingPOMDP(); io = stdout)

Print the discretization, the action set, and the space sizes. Cheap sanity check that a
config is the scenario you meant.
"""
function print_model_summary(config::StationkeepingPOMDP = StationkeepingPOMDP();
                             io::IO = stdout)
    println(io, "="^72)
    println(io, "Science/safety Enceladus stationkeeping POMDP")
    println(io, "="^72)
    println(io, "State = (achieved periapsis-altitude bin, per-band visit counts, ",
                "plume intensity, orbit-damage bin)")
    println(io, "  alt edges (km) : ", config.alt_edges, "  -> ", ALT_BINS,
                " (+ CRASHED, LOST)")
    println(io, "  residual (km)  : ", RESIDUAL_EDGES, "  -> ", RESIDUAL_BINS,
                "   [orbit damage; exactly observed]")
    println(io, "  bands          : ", config.band_names, "  bins=", config.band_bins,
                "  targets(km)=",
                [config.band_target_km[b] for b in config.band_names])
    println(io, "  visit cap      : ", config.visit_cap,
                "   (CORRECT holds ", config.correct_bin, ", which is not a science band)")
    println(io, "  actions        : ", actions(config))
    println(io, "  nav sigma (km) : ", config.sigma_nav_km)
    @printf(io, "  |S|=%d  |A|=%d  |O|=%d  discount=%.3f\n",
            n_states(config), n_actions(config), n_observations(config), config.discount)
    println(io, "Rewards: science=+", config.r_science, "  step=+", config.r_step_ok,
                "  crash=", config.r_crashed, "  lost=", config.r_lost,
                "  fuel_weight=", config.fuel_weight)
    return nothing
end

"""
    print_policy_table(policy, config = StationkeepingPOMDP(); io = stdout)

Print the greedy action a*(altitude, visits) for a belief concentrated on each state, one
table per orbit-damage bin.

  - `policy` — a solved policy supporting `action(policy, belief)`
  - `config` — the configuration the policy was solved against
  - `io` — output stream
  - `intensity` — which intensity level to tabulate, `1:plume_levels`

The tradeoff reads across a row; the damage response reads down the tables. The pattern to
look for is excursions at `R_OK` giving way to `CORRECT` at `R_DEGRADED`/`R_CRITICAL`.

NOTE: a concentrated belief is a diagnostic, not how the policy runs — it shows what the
policy would do knowing exactly where it was.
"""
function print_policy_table(policy, config::StationkeepingPOMDP = StationkeepingPOMDP();
                            io::IO = stdout, intensity::Int = 1)
    S    = states(config)
    vs   = visit_tuples(config)
    sidx = Dict(s => i for (i, s) in enumerate(S))

    println(io, "\nGreedy policy  a*(alt, visits | residual)   ",
                "[belief concentrated on the state, intensity=", intensity, "]")

    for res in RESIDUAL_BINS
        println(io, "\n  residual = ", res, "  (",
                res === :R_OK       ? "< $(RESIDUAL_EDGES[1]) km, healthy" :
                res === :R_DEGRADED ? "$(RESIDUAL_EDGES[1])-$(RESIDUAL_EDGES[2]) km, degraded" :
                                      ">= $(RESIDUAL_EDGES[2]) km, critical", ")")
        @printf(io, "  %-10s", "alt\\visits")
        for v in vs
            @printf(io, " %-14s", visit_label(config, v))
        end
        println(io)

        for alt in ALT_BINS
            @printf(io, "  %-10s", alt)
            for v in vs
                s = SKState(alt, v, intensity, res)
                bvec = zeros(Float64, length(S))
                bvec[sidx[s]] = 1.0
                b = SparseCat(S, bvec)
                @printf(io, " %-14s", action(policy, b))
            end
            println(io)
        end
    end
    return nothing
end