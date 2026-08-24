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
    println(io, "State = (dev safety bin, science coverage mask)")
    println(io, "  dev edges (km) : ", config.dev_edges, "  -> ", NONTERM_DEV,
                " (+ LOST, CRASHED)")
    println(io, "  bands          : ", config.band_names, "  targets(km)=",
                [config.band_target_km[b] for b in config.band_names])
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

Print the greedy action a*(dev, coverage) for a belief concentrated on each state,
grouped by coverage mask so the science-vs-safety tradeoff reads across the row.

A concentrated belief is a diagnostic, not how the policy runs — but it is the clearest
way to see what the policy would do if it knew exactly where it was.
"""
function print_policy_table(policy, config::StationkeepingPOMDP = StationkeepingPOMDP();
                            io::IO = stdout)
    S    = states(config)
    covs = 0:(n_cov(config) - 1)

    println(io, "\nGreedy policy  a*(dev, coverage)   [belief concentrated on the state]")
    @printf(io, "  %-8s", "dev\\cov")
    for c in covs
        @printf(io, " %-14s", cov_label(config, c))
    end
    println(io)

    for dev in NONTERM_DEV
        @printf(io, "  %-8s", dev)
        for c in covs
            s = SKState(dev, c)
            b = SparseCat(S, [x == s ? 1.0 : 0.0 for x in S])
            @printf(io, " %-14s", action(policy, b))
        end
        println(io)
    end
    return nothing
end