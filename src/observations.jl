"""
observations.jl — O[s, o]: a noisy read of the dev bin.

Nav noise blurs the measured apse deviation, so the SAFETY variable is only partially
observed — that is what makes this a POMDP rather than an MDP. The SCIENCE variable
(`cov`) is observed exactly: we know which excursions we commanded, so no belief is
needed over it. The observation space is therefore the dev bins alone.

Terminal states are self-announcing: a crash or an escape is unambiguous.
"""

"""
    observations(pomdp) -> Vector{Symbol}

The dev-bin observation labels. Same alphabet as `DEV_NEXT`.
"""
observations(pomdp::StationkeepingPOMDP) = collect(DEV_NEXT)

"""n_observations(pomdp) -> Int."""
n_observations(pomdp::StationkeepingPOMDP) = length(DEV_NEXT)

"""observation_index(pomdp) -> Dict{Symbol,Int}. Inverse of `observations`."""
observation_index(pomdp::StationkeepingPOMDP) =
    Dict(o => i for (i, o) in enumerate(observations(pomdp)))

"""
    observation_matrix(pomdp) -> Matrix{Float64}

O[s, o]: the probability of reading dev bin `o` while truly in state `s`.

Computed analytically rather than by Monte Carlo: place a Gaussian of width
`sigma_nav_km` at the bin's representative deviation, then integrate it over each bin's
edges via the Normal CDF. Deterministic, so the exported model is reproducible.

Rows are renormalized because the Gaussian puts mass below 0 km (a deviation magnitude
cannot be negative); that leaked mass is redistributed over the real bins.
"""
function observation_matrix(pomdp::StationkeepingPOMDP)
    S     = states(pomdp)
    n_obs = n_observations(pomdp)
    oidx  = observation_index(pomdp)
    O     = zeros(Float64, length(S), n_obs)
    edges = (-Inf, pomdp.dev_edges..., Inf)   # OK / DRIFT / FAR / LOST boundaries

    for (si, s) in enumerate(S)
        # A crash or an escape is observed unambiguously.
        if s.dev === :CRASHED
            O[si, oidx[:CRASHED]] = 1.0
            continue
        elseif s.dev === :LOST
            O[si, oidx[:LOST]] = 1.0
            continue
        end

        d = Normal(pomdp.dev_rep_km[s.dev], pomdp.sigma_nav_km)
        for oi in 1:4                          # OK, DRIFT, FAR, LOST
            O[si, oi] = cdf(d, edges[oi + 1]) - cdf(d, edges[oi])
        end
        O[si, :] ./= sum(O[si, :])
    end
    return O
end