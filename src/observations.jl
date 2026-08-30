"""
observations.jl — O[s, o]: a noisy read of the achieved periapsis ALTITUDE bin.

Nav noise blurs the measured altitude, so the altitude variable is only partially
observed — that is what makes this a POMDP rather than an MDP. The visit counts are
observed exactly, but NOT because they are known a priori: they are a deterministic
function of the observed altitude bin (see [`band_of_alt`](@ref)), so no belief is needed
over them and `policy_reconcentrate_cov!`'s hard projection stays licensed.

⚠️ That determinism buys soundness, not accuracy. Coverage is banked from the OBSERVED
altitude, so a true 32 km pass read as 27 km banks the wrong band. The observation model
below IS the misbin rate — read a row of `O` to see it directly.

⚠️ REPLACES the dev-norm observation, which was direction-blind: it could not distinguish
"6 km high" from "6 km low" from "6 km sideways", so the policy could not act on altitude.

Terminal states are self-announcing: a crash or an escape is unambiguous.
"""

"""
    observations(pomdp) -> Vector{Symbol}

The altitude-bin observation labels: the five live bins plus the two terminal outcomes.
"""
observations(pomdp::StationkeepingPOMDP) = collect(ALT_ALL)

"""n_observations(pomdp) -> Int."""
n_observations(pomdp::StationkeepingPOMDP) = length(ALT_ALL)

"""observation_index(pomdp) -> Dict{Symbol,Int}. Inverse of `observations`."""
observation_index(pomdp::StationkeepingPOMDP) =
    Dict(o => i for (i, o) in enumerate(observations(pomdp)))

"""
    observation_matrix(pomdp) -> Matrix{Float64}

O[s, o]: the probability of reading altitude bin `o` while truly in state `s`.

Computed analytically rather than by Monte Carlo: place a Gaussian of width
`sigma_nav_km` at the bin's representative altitude, then integrate it over each bin's
edges via the Normal CDF. Deterministic, so the exported model is reproducible.

Rows are renormalized because the Gaussian puts mass below 0 km (an altitude cannot be
negative); that leaked mass is redistributed over the real bins. The two terminal columns
get zero mass from live states — a live pass is never mistaken for a crash or an escape.
"""
function observation_matrix(pomdp::StationkeepingPOMDP)
    S     = states(pomdp)
    n_obs = n_observations(pomdp)
    oidx  = observation_index(pomdp)
    O     = zeros(Float64, length(S), n_obs)
    edges = (-Inf, pomdp.alt_edges..., Inf)   # the five live-bin boundaries

    for (si, s) in enumerate(S)
        # A crash or an escape is observed unambiguously.
        if isterminal_alt(s.alt)
            O[si, oidx[s.alt]] = 1.0
            continue
        end

        d = Normal(pomdp.alt_rep_km[s.alt], pomdp.sigma_nav_km)
        for (bi, b) in enumerate(ALT_BINS)
            O[si, oidx[b]] = cdf(d, edges[bi + 1]) - cdf(d, edges[bi])
        end
        O[si, :] ./= sum(O[si, :])
    end
    return O
end