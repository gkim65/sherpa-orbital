"""
plume.jl — P_θ(intensity | band): the plume altitude gradient.

A pass that lands in an altitude bin yields a sample INTENSITY, not just a tick in a
coverage box. `plume_gradient` (θ) controls how strongly intensity concentrates toward low
altitude, as a linear tilt in depth centered on the mean depth:

    P_θ(ℓ | b)  ∝  1 + λ_b · z_ℓ,      λ_b = clamp( θ · (d_b - d̄), -1, 1 )

with `d_b` the normalized band depth in [0, 1] (1 = lowest-altitude bin), `d̄` the mean
depth over `ALT_BINS`, and `z_ℓ` the centered level score in [-1, 1].

Centering on `d̄` makes θ redistribute science rather than create it: the tilts sum to zero
across bins, so deep bins gain what shallow bins lose and the mean of `E[intensity value]`
over the five bins holds at its θ = 0 value. Returns stay comparable across θ, which the
regret matrix requires.

λ is clamped to [-1, 1] to keep `1 + λ·z` non-negative. A clamped bin is degenerate and
further θ does not move it; with the default `alt_rep_km` the shallowest bin pins at
θ ≈ 1.87 and the usable sweep range is θ ∈ [0, 4]. Past that the two deepest bins pin
together, monotonicity in depth goes non-strict, and the five-bin mean drifts ~1.4%.

`z_ℓ` is centered so θ tilts the distribution without shifting its normalization with band
depth.

NOTE: this is a parameterized hypothesis, not a measured profile. Cassini plume transits
are at 95 km and 49 km, both above these bands, and give no yield-vs-altitude table. The
uncertainty is the point of the sweep; do not cite Cassini as if it pinned θ.

NOTE: θ enters `transition_matrix` analytically, so a sweep over it needs no
recalibration — only `noisy_thruster`/`thruster_kwargs` do. See `needs_recalibration`.

NOTE: intensity is observed, not hidden. It is what the instrument measured, so it enters
the state as an observed dimension and does not complicate the belief.
"""


"""
    plume_level_scores(k) -> Vector{Float64}

The centered level scores `z_ℓ`, evenly spaced over [-1, 1] with mean 0.

  - `k` — number of intensity levels

Returns a Vector{Float64} of length `k`. `k = 1` gives `[0.0]`: a single level carries no
gradient information, so the axis is inert.
"""
function plume_level_scores(k::Int)
    k >= 1 || error("plume_levels must be >= 1, got $k")
    k == 1 && return [0.0]
    return collect(range(-1.0, 1.0; length = k))
end

"""
    plume_band_depth(pomdp, bin::Symbol) -> Float64
    plume_band_depth(pomdp, b::Int) -> Float64

Normalized depth of an altitude bin: 1.0 for the lowest-altitude bin, 0.0 for the highest,
linear in the bin's representative altitude between.

  - `pomdp` — the configuration, supplying `alt_rep_km`
  - `bin` — an altitude bin symbol, or `b` a 1-based science-band index

Returns a depth in [0, 1].

Defined over every live altitude bin, not just the science bands: sampling is passive, so
every pass yields something and every altitude needs its own distribution. Keyed off
`alt_rep_km` rather than `band_target_km`, so depth reflects where a pass is BINNED rather
than what was commanded.
"""
function plume_band_depth(pomdp::StationkeepingPOMDP, bin::Symbol)
    alts = [pomdp.alt_rep_km[b] for b in ALT_BINS]
    lo, hi = minimum(alts), maximum(alts)
    hi ≈ lo && return 1.0
    return (hi - pomdp.alt_rep_km[bin]) / (hi - lo)
end

plume_band_depth(pomdp::StationkeepingPOMDP, b::Int) =
    plume_band_depth(pomdp, pomdp.band_bins[b])

"""
    mean_plume_depth(pomdp) -> Float64

Mean of `plume_band_depth` over `ALT_BINS`: the depth at which the tilt is zero.

  - `pomdp` — the configuration, supplying `alt_rep_km`

Returns a depth in [0, 1]. Bins deeper than this gain intensity as θ rises and shallower
ones lose it. Averaged over every live bin, not just the science bands, so the mean value
held invariant is the one a passive sampler actually sees.
"""
mean_plume_depth(pomdp::StationkeepingPOMDP) =
    sum(plume_band_depth(pomdp, b) for b in ALT_BINS) / length(ALT_BINS)

"""
    plume_intensity_dist(pomdp, b) -> Vector{Float64}

`P_θ(ℓ | bin)` over the `plume_levels` intensity levels, from the linear tilt in this
file's header.

  - `pomdp` — the configuration, supplying `plume_gradient` and `plume_levels`
  - `bin` — an altitude bin symbol, or `b` a 1-based science-band index

Returns a probability vector of length `plume_levels`, summing to 1. At
`plume_gradient = 0` it is uniform for every bin; as θ rises, mass shifts toward high
intensity in bins deeper than `d̄` and toward low intensity in bins shallower than it.
"""
function plume_intensity_dist(pomdp::StationkeepingPOMDP, bin::Symbol)
    k = pomdp.plume_levels
    k == 1 && return [1.0]
    z = plume_level_scores(k)
    d = plume_band_depth(pomdp, bin)
    λ = clamp(pomdp.plume_gradient * (d - mean_plume_depth(pomdp)), -1.0, 1.0)
    w = 1.0 .+ λ .* z
    return w ./ sum(w)
end

"""Band-index form; resolves through `band_bins` to the per-bin method above."""
plume_intensity_dist(pomdp::StationkeepingPOMDP, b::Int) =
    plume_intensity_dist(pomdp, pomdp.band_bins[b])

"""
    plume_intensity_value(pomdp, level) -> Float64

The science value multiplier of a realized intensity level.

  - `pomdp` — the configuration, supplying `plume_levels` and `intensity_value_min`
  - `level` — an INDEX into `1:plume_levels`, 1 = weakest sample

Returns the multiplier: `intensity_value_min` at level 1 rising evenly to 1.0 at level
`k`. At k = 3 the levels map to 0.3 / 0.65 / 1.0. `k = 1` pays 1.0.

NOTE: the argument is a level index, not a value — `plume_intensity_value(pomdp, 1)` is
"the value of the lowest level", not "the value 1".

NOTE: the mean multiplier is `(1 + min)/2`, not 1.0, so raising `plume_levels` with
`r_science` fixed reduces expected science. Only `k = 1` matches the unscaled reward.
"""
function plume_intensity_value(pomdp::StationkeepingPOMDP, level::Int)
    k = pomdp.plume_levels
    k == 1 && return 1.0
    1 <= level <= k || error("intensity level $level outside 1:$k")
    lo = pomdp.intensity_value_min
    return lo + (1.0 - lo) * (level - 1) / (k - 1)
end

"""
    plume_levels_range(pomdp) -> UnitRange{Int}

The support of the intensity state dimension.

  - `pomdp` — the configuration

Returns `1:plume_levels`.
"""
plume_levels_range(pomdp::StationkeepingPOMDP) = 1:pomdp.plume_levels