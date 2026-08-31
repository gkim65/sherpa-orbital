"""
plume.jl — P_θ(intensity | band): the plume altitude gradient, as ONE swept scalar.

WHAT THIS IS FOR
A pass that lands in a science band yields a SAMPLE INTENSITY, not just a tick in a
coverage box. `plume_gradient` (θ) controls how strongly intensity concentrates toward LOW
altitude. It is the third sweep axis, shaped identically to `sigma_nav_km` and
`noisy_thruster`: one field on `StationkeepingPOMDP`, no special-casing anywhere in the
build path.

⚠️ IT IS ALSO THE CHEAPEST AXIS TO SWEEP, contrary to the original plan's estimate of ~2
min per θ. The gradient enters `transition_matrix` ANALYTICALLY (the measured kernels are
altitude dynamics; intensity is layered on top), so no recalibration is needed and a solve
is ~2 s. Only `noisy_thruster`/`thruster_kwargs` force a re-measurement — see
`needs_recalibration`.

WHY THIS AXIS IS THE INTERESTING ONE
It turns reward hacking into the result. Under high θ, diving to LOW really IS optimal;
under low θ it is not. A policy solved for one θ and deployed at another pays real regret.
So the policy preferring LOW is not a defect to be tuned away — that preference being
WRONG under a different θ is the phenomenon being measured.

⚠️ THE FUNCTIONAL FORM, AND WHY THIS ONE (chosen 2026-08-30 — it goes in the paper).

    P_θ(ℓ | b)  ∝  exp( θ · d_b · z_ℓ )

with, for k levels indexed ℓ = 1..k (1 = lowest intensity, k = highest):

    d_b = normalized band DEPTH ∈ [0, 1]   (1 = lowest-altitude band, 0 = highest)
    z_ℓ = centered level score ∈ [-1, 1]   (evenly spaced, mean 0)

i.e. a SOFTMAX over level score, whose inverse temperature is `θ · d_b`. Three properties
made this the choice over a linear tilt:

  1. **θ = 0 is exactly uniform, for every band, by construction.** The exponent vanishes,
     so all bands get the identical distribution with no clamping and no special case. The
     sanity gate ("no altitude preference at θ = 0") is satisfied STRUCTURALLY, not by
     tuning — which matters, because a form that only approximately flattens at θ = 0 would
     make the gate a judgement call.
  2. **It is a valid distribution for every θ ≥ 0 with no clipping.** A linear tilt
     `P = 1/k + θ·(...)` goes NEGATIVE for large θ and needs a clamp, and clamping makes θ
     saturate — two different θ values then give the same POMDP, which is exactly what a
     sweep axis must not do.
  3. **θ is a single unbounded scalar with a physical reading** (gradient strength /
     inverse temperature). `theta_grid` can be a plain range.

`z_ℓ` is CENTERED (mean zero) so that θ tilts the distribution without also shifting its
normalization with band depth — an uncentered score would make deep bands uniformly
higher-probability in a way that is a normalization artifact rather than a gradient.

⚠️ HONEST LIMIT ON THE CASSINI CITATION. `aa50429-24_cassini.pdf` (Cassini INMS/CDA plume
transits) gives two flyby altitudes — E7 at 95 km and E21 at 49 km — BOTH ABOVE our 20–46
km bands, and the dominant structure in those profiles is HORIZONTAL (Tiger Stripe
crossings), not an altitude falloff. It therefore supports the qualitative claim "density
rises toward the source" and does NOT provide a yield-vs-altitude table. This softmax is a
PARAMETERIZED HYPOTHESIS swept over θ precisely because the real profile is unknown — that
uncertainty is the point of the study, not a gap in it. Do not cite Cassini as if it
pinned θ.

⚠️ INTENSITY IS OBSERVED, NOT HIDDEN. It is what the instrument measured on the pass, so it
enters the state as an observed dimension and does not complicate the belief the way a
latent plume variable would. See `states.jl` and `observations.jl`.
"""

"""
    plume_level_scores(k) -> Vector{Float64}

The centered level scores `z_ℓ ∈ [-1, 1]`, evenly spaced with mean 0. `k = 1` gives `[0.0]`
(a single level carries no gradient information, so the axis is inert — see
`plume_levels`).
"""
function plume_level_scores(k::Int)
    k >= 1 || error("plume_levels must be >= 1, got $k")
    k == 1 && return [0.0]
    return collect(range(-1.0, 1.0; length = k))
end

"""
    plume_band_depth(pomdp, b) -> Float64

Normalized DEPTH `d_b ∈ [0, 1]` of science band `b`: 1.0 for the lowest-altitude band,
0.0 for the highest, linear in the band's representative altitude in between.

Keyed off `alt_rep_km[band_bins[b]]` rather than `band_target_km`, so depth reflects where
a pass is BINNED (which is what the state records) rather than what was commanded. With a
single band, depth is 1.0 by convention.
"""
function plume_band_depth(pomdp::StationkeepingPOMDP, b::Int)
    alts = [pomdp.alt_rep_km[bin] for bin in pomdp.band_bins]
    lo, hi = minimum(alts), maximum(alts)
    hi ≈ lo && return 1.0
    return (hi - alts[b]) / (hi - lo)
end

"""
    plume_intensity_dist(pomdp, b) -> Vector{Float64}

`P_θ(ℓ | band b)` over the `plume_levels` intensity levels, from the softmax described in
this file's header. Sums to 1.

At `plume_gradient = 0` this is uniform for EVERY band — the θ = 0 sanity gate. As θ rises,
mass shifts toward high intensity in DEEP (low-altitude) bands and toward low intensity in
shallow ones.
"""
function plume_intensity_dist(pomdp::StationkeepingPOMDP, b::Int)
    k = pomdp.plume_levels
    k == 1 && return [1.0]
    z = plume_level_scores(k)
    d = plume_band_depth(pomdp, b)
    # Softmax with inverse temperature θ·d. Subtract the max for numerical stability; it
    # cancels in the normalization.
    e = pomdp.plume_gradient .* d .* z
    w = exp.(e .- maximum(e))
    return w ./ sum(w)
end

"""
    plume_intensity_value(pomdp, level) -> Float64

The science VALUE multiplier of a realized intensity level, in `[0, 1]`: level 1 pays the
least, level `plume_levels` pays the most, evenly spaced. `k = 1` pays 1.0.

⚠️ THIS IS WHAT MAKES THE GRADIENT BITE. The reward pays for the REALIZED intensity, not
for the band label (see `rewards.jl`), so a band whose intensity distribution is tilted
high by θ is genuinely worth more — that is the mechanism by which θ changes the optimal
policy rather than just relabeling states.

⚠️ The MEAN multiplier is 0.5 at k > 1, not 1.0, so raising `plume_levels` with
`r_science` fixed roughly HALVES expected science relative to the pre-intensity model.
Intentional and documented rather than rescaled: the k = 1 model is the one that matches
the old reward scale exactly.
"""
function plume_intensity_value(pomdp::StationkeepingPOMDP, level::Int)
    k = pomdp.plume_levels
    k == 1 && return 1.0
    1 <= level <= k || error("intensity level $level outside 1:$k")
    return (level - 1) / (k - 1)
end

"""
    plume_levels_range(pomdp) -> UnitRange{Int}

`1:plume_levels`. The support of the intensity state dimension.
"""
plume_levels_range(pomdp::StationkeepingPOMDP) = 1:pomdp.plume_levels