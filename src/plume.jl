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
    plume_band_depth(pomdp, bin::Symbol) -> Float64
    plume_band_depth(pomdp, b::Int) -> Float64

Normalized DEPTH `d ∈ [0, 1]` of an ALTITUDE BIN: 1.0 for the lowest-altitude bin, 0.0 for
the highest, linear in the bin's representative altitude in between.

⚠️ DEFINED OVER EVERY LIVE ALTITUDE BIN, NOT JUST THE SCIENCE BANDS (2026-08-31). It used to
span only `band_bins`, which meant the limit-cycle bin `A34_44` had no depth and therefore
no intensity distribution — so a `CORRECT` pass could not draw a sample at all and was
assigned the canonical "no sample" level. But sampling in the mission concept is PASSIVE:
the spacecraft collects by flying THROUGH the plume region, so EVERY pass yields something
and every altitude needs its own distribution. Spanning all of `ALT_BINS` also makes the
gradient continuous in altitude rather than defined at three isolated points.

Measured against the current `alt_rep_km`:

  | bin      | rep alt | depth |
  |----------|---------|-------|
  | BELOW_20 |    18.0 |  1.00 |
  | A20_27   |    23.5 |  0.80 |
  | A27_34   |    30.5 |  0.55 |
  | A34_44   |    37.2 |  0.31 |   <- the limit cycle: a real value, not a special case
  | ABOVE_44 |    46.0 |  0.00 |

Keyed off `alt_rep_km` rather than `band_target_km`, so depth reflects where a pass is
BINNED (which is what the state records) rather than what was commanded.

The `Int` method takes a SCIENCE BAND index and is retained for callers that think in
bands; it resolves through `band_bins` to the same per-bin scale, so the two agree.
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
    plume_intensity_dist(pomdp, b) -> Vector{Float64}

`P_θ(ℓ | band b)` over the `plume_levels` intensity levels, from the softmax described in
this file's header. Sums to 1.

At `plume_gradient = 0` this is uniform for EVERY band — the θ = 0 sanity gate. As θ rises,
mass shifts toward high intensity in DEEP (low-altitude) bands and toward low intensity in
shallow ones.
"""
function plume_intensity_dist(pomdp::StationkeepingPOMDP, bin::Symbol)
    k = pomdp.plume_levels
    k == 1 && return [1.0]
    z = plume_level_scores(k)
    d = plume_band_depth(pomdp, bin)
    # Softmax with inverse temperature θ·d. Subtract the max for numerical stability; it
    # cancels in the normalization.
    e = pomdp.plume_gradient .* d .* z
    w = exp.(e .- maximum(e))
    return w ./ sum(w)
end

"""Band-index form. Resolves through `band_bins` to the per-bin method above."""
plume_intensity_dist(pomdp::StationkeepingPOMDP, b::Int) =
    plume_intensity_dist(pomdp, pomdp.band_bins[b])

"""
    plume_intensity_value(pomdp, level) -> Float64

The science VALUE multiplier of a realized intensity LEVEL. `level` is an INDEX into
`1:plume_levels` (1 = weakest sample, `plume_levels` = strongest); the returned value is
what that level is worth. `k = 1` pays 1.0.

⚠️ READ THE SIGNATURE CAREFULLY — `plume_intensity_value(pomdp, 1)` is "the value of the
LOWEST level", not "the value 1". At k = 3 the levels map to `0.3, 0.65, 1.0`.

⚠️ **THE LOWEST LEVEL IS NEVER ZERO, AND THAT IS A FIX** (2026-08-31). The scale was
`(level - 1) / (k - 1)`, which put level 1 at exactly 0.0. That zeroed two distinct things,
and neither was intended:

  1. **A weak sample in a real science band paid nothing.** The spacecraft flew through the
     plume, took the weakest reading, and the reward scored it identically to not sampling.
     A weak sample is still a sample.
  2. **Every pass outside a science band paid nothing, unconditionally** — `transition.jl`
     used level 1 as a canonical "no sample" MARKER, so a `CORRECT` pass at the ~37 km limit
     cycle landed on the zero-valued level by construction. No coefficient anywhere else
     could rescue it: anything × 0.0 is 0.0. This is the same trap already documented for
     `r_crashed`/`r_lost`, reappearing in the science term.

Since every altitude bin now draws a genuine intensity (see `plume_band_depth` and
`transition.jl`), level 1 no longer means "nothing happened" — it means the weakest real
measurement — so a zero floor is wrong on its own terms.

The scale is `intensity_value_min` at level 1 rising evenly to 1.0 at level k.

⚠️ THE MEAN MULTIPLIER IS NOT 1.0 (it is `(1 + min)/2` at k > 1), so raising `plume_levels`
with `r_science` fixed still reduces expected science relative to the pre-intensity model.
The k = 1 model is the one that matches the old reward scale exactly.
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

`1:plume_levels`. The support of the intensity state dimension.
"""
plume_levels_range(pomdp::StationkeepingPOMDP) = 1:pomdp.plume_levels