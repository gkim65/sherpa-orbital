"""
calibration/calibrate.jl — MEASURE the dev-transition kernels from the Julia truth model.

WHAT THIS REPLACES
`artifacts/tables.json` has until now held HAND-TRANSCRIBED numbers, copied out of the
Python experiment logs (exp 11b for CORRECT/OBSERVE, exp 12 for EXCURSE) with the DRIFT/FAR
rows filled in by engineering judgement anchored on very few trials. That artifact's own
`meta.caveats` says so. This module measures them from the ported Julia truth model, so the
committed artifact becomes a machine-generated record with a reproducible provenance.

⚠️ THE MEASUREMENT IS THE PRODUCT — read the caveats before trusting a row.

  1. THE ORBIT IS THE KNOWN-DEFECTIVE ONE. The period-3 IC's periapsis sits at +87°
     latitude — the NORTH pole — not the south pole the science case wants (docs/todo.md
     "Step A′"). Every kernel here is conditional on that orbit. A row must NOT be
     presented as characterising the intended south-polar orbit.
  2. THE ORBIT IS VIOLENTLY UNSTABLE — it escapes in ~2 revs uncontrolled. This is why
     there is no longer an OBSERVE action to calibrate: a no-burn pass is not a decision on
     this orbit, it is a step toward losing the vehicle (see `actions`).
  3. EVERY ROW IS REPORTED WITH ITS TRIAL COUNT and a row below `MIN_TRIALS_TRUSTED` is
     flagged, not silently smoothed. A row with n = 0 is FILLED with a self-transition so
     the kernel stays a valid stochastic matrix, and named in `meta.unmeasured_rows`.
  4. `converged` IS THE WRONG SUCCESS TEST FOR AN EXCURSION. `mode = :altitude_position`
     leaves the apoapsis-POSITION block in the residual, which never clears
     `TARGET_TOL_KM` from a drifted state, so `converged` reads false on passes that hit
     the commanded altitude to ~0.05 km. The EXCURSE rows count a trial as
     non-converged on `peri_err_km`, the periapsis-altitude error alone.

METHOD — ONE WALKER, EVERY ACTION (unified 2026-08-31)
  `action_walk!` flies short walks of a SINGLE action, RESTARTED from the seed every
  `EXCURSE_PASSES_PER_RESTART` passes so each restart contributes a fresh departure from
  the origin bin, and carries the residual (orbit damage) along the walk so later passes
  are booked as degraded departures. Every action is measured this way — first from the
  nominal limit cycle, then seeded from a real halo-family member in each altitude bin so
  the bins a working controller never visits are measured rather than invented.

  The only per-action differences are ARGUMENTS: `:position` + nominal apse vectors for
  CORRECT, `:altitude_position` + a commanded altitude for each EXCURSE, and the matching
  success test (`converged` vs `peri_err_km` — see caveat 4). ONE KERNEL PER ACTION.

  ⚠️ CORRECT USED TO HAVE ITS OWN BESPOKE PATH — a sustained loop, written out inline
  twice — and it only ever flew from PRISTINE states. So every `CORRECT` row at
  `R_DEGRADED` / `R_CRITICAL` was n = 0, and the recovery behaviour the residual dimension
  exists to model ("the first CORRECT does not clear the damage, the second does") was not
  measured at all. Routing CORRECT through the shared walker is what fixed that.
            ⚠️ CHANGED 2026-08-31: the walk used to be carried forward across ALL trials,
            which made the trial count a WALK LENGTH. Because an accurate excursion settles
            after 1–2 passes, every later pass was recorded departing the DESTINATION bin
            and the origin rows stayed at n = 1 — see `excurse_walk!`.
            ⚠️ CHANGED 2026-08-30. Every EXCURSE_* used to SHARE one kernel, on exp 12's
            finding that excursion SAFETY did not differ meaningfully by band (only ΔV
            did). That is right for the safety question and wrong for the science one: a
            kernel also encodes WHERE YOU LAND, which is the point of aiming at a band.
            Pooling made EXCURSE_{LOW,MID,HIGH} identical in T — a degenerate action set
            in which nothing steers altitude. See `alt_kernel` in tables.jl.

⚠️ SEEDING IS WHAT MAKES THE ROWS MEAN ANYTHING, AND IT HAS A KNOWN BIAS. A seeded state is
a freshly-placed PERIODIC family member, which is the most stable state in the system; a row
measured only from seeds therefore understates how fast the orbit diverges. That bias is what
made the old OBSERVE rows come back as 100% self-transitions and taught the solved policy
that coasting was free — it chose OBSERVE twice and escaped at 3.78 d (2026-08-30). The
EXCURSE walks mitigate it by flying a few passes per restart rather than one, so the later
passes of each restart condition on drifted states. Prefer a row with sustained-loop trials in
it over one that is purely seeded. (Before 2026-08-31 the walker was never re-seeded at all,
which avoided the bias but produced n = 1 origin rows — a worse failure.)

⚠️ ROWS ARE CONDITIONED ON ORBIT DAMAGE AS OF 2026-08-31. A row's key is a
[`KernelKey`](@ref) — the altitude bin AND the residual bin — and its columns are the JOINT
successor `(alt, residual)`. Keyed on altitude alone, a row averages a FRESH departure with
a departure from an already-degraded orbit, and the average reports P(loss) = 0.0 for the
transition that actually loses the vehicle. That is not a calibration-effort problem (more
trials just average harder) and it cannot be fixed by raising the loss penalties (they
multiply P(loss), and anything × 0.0 is 0.0).

⚠️ AND THE ROW COUNT TRIPLED, ON THE SAME TRIAL BUDGET. Five altitude rows became fifteen
(alt × residual) rows per action. The trials are REDISTRIBUTED, not multiplied, so rows
that previously cleared `MIN_TRIALS_TRUSTED` may not any more — and many (alt, residual)
combinations are simply not reachable (the vehicle cannot be at 18 km with a pristine
residual after a violent transfer). Check `meta.trials` and `meta.unmeasured_rows`; an
unreachable row filled with a self-transition is not evidence about anything.

⚠️ TRUTH/ONBOARD SPLIT. `truth_eom!` is an argument and is integrated only in the coast
helpers; all planning goes through `solve_burn` on the onboard CR3BP.
"""

"""
    CalibrationRow

One measured kernel row: the counts over `ALT_ALL`, the normalized probabilities, and the
diagnostics needed to judge whether the row is trustworthy.
"""
struct CalibrationRow
    counts::Vector{Int}          # over kernel_columns() — the JOINT (alt, residual)
    probs::Vector{Float64}       # counts ./ n, or a fallback if n == 0
    n::Int
    n_nonconverged::Int          # solve_burn failures folded into this row
    dv_ms::Vector{Float64}       # per-trial ΔV, for the action_dv_cost proxy
end

"""Representative altitude (km) for seeding a bin: its midpoint, or a sensible interior
point for the two open-ended outer bins."""
function _bin_rep_alt(bin::Symbol, e::NTuple{4,<:Real})
    # NOT the midpoint (10 km): the continued family floor is 17.565 km, so a 10 km seed
    # finds no member and the row silently stays unmeasured. Seed just inside the family.
    bin === :BELOW_20 && return 18.0
    bin === :A20_27   && return 0.5 * (e[1] + e[2])
    bin === :A27_34   && return 0.5 * (e[2] + e[3])
    bin === :A34_44   && return 0.5 * (e[3] + e[4])
    # ABOVE_44 is open-ended, so pick a seed that is HOLDABLE rather than the geometric
    # continuation. Measured 2026-08-30: a family member holds its own altitude at 45 km but
    # ESCAPES by pass 3 at 55 and 60 km, so seeding at 55 would measure a doomed orbit and
    # report it as this bin's behaviour. 46 km is just inside the band and survivable.
    return 46.0
end

"""
    _bin_sample_alts(bin, e, n) -> Vector{Float64}

`n` seeding altitudes (km) SPREAD ACROSS a bin, rather than one representative point.

⚠️ WHY THE SPREAD EXISTS (2026-08-31). Seeding a bin at a single altitude measures one
canonical orbit and reports it as the behaviour of the whole bin. But a bin is a RANGE: a
20.5 km departure and a 26.5 km departure are both `A20_27` and do not behave the same. The
kernel row is supposed to mean "I am somewhere in this cell", so the measurement should
sample the cell, not a point in it.

Interior points only — the endpoints are deliberately inset by 15% of the bin width so a
sample never sits on a boundary where `alt_bin` could round it into the neighbour. The two
open-ended bins keep their measured, holdable anchors (see [`_bin_rep_alt`](@ref)) and are
spread over a narrow window around them rather than out into the region that escapes.
"""
function _bin_sample_alts(bin::Symbol, e::NTuple{4,<:Real}, n::Integer)
    n <= 1 && return [_bin_rep_alt(bin, e)]
    lo, hi = if bin === :BELOW_20
        # Floor is the continued family's 17.565 km; stay just inside it.
        (17.8, min(19.5, e[1] - 0.5))
    elseif bin === :A20_27
        (e[1], e[2])
    elseif bin === :A27_34
        (e[2], e[3])
    elseif bin === :A34_44
        (e[3], e[4])
    else
        # ABOVE_44 is open-ended but NOT holdable far above ~46 km (55 and 60 km escape by
        # pass 3, measured 2026-08-30), so spread over a narrow survivable window.
        (e[4] + 0.5, 48.0)
    end
    inset = 0.15 * (hi - lo)
    a, b = lo + inset, hi - inset
    return [a + (b - a) * (i - 1) / (n - 1) for i in 1:n]
end

"""
Samples per (altitude bin × action) seeding cell — how many DISPERSED departures to fly.

⚠️ THIS IS WHAT MAKES A ROW A DISTRIBUTION RATHER THAN A REPLICATE. Noise-free, the truth
integrator and the burn solver are both deterministic, so repeating a walk from the SAME
seed reproduces it exactly: a row reading n = 48 was 16 identical restarts of a 3-pass walk,
not 48 independent samples. Dispersing the departure state (position/velocity jitter) and
spreading the seed altitude across the bin are what make the replicates genuinely differ, so
`n` starts to mean sample size rather than repeat count.

⚠️ IT DOES NOT MAKE THE DANGEROUS ROWS SAFER, and should not be expected to. A cell
measured at P(LOST) = 1.00 over many DISPERSED samples is strong evidence; the point of
sampling is that the number means something, not that it softens.
"""
const SEED_SAMPLES_PER_CELL = 3

"""
Depth of the exhaustive action-sequence tree (see `tree_walk!` inside `calibrate_tables`).

Every action sequence of this length is flown from the nominal orbit, so the measurement
covers the CROSS PRODUCT of actions rather than one action at a time. That is what reaches
`(CORRECT, R_DEGRADED)` — a state CORRECT cannot produce for itself.

Cost before pruning is `(|A|^(d+1) - |A|) / (|A| - 1)` passes; at |A| = 4:

  | depth | passes | note                                   |
  |-------|--------|----------------------------------------|
  |     4 |    340 | thin                                   |
  |     5 |   1364 | comparable to the pre-tree calibration  |
  |     6 |   5460 | ~4x                                    |
  |     7 |  21844 | ~16x                                   |

Escaping branches prune their whole subtree, so the flown count runs well under these.
Raising this is a ONE-NUMBER change. There is deliberately no resume/extend path: the tree
is bottom-heavy (each level is ~|A| times the one before), so reusing a shallower tree can
never save more than ~1/|A| of the work — the frontier, which is the bulk, always has to be
flown. Threading is the speedup that actually pays; see `tree_walk!`.
"""
const TREE_DEPTH = 5

"""Minimum trials before a measured row is considered anything but indicative.

⚠️ READ THIS AS SAMPLE SIZE ONLY IF THE SAMPLES DIFFER. Noise-free with an undispersed seed
the environment is deterministic, so `n` counts REPLICATES of one trajectory, not draws from
a distribution — see [`SEED_SAMPLES_PER_CELL`](@ref). Check `meta.effort.seed_samples` before
reading a trial count as statistical confidence.
"""
const MIN_TRIALS_TRUSTED = 20

"""
Passes flown per EXCURSE restart before the walker is returned to its seed.

The trial budget is redistributed, not increased: `n_trials` total passes become
`n_trials / passes_per_restart` fresh DEPARTURES from the origin bin. Small because an
accurate excursion settles in 1-2 passes, so extra passes only re-measure the destination
bin; large enough to see whether the excursion actually survives the pass after arrival.
"""
const EXCURSE_PASSES_PER_RESTART = 3

# ── Coast helpers (the ONLY place truth is integrated) ─────────────────────────
"""
    _coast(truth_eom!, state, horizon, make_primary, arg) -> (:ok|:crash|:escape|:none, u)

Propagate under truth to the primary event, watching for a crash or an escape. Shares the
crash/escape shells with the rollout harness so calibration and rollout cannot disagree
about what "lost" means.
"""
function _coast(truth_eom!, state::AbstractVector{<:Real}, horizon::Real,
                make_primary, arg)
    r = _coast_to(truth_eom!, state, horizon, make_primary, arg;
                  rtol = RTOL_TRUTH, atol = ATOL_TRUTH)
    return r.outcome, r.u
end

_to_shell(eom!, s, h) = _coast(eom!, s, h, _terminal_shell_callback, CONTROL_ALT_KM)
_to_peri(eom!, s, h)  = _coast(eom!, s, h, _terminal_periapsis_callback_arg, nothing)

"""Map a terminal coast outcome onto its `ALT_ALL` label."""
_terminal_dev(outcome::Symbol) = outcome === :crash ? :CRASHED : :LOST

# ── Why the EXCURSE target is an ALTITUDE, not a position vector ───────────────
# Superseded 2026-08-30 by `mode = :altitude_position`; the helper this replaces
# (`_excurse_targets`, a radially-scaled waypoint) is gone. The measurements that forced
# the change are worth keeping:
#
# Aiming ONE impulse at a real family member's absolute apse POSITIONS from a drifted state
# fails on PHASE, not altitude. `next_apse_positions(member.ic)` returns where THAT member's
# periapsis sits at ITS OWN epoch. From the 31 km limit cycle vs the 30 km member, periapsis
# ALTITUDES differ ~22 km but periapsis POSITIONS are 585 km apart and apoapsis positions
# 2319 km apart, so the `:position` residual at ΔV = 0 is 2392 km:
#
#   | commanded | family-member apses | scaled waypoint     |
#   |-----------|---------------------|---------------------|
#   | 20 km     | 177.6 m/s -> ESCAPE | 2.39 m/s -> 37.9 km |
#   | 40 km     | 179.8 m/s -> 395 km | 1.51 m/s -> 42.8 km |
#   | 50 km     |  73.0 m/s -> 562 km | 1.23 m/s -> 45.3 km |
#
# Both columns are bad: the "real orbit" throws the vehicle out, and the waypoint only moves
# achieved periapsis 37.9 -> 45.3 km across a 30 km command range (~25% authority).
# `:altitude_position` constrains the periapsis ALTITUDE directly and delivers the command to
# ~0.2 km, which is what made these rows measurable at all.

"""
    CALIBRATION_EFFORT

The MEASUREMENT-EFFORT defaults, in one named place rather than scattered as inline
literals across the signature.

⚠️ EFFORT IS NOT θ. These control how HARD the environment is sampled — trial counts, loop
lengths, the RNG seed. The same environment measured with 8 or 24 trials is the SAME θ
measured with different confidence, so these are deliberately NOT fields on
`StationkeepingPOMDP` (which holds the environment) and are NOT part of a sweep grid. They
are recorded into `meta.effort` so an artifact says how hard it was measured, which is the
only way to tell a thin row that is thin BY CONFIGURATION from one thin because the walk
settled.

  - `n_steps`             — CORRECT passes to fly from the nominal limit cycle.
                            ⚠️ Was the length of ONE sustained loop; since the 2026-08-31
                            walker unification it is a total pass budget spread over
                            restarts, exactly like `excurse_trials`.
  - `horizon_s`           — ⚠️ DEAD as of 2026-08-31. It capped the wall-clock of the
                            bespoke sustained CORRECT loop, which no longer exists: every
                            action now goes through the restarting `action_walk!` or the
                            action tree, both bounded by pass COUNT rather than elapsed
                            time. Still accepted and recorded into `meta.effort` so older
                            drivers keep working and older artifacts stay comparable, but
                            changing it has NO effect on the measurement.
  - `excurse_trials`      — steps per primary excursion walk.
  - `seed_trials`         — CORRECT trials per seeded altitude bin.
  - `excurse_seed_trials` — EXCURSE trials per (bin × band) seed.
  - `seed_bins`           — seed the bins a working controller never visits at all.
  - `seed_samples`        — how many ALTITUDES to sample across each bin when seeding. The
                            cell's trial budget is split across them, so this trades
                            replicates for genuine spread rather than costing more.
  - `rng_seed`            — reproducibility seed. Only consumed when `noisy_thruster` is
                            on; noise-free the measurement is fully deterministic.

⚠️ `excurse_trials`/`excurse_seed_trials` were RAISED 8 → 24 on 2026-08-30 when the EXCURSE
kernel became per-action (splitting one pooled row three ways divides the trials). That did
NOT fix the thin ORIGIN rows, and more trials cannot: an accurate walk SETTLES into its
destination bin after 1–2 passes, so a longer walk feeds the destination row (n = 139) while
the origin row it departed keeps n = 1. Fixing that needs more RESTARTS, not longer walks —
deferred, see docs/todo.md.
"""
const CALIBRATION_EFFORT = (
    n_steps             = 120,
    horizon_s           = 25 * 86400.0,
    excurse_trials      = 24,
    seed_trials         = 8,
    excurse_seed_trials = 48,
    seed_bins           = true,
    # Seed altitudes sampled across each bin. The per-cell trial budget is DIVIDED across
    # these, not multiplied — see `SEED_SAMPLES_PER_CELL`.
    seed_samples        = SEED_SAMPLES_PER_CELL,
    # Exhaustive action-tree depth — the main coverage mechanism. See `TREE_DEPTH`.
    tree_depth          = TREE_DEPTH,
    rng_seed            = 0,
)

"""
    needs_recalibration(field::Symbol) -> Bool
    needs_recalibration(θ::NamedTuple)  -> Bool

Does changing this θ field require re-measuring the transition kernels?

⚠️ THIS DISTINCTION IS WORTH ~7 MINUTES PER θ. Some environment parameters enter T/O/R
ANALYTICALLY and cost nothing to sweep; others change the orbital dynamics and force a full
`calibrate_tables` run. Measured 2026-08-31:

  | θ field            | how it enters            | cost per θ |
  |--------------------|--------------------------|------------|
  | `sigma_nav_km`     | `observation_matrix`     | ~2 s       |
  | `plume_gradient`   | `transition_matrix`      | ~2 s       |
  | `noisy_thruster`   | RE-MEASURED kernels      | ~7 min     |
  | `thruster_kwargs`  | RE-MEASURED kernels      | ~7 min     |

So a 3-value nav sweep is seconds and a 3-value thruster sweep is ~20 minutes. A sweep
driver that does not know the difference either wastes 20 minutes or, worse, reuses stale
kernels for a dynamics-changing θ.
"""
needs_recalibration(field::Symbol) =
    field in (:noisy_thruster, :thruster_kwargs)
needs_recalibration(θ::NamedTuple) = any(needs_recalibration, keys(θ))

"""
    calibrate_tables(config::StationkeepingPOMDP; effort..., truth_eom!, verbose)
        -> (rows, diagnostics)

Measure the transition kernels for the environment described by `config`.

⚠️ THE CONFIG IS THE SINGLE OWNER OF θ (2026-08-31). This method reads `alt_edges`,
`band_names`, `band_target_km`, `noisy_thruster` and `thruster_kwargs` FROM the config.
Before this existed, `calibrate_tables` had its own keyword defaults for the same
quantities and every driver hand-mapped the config across — with defaults that silently
DISAGREED (`alt_edges` was `(20,30,40,50)` here vs `(20,27,34,44)` in the struct, and
`band_target_km` `25/35/45` vs `23.5/30.5/46`). Calling it without the hand-mapping
measured kernels keyed to bins the model does not use, and nothing errored.

Remaining keywords are MEASUREMENT EFFORT (see [`CALIBRATION_EFFORT`](@ref)), not
environment, plus `truth_eom!`.

    cfg  = StationkeepingPOMDP(; plume_gradient = 4.0)
    rows, diag = calibrate_tables(cfg; verbose = true)
    write_tables(tables_from_rows(rows, diag); path = theta_path("tables", (plume_gradient = 4.0,)))
"""
function calibrate_tables(config::StationkeepingPOMDP;
                          truth_eom! = cr3bp_j2_eom!,
                          truth_name::AbstractString = "CR3BP + Enceladus J2",
                          n_steps::Integer = CALIBRATION_EFFORT.n_steps,
                          horizon_s::Real = CALIBRATION_EFFORT.horizon_s,
                          excurse_trials::Integer = CALIBRATION_EFFORT.excurse_trials,
                          seed_trials::Integer = CALIBRATION_EFFORT.seed_trials,
                          excurse_seed_trials::Integer =
                              CALIBRATION_EFFORT.excurse_seed_trials,
                          seed_samples::Integer = CALIBRATION_EFFORT.seed_samples,
                          tree_depth::Integer = CALIBRATION_EFFORT.tree_depth,
                          seed_bins::Bool = CALIBRATION_EFFORT.seed_bins,
                          rng::AbstractRNG = Xoshiro(CALIBRATION_EFFORT.rng_seed),
                          kwargs...)
    return calibrate_tables(;
        # ── θ, read from the config: ONE owner ────────────────────────────────
        alt_edges       = config.alt_edges,
        band_names      = config.band_names,
        band_target_km  = config.band_target_km,
        noisy_thruster  = config.noisy_thruster,
        thruster_kwargs = config.thruster_kwargs,
        # ── effort + truth model ──────────────────────────────────────────────
        truth_eom!          = truth_eom!,
        truth_name          = truth_name,
        n_steps             = n_steps,
        horizon_s           = horizon_s,
        excurse_trials      = excurse_trials,
        seed_trials         = seed_trials,
        excurse_seed_trials = excurse_seed_trials,
        seed_samples        = seed_samples,
        tree_depth          = tree_depth,
        seed_bins           = seed_bins,
        rng                 = rng,
        rng_seed            = CALIBRATION_EFFORT.rng_seed,
        kwargs...)
end

"""
    calibrate_tables(; truth_eom!, n_steps, horizon_s, alt_edges, band_target_km,
                     mode, verbose) -> (rows, diagnostics)

Low-level form. Measure all kernels and return them with a per-row diagnostics dict.

⚠️ PREFER THE `calibrate_tables(config)` METHOD. This one's `alt_edges` /
`band_target_km` defaults are LEGACY and do not match `StationkeepingPOMDP`'s — calling it
bare measures kernels for bins the model does not use. It stays public only so a
measurement can be run against hand-specified bins without inventing a whole config.

  - `truth_eom!` — the truth model to calibrate against. Defaults to
    [`cr3bp_j2_eom!`](@ref) (CR3BP + Enceladus J2), which is the rung the committed
    hand-transcribed tables were measured on, so the diff is apples-to-apples.
  - `n_steps` — sustained-loop steps to log for the CORRECT rows.
  - `mode` — targeting mode for `solve_burn`. `:position` is Strategy 3 proper and matches
    how the kernels are defined; see caveat 4 above about its convergence.
    ⚠️ Applies to the CORRECT burns only — `excurse_walk!` always uses
    `:altitude_position`, which is why band delivery is ~0.2 km while `meta.mode` reads
    `:position`.
"""
function calibrate_tables(;
    truth_eom! = cr3bp_j2_eom!,
    truth_name::AbstractString = "CR3BP + Enceladus J2",
    # ── θ: THE ENVIRONMENT PARAMETERS ────────────────────────────────────────
    # These are the axes a POMDP FAMILY is swept along (see the module header). The kernels
    # this function returns ARE `T_θ`, so every θ needs its own `calibrate_tables` call.
    #
    # `noisy_thruster` closes a real gap: until 2026-08-30 every burn here was applied
    # PERFECTLY while the rollout harness routed ΔV through `apply_dv_noisy`, so the kernels
    # described a perfect-thruster world and understated excursion risk badly. Measured over
    # 12 seeds, holding a commanded band for 8 passes:
    #
    #   | band | kernel P(LOST), noise-free | actual, noisy thruster |
    #   |------|---------------------------|------------------------|
    #   | LOW  |                      0.19 |         1.00  (12/12)  |
    #   | MID  |                     0.056 |         1.00  (12/12)  |
    #   | HIGH |                      0.00 |         0.00  ( 0/12)  |
    #
    # A policy solved against the noise-free kernels therefore excurses freely and dies:
    # 100% survival noise-free, 0/10 noisy. Set `noisy_thruster = true` to measure the
    # environment the vehicle actually flies.
    #
    # ⚠️ NOISE-FREE IS STILL A LEGITIMATE θ, not a bug — it is the optimistic corner of the
    # family and the reference the regret of every other θ is measured against. Keep it
    # available; just do not mistake it for the deployment environment.
    noisy_thruster::Bool = false,
    rng::AbstractRNG = Xoshiro(0),
    thruster_kwargs::NamedTuple = NamedTuple(),
    # ⚠️ SOUTH POLAR, ALWAYS (2026-08-31). The plumes are at Enceladus's SOUTH pole, so the
    # science case only makes sense with periapsis over that hemisphere — and the plume
    # gradient θ is meaningless on a north-polar orbit. This default used to be
    # `PERIOD1_NORTH_IC_ND`, which repeatedly read as "we are measuring the wrong orbit".
    #
    # It was not actually wrong, and the reason is worth keeping: `PERIOD1_SOUTH_IC_ND =
    # mirror_z(PERIOD1_NORTH_IC_ND)` is an EXACT z-reflection, both primaries are
    # z-symmetric under CR3BP+J2, and every kernel here is keyed on periapsis ALTITUDE —
    # which is reflection-invariant. So the two ICs give identical kernels (see halo_ic.jl:
    # "every controller result measured on the north-polar IC transfers to this one
    # unchanged"), and the seeded walks already used the south family table.
    #
    # Defaulting to south anyway, permanently, so the config STATES the science case instead
    # of relying on a reader knowing the symmetry argument. Do not change this back.
    ic::AbstractVector{<:Real} = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND)),
    period_s::Real = PERIOD1_TRIPLE_PERIOD_S,
    n_steps::Integer = 120,
    # ⚠️ DEAD — see CALIBRATION_EFFORT. Recorded into meta, never read by the measurement.
    horizon_s::Real = 25 * 86400.0,
    alt_edges::NTuple{4,Float64} = (20.0, 30.0, 40.0, 50.0),
    band_names::NTuple{3,Symbol} = (:LOW, :MID, :HIGH),
    band_target_km::Dict{Symbol,Float64} =
        Dict(:LOW => 25.0, :MID => 35.0, :HIGH => 45.0),
    mode::Symbol = :position,
    excurse_trials::Integer = 8,
    family_table::Union{Nothing,Vector{NamedTuple}} = halo_family_table_cached(),
    seed_bins::Bool = true,
    # ⚠️ SET SO EVERY ROW CLEARS `MIN_TRIALS_TRUSTED`, not by feel. A row measured from 3
    # trials reads as a probability distribution but cannot distinguish a real one-third from
    # a coin flip that landed that way, and the solver will exploit the difference as if it
    # were signal. 8 CORRECT trials per bin plus the sustained loop's own visits, and
    # 8 EXCURSE trials per (bin × band), is what puts all five rows above 20.
    seed_trials::Integer = 8,
    # EXCURSE seeding runs per (bin × band), so it multiplies out faster than the
    # CORRECT seeding and gets its own knob rather than sharing one.
    excurse_seed_trials::Integer = 8,
    # Seed altitudes sampled across each bin (see `_bin_sample_alts`). The cell's trial
    # budget is DIVIDED across them, so this buys spread rather than costing more.
    seed_samples::Integer = SEED_SAMPLES_PER_CELL,
    # Base seed, retained for reproducibility of any stochastic path (`noisy_thruster`).
    rng_seed::Integer = 0,
    # ── The action tree ───────────────────────────────────────────────────────
    # Depth of the exhaustive action-sequence enumeration (see `tree_walk!`). 0 disables it.
    # Cost is (|A|^(d+1) - |A|)/(|A| - 1) passes before pruning; at |A| = 4 that is 1364 at
    # depth 5, 5460 at depth 6, 21844 at depth 7. Escapes prune whole subtrees, so the real
    # count is well below those.
    tree_depth::Integer = TREE_DEPTH,
    verbose::Bool = false,
)
    ic = collect(float.(ic))
    period_s = float(period_s)
    one_rev_s = period_s / 3          # 3 periapsis passes per period-3 orbit

    # Count-based apse search: `one_rev_s = T/3` is the inter-periapsis CONTROL cadence, not
    # an apse-to-apse revolution — as a search window it lands 0.136 s short of the first
    # apoapsis and yields a NaN target, which silently zeroes every CORRECT burn.
    r_peri_nom, r_apo_nom = next_apse_positions(ic; eom! = cr3bp_eom!)
    apo_nom_alt = norm(_enc_relative(r_apo_nom)) - R_ENCELADUS

    # Bin by achieved periapsis ALTITUDE. ⚠️ This must stay bit-identical to `alt_bin`
    # in states.jl, or the kernels are labelled with bins the model does not use.
    # ⚠️ ONE binning implementation, shared with the model. This used to be a hand-copied
    # inline closure carrying a comment that it "must stay bit-identical to `alt_bin` in
    # states.jl" — so it now simply IS `alt_bin`. See states.jl for the edges-only method.
    bin_of(h) = alt_bin(alt_edges, h)
    alt_of(u) = norm(_enc_relative(u[1:3])) - R_ENCELADUS

    # Apply a commanded ΔV the way the ENVIRONMENT does. Routed through the same
    # `apply_dv`/`apply_dv_noisy` pair the rollout harness uses, so a kernel measured here
    # and a rollout flown there cannot disagree about what the thruster does — which they
    # silently did until 2026-08-30 (calibration perfect, rollout noisy).
    exec_dv(dv) = noisy_thruster ? apply_dv_noisy(dv, rng; thruster_kwargs...)[1] :
                                   apply_dv(dv)

    # ⚠️ COLUMNS ARE THE JOINT (alt, residual) SUCCESSOR as of 2026-08-31, not the marginal
    # altitude. A row now answers "given I am in this bin with this much orbit damage and
    # take this action, where do I land AND how damaged am I then?" — the second half is
    # what lets the policy learn that the first CORRECT after an excursion does not clear
    # the damage and only the second one does.
    cols = kernel_columns()
    # ⚠️ KEYED PER BAND AS OF 2026-08-30. This replaces ONE pooled EXCURSE kernel.
    #
    # The old keys were just `(:CORRECT, :EXCURSE)`, on the exp-12 finding that excursion
    # RISK did not differ by band — only ΔV did. True for the SAFETY question, and wrong for
    # the SCIENCE question, because a kernel also encodes WHERE YOU LAND, which is the whole
    # point of aiming at a band. The file's own note said to re-check it after the band
    # rebase; this is that re-check.
    #
    # The measured consequence, from the artifact this replaces: the `A34_44` EXCURSE row
    # (from the limit cycle) read 1/3 A20_27, 1/3 A27_34, 1/3 ABOVE_44 over n = 3. That is
    # not "an excursion lands randomly" — it is the LOW walk landing in LOW, the MID walk in
    # MID and the HIGH walk in HIGH, one trial each, POOLED. Aiming was measured, then
    # averaged away.
    #
    # Why it stayed invisible: `action_dv_cost` gave the three actions distinct prices, so
    # the policy looked like it was choosing among three excursions when it was really
    # choosing among three prices. At `fuel_weight = 0` the reward spread across
    # EXCURSE_{LOW,MID,HIGH} is exactly 0.0 and the action set is degenerate — no action
    # raises P(landing in a deep band), so a plume-gradient θ cannot change behaviour.
    #
    # ⚠️ TRIAL COUNTS ARE DIVIDED, NOT MULTIPLIED. Splitting one pooled row into three gives
    # each new row ~1/3 of the trials, so rows that previously cleared `MIN_TRIALS_TRUSTED`
    # may no longer. Raise `excurse_trials` / `excurse_seed_trials` to compensate and CHECK
    # `meta.trials` — a thin row is exactly what this file exists to make auditable.
    action_keys = (:CORRECT, (Symbol("EXCURSE_", b) for b in band_names)...)
    # counts[action][KernelKey(alt_from, residual_from)] -> Vector{Int} over kernel_columns()
    all_keys = kernel_keys_all()
    counts = Dict(a => Dict(k => zeros(Int, N_KERNEL_COLS) for k in all_keys)
                  for a in action_keys)
    nonconv = Dict(a => Dict(k => 0 for k in all_keys)
                   for a in action_keys)
    dvs = Dict(a => Dict(k => Float64[] for k in all_keys)
               for a in action_keys)
    # ΔV per band, for the action_dv_cost proxy.
    band_dv = Dict(b => Float64[] for b in band_names)
    band_achieved_alt = Dict(b => Float64[] for b in band_names)

    """
    Record one measured transition.

    `from` is the FULL conditioning (altitude bin AND residual bin) the pass departed
    under; `to_alt` / `to_res` are the joint successor. A terminal successor carries
    `:R_OK` by convention — a lost orbit ran no onboard solve, so it has no residual (see
    `kernel_columns`).
    """
    bump!(a, from::KernelKey, to_alt::Symbol, to_res::Symbol = :R_OK) =
        counts[a][from][kernel_entry_index(to_alt, to_res)] += 1

    # ── ONE WALKER FOR EVERY ACTION ────────────────────────────────────────────
    # ⚠️ UNIFIED 2026-08-31. CORRECT used to be measured by a bespoke sustained loop
    # (written out inline TWICE — once here, once in the bin-seeding block) while the
    # EXCURSE actions went through a factored `excurse_walk!`. The two paths had drifted
    # into different restart policies, different seeding loops and different trial knobs.
    #
    # THAT ASYMMETRY CAUSED A SUBSTANTIVE BUG, not just duplication. `excurse_walk!` WALKS,
    # so damage accumulates across its passes and it naturally produced degraded-departure
    # rows. The CORRECT paths only ever flew from pristine states — a healthy limit cycle,
    # or a freshly-placed family member — so EVERY `CORRECT` row at `R_DEGRADED` /
    # `R_CRITICAL` came back n = 0. That is precisely the recovery behaviour the residual
    # dimension exists to model ("the first CORRECT does not clear the damage, the second
    # does"), and it was the one thing the measurement could not see. The file already
    # warned that "a hand-copied second loop is how the two would silently diverge"; they
    # diverged.
    #
    # Now every action — CORRECT included — goes through `action_walk!`. The legitimate
    # per-action differences survive as ARGUMENTS (targeting mode, aim point, success
    # test), not as separate implementations.
    st0, u0 = _to_peri(truth_eom!, ic, 4 * one_rev_s)
    base = st0 === :ok ? copy(u0) : copy(ic)

    """
        action_walk!(akey, start_state, n_trials; kwargs...)
        action_walk!(pattern, start_state, n_trials; kwargs...)

    Fly `n_trials` passes, restarting from `start_state` every `passes_per_restart`
    passes, and log each pass into the kernel of whichever action flew it.

    `akey` may be a single action (every pass flies it) or a PATTERN — a vector of actions
    cycled over the walk. The pattern form is what measures an action departing from a
    damage level that only ANOTHER action can produce; see `RECOVERY_PATTERNS`.

    The residual is carried ALONG the walk and reset on restart: it is the state of the
    ORBIT, not of the pass, so a transition is booked under the damage the PREVIOUS pass
    left behind. A fresh departure from a seed is undamaged by construction; pass 2 or 3
    of a walk departs under whatever the earlier passes accumulated. That conditioning is
    the whole reason the walk restarts rather than running once and long.

      - `mode_` / `peri_tgt` / `ra` / `rp` — the targeting. `:position` aims at nominal
        apse POSITION vectors (`rp`, `ra`) and is what CORRECT means; `:altitude_position`
        commands the periapsis ALTITUDE (`peri_tgt`) and keeps the apoapsis position
        constraints, which is what an EXCURSE means.
      - `band` — the science band an EXCURSE aims at, or `nothing` for CORRECT. Only used
        to accumulate the per-band ΔV / achieved-altitude diagnostics.
    """
    function action_walk!(akey::Symbol, start_state, n_trials;
                          mode_::Symbol, peri_tgt = nothing,
                          ra = nothing, rp = nothing,
                          band = nothing,
                          passes_per_restart::Integer = EXCURSE_PASSES_PER_RESTART)

        # ⚠️ RESTARTS, NOT ONE LONG WALK (fixed 2026-08-31). `walker` used to be initialised
        # ONCE outside this loop and carried forward, so `n_trials` was a WALK LENGTH, not a
        # trial count. An accurate excursion SETTLES — it reaches the commanded band on pass
        # 1–2 and stays — so every later pass was recorded departing the DESTINATION bin and
        # the origin row kept n = 1. That is why raising the trial counts 8 → 24 on
        # 2026-08-30 changed nothing for the rows the policy leans on hardest
        # (`EXCURSE_*` from the limit-cycle bin: "I am holding station, what if I dive?"),
        # which read as 100% success off a single sample — the same shape as the thin-kernel
        # bug that killed the OBSERVE action.
        #
        # Now: restart from `start_state` every `passes_per_restart` passes, so each restart
        # contributes a fresh DEPARTURE from the origin bin. `n_trials` is still the total
        # pass budget, so cost is unchanged; the passes are just redistributed.
        #
        # The trade-off is real and is the reason for short walks rather than one pass per
        # restart: `start_state` is a fresh PERIODIC family member, the most stable state in
        # the system, so restarting understates divergence. That seeding bias is what
        # corrupted the OBSERVE kernel. A few passes per restart sees whether the excursion
        # actually survives while still producing many departures.
        walker = copy(start_state)
        # ⚠️ THE RESIDUAL IS CARRIED ALONG THE WALK and RESET on each restart, exactly like
        # `walker` itself. It is the state of the ORBIT, so a fresh departure from the seed
        # is by construction undamaged (`:R_OK`) while pass 2 or 3 of a walk departs under
        # whatever damage the earlier passes accumulated. That is the conditioning the whole
        # dimension exists to record — without it, both get booked to the same row and
        # averaged, which is the defect being fixed.
        res_from = :R_OK
        for t in 1:n_trials
            # Fresh departure at the start of each restart block.
            if t > 1 && (t - 1) % passes_per_restart == 0
                walker = copy(start_state)
                res_from = :R_OK
            end

            from_alt = bin_of(alt_of(walker))
            # Out of the live bins -> this restart is done; begin the next one.
            if !(from_alt in ALT_BINS)
                walker = copy(start_state)
                res_from = :R_OK
                continue
            end
            from = KernelKey(from_alt, res_from)

            sh, sc = _to_shell(truth_eom!, walker, 4 * one_rev_s)
            if sh !== :ok
                bump!(akey, from, _terminal_dev(sh))
                walker = copy(start_state)   # terminal: restart, do not cancel the rest
                res_from = :R_OK
                continue
            end

            # The ONE targeting difference between the actions, as an argument rather than
            # a separate implementation. `:position` (CORRECT) aims at the nominal apse
            # POSITION vectors; `:altitude_position` (EXCURSE) commands the periapsis
            # ALTITUDE and keeps the apoapsis position constraints.
            b = mode_ === :altitude_position ?
                solve_burn(sc, one_rev_s; eom! = cr3bp_eom!, mode = :altitude_position,
                           peri_target_km = peri_tgt, r_apo_nom = ra) :
                solve_burn(sc, one_rev_s; eom! = cr3bp_eom!, mode = mode_,
                           r_peri_nom = rp, r_apo_nom = ra)

            # ⚠️ THE SUCCESS TEST DIFFERS BY MODE, and both are documented caveats.
            # In `:altitude_position` the total residual is dominated by the apoapsis
            # POSITION block and never clears `TARGET_TOL_KM` from a drifted state, so
            # `converged` would flag every excursion as failed; the delivery question is
            # whether the COMMANDED ALTITUDE was met, which is `peri_err_km`. In
            # `:position` there is no commanded altitude to check, and the ~8 km residual
            # floor means `converged` reads false on almost every healthy pass — it is
            # still the only signal available, so it is what gets counted.
            ok_ = mode_ === :altitude_position ?
                  (isfinite(b.peri_err_km) && b.peri_err_km < TARGET_TOL_KM) :
                  b.converged
            ok_ || (nonconv[akey][from] += 1)

            # ⚠️ A LOST APSE PAIR IS A LOSS, AND IT MUST BE CHARGED TO *THIS* BIN.
            #
            # When the onboard prediction cannot find the next apse pair, `solve_burn`
            # returns a non-finite residual and ΔV = 0 — the pass flies UNCONTROLLED. The
            # spacecraft is not yet outside the escape shell, so the coast below still
            # reaches a periapsis and the old code recorded that as an ordinary transition
            # (typically to ABOVE_50), then charged the escape on the FOLLOWING pass to
            # whatever bin the vehicle had fled to.
            #
            # That misattribution is not cosmetic — it is what made the kernel dangerous.
            # Measured 2026-08-30, holding the commanded 25 km from the drifted limit cycle:
            # passes 1 and 2 deliver 25.07 and 25.21 km, pass 3 loses the apse pair, and the
            # vehicle is gone by pass 4 — EVERY time. Yet `EXCURSE/A20_30` came back with
            # P(LOST) = 0.0 and a 71% self-transition, because the deaths were booked against
            # ABOVE_50 (whose 0.667 LOST mass is largely these misattributions). The solved
            # policy then chose EXCURSE_LOW three passes running and escaped at 5.3 d — it
            # was behaving correctly given a kernel that told it low excursions are free.
            #
            # A control step with no control is a loss of the orbit at the bin where the
            # decision was made, so record it there and stop the walk.
            if !isfinite(b.residual_km) || b.dv_mag_ms == 0.0
                bump!(akey, from, :LOST)
                push!(dvs[akey][from], 0.0)
                band === nothing || push!(band_dv[band], 0.0)
                verbose && @printf("  %-13s from %-16s NO APSE PAIR (ΔV=0) -> LOST\n",
                                   akey, string(from))
                walker = copy(start_state)   # terminal: restart, do not cancel the rest
                res_from = :R_OK
                continue
            end

            # The damage this pass leaves behind — the successor's residual bin, and the
            # conditioning the NEXT pass of this walk is booked under.
            res_to = residual_bin(b.residual_km)

            sp = copy(sc)
            sp[4:6] .+= exec_dv(b.dv)
            pe, ue = _to_peri(truth_eom!, sp, 4 * one_rev_s)
            if pe !== :ok
                bump!(akey, from, _terminal_dev(pe))
                push!(dvs[akey][from], b.dv_mag_ms)
                band === nothing || push!(band_dv[band], b.dv_mag_ms)
                walker = copy(start_state)   # terminal: restart, do not cancel the rest
                res_from = :R_OK
                continue
            end

            bump!(akey, from, bin_of(alt_of(ue)), res_to)
            push!(dvs[akey][from], b.dv_mag_ms)
            if band !== nothing
                push!(band_dv[band], b.dv_mag_ms)
                push!(band_achieved_alt[band], altitude(ue))
            end

            verbose && @printf("  %-13s from %-16s ΔV=%6.3f m/s -> alt %7.2f km (%s/%s)\n",
                               akey, string(from), b.dv_mag_ms, altitude(ue),
                               bin_of(alt_of(ue)), res_to)
            walker = copy(ue)          # continue the walk from here
            res_from = res_to
        end
        return nothing
    end

    # ── THE ACTION TREE ────────────────────────────────────────────────────────
    """
        tree_walk!(root_state, max_depth; cache) -> nothing

    Enumerate EVERY action sequence up to `max_depth` from `root_state`, booking each pass
    as one kernel sample.

    ⚠️ THIS IS THE COVERAGE MECHANISM (2026-08-31), and it replaces two failed attempts.
    A kernel row is `P(s' | s, a)`, so every reachable `(s, a)` needs visiting — but the
    single-action walks only ever fly ONE action per walk, which structurally cannot reach
    a state that another action produces. That is why every `CORRECT` row at `R_DEGRADED` /
    `R_CRITICAL` came back n = 0: CORRECT alone is a stable limit cycle at ~8 km residual
    and never damages its own orbit, so it can only DEPART degraded if something else
    degraded it first. The fix is not a longer walk or a bigger trial count; it is to fly
    the CROSS PRODUCT of actions.

    The two alternatives were tried and are worse:
      * placing the vehicle at a target damage level — there is nothing to place. Damage is
        a property of how far the orbit has drifted from what one impulse can fix, not a
        location.
      * perturbing a seed into a damaged state — breaks the onboard apse prediction and
        books solver failures as deaths. Measured 2026-08-31 from the 30.5 km member with
        identical targets: undispersed solves to residual 12.231 km / ΔV 1.884 m/s, while a
        0.5 km / 0.5 m/s perturbation gives residual = Inf and ΔV = 0. Whole-run symptom was
        CORRECT/A27_34/R_OK at P(LOST) = 0.471 — absurd, since correcting from a healthy mid
        orbit is the safe baseline. NOT a stability problem: a dispersed CORRECT walk
        survives at every scale tried and every seed altitude holds its own orbit for 8
        passes. The orbit is fine; the PLANNER is what the perturbation breaks, and it fails
        discontinuously, so shrinking the scale is not a fix.

    A tree needs neither: every node is a state the vehicle genuinely flew to, reached the
    way it would actually reach it.

    ⚠️ PARALLELISED BREADTH-FIRST, IN TWO PHASES PER LEVEL (2026-08-31). Each level's
    passes are independent — they share only their (already-computed) parent states — so
    they are flown with `@threads` and their outcomes collected into a per-pass result
    vector. NOTHING is booked during the parallel phase; the counts are folded in serially
    afterwards. That keeps `bump!`/`push!` off the threaded path entirely, so no locking is
    needed and, more importantly, the ARTIFACT IS BIT-IDENTICAL to a serial run: booking
    order is fixed by the level's node ordering, not by which thread finishes first.
    Julia must be started with `-t auto` (or `JULIA_NUM_THREADS`) for this to use more than
    one core; with one thread it degenerates to the serial walk.

    ⚠️ PREFIX SHARING IS EXACT ONLY BECAUSE THE DYNAMICS ARE DETERMINISTIC. A node's state
    is fully determined by its action prefix, so the 4 children of a node all branch from
    ONE parent state and a depth-d tree costs `(4^(d+1) - 4)/3` passes rather than
    `d * 4^d`. With `noisy_thruster = true` that no longer holds — each node would need
    several sampled children — so the cost model and this caching are both noise-free
    assumptions.

    ⚠️ COVERAGE IS REACHABILITY-WEIGHTED, NOT UNIFORM. States the vehicle passes through
    often get many samples; states reachable only at depth > `max_depth` get none, and
    `BELOW_20` / `ABOVE_44` may never appear at all from the nominal orbit. That is why the
    family-member seeding below is RETAINED — placement at an altitude is legitimate and
    was never the part that failed.

    Pre-resolved EXCURSE targets are passed in as `excurse_targets` so the family table is
    not queried inside the threaded region.
    """
    function tree_walk!(root_state, max_depth::Integer, excurse_targets)
        n_passes = 0
        n_nodes  = 0

        # `_fly_one` is PURE with respect to the accumulators: it integrates and solves,
        # and RETURNS what happened. Nothing here touches `counts`/`dvs`/`nonconv`, which
        # is what makes the threaded phase safe without locks.
        function _fly_one(state, res_from::Symbol, a::Symbol)
            from_alt = bin_of(alt_of(state))
            from_alt in ALT_BINS ||
                return (kind = :dead, from = nothing, a = a)
            from = KernelKey(from_alt, res_from)

            sh, sc = _to_shell(truth_eom!, state, 4 * one_rev_s)
            sh === :ok ||
                return (kind = :terminal, from = from, a = a, out = _terminal_dev(sh))

            # Same per-action targeting split as `action_walk!`.
            b = if a === :CORRECT
                solve_burn(sc, one_rev_s; eom! = cr3bp_eom!, mode = mode,
                           r_peri_nom = r_peri_nom, r_apo_nom = r_apo_nom)
            else
                tgt = get(excurse_targets, a, nothing)
                tgt === nothing ? nothing :
                    solve_burn(sc, one_rev_s; eom! = cr3bp_eom!,
                               mode = :altitude_position,
                               peri_target_km = tgt, r_apo_nom = r_apo_nom)
            end
            b === nothing && return (kind = :dead, from = from, a = a)

            ok_ = a === :CORRECT ? b.converged :
                  (isfinite(b.peri_err_km) && b.peri_err_km < TARGET_TOL_KM)

            # Lost apse pair: an uncontrolled pass, charged to THIS bin. Branch dies.
            if !isfinite(b.residual_km) || b.dv_mag_ms == 0.0
                return (kind = :noapse, from = from, a = a, ok = ok_)
            end

            res_to = residual_bin(b.residual_km)
            sp = copy(sc)
            sp[4:6] .+= exec_dv(b.dv)
            pe, ue = _to_peri(truth_eom!, sp, 4 * one_rev_s)
            pe === :ok ||
                return (kind = :terminal_after, from = from, a = a, ok = ok_,
                        dv = b.dv_mag_ms, out = _terminal_dev(pe))
            return (kind = :ok, from = from, a = a, ok = ok_, dv = b.dv_mag_ms,
                    to_alt = bin_of(alt_of(ue)), to_res = res_to,
                    state = copy(ue))
        end

        # BFS. `frontier` holds the nodes whose children are flown next.
        frontier = [(state = collect(float.(root_state)), res = :R_OK)]
        for _ in 1:max_depth
            isempty(frontier) && break

            # One work item per (parent node × action) — all independent.
            jobs = [(pi_, a) for pi_ in eachindex(frontier) for a in action_keys]
            results = Vector{Any}(undef, length(jobs))

            # ── PARALLEL PHASE: integrate + solve only. No accumulator touched. ──
            Threads.@threads for j in eachindex(jobs)
                pi_, a = jobs[j]
                node = frontier[pi_]
                results[j] = _fly_one(node.state, node.res, a)
            end

            # ── SERIAL PHASE: book in a FIXED order, so the artifact is reproducible
            # regardless of thread scheduling. ──
            next_frontier = NamedTuple[]
            for j in eachindex(jobs)
                r = results[j]
                r.kind === :dead && continue
                n_passes += 1
                if r.kind !== :terminal
                    get(r, :ok, true) || (nonconv[r.a][r.from] += 1)
                end
                if r.kind === :terminal
                    bump!(r.a, r.from, r.out)
                elseif r.kind === :noapse
                    bump!(r.a, r.from, :LOST)
                    push!(dvs[r.a][r.from], 0.0)
                elseif r.kind === :terminal_after
                    bump!(r.a, r.from, r.out)
                    push!(dvs[r.a][r.from], r.dv)
                else
                    bump!(r.a, r.from, r.to_alt, r.to_res)
                    push!(dvs[r.a][r.from], r.dv)
                    push!(next_frontier, (state = r.state, res = r.to_res))
                    n_nodes += 1
                end
            end
            frontier = next_frontier
        end

        verbose && @printf("  tree: depth %d, %d passes flown, %d live nodes, %d threads\n",
                           max_depth, n_passes, n_nodes, Threads.nthreads())
        return (passes = n_passes, nodes = n_nodes)
    end

    # Resolve the EXCURSE aim points ONCE, outside the threaded region — the family table
    # is a shared structure and `retarget_to_altitude` has no documented thread safety.
    excurse_targets = Dict{Symbol,Float64}()
    if family_table !== nothing
        for band in band_names
            m = retarget_to_altitude(family_table, band_target_km[band])
            m === nothing && continue
            excurse_targets[Symbol("EXCURSE_", band)] = m.info.periapsis_alt_km
        end
    end

    tree_stats = tree_depth > 0 ?
        tree_walk!(base, tree_depth, excurse_targets) : (passes = 0, nodes = 0)
    tree_passes, tree_nodes = tree_stats.passes, tree_stats.nodes

    # ── PRIMARY WALKS from the nominal limit cycle, one per action ─────────────
    # CORRECT goes through the SAME walker as the excursions now. `n_steps` is its pass
    # budget (it used to be the length of a single sustained loop); with restarts it is
    # spread over many fresh departures instead, exactly as the EXCURSE budget is.
    action_walk!(:CORRECT, copy(base), n_steps;
                 mode_ = mode, rp = collect(r_peri_nom), ra = collect(r_apo_nom))

    for band in band_names
        target = band_target_km[band]
        member = family_table === nothing ? nothing :
                 retarget_to_altitude(family_table, target)
        if member === nothing
            verbose && @info "excurse: no family member at $(target) km for $band — skipped"
            continue
        end
        # `:altitude_position` (2026-08-30): the periapsis is commanded as an ALTITUDE, so
        # the family member's role is only to confirm the band is a realisable orbit — the
        # radius no longer has to be smuggled in through a scaled position vector, and the
        # phase-matching problem noted at the top of this file does not arise for the
        # periapsis half at all. The apoapsis anchor stays our own nominal.
        action_walk!(Symbol("EXCURSE_", band), copy(base), excurse_trials;
                     mode_ = :altitude_position,
                     peri_tgt = member.info.periapsis_alt_km,
                     ra = collect(r_apo_nom), band = band)
    end

    # ── SEED EVERY ACTION FROM EVERY ALTITUDE BIN ──────────────────────────────
    # ⚠️ WITHOUT SEEDING, A KERNEL DESCRIBES ONE BIN. Measured 2026-08-30: the primary walks
    # all start from the IC limit cycle (~37 km) and, once `:altitude_position` made the aim
    # accurate, they SETTLE — 8 passes commanded 35 km go 34.97, 34.95, 34.94 … and never
    # leave the bin they started in. Rows are keyed by the bin a pass STARTS in, so covering
    # them means starting passes in them. A working controller has no reason to visit
    # BELOW_20 or ABOVE_44, yet the solver queries those states, and filling them with a
    # guess would teach the policy something never measured.
    #
    # ⚠️ ONE LOOP FOR ALL ACTIONS (2026-08-31). This used to be TWO blocks — one seeding the
    # EXCURSE walks per (bin × band), one seeding CORRECT per bin with its own hand-written
    # copy of the burn/coast/book logic. Merging them is what finally measures CORRECT from
    # a DEGRADED orbit: the shared walker accumulates damage across a restart block, so
    # `CORRECT` now produces `R_DEGRADED` / `R_CRITICAL` departures the same way the
    # excursions always did. Those rows were ALL n = 0 before, which meant the recovery
    # behaviour the residual dimension exists to model was pure inference.
    #
    # ⚠️ AND THE LOW BINS ARE REACHABLE — the earlier read that they were not was wrong.
    # Commanding 20 or 25 km from the 37 km limit cycle escapes, but that is the TRANSFER
    # failing, not the destination: placed ON the 18/20/22/25/28 km family members and told
    # to hold their own altitude, `:altitude_position` holds every one of them to ~0.09 km
    # for 12 passes with real burns. `ABOVE_44` is the genuinely hard bin — 55 and 60 km
    # escape by the third pass.
    # ⚠️ SEEDS ARE SPREAD ACROSS EACH BIN, BUT NOT DISPERSED (2026-08-31). Sampling several
    # ALTITUDES per bin is sound — a bin is a range, and one point should not stand in for
    # it — and every sampled seed was checked to hold its own orbit for 8 passes. Adding a
    # state-space PERTURBATION on top was tried and rejected: it breaks the onboard apse
    # prediction rather than the orbit, so the kernel records solver failures as deaths.
    # See `tree_walk!` for the measurements; reachability is handled by the tree instead.
    if seed_bins && family_table !== nothing
        for bin in ALT_BINS
            alts = _bin_sample_alts(bin, alt_edges, seed_samples)
            # Split each cell's budget across the samples (at least one pass each).
            corr_each = max(1, seed_trials ÷ length(alts))
            exc_each  = max(1, excurse_seed_trials ÷ length(alts))

            for rep in alts
                m = retarget_to_altitude(family_table, rep)
                if m === nothing
                    verbose && @info "seed: no family member near $(rep) km for $bin — sample skipped"
                    continue
                end
                seed_ic = collect(float.(m.ic))
                ps, us = _to_peri(truth_eom!, seed_ic, 4 * one_rev_s)
                ps === :ok || continue
                # Apse anchors from the SEEDED orbit, not the nominal one: from an 18 km
                # halo the nominal 37 km apses are not what this vehicle is holding, so
                # targeting them would measure a TRANSFER rather than stationkeeping there.
                rp_s, ra_s = next_apse_positions(seed_ic; eom! = cr3bp_eom!)

                # CORRECT: hold the seeded orbit.
                action_walk!(:CORRECT, copy(us), corr_each;
                             mode_ = mode, rp = rp_s, ra = ra_s)

                # EXCURSE_<band>: aim at each band's commanded altitude from this bin.
                for band in band_names
                    tgt_m = retarget_to_altitude(family_table, band_target_km[band])
                    tgt_m === nothing && continue
                    action_walk!(Symbol("EXCURSE_", band), copy(us), exc_each;
                                 mode_ = :altitude_position,
                                 peri_tgt = tgt_m.info.periapsis_alt_km,
                                 ra = ra_s, band = band)
                end
            end
        end
    end

    # ── Assemble ──────────────────────────────────────────────────────────────
    rows = Dict{Symbol,Dict{KernelKey,CalibrationRow}}()
    for a in action_keys
        rows[a] = Dict{KernelKey,CalibrationRow}()
        for k in all_keys
            c = counts[a][k]
            n = sum(c)
            rows[a][k] = CalibrationRow(c, n == 0 ? Float64[] : c ./ n, n,
                                        nonconv[a][k], dvs[a][k])
        end
    end

    diagnostics = Dict{String,Any}(
        # Total CORRECT passes actually logged. Was the length of the bespoke sustained
        # loop; now it is just how many CORRECT trials the unified walker booked.
        "n_loop_steps"       => sum(sum(counts[:CORRECT][k]) for k in all_keys),
        "rows"               => rows,
        "band_dv_ms"         => Dict(string(b) => band_dv[b] for b in band_names),
        "band_achieved_alt"  => Dict(string(b) => band_achieved_alt[b] for b in band_names),
        "truth_name"         => truth_name,
        # θ, recorded so an artifact is self-describing: a kernel set is only meaningful
        # together with the environment it was measured in.
        "noisy_thruster"     => noisy_thruster,
        "thruster_kwargs"    => Dict(string(k) => v for (k, v) in pairs(thruster_kwargs)),
        "band_target_km"     => Dict(string(b) => band_target_km[b] for b in band_names),
        "mode"               => string(mode),
        "alt_edges"          => collect(alt_edges),
        # The residual (orbit-damage) discretization these rows are conditioned on. Part of
        # the STATE-SPACE definition, not θ: it must be identical across every θ in a sweep
        # (only T_θ and O_θ may vary), so it is recorded to make a mismatch detectable.
        "residual_edges"     => collect(RESIDUAL_EDGES),
        "residual_bins"      => string.(collect(RESIDUAL_BINS)),
        "min_trials_trusted" => MIN_TRIALS_TRUSTED,
        # MEASUREMENT EFFORT, recorded separately from θ. Without this you cannot tell a
        # thin row that is thin BY CONFIGURATION from one thin because the walk settled.
        "effort" => Dict{String,Any}(
            "n_steps"             => n_steps,
            "horizon_s"           => horizon_s,
            "excurse_trials"      => excurse_trials,
            "seed_trials"         => seed_trials,
            "excurse_seed_trials" => excurse_seed_trials,
            "seed_bins"           => seed_bins,
            # ⚠️ READ THIS BEFORE READING A TRIAL COUNT. With `seed_samples = 1` and a
            # noise-free thruster the environment is deterministic, so `n` counts
            # REPLICATES of one trajectory rather than samples from a distribution.
            "seed_samples"        => seed_samples,
            "tree_depth"          => tree_depth,
            "tree_passes"         => tree_passes,
            "tree_nodes"          => tree_nodes,
            "rng_seed"            => rng_seed,
        ),
    )
    return rows, diagnostics
end
"""
    _fill_unmeasured_row(rows, action, key) -> Vector{Float64}

The distribution to use for a row with `n = 0`, i.e. one the measurement never visited.

⚠️ A SELF-TRANSITION FILL IS NOT NEUTRAL, AND IT BIT US TWICE (2026-08-31). Filling an
unmeasured row with "you stay exactly where you are" hands the solver an action that is
free, perfectly safe, and — because the coverage gate keys off the landed bin — often
SCIENCE-BANKING as well. A solver maximising value will find that action and take it. This
is the identical failure mode that killed the OBSERVE action (four rows measured as 100%
self-transitions taught the policy that coasting was free; it coasted and escaped at
3.78 d), and it reappeared the moment the residual split created 34 unmeasured rows: with
a self-transition fill, `EXCURSE_LOW` from `A27_34/R_CRITICAL` looked like a safe +10.5
while the measured `EXCURSE_MID` from the same state was a −157.5, so the policy chose the
unmeasured one.

The fill therefore BACKS OFF ALONG THE DAMAGE AXIS instead: reuse the SAME action's
measured row at the nearest LESS-DEGRADED residual bin. That is the conservative reading of
the one monotonicity the measurements do support — more damage never makes an action safer
(measured: `EXCURSE_LOW/A20_27` goes P(LOST) 0.000 → 0.000 → 1.000 across R_OK →
R_DEGRADED → R_CRITICAL; `EXCURSE_MID/A27_34` goes 0.000 → 0.000 → 0.800). So inheriting
the healthier bin's outcome UNDERSTATES the risk rather than fabricating safety, and it
never invents a free action.

If no less-degraded row was measured either, fall back to the self-transition — there is
genuinely nothing to infer from — and rely on `meta.unmeasured_rows` to flag it. Such a row
is not evidence, and a policy leaning on one means nothing.
"""
function _fill_unmeasured_row(rows::Dict{Symbol,Dict{KernelKey,CalibrationRow}},
                              action::Symbol, key::KernelKey)
    ri = residual_index(key.residual)
    # Walk DOWN the damage axis: R_CRITICAL falls back to R_DEGRADED, then to R_OK.
    for j in (ri - 1):-1:1
        donor = rows[action][KernelKey(key.alt, RESIDUAL_BINS[j])]
        donor.n == 0 && continue
        return collect(donor.probs)
    end
    # Nothing measured for this action at this altitude at any damage level.
    fill_row = zeros(Float64, N_KERNEL_COLS)
    fill_row[kernel_entry_index(key.alt, key.residual)] = 1.0
    return fill_row
end

"""
    tables_from_rows(rows, diagnostics) -> AltTables

Assemble measured [`CalibrationRow`](@ref)s into an [`AltTables`](@ref) ready for
[`write_tables`](@ref). This is the step that was missing while `artifacts/tables.json`
held hand-transcribed numbers: `calibrate_tables` measured the rows and nothing turned
them into the artifact, so the numbers were copied across by hand and could drift from the
run that produced them with nothing to detect it.

⚠️ A ROW WITH `n = 0` IS NOT MEASURED. `calibrate_tables(seed_bins = true)` seeds a trial
from every altitude bin precisely so this does not happen, but many (altitude × residual)
combinations are simply not reachable — the vehicle cannot arrive at 18 km with a pristine
solve after a violent transfer — so the residual split leaves ~34 of 60 rows empty. They
are filled by [`_fill_unmeasured_row`](@ref), which inherits the same action's measured
behaviour at the nearest LESS-DEGRADED damage bin rather than fabricating a self-transition;
read that function's docstring before changing it, because the obvious fill is actively
dangerous here. `meta.trials` records the count for every row so the fill is auditable
rather than invisible. A policy is no more trustworthy than its least-visited row — check
`meta.trials` against [`MIN_TRIALS_TRUSTED`](@ref) before believing what the policy does in
a rarely-visited bin.
"""
function tables_from_rows(rows::Dict{Symbol,Dict{KernelKey,CalibrationRow}},
                          diagnostics::AbstractDict)
    kernels = Dict{Symbol,Dict{KernelKey,Vector{Float64}}}()
    trials  = Dict{String,Dict{String,Int}}()
    unmeasured = String[]

    # Whatever action keys `calibrate_tables` produced — `:CORRECT` plus one
    # `:EXCURSE_<BAND>` per band as of 2026-08-30 (previously a single pooled `:EXCURSE`).
    for a in sort(collect(keys(rows)))
        kernels[a] = Dict{KernelKey,Vector{Float64}}()
        trials[string(a)] = Dict{String,Int}()
        for key in kernel_keys_all()
            row = rows[a][key]
            if row.n == 0
                push!(unmeasured, "$a/$key")
                kernels[a][key] = _fill_unmeasured_row(rows, a, key)
            else
                kernels[a][key] = collect(row.probs)
            end
            trials[string(a)][string(key)] = row.n
        end
    end

    meta = Dict{String,Any}(
        "generated_by"       => "calibrate_tables + tables_from_rows",
        "truth_name"         => get(diagnostics, "truth_name", "unknown"),
        "mode"               => get(diagnostics, "mode", "unknown"),
        # ── θ: WHICH ENVIRONMENT THESE KERNELS DESCRIBE ──────────────────────
        # A kernel set is meaningless without the environment it was measured in, and this
        # artifact is one member of a parameterized POMDP family (shared S/A/O/R, varying
        # T_θ and O_θ), not "the" model. Recorded here so the committed JSON is
        # self-describing rather than relying on whoever ran it to remember.
        "theta" => Dict{String,Any}(
            # `truth_eom` lives ONLY here, not on `StationkeepingPOMDP`: `build_pomdp` never
            # touches the truth model (it loads pre-measured kernels), so a copy on the
            # config would be a second owner free to disagree with the artifact that the
            # kernels were actually measured against. The artifact is the record.
            "truth_eom"       => get(diagnostics, "truth_name", "unknown"),
            "noisy_thruster"  => get(diagnostics, "noisy_thruster", false),
            "thruster_kwargs" => get(diagnostics, "thruster_kwargs", Dict{String,Any}()),
            "band_target_km"  => get(diagnostics, "band_target_km", Dict{String,Any}()),
            "alt_edges"       => get(diagnostics, "alt_edges", Float64[]),
        ),
        "alt_edges"          => get(diagnostics, "alt_edges", Float64[]),
        # STATE-SPACE definition, not θ — identical across every θ in a sweep.
        "residual_edges"     => get(diagnostics, "residual_edges", collect(RESIDUAL_EDGES)),
        "residual_bins"      => get(diagnostics, "residual_bins",
                                    string.(collect(RESIDUAL_BINS))),
        "n_loop_steps"       => get(diagnostics, "n_loop_steps", 0),
        "effort"             => get(diagnostics, "effort", Dict{String,Any}()),
        "trials"             => trials,
        "unmeasured_rows"    => unmeasured,
        "min_trials_trusted" => MIN_TRIALS_TRUSTED,
        "caveats"            => [
            "Rows listed in unmeasured_rows have n = 0 and are FILLED, not measured. " *
            "They are not evidence. The fill inherits the SAME action's measured row at " *
            "the nearest less-degraded residual bin (see _fill_unmeasured_row) — never a " *
            "self-transition, which would hand the solver a free risk-free action.",
            "Rows with 0 < n < min_trials_trusted are indicative only.",
            "Noise-free unless the run was configured otherwise, so any survival number " *
            "derived from these kernels is an UPPER BOUND, not feasibility.",
            "Rows are conditioned on the RESIDUAL (orbit-damage) bin as well as the " *
            "altitude bin, so the row count is 3x what it was before 2026-08-31 and the " *
            "SAME trial budget is spread across all of them. Expect thinner rows; check " *
            "meta.trials rather than assuming the previous counts carried over.",
        ],
    )
    # Split into the CORRECT kernel and the per-action EXCURSE kernels.
    excurse = Dict{Symbol,Dict{KernelKey,Vector{Float64}}}(
        a => k for (a, k) in kernels if a !== :CORRECT)
    return AltTables(kernels[:CORRECT], excurse, meta)
end
