"""
calibration/calibrate.jl — measure the transition kernels from the truth model.

Rows are keyed [`KernelKey`](@ref) — altitude bin and residual bin — and their columns are
the joint successor `(alt, residual)`. Two sources feed them:

  - `tree_walk!` — an exhaustive action-sequence tree from the nominal orbit, threaded.
    The bulk of the samples, and the only source that reaches a state one action produces
    for another to depart from.
  - `action_walk!` — single-action walks from the nominal orbit and from seeded
    halo-family members in each altitude bin. Serial. The only source for `BELOW_20`, and
    for the observed losses in some `R_CRITICAL` rows.

Every row carries its trial count. A row below `MIN_TRIALS_TRUSTED` is flagged rather than
smoothed; an `n = 0` row is filled by [`_fill_unmeasured_row`](@ref) and named in
`meta.unmeasured_rows`.

NOTE: truth/onboard split. `truth_eom!` is an argument, integrated only in the coast
helpers. All planning goes through `solve_burn` on the onboard CR3BP.

NOTE: seeded rows are optimistic. A seed is a freshly-placed periodic family member, the
most stable state in the system, so a row measured only from seeds understates how fast the
orbit diverges.

NOTE: `converged` is not the success test for an excursion. In `:altitude_position` the
apoapsis-position block never clears `TARGET_TOL_KM` from a drifted state, so `converged`
reads false on passes that hit the commanded altitude to well under a km. EXCURSE rows
score on `peri_err_km` instead.

NOTE: only the tree is threaded; run with `-t auto`. Output is bit-identical either way,
because the tree books its counts serially in a fixed order.
"""

"""
    CalibrationRow

One measured kernel row.

  - `counts` — successor counts over [`kernel_columns`](@ref), the joint (alt, residual)
  - `probs` — `counts ./ n`, or empty when `n == 0`
  - `n` — trials contributing to this row
  - `n_nonconverged` — solve failures folded into it
  - `dv_ms` — per-trial ΔV (m/s), for the `action_dv_cost` proxy
"""
struct CalibrationRow
    counts::Vector{Int}          # over kernel_columns() — the JOINT (alt, residual)
    probs::Vector{Float64}       # counts ./ n, or a fallback if n == 0
    n::Int
    n_nonconverged::Int          # solve_burn failures folded into this row
    dv_ms::Vector{Float64}       # per-trial ΔV, for the action_dv_cost proxy
end

"""
    _bin_rep_alt(bin, e) -> Float64

Representative seeding altitude (km) for an altitude bin: its midpoint, or an interior
point for the two open-ended outer bins.

  - `bin` — the altitude bin
  - `e` — the four bin edges (km)
"""
function _bin_rep_alt(bin::Symbol, e::NTuple{4,<:Real})
    # Not the midpoint: the continued family floor is ~17.6 km, so a lower seed finds no
    # member and the row silently stays unmeasured.
    bin === :BELOW_20 && return 18.0
    bin === :A20_27   && return 0.5 * (e[1] + e[2])
    bin === :A27_34   && return 0.5 * (e[2] + e[3])
    bin === :A34_44   && return 0.5 * (e[3] + e[4])
    # ABOVE_44 is open-ended, so seed somewhere HOLDABLE rather than at the geometric
    # continuation: a member holds its own altitude at 45 km but escapes by pass 3 at 55.
    return 46.0
end

"""
    _bin_sample_alts(bin, e, n) -> Vector{Float64}

Seeding altitudes spread across a bin, rather than one representative point.

  - `bin` — the altitude bin
  - `e` — the four bin edges (km)
  - `n` — how many altitudes to return; `n <= 1` falls back to [`_bin_rep_alt`](@ref)

Returns `n` altitudes in km. A row means "somewhere in this cell", so the measurement
samples the cell rather than a point in it.

NOTE: interior points only — endpoints are inset by 15% of the bin width so a sample never
sits on a boundary where `alt_bin` could round it into the neighbour. The open-ended bins
are spread over a narrow holdable window rather than out into the region that escapes.
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
Seed altitudes per (altitude bin x action) cell. The cell's trial budget is divided across
them, so this buys spread rather than costing more.

NOTE: noise-free, the truth integrator and burn solver are both deterministic, so restarts
from the SAME seed reproduce each other exactly. Spreading the seed altitude across the bin
is what makes `n` a sample size rather than a repeat count.
"""
const SEED_SAMPLES_PER_CELL = 3

"""
Depth of the exhaustive action-sequence tree (see `tree_walk!` inside `calibrate_tables`).

Every action sequence of this length is flown from the nominal orbit, so the measurement
covers the cross product of actions rather than one action at a time. That is what reaches
`(CORRECT, R_DEGRADED)`, a state CORRECT cannot produce for itself.

Cost before pruning is `(|A|^(d+1) - |A|) / (|A| - 1)` passes — at |A| = 4, 1364 at depth 5
and 21844 at depth 7. Escaping branches prune their whole subtree, so the flown count runs
well under that: depth 7 flies ~10,800. `experiments/calibrate.jl` takes the depth as its
first argument.

NOTE: there is no resume path. The tree is bottom-heavy, so reusing a shallower one can
never save more than ~1/|A| of the work; the frontier always has to be flown.
"""
const TREE_DEPTH = 7

"""Minimum trials before a measured row is anything but indicative.

NOTE: a trial count is a sample size only if the samples differ. Noise-free the environment
is deterministic, so restarts from one seed are replicates rather than draws — check
`meta.effort.seed_samples` before reading `n` as statistical confidence.
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

# An EXCURSE commands the periapsis ALTITUDE (`mode = :altitude_position`), not a position
# vector. Aiming one impulse at a real family member's absolute apse positions fails on
# PHASE — those are where that member sits at its own epoch, which is unrelated to ours —
# and a radially-scaled waypoint has too little authority to move achieved periapsis. See
# `_sarsop_band_targets` in common/simulate.jl.

"""
    CALIBRATION_EFFORT

Measurement-effort defaults, in one named place rather than as inline literals across the
signature. Recorded into `meta.effort`, so an artifact says how hard it was measured.

  - `n_steps`             — total CORRECT passes from the nominal limit cycle, spread over
                            restarts
  - `horizon_s`           — accepted and recorded, but see the note below
  - `excurse_trials`      — passes per primary excursion walk
  - `seed_trials`         — CORRECT passes per seeded altitude bin
  - `excurse_seed_trials` — EXCURSE passes per (bin x band) seed
  - `seed_bins`           — also seed the bins a working controller never visits
  - `seed_samples`        — altitudes sampled across each bin when seeding; the cell's
                            budget is split across them, so this buys spread not cost
  - `tree_depth`          — action-tree depth, the main coverage mechanism
  - `rng_seed`            — only consumed when `noisy_thruster` is on; noise-free the
                            measurement is fully deterministic

NOTE: effort is not θ. These say how hard the environment was sampled, not which
environment it is, so they are deliberately not fields on `StationkeepingPOMDP` and are not
part of a sweep grid.

NOTE: `horizon_s` is dead. It capped the wall clock of a sustained CORRECT loop that no
longer exists — every action now goes through `action_walk!` or the tree, both bounded by
pass count. Still accepted so older drivers keep working, but changing it does nothing.

NOTE: raising the trial counts does not fix a thin ORIGIN row. An accurate walk settles
into its destination bin after 1-2 passes, so a longer walk feeds the destination row while
the origin row it departed stays thin. More restarts, not longer walks.
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

  - `field` / `θ` — a single field name, or a NamedTuple whose keys are checked

Returns `true` for `:noisy_thruster` and `:thruster_sigma_pct`, which change the dynamics and
force a full `calibrate_tables` run (minutes per θ). `sigma_nav_km` enters
`observation_matrix` analytically and `plume_gradient` enters `transition_matrix`
analytically, so both are seconds per θ and reuse the committed kernels.

So a 3-value nav sweep is seconds and a 3-value thruster sweep is ~20 minutes. A sweep
driver that does not know the difference either wastes 20 minutes or, worse, reuses stale
kernels for a dynamics-changing θ.
"""
needs_recalibration(field::Symbol) =
    field in (:noisy_thruster, :thruster_sigma_pct)
needs_recalibration(θ::NamedTuple) = any(needs_recalibration, keys(θ))

"""
    calibrate_tables(config::StationkeepingPOMDP; effort..., truth_eom!, verbose)
        -> (rows, diagnostics)

Measure the transition kernels for the environment described by `config`.

  - `config` — the scenario. θ is read from it: `alt_edges`, `band_names`,
    `band_target_km`, `noisy_thruster`, `thruster_sigma_pct`
  - `truth_eom!`, `truth_name` — the truth model to measure against, and its label
  - remaining keywords — measurement effort, see [`CALIBRATION_EFFORT`](@ref)

Returns `(rows, diagnostics)`: a `Dict{Symbol,Dict{KernelKey,CalibrationRow}}` per action,
and the per-row diagnostics that [`tables_from_rows`](@ref) turns into an artifact.

    cfg  = StationkeepingPOMDP(; plume_gradient = 4.0)
    rows, diag = calibrate_tables(cfg; verbose = true)
    write_tables(tables_from_rows(rows, diag);
                 path = theta_path("tables", (plume_gradient = 4.0,)))

NOTE: prefer this method over the keyword-only one. The config is the single owner of θ, so
there is nothing to hand-map and nothing to disagree.
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
        thruster_sigma_pct = config.thruster_sigma_pct,
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

Low-level form: measure all kernels against hand-specified bins, without a config.

  - `truth_eom!` — truth model to measure against; defaults to [`cr3bp_j2_eom!`](@ref)
  - `alt_edges`, `band_names`, `band_target_km` — θ, the discretization and aim points
  - `noisy_thruster`, `thruster_sigma_pct` — whether burns execute with error, and the 1σ
    magnitude error in percent
  - `ic`, `period_s` — the reference orbit and its period (s)
  - `mode` — targeting mode for the CORRECT burns
  - `family_table` — prebuilt halo family for seeding; `nothing` uses the cached span
  - remaining keywords — measurement effort, see [`CALIBRATION_EFFORT`](@ref)

Returns `(rows, diagnostics)`.

NOTE: prefer `calibrate_tables(config)`. This method's `alt_edges` / `band_target_km`
defaults do not match `StationkeepingPOMDP`'s, so calling it bare measures kernels keyed to
bins the model does not use — and nothing errors.

NOTE: `mode` applies to the CORRECT burns only. Every EXCURSE uses `:altitude_position`
regardless, which is why band delivery is ~0.2 km while `meta.mode` reads `:position`.
"""
function calibrate_tables(;
    truth_eom! = cr3bp_j2_eom!,
    truth_name::AbstractString = "CR3BP + Enceladus J2",
    # ── θ: the environment parameters ────────────────────────────────────────
    # The kernels this function returns ARE `T_θ`, so every θ needs its own call.
    #
    # Noise-free is the optimistic corner of the family and the reference other θ are
    # measured against — a legitimate θ, but not the deployment environment. Turning noise
    # on raises measured P(LOST) for the LOW and MID excursions sharply.
    noisy_thruster::Bool = false,
    rng::AbstractRNG = Xoshiro(0),
    thruster_sigma_pct::Real = THRUSTER_SIGMA_PCT_B24_MODEL2,
    # South-polar, because the plumes are at the south pole and the plume gradient θ is
    # meaningless over the north. The north-polar member gives identical kernels — it is an
    # exact z-reflection, the dynamics are z-symmetric, and every row is keyed on periapsis
    # ALTITUDE, which the reflection preserves — but defaulting to south states the science
    # case rather than relying on a reader knowing that.
    ic::AbstractVector{<:Real} = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND)),
    period_s::Real = PERIOD1_TRIPLE_PERIOD_S,
    n_steps::Integer = 120,
    # Dead — recorded into meta, never read by the measurement. See CALIBRATION_EFFORT.
    horizon_s::Real = 25 * 86400.0,
    alt_edges::NTuple{4,Float64} = (20.0, 30.0, 40.0, 50.0),
    band_names::NTuple{3,Symbol} = (:LOW, :MID, :HIGH),
    band_target_km::Dict{Symbol,Float64} =
        Dict(:LOW => 25.0, :MID => 35.0, :HIGH => 45.0),
    mode::Symbol = :position,
    excurse_trials::Integer = 8,
    family_table::Union{Nothing,Vector{NamedTuple}} = halo_family_table_cached(),
    seed_bins::Bool = true,
    # Sized so rows clear `MIN_TRIALS_TRUSTED`. A row measured from a handful of trials
    # reads as a distribution but cannot tell a real one-third from a coin flip that landed
    # that way, and the solver will exploit the difference as if it were signal.
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
    one_rev_s = period_s / 3          # `period_s` defaults to 3x the true period

    # Count-based apse search: `one_rev_s = T/3` is the inter-periapsis CONTROL cadence, not
    # an apse-to-apse revolution — as a search window it lands 0.136 s short of the first
    # apoapsis and yields a NaN target, which silently zeroes every CORRECT burn.
    r_peri_nom, r_apo_nom = next_apse_positions(ic; eom! = cr3bp_eom!)
    apo_nom_alt = norm(_enc_relative(r_apo_nom)) - R_ENCELADUS

    # Bin by achieved periapsis altitude, through the model's own `alt_bin` so there is one
    # implementation rather than two that must stay bit-identical.
    bin_of(h) = alt_bin(alt_edges, h)
    alt_of(u) = norm(_enc_relative(u[1:3])) - R_ENCELADUS

    # Apply a commanded ΔV the way the environment does, through the same
    # `apply_dv`/`apply_dv_noisy` pair the rollout harness uses, so a kernel measured here
    # and a rollout flown there cannot disagree about what the thruster does.
    exec_dv(dv) = noisy_thruster ?
        apply_dv_noisy(dv, rng; sigma_pct = thruster_sigma_pct)[1] :
                                   apply_dv(dv)

    # Columns are the JOINT (alt, residual) successor, so a row answers both "where do I
    # land" and "how damaged am I then" — the second half is what lets the policy learn
    # that the first CORRECT after an excursion does not clear the damage.
    cols = kernel_columns()
    # One kernel per EXCURSE action, not one pooled across bands: a kernel encodes where
    # you LAND, and pooling makes the three excursions identical in T.
    #
    # NOTE: per-band keying DIVIDES the trials rather than multiplying them, so rows that
    # cleared `MIN_TRIALS_TRUSTED` under a pooled kernel may not. Check `meta.trials`.
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

    # ── One walker for every action ────────────────────────────────────────────
    # Every action, CORRECT included, goes through `action_walk!`. The legitimate per-action
    # differences are arguments — targeting mode, aim point, success test — not separate
    # implementations: a second hand-copied loop is how the two would silently diverge.
    st0, u0 = _to_peri(truth_eom!, ic, 4 * one_rev_s)
    base = st0 === :ok ? copy(u0) : copy(ic)

    """
        action_walk!(akey, start_state, n_trials; kwargs...)

    Fly `n_trials` passes of one action, restarting from `start_state` every
    `passes_per_restart` passes, and book each pass into that action's kernel.

      - `akey` — the action every pass flies
      - `start_state` — the state each restart departs from (km, km/s)
      - `n_trials` — total pass budget, spread over restarts
      - `mode_` — `:position` aims at nominal apse position vectors (CORRECT);
        `:altitude_position` commands the periapsis altitude (EXCURSE)
      - `peri_tgt` — commanded periapsis altitude (km), for `:altitude_position`
      - `rp`, `ra` — nominal periapsis / apoapsis position targets, for `:position`
      - `band` — science band an EXCURSE aims at, or `nothing`; only used for the per-band
        ΔV and achieved-altitude diagnostics
      - `passes_per_restart` — passes flown before returning to `start_state`

    The residual is carried along the walk and reset on restart: it is the state of the
    ORBIT, so a transition is booked under the damage the PREVIOUS pass left behind. A
    fresh departure from a seed is undamaged by construction; later passes of a walk depart
    under whatever the earlier ones accumulated.
    """
    function action_walk!(akey::Symbol, start_state, n_trials;
                          mode_::Symbol, peri_tgt = nothing,
                          ra = nothing, rp = nothing,
                          band = nothing,
                          passes_per_restart::Integer = EXCURSE_PASSES_PER_RESTART)

        # Restarts, not one long walk. An accurate excursion SETTLES — it reaches the
        # commanded band in a pass or two and stays — so a single long walk books every
        # later pass as departing the DESTINATION bin and leaves the origin row at n = 1.
        # Restarting spends the same pass budget on many fresh departures instead.
        #
        # A few passes per restart rather than one, because `start_state` is a periodic
        # family member and the most stable state in the system: restarting understates
        # divergence, so the later passes of each restart are what see a drifted departure.
        walker = copy(start_state)
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

            # The one targeting difference between the actions, as an argument rather than
            # a separate implementation.
            b = mode_ === :altitude_position ?
                solve_burn(sc, one_rev_s; eom! = cr3bp_eom!, mode = :altitude_position,
                           peri_target_km = peri_tgt, r_apo_nom = ra) :
                solve_burn(sc, one_rev_s; eom! = cr3bp_eom!, mode = mode_,
                           r_peri_nom = rp, r_apo_nom = ra)

            # The success test differs by mode. In `:altitude_position` the total residual
            # is dominated by the apoapsis-position block and never clears `TARGET_TOL_KM`
            # from a drifted state, so `converged` would flag every excursion as failed;
            # the delivery question is `peri_err_km`. In `:position` there is no commanded
            # altitude to check and the residual floor means `converged` reads false on
            # almost every healthy pass, but it is the only signal available.
            ok_ = mode_ === :altitude_position ?
                  (isfinite(b.peri_err_km) && b.peri_err_km < TARGET_TOL_KM) :
                  b.converged
            ok_ || (nonconv[akey][from] += 1)

            # A lost apse pair is a LOSS, charged to THIS bin. When the onboard prediction
            # cannot find the next apse pair, `solve_burn` returns a non-finite residual and
            # ΔV = 0 and the pass flies uncontrolled. The vehicle is not yet outside the
            # escape shell, so the coast below still reaches a periapsis — booking that as
            # an ordinary transition hides the death and charges the eventual escape to
            # whatever bin the vehicle fled to, which reads as "this excursion is free".
            #
            # A control step with no control is a loss at the bin where the decision was
            # made, so record it there and end the walk.
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
        tree_walk!(root_state, max_depth, excurse_targets) -> (passes, nodes)

    Enumerate every action sequence up to `max_depth` from `root_state`, booking each pass
    as one kernel sample.

      - `root_state` — the state every sequence departs from (km, km/s)
      - `max_depth` — sequence length; see [`TREE_DEPTH`](@ref) for the cost
      - `excurse_targets` — commanded altitude per EXCURSE action, resolved before the
        threaded region so the family table is not queried inside it

    Returns `(passes, nodes)`: passes flown and live nodes reached.

    This is the coverage mechanism. A kernel row is `P(s' | s, a)`, so every reachable
    `(s, a)` needs visiting — and a single-action walk structurally cannot reach a state
    that a DIFFERENT action produces. CORRECT alone is a stable limit cycle and never
    damages its own orbit, so `CORRECT` from a degraded departure only exists if something
    else degraded it first. Flying the cross product is what reaches those rows.

    NOTE: parallelised breadth-first in two phases per level. Passes are flown with
    `@threads` and NOTHING is booked during that phase; counts are folded in serially
    afterwards, in the level's node order. So no locking is needed. Start Julia with
    `-t auto` to use more than one core.

    NOTE: bit-identical to a serial run ONLY when `noisy_thruster = false`. The serial fold
    fixes the booking order, but under noise `exec_dv` draws from a single shared `rng`
    inside the threaded phase, so the draw ORDER varies with thread interleaving and
    `rng_seed` does not pin the result. Runs then differ in which rows get measured at all:
    at `(model = :gaussian_pct, sigma_pct = 2.0)`, depth 7, a `-t auto` run and a `-t 1`
    run left 11 and 8 rows unmeasured respectively from the same seed. Calibrate a noisy
    artifact with `-t 1` when it needs to be reproducible.

    NOTE: prefix sharing is exact only because the dynamics are deterministic — a node's
    state is fully determined by its action prefix, so children branch from one parent
    state. Under `noisy_thruster = true` each node would need several sampled children, and
    both the cost model and this sharing stop holding.

    NOTE: coverage is reachability-weighted, not uniform. States the vehicle passes through
    often get many samples; `BELOW_20` is not reached from the nominal orbit at all, which
    is why the family-member seeding below is retained.
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

    # ── Seed every action from every altitude bin ──────────────────────────────
    # Rows are keyed by the bin a pass STARTS in, so covering a bin means starting passes
    # in it. The primary walks all depart the ~37 km limit cycle and, once the aim is
    # accurate, they settle and never leave the bin they started in — so without seeding a
    # kernel describes one bin. A working controller has no reason to visit `BELOW_20`, yet
    # the solver queries those states, and filling them with a guess would teach the policy
    # something never measured.
    #
    # Seeds are spread across each bin (a bin is a range, and one point should not stand in
    # for it) but never state-space DISPERSED: a perturbation breaks the onboard apse
    # prediction rather than the orbit, so the kernel would record solver failures as
    # deaths. Reachability is the tree's job instead.
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
        # `nothing` when noise-free: burns applied ΔV exactly, so no σ was in effect and
        # recording one would imply a law this artifact was not measured under.
        "thruster_sigma_pct" => noisy_thruster ? thruster_sigma_pct : nothing,
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
            # Read this before reading a trial count. With `seed_samples = 1` and a
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

The distribution to use for a row the measurement never visited.

  - `rows` — all measured rows, to donate from
  - `action` — the action whose row is being filled
  - `key` — the unmeasured (altitude, residual) conditioning

Returns a probability vector over [`kernel_columns`](@ref).

The fill BACKS OFF ALONG THE DAMAGE AXIS: reuse the same action's measured row at the
nearest less-degraded residual bin. That is the conservative reading of the one
monotonicity the measurements support — more damage never makes an action safer — so it
understates risk rather than fabricating safety.

NOTE: do not "simplify" this to a self-transition. Filling an unmeasured row with "you
stay where you are" hands the solver an action that is free, perfectly safe, and (because
the coverage gate keys off the landed bin) often science-banking as well. A solver
maximising value will find that action and take it.

If no less-degraded row was measured either, it falls back to the self-transition because
there is genuinely nothing to infer from. `meta.unmeasured_rows` flags those; a policy
leaning on one means nothing.
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
[`write_tables`](@ref).

  - `rows` — per-action measured rows from [`calibrate_tables`](@ref)
  - `diagnostics` — the accompanying diagnostics dict, recorded into `meta`

Returns an [`AltTables`](@ref) with θ, effort, per-row trial counts and the list of
unmeasured rows in its metadata.

NOTE: a row with `n = 0` is not measured. Seeding covers the bins a controller never
visits, but many (altitude, residual) combinations are simply unreachable — the vehicle
cannot arrive low with a pristine solve after a violent transfer. Those are filled by
[`_fill_unmeasured_row`](@ref); read its docstring before changing the strategy, because
the obvious fill is actively dangerous.

NOTE: a policy is no more trustworthy than its least-visited row. Check `meta.trials`
against [`MIN_TRIALS_TRUSTED`](@ref) before believing what a policy does in a rare bin.
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
            "thruster_sigma_pct" => get(diagnostics, "thruster_sigma_pct", nothing),
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
