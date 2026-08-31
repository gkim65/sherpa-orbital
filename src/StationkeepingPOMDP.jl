"""
StationkeepingPOMDP.jl — the science-vs-safety Enceladus stationkeeping POMDP.

The struct IS the configuration: every hyperparameter is a keyword field with a literal
default, so a scenario is one constructor call and nothing is hidden in globals.

    pomdp = StationkeepingPOMDP()                              # baseline scenario
    pomdp = StationkeepingPOMDP(; r_science = 40.0)             # value science more
    pomdp = StationkeepingPOMDP(; dev_edges = (10.0, 50.0, 150.0))  # tighter safety bins

Formulation
  State   : (alt, visits). `alt` = achieved periapsis-ALTITUDE bin, whose support also
            carries the two absorbing outcomes (CRASHED/LOST) — there is no separate
            safety variable. `visits` = per-band sample COUNT saturating at `visit_cap`,
            so revisiting a band keeps paying. |S| = 5*(cap+1)^n_bands + 2 = 322 by
            default.
  Action  : CORRECT / EXCURSE_{LOW,MID,HIGH}. Every action burns — there is no OBSERVE
            (removed 2026-08-30: a no-burn coast loses this orbit; see `actions`). Actions
            encode INTENT only; the burn VECTOR is solved live by the planner (a
            fixed-direction menu was shown to fail — see experiments/studies exp 04).
            All THREE science bands get an EXCURSE (|A| = 4): the limit cycle now lives in
            its own non-science bin, so no band duplicates what CORRECT already does.
  Obs     : noisy read of the achieved periapsis altitude (Gaussian nav noise, σ =
            `sigma_nav_km`), binned by `alt_bin`. Coverage is banked from this OBSERVED
            altitude, so it is deterministic given the observation but not correct — see
            `alt_bin` for the ~15–20% edge-driven misbin rate.
  Reward  : +r_science per band sample while under `visit_cap`, − fuel per burn, large −
            on CRASHED/LOST.

⚠️ 2026-08-30 REDESIGN. This replaces the (dev, cov) state space, whose observation was a
binned deviation NORM — direction-blind, so the policy structurally could not tell "6 km
high" from "6 km low" and could not act on altitude at all. See `states.jl` for the two
deliberate consequences (no `dev` dimension; coverage gated on the observed altitude).
⚠️ The T/O artifacts in `artifacts/tables.json` are calibrated against the OLD state space
and do NOT describe this one. Kernels must be re-measured before any result is quoted.

The transition/observation tables are measured artifacts, not analytic guesses. See
`tables.jl` and `artifacts/tables.json`.
"""

"""
    StationkeepingPOMDP(; kwargs...)

Configuration + model for the science/safety stationkeeping POMDP. All units are km,
km/s, m/s (ΔV costs), matching the Python truth model's conventions.

# Altitude discretization
- `alt_edges`: periapsis-altitude bin edges (km), half-open [lo, hi), giving the five live
  bins BELOW_20 / A20_27 / A27_34 / A34_44 / ABOVE_44. ⚠️ Rebased 2026-08-30 so the
  CORRECT limit cycle (37.17 km) sits in A34_44, which is NOT a science band.
- `alt_rep_km`: representative altitude (km) per live bin — the mean of the nav
  observation model for that bin.

# Science bands
- `band_names`: the science bands, in visit-tuple order (index 1 = first band).
- `band_bins`: which altitude bin each band corresponds to. This is the coverage gate:
  an observed altitude banks a band when it lands in that band's bin.
- `band_target_km`: commanded periapsis-altitude target per band (km), set to the BIN
  MIDPOINTS. ⚠️ The old "achieving X requires COMMANDING ≈ X − 6" no longer holds: that was
  the `:position` residual floor, and `EXCURSE_*` now targets the commanded altitude
  directly via `mode = :altitude_position`, which delivers it to ~0.2 km (commanded
  25/35/45 → achieved 25.07/34.94/44.68 km, measured 2026-08-30). Command what you want.
- `visit_cap`: max counted samples per band. Saturates the visit counts, so |S| grows as
  `(visit_cap + 1)^n_bands` — EXPONENTIAL in bands. `cap = 3` is 64 combos; a target of 20
  would be 9261 and needs a different encoding, not a bigger cap.

# Noise hyperparameters (swept)
- `sigma_nav_km`: 1σ Gaussian nav noise on the measured altitude (km). Ours is a
  deliberately conservative 2.0 vs MacKenzie §C.1.1.2's ~0.3–1 km. Drives the coverage
  misbin rate — see `alt_bin`.
- `noisy_thruster`: whether burns execute with error. `false` applies commanded ΔV exactly
  (the optimistic corner of the family); `true` routes through `apply_dv_noisy`.
  ⚠️ RECALIBRATION AXIS — changing it requires re-running `calibrate_tables`, unlike
  `sigma_nav_km`/`plume_gradient` which are analytic.
- `thruster_kwargs`: forwarded to `sample_eta_eff` to pick the error law — `model`
  (`:uniform` | `:gaussian_pct`), `sigma_pct`, `eta_min`, `eta_max`. MacKenzie Exhibit B-24
  (Cassini) reports a SYMMETRIC 1σ magnitude error of 0.7% (Model 1) / 2.0% (Model 2), so
  `(model = :gaussian_pct, sigma_pct = 2.0)` is the cited choice. The `:uniform` legacy law
  is 0–20% underburn, mean 10% short and NEVER over — ~10× B-24's error plus a
  one-directional bias — so do not use it for a quoted result.
- `plume_gradient`: θ, the altitude-gradient strength of plume intensity (softmax inverse
  temperature over band depth). 0 = no gradient, all bands identical. See `plume.jl`.
- `plume_levels`: k, the number of observed intensity levels. 1 disables the dimension.

# Rewards
- `r_science`: reward for sampling a band not yet in `cov`.
- `repeat_factor`: multiplies `r_science` for samples taken AFTER a band hits `visit_cap`.
  Replaces a hard zero — you do not stop collecting plume material after four passes.
- `r_step_ok`: small living reward for surviving a non-terminal step.
- `r_crashed`, `r_lost`: mission-loss penalties.
- `fuel_weight`: multiplies `action_dv_cost` (m/s) as a penalty.
- `action_dv_cost`: ΔV cost proxy (m/s) per action.

# Solver
- `discount`: POMDP discount factor.
"""
Base.@kwdef struct StationkeepingPOMDP
    # -- altitude discretization ----------------------------------------------
    # ⚠️ EDGES CHOSEN AGAINST THE MEASURED LIMIT CYCLE, not by round numbers (2026-08-30).
    # The previous (20, 30, 40, 50) edges put the CORRECT limit cycle INSIDE a science band:
    # noise-free the cycle holds 37.17 km with 96% of 295 passes inside 36–38 km, which sits
    # squarely in the old `A30_40` = MID band. So MID was banked by doing nothing, and with
    # the (known-unphysical, 10%-mean-underburn) `:uniform` thruster the cycle smears up to a
    # 40.55 km median and gifted the old `A40_50` = HIGH band too. Two of three bands were
    # free, so the policy collected 8 of 12 science points at zero risk and then — correctly —
    # refused the only band that cost anything. The science/safety tradeoff the POMDP exists
    # to study was largely vacuous.
    #
    # These edges instead BRACKET the limit cycle in its own band (`A34_44`, deliberately not
    # a science band) so every science band requires a real, chosen maneuver:
    #
    #   | bin      | range (km) | what reaches it                                  |
    #   |----------|------------|--------------------------------------------------|
    #   | BELOW_20 | < 20       | off-nominal low; holdable (18 km holds steady)    |
    #   | A20_27   | 20–27      | LOW band — a real excursion (P(LOST) ≈ 0.19)     |
    #   | A27_34   | 27–34      | MID band — a real excursion, below the cycle      |
    #   | A34_44   | 34–44      | THE LIMIT CYCLE. Not a science band.              |
    #   | ABOVE_44 | > 44       | HIGH band — above the cycle, still holdable (45)  |
    #
    # Constrained by measurement, not taste: the continued family spans 17.57–63.13 km, and
    # holding a member on its own altitude works at 18/20/22/25/28/32/45 km but ESCAPES by
    # pass 3 at 55 and 60 km. So the top band has to stay near 45 km, and 50+ is not a science
    # band that can be held.
    alt_edges::NTuple{4,Float64}    = (20.0, 27.0, 34.0, 44.0)
    alt_rep_km::Dict{Symbol,Float64} = Dict(
        :BELOW_20 => 18.0, :A20_27 => 23.5, :A27_34 => 30.5,
        :A34_44   => 37.2, :ABOVE_44 => 46.0,
    )

    # -- science bands --------------------------------------------------------
    band_names::NTuple{3,Symbol}    = (:LOW, :MID, :HIGH)
    # ⚠️ HIGH IS `ABOVE_44`, NOT the limit-cycle bin. `A34_44` holds the CORRECT cycle
    # (37.17 km) and is deliberately absent here — a bin the controller already occupies
    # cannot be a science objective, which is the whole point of the `alt_edges` rebase.
    band_bins::NTuple{3,Symbol}     = (:A20_27, :A27_34, :ABOVE_44)
    # Commanded periapsis altitude per band (km). These are REAL halo-family members
    # (the continued family spans 17.565–63.126 km, so all three exist) and the excursion
    # reference PERSISTS until CORRECT clears it — so they are reached by settling over
    # several passes, not in one impulse.
    #
    # ⚠️ ONE COMMAND PER BAND, PLACED INSIDE ITS BIN AND AWAY FROM ITS EDGES (2026-08-30).
    # Commanding a bin EDGE was a real defect: `EXCURSE_*` now delivers the commanded
    # altitude to ~0.2 km, so a command at 20.0 for the [20,27) bin lands ~0.1 km from the
    # boundary and nav noise misbins it constantly. (Under the old ~25%-authority targeting
    # this was invisible — every command landed near 37 km regardless.) Interior points:
    #
    #   | band | bin       | commanded | measured achieved            |
    #   |------|-----------|-----------|------------------------------|
    #   | LOW  | [20, 27)  |   23.5 km | 25 km delivered to 0.2 km    |
    #   | MID  | [27, 34)  |   30.5 km | 30 km delivered to 0.09 km   |
    #   | HIGH | [44, ∞)   |   46.0 km | 45 km delivered to 0.3 km    |
    #
    # ⚠️ HIGH IS CONSTRAINED FROM ABOVE BY THE ORBIT, not by preference. Holding a family
    # member on its own altitude works at 45 km but ESCAPES by pass 3 at 55 and 60 km, so
    # 46 km is near the top of what can actually be held. Do not raise it without measuring.
    #
    # ⚠️ AND LOW IS THE EXPENSIVE ONE: `EXCURSE` from `A20_27` measures P(LOST) ≈ 0.19,
    # because the transfer down from the 37 km cycle drives apoapsis past MacKenzie's 1110 km
    # ceiling and the onboard model then loses the apse pair (ΔV = 0, uncontrolled pass).
    # That risk is real and measured — it is not a targeting defect to be tuned away.
    #
    # ⚠️ `band_bins` — not this — is what the state space, action set and rewards key off
    # (see `states.jl` `bank_visit`, `actions.jl` `excursion_bands`). Changing these commanded
    # altitudes leaves |S|, |A| and the reward structure untouched.
    band_target_km::Dict{Symbol,Float64} =
        Dict(:LOW => 23.5, :MID => 30.5, :HIGH => 46.0)
    # ⚠️ THE `InexactError: Int64(NaN)` THIS ONCE WORKED AROUND IS NOW DIAGNOSED, and the
    # workaround is no longer what avoids it — pass `use_binning = false` to `SARSOPSolver`
    # (see `experiments/example.jl`). Root cause, found 2026-08-30:
    #
    # `NativeSARSOP.entropy` has two methods and only the dense one guards the `p log p`
    # term. `tree.b` is hard-typed `Vector{SparseVector}`, so the UNGUARDED one always runs:
    #
    #     entropy(b::AbstractVector) : b_i > 0 && (ent -= b_i * log(b_i))       # correct
    #     entropy(b::SparseVector)   : for b_i in b.nzval; ent -= b_i*log(b_i)  # no guard
    #
    # `nzval` is the STRUCTURAL sparsity pattern, so an entry can sit in it at exactly 0.0.
    # `tree.jl`'s belief update (`bp.nzval ./= po`, ~line 308) has no `dropzeros!` after the
    # observation-weighting step, unlike the prediction step just above it — so an entry
    # zeroed by an observation column stays in the pattern and `0.0 * log(0.0) = NaN`. That
    # NaN reaches `get_interval_idx`, where `Int(floor(NaN/interval) + 1)` throws. Reproduced
    # in three lines with no dynamics involved:
    #
    #     v = SparseVector(3,[1,2,3],[0.5,0.5,0.0]); entropy(v)   # NaN
    #     entropy(Vector(v))                                      # 0.693
    #
    # `entropy` is reached ONLY from the two `use_binning` call sites, which is why disabling
    # binning fixes it outright. Binning is a search heuristic for grouping similar beliefs;
    # switching it off changes the exploration bookkeeping, not the POMDP or the solution.
    #
    # The old note here guessed "the upper and lower bounds coincide and a bin index goes
    # 0/0", and also asserted the model has "no zero-mass observation column" — both wrong.
    # `observation_matrix` gives every LIVE state exactly 0.0 in the CRASHED and LOST
    # columns, deliberately and correctly (a live pass is not a crash; those are categorical
    # events, not noisy altitude reads), and that is precisely the zero involved.
    #
    # Why caps 1/2/4/5 solved and 3 did not is then incidental, not structural: the cap
    # changes which beliefs SARSOP explores, hence whether it happens to step on a zeroed
    # entry. Same reason `r_science = 0` "fixed" it. Do NOT detune the science objective to
    # dodge this — that changes the problem being solved.
    visit_cap::Int                  = 4
    # The bin CORRECT already holds — measured 37.17 km noise-free, 96% of 295 passes inside
    # 36–38 km. It is `A34_44`, which is NOT in `band_bins`, so `excursion_bands` generates an
    # EXCURSE for all THREE science bands now (|A| = 4: CORRECT + LOW/MID/HIGH). Under the old
    # edges this bin WAS the MID band, which is why MID had no action and was banked free.
    correct_bin::Symbol             = :A34_44

    # -- noise hyperparameters (swept) ----------------------------------------
    sigma_nav_km::Float64           = 2.0
    # ⚠️ REPLACED `thruster_model` / `thruster_sigma_pct` / `eta_eff_min` / `eta_eff_max`
    # (2026-08-31). Those four fields were DEAD — grep found no reader anywhere in `src/`
    # outside this struct, so sweeping them changed nothing (measured: 1/3 distinct policies
    # across thruster_sigma_pct ∈ {0.7, 2.0, 5.0}). They duplicated, and silently disagreed
    # with, the arguments `calibrate_tables` actually uses.
    #
    # `noisy_thruster` is the live path: `calibrate_tables` routes burns through
    # `apply_dv_noisy` when it is true, so this genuinely re-measures T. `thruster_kwargs`
    # forwards to `sample_eta_eff` (`model`, `sigma_pct`, `eta_min`, `eta_max`), keeping the
    # B-24 presets reachable without four more struct fields to fall out of sync.
    #
    # ⚠️ THIS IS A RECALIBRATION AXIS, not an analytic one — see `needs_recalibration`.
    # Measured consequence of turning it on (2026-08-30): P(LOST) for LOW/MID goes 0.19 /
    # 0.056 → 1.00 / 1.00 over 12 seeds. Noise-free is a legitimate optimistic θ, not a bug,
    # but it is NOT the deployment environment.
    noisy_thruster::Bool            = false
    thruster_kwargs::NamedTuple     = NamedTuple()
    # ⚠️ THE THIRD SWEEP AXIS, SHAPED LIKE THE TWO ABOVE — one scalar, no special-casing.
    # `plume_gradient` (θ) is the inverse temperature of a softmax over normalized band
    # DEPTH; see `plume.jl` for the functional form and why softmax rather than a linear
    # tilt. θ = 0 makes every band's intensity distribution IDENTICAL and uniform, which is
    # the sanity gate: with no gradient there is no reason to prefer any altitude. Rising θ
    # concentrates intensity mass toward the LOW band.
    #
    # ⚠️ UNLIKE the two axes above, this one changes `T`, so a sweep over it needs its own
    # `calibrate_tables` call per θ (~2 min). `sigma_nav_km` only enters the ANALYTIC
    # `observation_matrix` and needs no re-measurement at all. Cost a sweep accordingly.
    plume_gradient::Float64         = 0.0
    # Number of observed intensity levels a pass can yield. `1` disables the dimension
    # entirely (|S| unchanged), so the pre-intensity model stays reachable for comparison.
    # ⚠️ |S| scales LINEARLY in this: 5*(cap+1)^bands*k + 2, so k=3 is 1877 and k=5 is 3127.
    plume_levels::Int               = 3

    # -- rewards -------------------------------------------------------------
    r_science::Float64              =   20.0
    r_step_ok::Float64              =    0.5
    r_crashed::Float64              = -200.0
    r_lost::Float64                 = -200.0
    # ⚠️ ZEROED 2026-08-30 (user's call). The study is SCIENCE YIELD UNDER ENVIRONMENTAL
    # UNCERTAINTY, not fuel feasibility, and zeroing the weight sidesteps an unresolved
    # ΔV-budget question rather than pretending to answer it. The fuel machinery is
    # deliberately LEFT INTACT — `action_dv_cost` is still measured and still multiplied
    # here — so restoring the tradeoff is a one-field change, not a re-implementation.
    fuel_weight::Float64            =    0.0
    # Diminishing-returns factor for samples past `visit_cap`. ⚠️ REPLACES A HARD CLIFF: the
    # reward is `visit_total(sp) - visit_total(s)` and the counts SATURATE, so before this
    # existed visits 1–4 paid full `r_science` and visit 5+ paid exactly zero — once all
    # three bands capped, NO action earned science and the policy was indifferent between
    # sampling and not. 0.2 keeps long rollouts informative and is the more honest model.
    repeat_factor::Float64          =    0.2

    # ΔV cost proxy (m/s) per action.
    # ⚠️ MEASURED, not placeholders (2026-08-30). Per-pass ΔV medians from the calibration
    # run that produced the committed kernels — `experiments/calibrate.jl` prints these, so
    # they are re-derivable rather than transcribed. The old 1.3 / 2.1 / 2.1 values predated
    # both the band rebase and `:altitude_position`; EXCURSE_MID did not exist at all,
    # because the limit-cycle bin used to BE the MID band.
    #
    #   | action       |  n | median | mean  |
    #   |--------------|----|--------|-------|
    #   | CORRECT      | 90 |  1.278 | 1.314 |
    #   | EXCURSE_LOW  | 24 |  3.292 | 3.443 |
    #   | EXCURSE_MID  | 36 |  3.368 | 3.562 |
    #   | EXCURSE_HIGH | 48 |  5.378 | 5.824 |
    #
    # Medians, not means: the means are pulled up by the first pass of each settling walk,
    # which costs more than the steady-state hold this proxy is meant to price.
    #
    # ⚠️ HIGH IS THE EXPENSIVE BAND BUT ALSO THE SAFE ONE — measured P(LOST) = 0.0 over 45
    # trials, against 0.19 for LOW at half the ΔV. That asymmetry is the tradeoff the policy
    # exists to make; do not flatten these costs to equal values.
    action_dv_cost::Dict{Symbol,Float64} = Dict(
        :CORRECT      => 1.278,
        :EXCURSE_LOW  => 3.292,
        :EXCURSE_MID  => 3.368,
        :EXCURSE_HIGH => 5.378,
    )

    # -- solver --------------------------------------------------------------
    discount::Float64               = 0.95

    # -- measured tables ------------------------------------------------------
    # Path to the measured T/O artifact. `nothing` uses the packaged default.
    tables_path::Union{Nothing,String} = nothing
end