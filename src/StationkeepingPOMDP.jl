"""
StationkeepingPOMDP.jl — the science-vs-safety Enceladus stationkeeping POMDP.

The struct IS the configuration: every hyperparameter is a keyword field with a literal
default, so a scenario is one constructor call and nothing is hidden in globals.

    pomdp = StationkeepingPOMDP()                              # baseline scenario
    pomdp = StationkeepingPOMDP(; r_science = 40.0)             # value science more
    pomdp = StationkeepingPOMDP(; plume_gradient = 4.0)         # steep plume gradient

Formulation
  State   : (alt, visits, intensity, residual). `alt` = achieved periapsis-ALTITUDE bin,
            whose support also carries the two absorbing outcomes (CRASHED/LOST) — there
            is no separate safety variable. `visits` = per-band sample COUNT saturating at
            `visit_cap`, so revisiting a band keeps paying. `intensity` = the plume sample
            the pass yielded; `residual` = the orbit-damage bin. |S| = 5627 by default.
  Action  : CORRECT / EXCURSE_{LOW,MID,HIGH}, |A| = 4. Every action burns — there is no
            OBSERVE, because a no-burn coast loses this orbit (see `actions`). Actions
            encode INTENT only; the burn VECTOR is solved live by the planner.
  Obs     : noisy read of the achieved periapsis altitude (Gaussian nav noise, σ =
            `sigma_nav_km`), binned by `alt_bin`. Coverage is banked from this OBSERVED
            altitude, so it is deterministic given the observation but not correct — see
            `alt_bin` for the ~15–20% edge-driven misbin rate.
  Reward  : +r_science per band sample while under `visit_cap`, − fuel per burn, large −
            on CRASHED/LOST.

The transition/observation tables are measured artifacts, not analytic guesses. See
`tables.jl` and `artifacts/tables.json`.
"""

"""
    StationkeepingPOMDP(; kwargs...)

Configuration + model for the science/safety stationkeeping POMDP. All units are km,
km/s, m/s (ΔV costs), matching the Python truth model's conventions.

# Altitude discretization
- `alt_edges`: periapsis-altitude bin edges (km), half-open [lo, hi), giving the five live
  bins BELOW_20 / A20_27 / A27_34 / A34_44 / ABOVE_44. Chosen so the CORRECT limit cycle
  (37.17 km) sits in A34_44, as the default nominal orbit rather than a science band.
- `alt_rep_km`: representative altitude (km) per live bin — the mean of the nav
  observation model for that bin.

# Science bands
- `band_names`: the science bands, in visit-tuple order (index 1 = first band).
- `band_bins`: which altitude bin each band corresponds to. This is the coverage gate:
  an observed altitude banks a band when it lands in that band's bin.
- `band_target_km`: commanded periapsis-altitude target per band (km), set to the BIN
  MIDPOINTS. `EXCURSE_*` targets the commanded altitude directly via
  `mode = :altitude_position`, which delivers it to ~0.2 km — command what you want.
- `visit_cap`: max counted samples per band. Saturates the visit counts, so |S| grows as
  `(visit_cap + 1)^n_bands` — EXPONENTIAL in bands. `cap = 3` is 64 combos; a target of 20
  would be 9261 and needs a different encoding, not a bigger cap.

# Noise hyperparameters (swept)
- `sigma_nav_km`: 1σ Gaussian nav noise on the measured altitude (km). Deliberately
  conservative — see `SIGMA_NAV_POS` in `constants.jl`. Drives the coverage misbin rate,
  see `alt_bin`.
- `noisy_thruster`: whether burns execute with error. `false` applies commanded ΔV exactly;
  `true` routes through `apply_dv_noisy`. NOTE: recalibration axis — changing it requires
  re-running `calibrate_tables`, unlike `sigma_nav_km`/`plume_gradient` which are analytic.
- `thruster_kwargs`: forwarded to `sample_eta_eff` to pick the error law — `model`
  (`:uniform` | `:gaussian_pct`), `sigma_pct`, `eta_min`, `eta_max`. See `thruster.jl` for
  which law is defensible and which is the default.
- `plume_gradient`: θ, the altitude-gradient strength of plume intensity (slope of a linear
  tilt in band depth). 0 = no gradient, all bands identical. See `plume.jl`.
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
    # THE EDGES BRACKET THE CORRECT LIMIT CYCLE IN ITS OWN NON-SCIENCE BIN (`A34_44`), so
    # every science band requires a real, chosen maneuver. Edges that put the cycle inside a
    # science band make that band free and the science/safety tradeoff vacuous.
    #
    #   | bin      | range (km) | what reaches it                                  |
    #   |----------|------------|--------------------------------------------------|
    #   | BELOW_20 | < 20       | off-nominal low; holdable (18 km holds steady)    |
    #   | A20_27   | 20–27      | LOW band — a real excursion (P(LOST) ≈ 0.19)     |
    #   | A27_34   | 27–34      | MID band — a real excursion, below the cycle      |
    #   | A34_44   | 34–44      | THE LIMIT CYCLE. Not a science band.              |
    #   | ABOVE_44 | > 44       | HIGH band — above the cycle, still holdable (45)  |
    #
    # THE TOP BAND IS CONSTRAINED BY THE ORBIT. The continued family spans 17.57–63.13 km,
    # but holding a member on its own altitude works only up to ~45 km — 55 and 60 km escape
    # by pass 3. Do not move the top edge up without measuring.
    alt_edges::NTuple{4,Float64}    = (20.0, 27.0, 34.0, 44.0)
    alt_rep_km::Dict{Symbol,Float64} = Dict(
        :BELOW_20 => 18.0, :A20_27 => 23.5, :A27_34 => 30.5,
        :A34_44   => 37.2, :ABOVE_44 => 46.0,
    )

    # -- science bands --------------------------------------------------------
    band_names::NTuple{3,Symbol}    = (:LOW, :MID, :HIGH)
    # HIGH IS `ABOVE_44`, NOT the limit-cycle bin. `A34_44` holds the CORRECT cycle
    # (37.17 km) and is deliberately absent here — a bin the controller already occupies
    # cannot be a science objective.
    band_bins::NTuple{3,Symbol}     = (:A20_27, :A27_34, :ABOVE_44)
    # Commanded periapsis altitude per band (km). All three are realisable halo-family
    # members, and the excursion reference PERSISTS until CORRECT clears it — so a band is
    # reached by settling over several passes, not in one impulse.
    #
    # EACH COMMAND SITS IN ITS BIN'S INTERIOR, AWAY FROM THE EDGES. Delivery is ~0.2 km,
    # so a command placed on a boundary lands close enough to it that nav noise misbins the
    # pass constantly.
    #
    # HIGH IS CONSTRAINED FROM ABOVE BY THE ORBIT, not by preference — 46 km is near the
    # top of what can be held (55 and 60 km escape by pass 3). Do not raise it without
    # measuring.
    #
    # AND LOW IS THE EXPENSIVE ONE: `EXCURSE` into `A20_27` measures P(LOST) ≈ 0.19,
    # because the transfer down from the 37 km cycle drives apoapsis past MacKenzie's 1110 km
    # ceiling and the onboard model then loses the apse pair (ΔV = 0, uncontrolled pass).
    # That risk is real and measured — it is not a targeting defect to be tuned away.
    #
    # `band_bins` — not this — is what the state space, action set and rewards key off
    # (see `actions.jl` `excursion_bands`). Changing these commanded altitudes leaves |S|,
    # |A| and the reward structure untouched.
    band_target_km::Dict{Symbol,Float64} =
        Dict(:LOW => 23.5, :MID => 30.5, :HIGH => 46.0)
    # NOTE: solve with `use_binning = false` (see `experiments/example.jl`).
    # NativeSARSOP's `entropy(::SparseVector)` does not guard the `p log p` term and
    # iterates the structural sparsity pattern, which holds entries at exactly 0.0 —
    # `observation_matrix` gives every live state 0.0 in the CRASHED and LOST columns. That
    # yields a NaN entropy and an `InexactError: Int64(NaN)`. Binning is a belief-grouping
    # search heuristic, so disabling it changes the exploration bookkeeping, not the POMDP
    # or its solution. Detuning `visit_cap` or `r_science` also silences the error, by
    # changing which beliefs get explored; that changes the problem being solved.
    visit_cap::Int                  = 4
    # Altitude bin the CORRECT limit cycle settles in (37.17 km, 96% of passes in
    # 36-38 km). Not in `band_bins`, so `excursion_bands` gives all three science bands an
    # EXCURSE action (|A| = 4).
    correct_bin::Symbol             = :A34_44

    # -- noise hyperparameters (swept) ----------------------------------------
    sigma_nav_km::Float64           = 2.0
    # Whether calibration burns execute with error. `thruster_kwargs` forwards to
    # `sample_eta_eff` (`model`, `sigma_pct`, `eta_min`, `eta_max`) to pick the error law.
    #
    # NOTE: recalibration axis — changing either forces a fresh `calibrate_tables` run, see
    # `needs_recalibration`. Turning noise on raises measured P(LOST) for the LOW and MID
    # excursions sharply. Noise-free is the optimistic corner of the family, not the
    # deployment environment.
    noisy_thruster::Bool            = false
    thruster_kwargs::NamedTuple     = NamedTuple()
    # Slope of a linear tilt in band depth, centered on the mean depth over `ALT_BINS`; see
    # `plume.jl` for the functional form. 0 makes every bin's intensity distribution
    # identical and uniform; rising values shift intensity toward the deeper bins and away
    # from the shallower ones, holding the five-bin mean fixed.
    #
    # NOTE: usable range is [0, 4]. The tilt is clamped to keep probabilities non-negative,
    # so the shallowest bin pins at about 1.87 and the two deepest by 4.
    #
    # NOTE: enters `transition_matrix` analytically, so a sweep needs no recalibration.
    plume_gradient::Float64         = 0.0
    # Number of observed intensity levels a pass can yield. `1` disables the dimension.
    #
    # NOTE: |S| scales linearly in this, so it is not a legal sweep axis — S must be fixed
    # across a POMDP family.
    plume_levels::Int               = 3
    # Science value of the weakest intensity level; the strongest is 1.0 and the levels
    # between are evenly spaced. At k = 3 this gives 0.3 / 0.65 / 1.0.
    #
    # NOTE: must stay nonzero. Level 1 is also what a pass outside every science band
    # carries, so a zero floor makes a CORRECT pass earn nothing regardless of the rest of
    # the reward.
    #
    # NOTE: a modelling choice, not a measured quantity — Cassini gives no
    # yield-vs-altitude table at these altitudes. Sweep it rather than trusting it.
    intensity_value_min::Float64    = 0.3
    # Science yield multiplier by orbit-damage bin, in `RESIDUAL_BINS` order. A sample
    # taken from a degraded orbit is worth less: pointing and geometry are worse. Same
    # nonzero-floor-to-1.0 scale as `intensity_value_min`, so one damage level and one
    # intensity level are comparable in magnitude.
    #
    # NOTE: keys on the SUCCESSOR's damage, not the departing state (see `rewards.jl`). A
    # departing-state penalty is identical across actions and cancels out of the comparison.
    #
    # NOTE: a modelling choice, not measured. Set all three to 1.0 to disable the term.
    damage_yield::NTuple{3,Float64} = (1.0, 0.65, 0.3)

    # -- rewards -------------------------------------------------------------
    r_science::Float64              =   20.0
    r_step_ok::Float64              =    0.5
    r_crashed::Float64              = -200.0
    r_lost::Float64                 = -200.0
    # Multiplies `action_dv_cost` as a penalty. Zero by default: the study is science
    # yield under environmental uncertainty, not fuel feasibility. The fuel machinery is
    # intact, so raising this restores the tradeoff with no other change.
    fuel_weight::Float64            =    0.0
    # Diminishing-returns factor for samples taken after a band hits `visit_cap`.
    #
    # NOTE: must stay nonzero. The science reward is `visit_total(sp) - visit_total(s)` and
    # the counts saturate, so at zero no action earns science once every band has capped and
    # the policy goes indifferent between sampling and not.
    repeat_factor::Float64          =    0.2

    # ΔV cost proxy (m/s) per action. Per-pass MEDIANS, not means. 
    # To regenerate, run `julia --project=experiments -t auto experiments/calibrate.jl`.
    # It prints n, median and mean per action under "Measured ΔV per action (m/s)"; copy
    # the median column here.
    #
    # NOTE: only consumer is `rewards.jl`, multiplied by `fuel_weight`, which defaults to
    # 0.0 — so these are inert unless the fuel tradeoff is switched on. They are still
    # written into the exported policy artifact by `export.jl`.
    #
    # NOTE: LOW is both the most expensive excursion and the most dangerous (measured
    # P(LOST) = 0.19, against 0.0 for HIGH). Do not flatten these to equal values.
    action_dv_cost::Dict{Symbol,Float64} = Dict(
        :CORRECT      => 1.502,
        :EXCURSE_LOW  => 4.862,
        :EXCURSE_MID  => 3.565,
        :EXCURSE_HIGH => 3.498,
    )

    # -- solver --------------------------------------------------------------
    discount::Float64               = 0.95

    # -- measured tables ------------------------------------------------------
    # Path to the measured T/O artifact. `nothing` uses the packaged default.
    tables_path::Union{Nothing,String} = nothing
end