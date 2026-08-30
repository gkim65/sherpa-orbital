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
  Action  : OBSERVE / CORRECT / EXCURSE_{LOW,HIGH}. Actions encode INTENT only; the burn
            VECTOR is solved live by the planner (a fixed-direction menu was shown to
            fail — see experiments/studies exp 04). There is no EXCURSE_MID: CORRECT
            already holds ~31–37 km, which IS the middle band, so a MID excursion would
            duplicate it.
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
  bins BELOW_20 / A20_30 / A30_40 / A40_50 / ABOVE_50.
- `alt_rep_km`: representative altitude (km) per live bin — the mean of the nav
  observation model for that bin.

# Science bands
- `band_names`: the science bands, in visit-tuple order (index 1 = first band).
- `band_bins`: which altitude bin each band corresponds to. This is the coverage gate:
  an observed altitude banks a band when it lands in that band's bin.
- `band_target_km`: commanded periapsis-altitude target per band (km). ⚠️ Achieving X
  requires COMMANDING ≈ X − 6 (the measured `:position` residual floor, characterised and
  deliberately not fixed — `docs/session-log/2026-08-29-commanded-altitude-hold.md` §4d).
- `visit_cap`: max counted samples per band. Saturates the visit counts, so |S| grows as
  `(visit_cap + 1)^n_bands` — EXPONENTIAL in bands. `cap = 3` is 64 combos; a target of 20
  would be 9261 and needs a different encoding, not a bigger cap.

# Noise hyperparameters (swept)
- `sigma_nav_km`: 1σ Gaussian nav noise on the measured altitude (km). Ours is a
  deliberately conservative 2.0 vs MacKenzie §C.1.1.2's ~0.3–1 km. Drives the coverage
  misbin rate — see `alt_bin`.
- `thruster_model`: `:uniform` (legacy `η_eff ~ U(eta_eff_min, eta_eff_max)`) or
  `:gaussian_pct` (symmetric `η_eff ~ N(1, thruster_sigma_pct/100)`).
  ⚠️ The `:uniform` default is a KNOWN-WRONG model kept only for continuity: it is 0–20%
  underburn, mean 10% short, and NEVER over, whereas MacKenzie Exhibit B-24 (Cassini)
  reports a SYMMETRIC 1σ magnitude error of 0.7% (Model 1) / 2.0% (Model 2). That is ~10×
  too much error plus a systematic one-directional bias, and it is known to flip a 0/5
  survival result to 5/5 (`docs/todo.md`). Sweep it; do not trust the default.
- `thruster_sigma_pct`: 1σ magnitude error (%) for `:gaussian_pct`. B-24 presets are 0.7
  and 2.0.
- `eta_eff_min`, `eta_eff_max`: bounds for `:uniform`.

# Rewards
- `r_science`: reward for sampling a band not yet in `cov`.
- `r_step_ok`: small living reward for surviving a non-terminal step.
- `r_crashed`, `r_lost`: mission-loss penalties.
- `fuel_weight`: multiplies `action_dv_cost` (m/s) as a penalty.
- `action_dv_cost`: ΔV cost proxy (m/s) per action.

# Solver
- `discount`: POMDP discount factor.
"""
Base.@kwdef struct StationkeepingPOMDP
    # -- altitude discretization ----------------------------------------------
    # Edges at the science-range boundaries, so "in the 20–50 km science range" is exactly
    # the three middle bins and needs no arithmetic.
    alt_edges::NTuple{4,Float64}    = (20.0, 30.0, 40.0, 50.0)
    alt_rep_km::Dict{Symbol,Float64} = Dict(
        :BELOW_20 => 15.0, :A20_30 => 25.0, :A30_40 => 35.0,
        :A40_50   => 45.0, :ABOVE_50 => 60.0,
    )

    # -- science bands --------------------------------------------------------
    band_names::NTuple{3,Symbol}    = (:LOW, :MID, :HIGH)
    band_bins::NTuple{3,Symbol}     = (:A20_30, :A30_40, :A40_50)
    # Commanded periapsis altitude per band (km). These are REAL halo-family members
    # (the continued family spans 17.565–63.126 km, so all three exist) and the excursion
    # reference PERSISTS until CORRECT clears it — so they are reached by settling over
    # several passes, not in one impulse.
    # ⚠️ Do NOT re-tune these to match a single-pass achieved altitude. Single-impulse
    # authority is only ~25% (commanding 20 km from the 31 km limit cycle reaches ~38 km on
    # one pass), which is why the reference persists; "spend another cycle getting there"
    # is the mechanism, and lowering the command to flatter a one-pass number would remove
    # the very behaviour the policy is meant to learn.
    band_target_km::Dict{Symbol,Float64} =
        Dict(:LOW => 20.0, :MID => 30.0, :HIGH => 40.0)
    # ⚠️ 4, NOT 3. `visit_cap = 3` (322 states) makes NativeSARSOP die with
    # `InexactError: Int64(NaN)` inside its belief-binning bound initialization. Measured
    # 2026-08-30: it is NOT a size limit and NOT a malformed model — caps 1, 2, 4 and 5
    # all solve (42 / 137 / 627 / 1082 states), and cap 3 itself solves at discount = 0.90
    # or with r_science = 0, at every precision tried. T and O are finite, normalized, and
    # have no zero-mass observation column. It is a solver-side numerical edge case where
    # the upper and lower bounds coincide and a bin index goes 0/0.
    # 4 is chosen over 2 because it is the nearest working value ABOVE 3, so the science
    # resolution goes up rather than down. Revisit if NativeSARSOP is updated.
    visit_cap::Int                  = 4
    # The bin CORRECT already holds (nominal periapsis 30.98 km, limit cycle ~37 km), so
    # no EXCURSE action is generated for it. See `excursion_bands`.
    correct_bin::Symbol             = :A30_40

    # -- noise hyperparameters (swept) ----------------------------------------
    sigma_nav_km::Float64           = 2.0
    thruster_model::Symbol          = :uniform
    thruster_sigma_pct::Float64     = 2.0
    eta_eff_min::Float64            = ETA_EFF_MIN
    eta_eff_max::Float64            = ETA_EFF_MAX

    # -- rewards -------------------------------------------------------------
    r_science::Float64              =   20.0
    r_step_ok::Float64              =    0.5
    r_crashed::Float64              = -200.0
    r_lost::Float64                 = -200.0
    fuel_weight::Float64            =    1.0

    # ΔV cost proxy (m/s) per action.
    # ⚠️ PLACEHOLDERS, NOT MEASUREMENTS. The old 2.1 / 3.1 / 9.9 values were measured in
    # exp 12 for bands at 40 / 70 / 120 km, which no longer exist — the bands are now
    # 20–30 / 30–40 / 40–50 km and much closer together, so those numbers do not transfer.
    # CORRECT's 1.3 is the one survivor with support (exp 11b, ~1.3 m/s per step, and the
    # measured limit cycle sits at 1.32 m/s per pass). Re-measure with the kernels.
    action_dv_cost::Dict{Symbol,Float64} = Dict(
        :OBSERVE      => 0.0,
        :CORRECT      => 1.3,
        :EXCURSE_LOW  => 2.1,
        :EXCURSE_HIGH => 2.1,
    )

    # -- solver --------------------------------------------------------------
    discount::Float64               = 0.95

    # -- measured tables ------------------------------------------------------
    # Path to the measured T/O artifact. `nothing` uses the packaged default.
    tables_path::Union{Nothing,String} = nothing
end