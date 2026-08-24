"""
StationkeepingPOMDP.jl — the science-vs-safety Enceladus stationkeeping POMDP.

The struct IS the configuration: every hyperparameter is a keyword field with a literal
default, so a scenario is one constructor call and nothing is hidden in globals.

    pomdp = StationkeepingPOMDP()                              # baseline scenario
    pomdp = StationkeepingPOMDP(; r_science = 40.0)             # value science more
    pomdp = StationkeepingPOMDP(; dev_edges = (10.0, 50.0, 150.0))  # tighter safety bins

Formulation
  State   : (dev, cov). `dev` = apse-position deviation bin (OK/DRIFT/FAR, + terminal
            LOST/CRASHED) — the SAFETY variable. `cov` = 3-bit mask over science
            altitude bands (LOW/MID/HIGH) — the SCIENCE variable. |S| = 3*8 + 2 = 26.
  Action  : OBSERVE / CORRECT / EXCURSE_{LOW,MID,HIGH}. Actions encode INTENT only; the
            burn VECTOR is solved live by the planner (a fixed-direction menu was shown
            to fail — see experiments/studies exp 04).
  Obs     : noisy read of the dev bin (Gaussian nav noise on the apse deviation). `cov`
            is known exactly, since we know which excursions we commanded.
  Reward  : +r_science per NEWLY sampled band, − fuel per burn, large − on CRASHED/LOST.

The transition/observation tables are measured artifacts, not analytic guesses. See
`tables.jl` and `artifacts/tables.json`.
"""

"""
    StationkeepingPOMDP(; kwargs...)

Configuration + model for the science/safety stationkeeping POMDP. All units are km,
km/s, m/s (ΔV costs), matching the Python truth model's conventions.

# Safety (dev) discretization
- `dev_edges`: apse-deviation bin edges (km). Half-open [lo, hi): a deviation below
  `dev_edges[1]` is OK, below `[2]` is DRIFT, below `[3]` is FAR, else LOST.
- `sigma_nav_km`: 1-sigma Gaussian nav noise on the measured apse deviation (km).
- `dev_rep_km`: representative deviation (km) per non-terminal bin, used as the mean of
  the nav observation model.

# Science (cov) bands
- `band_names`: the science altitude bands, in bit order (bit 1 = first band).
- `band_target_km`: commanded periapsis-altitude target per band (km).

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
    # -- safety discretization ------------------------------------------------
    dev_edges::NTuple{3,Float64}    = (15.0, 60.0, 200.0)
    sigma_nav_km::Float64           = 2.0
    dev_rep_km::Dict{Symbol,Float64} = Dict(:OK => 7.0, :DRIFT => 35.0, :FAR => 120.0)

    # -- science bands --------------------------------------------------------
    band_names::NTuple{3,Symbol}    = (:LOW, :MID, :HIGH)
    band_target_km::Dict{Symbol,Float64} =
        Dict(:LOW => 40.0, :MID => 70.0, :HIGH => 120.0)

    # -- rewards -------------------------------------------------------------
    r_science::Float64              =   20.0
    r_step_ok::Float64              =    0.5
    r_crashed::Float64              = -200.0
    r_lost::Float64                 = -200.0
    fuel_weight::Float64            =    1.0

    # ΔV cost proxy (m/s) per action. CORRECT from exp 11b (~1.3/step); EXCURSE costs
    # (incl. the recovery burn) measured in exp 12: LOW 2.1, MID 3.1, HIGH 9.9 m/s.
    action_dv_cost::Dict{Symbol,Float64} = Dict(
        :OBSERVE      => 0.0,
        :CORRECT      => 1.3,
        :EXCURSE_LOW  => 2.1,
        :EXCURSE_MID  => 3.1,
        :EXCURSE_HIGH => 9.9,
    )

    # -- solver --------------------------------------------------------------
    discount::Float64               = 0.95

    # -- measured tables ------------------------------------------------------
    # Path to the measured T/O artifact. `nothing` uses the packaged default.
    tables_path::Union{Nothing,String} = nothing
end