"""
model.jl — assemble the config + measured tables into a solvable POMDP.

The seam between a scenario (a `StationkeepingPOMDP` config plus measured `AltTables`) and
something a solver can consume (a QuickPOMDP). The library declares no solver dependency,
so the choice of solver belongs to the caller in `experiments/`.
"""

"""
    build_pomdp(config = StationkeepingPOMDP(); tables = nothing) -> QuickPOMDP

Assemble the science/safety stationkeeping POMDP.

  - `config` — the scenario; every hyperparameter is a field with a literal default
  - `tables` — measured kernels; `nothing` loads the artifact at `config.tables_path`, or
    the packaged default

Returns a `QuickPOMDP` over `SKState`.

    using SherpaOrbital, NativeSARSOP
    pomdp  = build_pomdp(StationkeepingPOMDP(; r_science = 40.0))
    policy = solve(SARSOPSolver(; precision = 1e-3, use_binning = false), pomdp)

NOTE: `use_binning = false` is required for NativeSARSOP on this model — see `visit_cap`
in `StationkeepingPOMDP.jl`.
"""
function build_pomdp(config::StationkeepingPOMDP = StationkeepingPOMDP();
                     tables::Union{Nothing,AltTables} = nothing)
    tbl = tables === nothing ?
        load_tables(something(config.tables_path, DEFAULT_TABLES_PATH)) : tables

    S = states(config)
    A = actions(config)
    Ω = observations(config)

    T = transition_matrix(config, tbl)
    O = observation_matrix(config)
    r = reward_function(config, T)

    sidx = state_index(config)
    aidx = action_index(config)

    return QuickPOMDP(
        states        = S,
        actions       = A,
        observations  = Ω,
        discount      = config.discount,
        isterminal    = s -> isterminal_state(s),

        # Start on the nominal orbit: the `correct_bin` altitude, nothing banked, and an
        # undamaged residual. There is no separate "on target" state — altitude is the
        # state variable, and starting anywhere but `:R_OK` would presume damage the
        # mission has not yet accumulated.
        initialstate  = Deterministic(
            SKState(config.correct_bin, _zero_visits(config), 1, :R_OK)),

        transition    = (s, a)  -> SparseCat(S, T[sidx[s], aidx[a], :]),

        # The observation depends only on the landed state's altitude bin.
        observation   = (a, sp) -> SparseCat(Ω, O[sidx[sp], :]),

        reward        = r,
    )
end

"""
    model_tables(config = StationkeepingPOMDP(); tables = nothing) -> (T, O)

The raw transition and observation arrays for a config, without building a POMDP.

  - `config` — the scenario
  - `tables` — measured kernels; `nothing` loads the artifact at `config.tables_path`

Returns `(T, O)`, sized `|S| x |A| x |S|` and `|S| x |O|`. Useful for exporting the model
or checking a hand-edited artifact.
"""
function model_tables(config::StationkeepingPOMDP = StationkeepingPOMDP();
                      tables::Union{Nothing,AltTables} = nothing)
    tbl = tables === nothing ?
        load_tables(something(config.tables_path, DEFAULT_TABLES_PATH)) : tables
    return transition_matrix(config, tbl), observation_matrix(config)
end