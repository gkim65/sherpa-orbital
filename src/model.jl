"""
model.jl — assemble the config + measured tables into a solvable POMDP.

This is the seam between "the scenario" (a `StationkeepingPOMDP` config plus a measured
`AltTables`) and "a thing a solver can consume" (a QuickPOMDP). Keeping the solver out of
this file is deliberate: the library declares no solver dependency, so the choice of
solver belongs to the caller in `experiments/`.
"""

"""
    build_pomdp(config = StationkeepingPOMDP(); tables = nothing) -> QuickPOMDP

Assemble the science/safety stationkeeping POMDP. `tables` defaults to the measured
artifact at `config.tables_path` (or the packaged default).

    using SherpaOrbital, NativeSARSOP
    pomdp  = build_pomdp(StationkeepingPOMDP(; r_science = 40.0))
    policy = solve(SARSOPSolver(; precision = 1e-3), pomdp)
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

        # Start on the nominal orbit, no science banked yet. Nominal periapsis is
        # 30.98 km, so the initial altitude bin is `correct_bin` (A30_40) — NOT a
        # separate "on target" state, since altitude is now the state variable.
        initialstate  = Deterministic(
            SKState(config.correct_bin, _zero_visits(config))),

        transition    = (s, a)  -> SparseCat(S, T[sidx[s], aidx[a], :]),

        # The observation depends only on the landed state's altitude bin.
        observation   = (a, sp) -> SparseCat(Ω, O[sidx[sp], :]),

        reward        = r,
    )
end

"""
    model_tables(config = StationkeepingPOMDP(); tables = nothing) -> (T, O)

The raw T[s,a,s'] and O[s,o] arrays for a config, without building a POMDP. Useful for
exporting the model, or for checking a hand-edited artifact.
"""
function model_tables(config::StationkeepingPOMDP = StationkeepingPOMDP();
                      tables::Union{Nothing,AltTables} = nothing)
    tbl = tables === nothing ?
        load_tables(something(config.tables_path, DEFAULT_TABLES_PATH)) : tables
    return transition_matrix(config, tbl), observation_matrix(config)
end