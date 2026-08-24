"""
SherpaOrbital — offline POMDP stationkeeping for the Enceladus Orbilander.

SHERPA-RPA Direction 3. The spacecraft holds a period-3 L1 halo orbit around Enceladus
and must stationkeep without ground contact, trading SCIENCE (sampling a range of
periapsis altitudes) against SAFETY (not crashing or escaping an unstable orbit).

    using SherpaOrbital, NativeSARSOP

    pomdp  = build_pomdp(StationkeepingPOMDP(; r_science = 40.0))
    policy = solve(SARSOPSolver(; precision = 1e-3), pomdp)
    print_policy_table(policy)

The library is solver-agnostic: no solver is a dependency, so `experiments/` chooses one.

References
  MacKenzie et al. (2020), Enceladus Orbilander Mission Concept Study — orbit and
    thruster parameters, §B.2.3.
  Kim et al. (2025) — the POMDP/Bayesian-network formulation style this follows.
"""
module SherpaOrbital

using Distributions
using JSON
using OrdinaryDiffEq
using POMDPs
using POMDPTools
using QuickPOMDPs
using Printf

# Physics layer: constants, then dynamics (truth/onboard split preserved), then
# propagation. Nothing below may hardcode a physical constant.
include("constants.jl")
include("dynamics/cr3bp.jl")
include("dynamics/cr3bp_j2.jl")
include("dynamics/cr3bp_saturn_j2.jl")
include("dynamics/integrator.jl")

# Configuration struct first: everything below dispatches on it.
include("StationkeepingPOMDP.jl")
include("states.jl")
include("actions.jl")
include("observations.jl")
include("tables.jl")
include("transition.jl")
include("rewards.jl")
include("model.jl")
include("export.jl")
include("common/report.jl")

export
    # dynamics — onboard model (CR3BP) and truth models (+J2). Kept separate.
    cr3bp_eom!,
    cr3bp_j2_eom!,
    cr3bp_saturn_enc_j2_eom!,
    jacobi_constant,
    libration_points_x,
    # propagation + events
    propagate,
    propagate_to_apoapsis,
    propagate_to_periapsis,
    propagate_n_orbits,
    collect_apses,
    r_enceladus,
    rdot_enceladus,
    altitude,
    # configuration + model
    StationkeepingPOMDP,
    build_pomdp,
    model_tables,
    # state space
    SKState,
    dev_bin,
    cov_has,
    cov_set,
    cov_count,
    cov_label,
    state_label,
    # measured tables
    DevTables,
    load_tables,
    write_tables,
    validate_tables,
    # export + reporting
    export_policy,
    print_policy_table,
    print_model_summary

end # module SherpaOrbital