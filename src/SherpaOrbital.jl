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
using LinearAlgebra
using OrdinaryDiffEq
using POMDPs
using POMDPTools
using QuickPOMDPs
using Printf
using Random
using Statistics

# Physics layer: constants, then dynamics (truth/onboard split preserved), then
# propagation. Nothing below may hardcode a physical constant.
include("constants.jl")
include("dynamics/cr3bp.jl")
include("dynamics/cr3bp_j2.jl")
include("dynamics/cr3bp_saturn_j2.jl")
include("dynamics/integrator.jl")

# Orbit generation: unit conversions first (halo_ic depends on them), then the
# differential corrector. NOTE: halo_ic.jl works in NON-DIMENSIONAL units.
include("orbital_elements.jl")
include("halo_ic.jl")

# Control layer. The planner is the ONBOARD model (CR3BP only) and lives in the library
# because every baseline shares it; the MPC baseline is truth-model-agnostic on top of it.
include("planner.jl")
include("baselines/mpc.jl")

# Spacecraft models. Deliberately upstream of the dynamics: they map commanded ΔV → applied
# ΔV and true state → noisy observation, and never call an integrator. Both are stochastic
# and take an explicit `rng`; neither touches the global RNG.
include("spacecraft/thruster.jl")
include("spacecraft/nav.jl")

# Configuration struct first: everything below dispatches on it.
include("StationkeepingPOMDP.jl")
# P_θ(intensity | band) — the plume gradient, the third sweep axis. Before `states.jl`,
# which enumerates the intensity dimension it defines.
include("plume.jl")
include("states.jl")
include("actions.jl")
include("observations.jl")
include("tables.jl")
include("transition.jl")
include("rewards.jl")
include("model.jl")
include("export.jl")
include("common/report.jl")

# The unified rollout harness. Last, because a SARSOP controller consumes the exported
# policy artifact and the (dev, cov) state helpers above.
include("common/simulate.jl")

# Calibration: MEASURE the transition kernels from the truth model. Depends on the coast
# helpers and geometry in common/simulate.jl, so it comes last.
include("calibration/calibrate.jl")

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
    # unit conversions + orbital elements
    cr3bp_to_nondim,
    nondim_to_cr3bp,
    ic_nondim_to_physical,
    r_from_enceladus,
    altitude_from_enceladus,
    cartesian_to_keplerian,
    # halo IC differential corrector (NON-DIMENSIONAL internals)
    PERIOD1_NORTH_IC_ND,
    PERIOD1_TRIPLE_PERIOD_S,
    PERIOD1_PERIOD_ND,
    PERIOD1_SOUTH_IC_ND,
    HALO_PERIOD_S,
    mirror_z,
    cr3bp_nd,
    cr3bp_stm_nd!,
    cr3bp_stm_jacobian_nd,
    propagate_half_period_nd,
    richardson_ic,
    seed_scan,
    differential_corrector,
    # family continuation — x0 as the family parameter (see corrector_free_z0's note on why
    # differential_corrector's z0-fixed split cannot continue the family)
    corrector_free_z0,
    halo_family_member,
    halo_family_table,
    halo_family_span,
    halo_family_table_cached,
    retarget_to_altitude,
    monodromy_matrix,
    stability_index,
    stability_index_trace,
    e_folding_time_s,
    characterise_orbit,
    find_halo_ic,
    # onboard burn planner (CR3BP only — never a truth EOM)
    CONTROL_ALT_KM,
    PERIAPSIS_ALT_TARGET,
    APOAPSIS_ALT_TARGET,
    TARGET_TOL_KM,
    ESCAPE_ALT_KM,
    escape_callback,
    predict_apses,
    predict_apse_states,
    next_apses,
    next_apse_positions,
    validate_apse_targets,
    apse_residual,
    solve_burn,
    # baselines — truth-model-agnostic (the truth EOM is an argument)
    run_mpc,
    # spacecraft models (stochastic; explicit rng, never the global one)
    ETA_EFF_MIN,
    ETA_EFF_MAX,
    apply_dv,
    apply_dv_noisy,
    sample_eta_eff,
    observe_position,
    observe_altitude,
    observe_deviation,
    # unified rollout harness — one world, controller swapped by type
    AbstractController,
    MPCController,
    SARSOPController,
    controller_type,
    load_policy,
    run_rollout,
    summarize_rollouts,
    # calibration — measure the kernels rather than transcribe them
    CalibrationRow,
    calibrate_tables,
    tables_from_rows,
    MIN_TRIALS_TRUSTED,
    CALIBRATION_EFFORT,
    needs_recalibration,
    # configuration + model
    StationkeepingPOMDP,
    build_pomdp,
    model_tables,
    # state space
    SKState,
    ALT_BINS,
    ALT_ALL,
    TERMINAL_ALT,
    isterminal_alt,
    isterminal_state,
    alt_bin,
    band_of_alt,
    visit_inc,
    visit_total,
    visit_tuples,
    visit_label,
    state_label,
    # plume gradient — P_θ(intensity | band), the third sweep axis
    plume_level_scores,
    plume_band_depth,
    plume_intensity_dist,
    plume_intensity_value,
    plume_levels_range,
    # measured tables
    AltTables,
    load_tables,
    write_tables,
    validate_tables,
    # export + reporting
    export_policy,
    theta_slug,
    theta_path,
    print_policy_table,
    print_model_summary

end # module SherpaOrbital