#=
sweep.jl — calibrate, solve and fly every controller across a parameter sweep.

    julia --project=experiments -t auto experiments/sweep.jl [options]

Options are environment variables, so a new sweep is one line:

    KEY=sigma_nav_km VALS=0,0.1,0.3,1,2 SEEDS=3 julia --project=experiments -t auto experiments/sweep.jl
    KEY=thruster_sigma_pct VALS=0.7,2 SEEDS=3 julia ...
    KEY=plume_gradient VALS=0,1.5,4 SEEDS=3 julia ...

    KEY     swept config field                      (default sigma_nav_km)
    VALS    comma-separated values                   (default 0,0.1,0.3,1,2)
    SEEDS   rollouts per cell                        (default 3)
    WORKERS, WORKER
            split seeds across WORKERS processes;     (default 1, 0)
            this one is WORKER (0-based)
    ARMS    comma-separated controller labels        (default all)
    DAYS    rollout horizon in days                  (default 30)
    PLUME   plume_gradient, when not the swept key   (default 1.5)
    OUT     checkpoint root                          (default artifacts/sweeps/<KEY>)

EVERYTHING IS RESUMABLE AND KEYED BY VALUE. Kernels, policies and rollouts are each skipped
if already on disk, so:

  - killing the script keeps every rollout already flown
  - rerunning with more SEEDS flies only the new seeds
  - rerunning with more VALS calibrates, solves and flies only the new levels

so a coarse sweep can be filled in later without redoing any of it.

To split a level across PROCESSES, run N workers over the same command with WORKER = 0..N-1;
each takes every Nth seed. See `experiments/cluster/run_sweep.sh`, which solves once and then
fans the rollouts out.

    WORKERS=8 WORKER=0 KEY=sigma_nav_km VALS=0.3 SEEDS=100 julia ... &
    WORKERS=8 WORKER=1 KEY=sigma_nav_km VALS=0.3 SEEDS=100 julia ... &

Per level, for a recalibration axis: ~3 min calibrate + ~9 min solve + ~1 min rollouts.
`needs_recalibration` decides — `sigma_nav_km`, `thruster_sigma_pct` and `noisy_thruster`
change the dynamics the kernels measure; `plume_gradient` enters the reward analytically and
reuses one kernel set.

Analyse afterwards with `load_sweep(root)`; the per-step trace is stored, so a new metric or
figure needs no re-run.
=#

using SherpaOrbital
using SARSOP, POMDPs
using Printf, Random, Statistics

const KEY   = Symbol(get(ENV, "KEY", "sigma_nav_km"))
const VALS  = parse.(Float64, split(get(ENV, "VALS", "0,0.1,0.3,1,2"), ","))
const SEEDS = parse(Int, get(ENV, "SEEDS", "3"))
# Seed shard, for splitting one level across several PROCESSES. Launch N workers with
# WORKER = 0..N-1 and each takes every Nth seed, so they write disjoint checkpoint files and
# need no locking. Striding rather than blocking keeps the shards balanced when rollout cost
# varies with the seed — a block of early seeds is not systematically cheaper than a block
# of late ones, but a run that escapes on pass 3 costs far less than one that flies 60.
const WORKERS = parse(Int, get(ENV, "WORKERS", "1"))
const WORKER  = parse(Int, get(ENV, "WORKER", "0"))
0 <= WORKER < WORKERS || error("WORKER must be in 0:$(WORKERS - 1), got $WORKER")
const DAYS  = parse(Float64, get(ENV, "DAYS", "30"))
const PLUME = parse(Float64, get(ENV, "PLUME", "1.5"))
const OUT   = get(ENV, "OUT", joinpath("artifacts", "sweeps", String(KEY)))
const ALL_ARMS = ["POMDP", "Threshold", "Cyclic k=1", "Cyclic k=2", "Cyclic k=3",
                  "Greedy", "MPC hold"]
# An EMPTY `ARMS` means "all", not "one arm with no name": a shell driver that always
# forwards the variable passes "" when the caller did not set it, and `haskey` is true for
# that, so splitting it yields a single empty arm and the run dies on the first cell.
const ARMS  = let a = strip(get(ENV, "ARMS", ""))
    isempty(a) ? ALL_ARMS : [strip(x) for x in split(a, ",") if !isempty(strip(x))]
end
const RECAL = needs_recalibration(KEY)

state0   = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND))
period_s = PERIOD1_TRIPLE_PERIOD_S
horizon  = DAYS * 24 * 3600.0

"""Config for one cell: the baseline scenario with the swept field overridden."""
cell_config(v) = StationkeepingPOMDP(; plume_gradient = PLUME,
                                     NamedTuple{(KEY,)}((v,))...)

"""
Kernels for one cell, calibrated if absent.

On a recalibration axis the planner solves from a navigation estimate at the cell's own
sigma, so the kernels model the same information the rollouts will fly with. Off one, the
packaged artifact for the config's thruster settings is reused.
"""
function cell_tables(cfg)
    nav = RECAL ? cfg.sigma_nav_km : 0.0
    path = tables_path_for(cfg; nav_sigma_km = nav)
    if isfile(path)
        @printf("  kernels: %s\n", basename(path)); flush(stdout)
        return load_tables(path)
    end
    @printf("  calibrating (nav_sigma=%.2f) ...\n", nav); flush(stdout)
    t0 = time()
    rows, diag = calibrate_tables(cfg; nav_sigma_km = nav)
    tbl = tables_from_rows(rows, diag)
    write_tables(tbl; path = path)
    @printf("    %.1f min -> %s\n", (time() - t0) / 60, basename(path)); flush(stdout)
    return tbl
end

"""
Policy for one cell, solved if absent.

Saved as SARSOP's own `.out`, not the JSON archive: the JSON carries the dense
`T[s][a][s']` (gigabytes at |S| = 5627) while `SARSOPController(policy, config)` rebuilds
T and O from the config, so the archive buys nothing here.
"""
function cell_policy(cfg, tbl, v)
    # SARSOP opens these paths directly and does not create the directory, so a fresh
    # clone has nowhere to write and the solve dies on the first level.
    mkpath(joinpath("artifacts", "solver"))
    stem = joinpath("artifacts", "solver",
                    string(KEY, "=", v, "_plume=", PLUME))
    pomdp = build_pomdp(cfg; tables = tbl)
    if isfile("$stem.out")
        @printf("  policy: %s.out\n", basename(stem)); flush(stdout)
        return SARSOP.load_policy(pomdp, "$stem.out")
    end
    println("  solving ..."); flush(stdout)
    t0 = time()
    p = solve(SARSOP.SARSOPSolver(; precision = 1e-3, timeout = 1800.0, verbose = false,
                                  pomdp_filename  = "$stem.pomdpx",
                                  policy_filename = "$stem.out"), pomdp)
    @printf("    %.1f min -> %s.out\n", (time() - t0) / 60, basename(stem)); flush(stdout)
    return p
end

"""Fresh controller per rollout — every controller carries live state."""
function build_arm(arm::AbstractString, cfg, policy, tbl)
    arm == "POMDP"     && return SARSOPController(policy, cfg; ref_ic = state0,
                                                  tables = tbl)
    arm == "MPC hold"  && return MPCController(; ref_ic = state0, mode = :position,
                                               nav_sigma_km = cfg.sigma_nav_km)
    arm == "Greedy"    && return GreedyController(scripted_core(cfg; ref_ic = state0))
    arm == "Threshold" && return ThresholdController(scripted_core(cfg; ref_ic = state0);
                                                     max_residual = "R_OK")
    m = match(r"Cyclic k=(\d+)", arm)
    m === nothing && error("unknown arm $arm; known: $(join(ALL_ARMS, ", "))")
    return CyclicController(scripted_core(cfg; ref_ic = state0), parse(Int, m[1]))
end

@printf("sweep %s = %s\n", KEY, join(VALS, ", "))
@printf("arms: %s\n", join(ARMS, ", "))
@printf("%d seeds, %.0f d, plume=%.1f, recalibrate=%s -> %s\n",
        SEEDS, DAYS, PLUME, RECAL, OUT)
WORKERS > 1 && @printf("worker %d of %d: seeds %s\n", WORKER, WORKERS,
                       join(WORKER:WORKERS:min(SEEDS - 1, WORKER + 4WORKERS), ",") *
                       (WORKER + 4WORKERS < SEEDS - 1 ? ",..." : ""))
flush(stdout)

t_start = time()
for v in VALS
    cfg = cell_config(v)
    @printf("\n=== %s = %s ===\n", KEY, v); flush(stdout)

    tbl = cell_tables(cfg)
    policy = "POMDP" in ARMS ? cell_policy(cfg, tbl, v) : nothing

    for arm in ARMS
        dir  = cell_dir(OUT, NamedTuple{(KEY,)}((v,)), arm)
        # Resume per SEED, not per count: with several processes sharding one cell the
        # checkpoints do not arrive in order, so a count says nothing about which seeds this
        # shard still owes.
        # Resume per SEED, not per count: with several workers sharding one cell the
        # checkpoints do not arrive in order, so a count says nothing about which seeds
        # this worker still owes.
        todo = [k for k in WORKER:WORKERS:(SEEDS - 1)
                if !isfile(joinpath(dir, rollout_filename(k)))]
        if isempty(todo)
            @printf("  %-11s shard banked, skipping\n", arm); flush(stdout)
            continue
        end
        for k in todo
            t0 = time()
            res = run_rollout(build_arm(arm, cfg, policy, tbl), state0, cr3bp_j2_eom!,
                              period_s, horizon;
                              rng = Xoshiro(k),
                              noisy_thruster = cfg.noisy_thruster,
                              thruster_sigma_pct = cfg.thruster_sigma_pct)
            save_rollout(dir, k, res, cfg; seed = k, arm = arm,
                         theta = NamedTuple{(KEY,)}((v,)),
                         nav_plan_sigma_km = RECAL ? cfg.sigma_nav_km : 0.0,
                         wall_s = time() - t0)
            sci = science_trace(res, cfg)
            @printf("  %-11s seed %d  %-7s sci=%6.1f dV=%6.1f bands=%d  %.0fs\n",
                    arm, k, String(res.outcome),
                    isempty(sci.cumulative) ? 0.0 : last(sci.cumulative),
                    res.total_dv_ms, res.n_bands, time() - t0)
            flush(stdout)
        end
    end
end

@printf("\ndone in %.1f min -> %s\n", (time() - t_start) / 60, OUT)
@printf("analyse with: load_sweep(\"%s\")\n", OUT)