#=
sweep.jl — fly every controller across a parameter sweep, checkpointing as it goes.

    julia --project=experiments experiments/sweep.jl [options]

Options are environment variables, so a new sweep is one line:

    KEY=sigma_nav_km VALS=2,4,6,8 SEEDS=3 julia --project=experiments experiments/sweep.jl
    KEY=thruster_sigma_pct VALS=0.7,2,5 SEEDS=5 ARMS=POMDP,Threshold julia ...
    KEY=plume_gradient VALS=0,1.5,4 DAYS=15 julia ...

    KEY     swept config field                        (default sigma_nav_km)
    VALS    comma-separated values                     (default 2,4,6,8)
    SEEDS   rollouts per cell                          (default 3)
    ARMS    comma-separated controller labels          (default all)
    DAYS    rollout horizon in days                    (default 30)
    PLUME   plume_gradient, when not the swept key     (default 1.5)
    OUT     checkpoint root                            (default artifacts/sweeps/<KEY>)

EVERY ROLLOUT IS WRITTEN AS IT COMPLETES. Killing this script keeps everything already
flown; rerunning the same command resumes from the checkpoints on disk rather than
re-flying them. Analyse afterwards with `load_sweep(root)` — the per-step trace is stored,
so a new metric or figure needs no re-run.

NOTE: the POMDP arm reuses the policy solved at PLUME while the world varies, which is the
deployment question — a precomputed policy meeting conditions it was not solved for. It is
deliberately not re-solved per cell. Sweeping `plume_gradient` itself therefore needs a
policy per value; solve them first with `experiments/example.jl`.

NOTE: `sigma_nav_km` is passed to the SARSOP controller explicitly. Its constructor
defaults to the `SIGMA_NAV_POS` constant, not the config, so omitting it would fly the
POMDP arm at 2 km while the scripted arms saw the swept value.
=#

using SherpaOrbital
using Printf, Random, Statistics

const KEY   = Symbol(get(ENV, "KEY", "sigma_nav_km"))
const VALS  = parse.(Float64, split(get(ENV, "VALS", "2,4,6,8"), ","))
const SEEDS = parse(Int, get(ENV, "SEEDS", "3"))
const DAYS  = parse(Float64, get(ENV, "DAYS", "30"))
const PLUME = parse(Float64, get(ENV, "PLUME", "1.5"))
const OUT   = get(ENV, "OUT", joinpath("artifacts", "sweeps", String(KEY)))
const ALL_ARMS = ["POMDP", "Threshold", "Cyclic k=1", "Cyclic k=2", "Cyclic k=3",
                  "Greedy", "MPC hold"]
const ARMS  = haskey(ENV, "ARMS") ? split(ENV["ARMS"], ",") : ALL_ARMS

state0   = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND))
period_s = PERIOD1_TRIPLE_PERIOD_S
horizon  = DAYS * 24 * 3600.0

"""Config for one sweep cell: the baseline scenario with the swept field overridden."""
cell_config(v) = StationkeepingPOMDP(; plume_gradient = PLUME,
                                     NamedTuple{(KEY,)}((v,))...)

"""Fresh controller per rollout — every controller carries live state."""
function build_arm(arm::AbstractString, cfg::StationkeepingPOMDP, policy)
    arm == "POMDP"      && return SARSOPController(policy; ref_ic = state0,
                                                   sigma_nav_km = cfg.sigma_nav_km)
    arm == "MPC hold"   && return MPCController(; ref_ic = state0, mode = :position)
    arm == "Greedy"     && return GreedyController(scripted_core(cfg; ref_ic = state0))
    arm == "Threshold"  && return ThresholdController(scripted_core(cfg; ref_ic = state0);
                                                      max_residual = "R_OK")
    m = match(r"Cyclic k=(\d+)", arm)
    m === nothing && error("unknown arm $arm; known: $(join(ALL_ARMS, ", "))")
    return CyclicController(scripted_core(cfg; ref_ic = state0), parse(Int, m[1]))
end

@printf("sweep %s = %s\n", KEY, join(VALS, ", "))
@printf("arms: %s\n", join(ARMS, ", "))
@printf("%d seeds, %.0f d, plume=%.1f -> %s\n\n", SEEDS, DAYS, PLUME, OUT)

policy_cache = Dict{Float64,Any}()
t_start = time()

for v in VALS
    cfg = cell_config(v)
    # The policy depends on plume_gradient only; cache per value so a nav or thruster sweep
    # loads the ~1.6 GB artifact once rather than per cell.
    policy = if "POMDP" in ARMS
        get!(policy_cache, cfg.plume_gradient) do
            p = theta_path("policy", (plume_gradient = cfg.plume_gradient,))
            isfile(p) ? load_policy(p) : (@warn "no policy at $p — POMDP arm skipped"; nothing)
        end
    else
        nothing
    end

    for arm in ARMS
        arm == "POMDP" && policy === nothing && continue
        dir  = cell_dir(OUT, NamedTuple{(KEY,)}((v,)), arm)
        done = n_checkpoints(dir)
        if done >= SEEDS
            @printf("%-11s %s=%-5s  %d/%d already banked, skipping\n",
                    arm, KEY, v, done, SEEDS)
            flush(stdout)
            continue
        end

        for k in done:(SEEDS - 1)
            t0 = time()
            res = run_rollout(build_arm(arm, cfg, policy), state0, cr3bp_j2_eom!,
                              period_s, horizon;
                              rng = Xoshiro(k),
                              noisy_thruster = cfg.noisy_thruster,
                              thruster_sigma_pct = cfg.thruster_sigma_pct)
            save_rollout(dir, k, res, cfg; seed = k, arm = arm,
                         theta = NamedTuple{(KEY,)}((v,)),
                         wall_s = time() - t0)
            sci = science_trace(res, cfg)
            @printf("%-11s %s=%-5s seed %d  %-7s sci=%6.1f dV=%6.1f  %.0fs\n",
                    arm, KEY, v, k, String(res.outcome),
                    isempty(sci.cumulative) ? 0.0 : last(sci.cumulative),
                    res.total_dv_ms, time() - t0)
            flush(stdout)
        end
    end
end

@printf("\ndone in %.1f min -> %s\n", (time() - t_start) / 60, OUT)
@printf("analyse with: load_sweep(\"%s\")\n", OUT)