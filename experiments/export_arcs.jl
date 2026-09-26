#=
export_arcs.jl — dump the flown geometry of one rollout per navigation level to .npz,
for the Python 3D plotter.

    julia --project=experiments experiments/export_arcs.jl [sigmas] [out]

    sigmas  comma-separated navigation levels   (default 0.0,0.05,0.1)
    out     output .npz                          (default figures/arcs.npz)

Julia computes, Python plots. CairoMakie has no depth buffer — it sorts primitives by draw
order, so a surface cannot occlude a line behind it whatever the geometry says — and any
alpha below 1 on a surface emits a PDF soft mask per quad, which costs megabytes. mplot3d
z-sorts polygons and emits one graphics state, so the 3D trajectory view is markedly better
there. The 2D figures stay in Julia, where Makie is the better tool.

Requires a solved policy and matching kernels per level, i.e. a sweep that has reached them:

    KEY=sigma_nav_km VALS=0,0.05,0.1 SEEDS=1 julia --project=experiments experiments/sweep.jl

NOTE: `n_arc` is what makes the harness record the geometry. Nothing else stored can
reconstruct it — the checkpoints keep per-pass scalars, and re-propagating from a stored
state gives an uncontrolled coast that escapes, because the burns are not in that state.
=#

using SherpaOrbital
using SARSOP, POMDPs
using NPZ, Printf, Random

const SIGMAS = parse.(Float64, split(length(ARGS) >= 1 ? ARGS[1] : "0.0,0.05,0.1", ","))
const OUT    = length(ARGS) >= 2 ? ARGS[2] : joinpath("figures", "arcs.npz")
const SEED   = parse(Int, get(ENV, "SEED", "0"))
const DAYS   = parse(Float64, get(ENV, "DAYS", "30"))
const PLUME  = parse(Float64, get(ENV, "PLUME", "1.5"))
const N_ARC  = parse(Int, get(ENV, "N_ARC", "60"))

state0 = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND))

# Action -> code. 0 is the initial coast, which was flown under no burn at all.
const ACTION_NAMES = ["INITIAL", "CORRECT", "EXCURSE_LOW", "EXCURSE_MID", "EXCURSE_HIGH"]
const ACTION_CODE  = Dict(n => i - 1 for (i, n) in enumerate(ACTION_NAMES))

"""One rollout at `sig`, returned as a flat point list plus per-arc index ranges."""
function arcs_for(sig)
    cfg = StationkeepingPOMDP(; plume_gradient = PLUME, sigma_nav_km = sig)
    tpath = tables_path_for(cfg; nav_sigma_km = sig)
    isfile(tpath) || error("no kernels at $tpath — run the sweep for sigma = $sig")
    tbl = load_tables(tpath)
    stem = joinpath("artifacts", "solver", "sigma_nav_km=$(sig)_plume=$(PLUME)")
    isfile("$stem.out") || error("no policy at $stem.out — run the sweep for sigma = $sig")
    pol = SARSOP.load_policy(build_pomdp(cfg; tables = tbl), "$stem.out")

    res = run_rollout(SARSOPController(pol, cfg; ref_ic = state0, tables = tbl),
                      state0, cr3bp_j2_eom!, PERIOD1_TRIPLE_PERIOD_S, DAYS * 86400.0;
                      rng = Xoshiro(SEED), n_arc = N_ARC,
                      noisy_thruster = cfg.noisy_thruster,
                      thruster_sigma_pct = cfg.thruster_sigma_pct)

    # Flat Nx3 of Enceladus-relative points, with a start index per arc: npz has no ragged
    # arrays, so the consumer slices with `starts`.
    # Actions as integer CODES: npz cannot hold a string array, so the consumer maps the
    # codes through `action_names`, written once alongside.
    pts = Float64[]
    starts = Int[]
    acts = Int[]
    for a in res.arcs
        size(a.xyz, 2) == 0 && continue
        push!(starts, length(pts) ÷ 3)          # 0-based, for numpy
        push!(acts, get(ACTION_CODE, String(a.action), 0))
        for i in 1:size(a.xyz, 2)
            r = SherpaOrbital._enc_relative(a.xyz[:, i])
            append!(pts, r)
        end
    end
    xyz = permutedims(reshape(pts, 3, :))       # N x 3
    return (xyz = xyz, starts = starts, actions = acts,
            outcome = String(res.outcome), n_pass = length(res.steps),
            peri_alts = [s.peri_alt_km for s in res.steps])
end

payload = Dict{String,Any}(
    "sigmas"      => SIGMAS,
    "r_enceladus" => SherpaOrbital.R_ENCELADUS,
    "days"        => DAYS,
    "seed"        => SEED,
    "plume"       => PLUME,
    # The action-name table as UTF-8 BYTES: npz holds neither string arrays nor bare
    # strings, so the consumer decodes this and splits on the comma. The codes in
    # `actions_*` index into it.
    "action_names_utf8" => Vector{UInt8}(join(ACTION_NAMES, ",")),
)
for sig in SIGMAS
    a = arcs_for(sig)
    k = replace(string(sig), "." => "p")        # npz keys cannot carry a dot cleanly
    payload["xyz_$k"]     = a.xyz
    payload["starts_$k"]  = a.starts
    payload["actions_$k"] = a.actions
    payload["perialt_$k"] = a.peri_alts
    @printf("  sigma=%-5s %2d passes, %3d arcs, %5d points, %s\n",
            sig, a.n_pass, length(a.starts), size(a.xyz, 1), a.outcome)
end

mkpath(dirname(OUT))
npzwrite(OUT, payload)
@printf("wrote %s  (%.0f KB)\n", OUT, filesize(OUT) / 1024)