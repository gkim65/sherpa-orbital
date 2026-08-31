#=
calibrate.jl — MEASURE the transition kernels and write artifacts/tables.json.

    julia --project=experiments -t auto experiments/calibrate.jl [tree_depth]

This is the generator for `artifacts/tables.json`. Before 2026-08-30 that artifact held
hand-transcribed numbers and this step did not exist, so the committed kernels could drift
from the run that produced them with nothing to detect it. Run this whenever the state
space, the bands, the orbit, or the truth model changes — the kernels are conditional on
all four.

⚠️ `-t auto` MATTERS. The action tree is threaded (see `tree_walk!`); single-threaded it
still works and gives BIT-IDENTICAL output, just several times slower.

⚠️ THE DEPTH IS AN ARGUMENT, so exploring it never needs a throwaway script. It used to be
reachable only by editing `CALIBRATION_EFFORT`, which is why depth experiments kept getting
run from `scratch/` — and those scripts do not call `write_tables`, so the measurement was
thrown away every time. Pass the depth here and the artifact is always written.

Prints per-row trial counts and a coverage summary before writing, because a row measured
from too few trials is the main way a plausible-looking artifact goes wrong. Rows reported
as UNMEASURED are FILLED (see `_fill_unmeasured_row`); they are not evidence and the
policy's behaviour in those bins means nothing.
=#

using SherpaOrbital
using SherpaOrbital: ALT_BINS, ALT_ALL, RESIDUAL_BINS, RESIDUAL_EDGES,
                     kernel_keys_all, kernel_columns, kernel_entry_index, KernelKey,
                     residual_index, isterminal_alt
using Printf
using Statistics: median, mean

# ── Scenario ─────────────────────────────────────────────────────────────────
# Must match the StationkeepingPOMDP config the policy will be solved against: the kernels
# are keyed by altitude bin, so different edges mean different rows.
config = StationkeepingPOMDP()

tree_depth = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : CALIBRATION_EFFORT.tree_depth

println("Measuring kernels against the truth model ...")
println("  alt edges (km) : ", config.alt_edges)
println("  residual (km)  : ", RESIDUAL_EDGES, " -> ", RESIDUAL_BINS)
println("  bands          : ", config.band_names, " -> ", config.band_bins,
        "  commanded(km)=", [config.band_target_km[b] for b in config.band_names])
println("  rows/action    : ", length(kernel_keys_all()),
        " (alt x residual)   columns: ", length(kernel_columns()), " joint (alt, residual)")
println("  tree depth     : ", tree_depth, "   threads: ", Threads.nthreads())
println()

# ⚠️ THE CONFIG IS THE ONLY SOURCE OF θ (2026-08-31). This used to hand-map `alt_edges` /
# `band_names` / `band_target_km` across one by one, against `calibrate_tables` defaults
# that silently DISAGREED with the struct's — so forgetting one argument measured kernels
# keyed to the wrong bins with no error. Effort knobs come from `CALIBRATION_EFFORT`.
t_measure = @elapsed ((rows, diag) =
    calibrate_tables(config; tree_depth = tree_depth, verbose = false))
let e = diag["effort"]
    @printf("Measured in %.1f s   (tree: %d passes, %d live nodes)\n",
            t_measure, e["tree_passes"], e["tree_nodes"])
end

# ── Per-row provenance ───────────────────────────────────────────────────────
println("\nMeasured rows (n = trials; a row with n = 0 is FILLED, not measured).")
println("Rows are keyed (altitude bin / residual bin); columns are the JOINT successor.")
@printf("  %-13s %-20s %6s %7s %8s  %s\n",
        "action", "from (alt/residual)", "n", "nonconv", "P(LOST)", "top successors")
cols = kernel_columns()
for a in sort(collect(keys(rows))), key in kernel_keys_all()
    r = rows[a][key]
    if r.n == 0
        @printf("  %-13s %-20s %6d %7d %8s  %s\n", a, string(key), 0, 0, "-",
                "(UNMEASURED -> self-transition)")
        continue
    end
    p_lost = r.probs[SherpaOrbital.kernel_entry_index(:LOST, :R_OK)]
    # The three largest successor masses, named — a 17-wide raw vector is unreadable.
    ord = sortperm(r.probs; rev = true)
    top = join([@sprintf("%s/%s=%.2f", cols[i][1], cols[i][2], r.probs[i])
                for i in ord[1:min(3, end)] if r.probs[i] > 0.0], " ")
    @printf("  %-13s %-20s %6d %7d %8.3f  %s\n",
            a, string(key), r.n, r.n_nonconverged, p_lost, top)
end

# ── Did the residual conditioning actually buy anything? ─────────────────────
# The dimension was added on the claim that P(loss) is NOT constant across damage bins for
# a fixed (action, altitude). If it comes back flat, the encoding is not earning its |S|.
println("\nP(LOST) by damage bin, for each (action, altitude) — the reason the dimension exists:")
@printf("  %-13s %-10s %s\n", "action", "alt", join([@sprintf("%-22s", r) for r in RESIDUAL_BINS]))
for a in sort(collect(keys(rows))), alt in ALT_BINS
    cells = String[]
    for res in RESIDUAL_BINS
        r = rows[a][SherpaOrbital.KernelKey(alt, res)]
        push!(cells, r.n == 0 ? @sprintf("%-22s", "  -    (n=0)") :
              @sprintf("%-22s", @sprintf("%.3f (n=%d)",
                       r.probs[SherpaOrbital.kernel_entry_index(:LOST, :R_OK)], r.n)))
    end
    @printf("  %-13s %-10s %s\n", a, alt, join(cells))
end

# ── Coverage: how much of the model is actually MEASURED? ────────────────────
# The headline number for "is this artifact good enough". Rows are counted three ways
# because "45/60" alone is misleading in both directions: a row with n = 1 is technically
# measured but is a single deterministic trajectory, and a filled row in a cell the vehicle
# never occupies costs nothing.
let nmeas = 0, nthin = 0, ntot = 0
    for a in sort(collect(keys(rows))), k in kernel_keys_all()
        ntot += 1
        n = rows[a][k].n
        n > 0 && (nmeas += 1)
        0 < n < MIN_TRIALS_TRUSTED && (nthin += 1)
    end
    println("\n", "="^72)
    @printf("COVERAGE  %d / %d rows measured (%.0f%%)   thin (0 < n < %d): %d   filled: %d\n",
            nmeas, ntot, 100nmeas / ntot, MIN_TRIALS_TRUSTED, nthin, ntot - nmeas)
end

# ── Can CORRECT recover a degraded orbit? ────────────────────────────────────
# THE question the residual dimension was added to answer. Before the action tree this was
# unanswerable: every CORRECT row at R_DEGRADED/R_CRITICAL had n = 0, because CORRECT flown
# alone is a stable limit cycle and cannot damage its own orbit. A row here with n > 0 is
# measured evidence about the "first correction does not clear it, the second does" finding.
println("\nCORRECT from a DEGRADED orbit — does correcting reduce the damage bin?")
let any_ = false
    for alt in ALT_BINS, res in (:R_DEGRADED, :R_CRITICAL)
        r = rows[:CORRECT][KernelKey(alt, res)]
        r.n == 0 && continue
        any_ = true
        ri = residual_index(res)
        p_better = 0.0
        for (i, (a2, r2)) in enumerate(kernel_columns())
            if !isterminal_alt(a2) && residual_index(r2) < ri
                p_better += r.probs[i]
            end
        end
        @printf("  CORRECT %-10s %-12s P(damage decreases) = %.3f   P(LOST) = %.3f   (n=%d)\n",
                alt, res, p_better,
                r.probs[kernel_entry_index(:LOST, :R_OK)], r.n)
    end
    any_ || println("  (none measured — the tree never reached a degraded CORRECT departure)")
end

# ── Where the excursions actually went ───────────────────────────────────────
# ── Measured ΔV per action, for `action_dv_cost` ─────────────────────────────
# The config's `action_dv_cost` is a REWARD-SHAPING proxy, and its entries were flagged as
# placeholders needing re-measurement. Print the measured medians so they can be set from
# data rather than carried forward by hand.
println("\nMeasured ΔV per action (m/s) — set config.action_dv_cost from these:")
let corr = vcat((rows[:CORRECT][k].dv_ms for k in kernel_keys_all())...)
    isempty(corr) || @printf("  CORRECT      n=%3d  median %6.3f  mean %6.3f\n",
                             length(corr), median(corr), mean(corr))
end
for b in config.band_names
    dv = diag["band_dv_ms"][string(b)]
    isempty(dv) && continue
    @printf("  EXCURSE_%-5s n=%3d  median %6.3f  mean %6.3f\n",
            b, length(dv), median(dv), mean(dv))
end

println("\nBand COMMANDED vs ACHIEVED periapsis altitude (the excursion sanity check):")
for b in config.band_names
    got = diag["band_achieved_alt"][string(b)]
    # Summary, not the raw vector: these run to hundreds of entries and the question is
    # only whether the command was delivered.
    @printf("  %-5s commanded %5.1f km -> achieved n=%3d  min %6.2f  median %6.2f  max %6.2f\n",
            b, config.band_target_km[b], length(got),
            isempty(got) ? NaN : minimum(got),
            isempty(got) ? NaN : median(got),
            isempty(got) ? NaN : maximum(got))
end

# ── Assemble + write ─────────────────────────────────────────────────────────
tables = tables_from_rows(rows, diag)
validate_tables(tables)

unmeasured = tables.meta["unmeasured_rows"]
if !isempty(unmeasured)
    println("\n⚠️  ", length(unmeasured), " row(s) UNMEASURED and filled: ",
            join(unmeasured, ", "))
    println("    The policy's behaviour in those bins is not supported by measurement.")
end

path = write_tables(tables)
println("\nWrote kernels -> $path")