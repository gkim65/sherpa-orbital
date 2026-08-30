#=
calibrate.jl — MEASURE the transition kernels and write artifacts/tables.json.

    julia --project=experiments experiments/calibrate.jl

This is the generator for `artifacts/tables.json`. Before 2026-08-30 that artifact held
hand-transcribed numbers and this step did not exist, so the committed kernels could drift
from the run that produced them with nothing to detect it. Run this whenever the state
space, the bands, the orbit, or the truth model changes — the kernels are conditional on
all four.

Prints the per-row trial counts before writing, because a row measured from too few trials
is the main way a plausible-looking artifact goes wrong. Rows reported as UNMEASURED are
filled with a self-transition so the kernel is a valid stochastic matrix; they are not
evidence and the policy's behaviour in those bins means nothing.
=#

using SherpaOrbital
using SherpaOrbital: ALT_BINS, ALT_ALL
using Printf

# ── Scenario ─────────────────────────────────────────────────────────────────
# Must match the StationkeepingPOMDP config the policy will be solved against: the kernels
# are keyed by altitude bin, so different edges mean different rows.
config = StationkeepingPOMDP()

println("Measuring kernels against the truth model ...")
println("  alt edges (km) : ", config.alt_edges)
println("  bands          : ", config.band_names, " -> ", config.band_bins,
        "  commanded(km)=", [config.band_target_km[b] for b in config.band_names])
println()

rows, diag = calibrate_tables(;
    alt_edges      = config.alt_edges,
    band_names     = config.band_names,
    band_target_km = config.band_target_km,
    verbose        = true,
)

# ── Per-row provenance ───────────────────────────────────────────────────────
println("\nMeasured rows (n = trials; a row with n = 0 is FILLED, not measured):")
@printf("  %-9s %-10s %6s %7s  %s\n", "action", "from", "n", "nonconv", "P(next)")
for a in (:CORRECT, :OBSERVE, :EXCURSE), b in ALT_BINS
    r = rows[a][b]
    @printf("  %-9s %-10s %6d %7d  %s\n", a, b, r.n, r.n_nonconverged,
            r.n == 0 ? "(UNMEASURED -> self-transition)" :
                       string(round.(r.probs, digits = 3)))
end

# ── Where the excursions actually went ───────────────────────────────────────
println("\nBand COMMANDED vs ACHIEVED periapsis altitude (the excursion sanity check):")
for b in config.band_names
    got = diag["band_achieved_alt"][string(b)]
    @printf("  %-5s commanded %5.1f km -> achieved %s\n",
            b, config.band_target_km[b],
            isempty(got) ? "(no surviving trial)" : string(round.(got, digits = 2)))
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