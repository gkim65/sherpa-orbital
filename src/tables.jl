"""
tables.jl — the measured transition kernels, as a versioned artifact.

WHY THIS IS A FILE AND NOT A CONSTANT
The dev-transition kernels are MEASURED from CR3BP+J2 experiments, not derived
analytically. Previously they lived as literal constants hand-transcribed from
experiment output, which meant re-running an experiment could silently leave the model
stale with nothing to detect it. They are now a generated artifact carrying its own
provenance: `artifacts/tables.json` records the numbers AND which experiment produced
them, so a re-measurement shows up as a reviewable diff.

Format is JSON rather than TOML/binary deliberately: these are a scientific artifact
committed to git, so being able to eyeball a changed probability in review matters more
than compactness. Swapping to JLD2 later is a drop-in behind `load_tables`/`write_tables`.

Rows are P(next alt, next residual | alt, residual, action) — a JOINT distribution over
ALT_ALL × RESIDUAL_BINS, flattened with the residual varying FASTEST (see
[`kernel_entry_index`](@ref)).

⚠️ REKEYED 2026-08-30 from dev bins to ALTITUDE bins. The committed `artifacts/tables.json`
was measured against the old (dev, cov) state space and does NOT describe this model;
`load_tables` rejects it rather than silently misaligning the kernel columns. Re-measure
with `calibrate_tables` before quoting any result.

⚠️ REKEYED AGAIN 2026-08-31 to carry the RESIDUAL (orbit-damage) dimension. Rows are now
keyed `[action][(alt_bin, residual_bin)]` and their columns are the JOINT successor
`(alt, residual)`, because the whole point of the dimension is that the transition depends
on accumulated damage — a kernel keyed on altitude alone averages "fresh" and "degraded"
into one number and reports P(loss) = 0.0 for the transition that actually kills the
vehicle. Format marker goes 2 → 3; `load_tables` rejects format 2 rather than remapping,
since the residual conditioning was never recorded in it and cannot be recovered.
"""

const DEFAULT_TABLES_PATH =
    normpath(joinpath(@__DIR__, "..", "artifacts", "tables.json"))

"""
    KernelKey

The conditioning of one measured row: which altitude bin the pass departed from, and how
DEGRADED the orbit was when it did (`residual ∈ RESIDUAL_BINS`).

⚠️ THE RESIDUAL HALF IS THE 2026-08-31 ADDITION and it is the entire point of the change.
Keyed on `alt` alone, the row for "excurse from the limit cycle" pools a fresh departure
(which is safe) with a departure from an orbit already 30 km out of trim (which is not),
and reports their average — P(loss) = 0.0, because fresh departures dominate the sample.
"""
struct KernelKey
    alt::Symbol
    residual::Symbol
end

Base.show(io::IO, k::KernelKey) = print(io, k.alt, "/", k.residual)

"""
    kernel_keys_all() -> Vector{KernelKey}

Every (live altitude bin × residual bin) conditioning, in the canonical order rows are
enumerated and reported in. Altitude varies slowest.
"""
kernel_keys_all() = [KernelKey(a, r) for a in ALT_BINS for r in RESIDUAL_BINS]

"""
    kernel_entry_index(alt, residual) -> Int
    kernel_columns() -> Vector{Tuple{Symbol,Symbol}}

The flattening convention for a kernel ROW: the successor is a JOINT `(alt, residual)`, and
the columns run over `ALT_ALL × RESIDUAL_BINS` with the RESIDUAL varying FASTEST.

The two absorbing altitudes are carried once each rather than once per residual bin — a
lost orbit ran no onboard solve, so it has no residual — which keeps the terminal columns
from splitting into unreachable duplicates. They are placed LAST, after the live block.

Row length is therefore `|ALT_BINS| * |RESIDUAL_BINS| + |TERMINAL_ALT|` = 5*3 + 2 = 17.
"""
function kernel_columns()
    cols = Tuple{Symbol,Symbol}[]
    for a in ALT_BINS, r in RESIDUAL_BINS
        push!(cols, (a, r))
    end
    for a in TERMINAL_ALT
        push!(cols, (a, :R_OK))
    end
    return cols
end

const N_KERNEL_COLS = length(ALT_BINS) * length(RESIDUAL_BINS) + length(TERMINAL_ALT)

function kernel_entry_index(alt::Symbol, residual::Symbol)
    if isterminal_alt(alt)
        return length(ALT_BINS) * length(RESIDUAL_BINS) +
               findfirst(==(alt), TERMINAL_ALT)
    end
    ai = findfirst(==(alt), ALT_BINS)
    ri = findfirst(==(residual), RESIDUAL_BINS)
    return (ai - 1) * length(RESIDUAL_BINS) + ri
end

"""
    AltTables

Measured transition kernels, keyed by action group then by [`KernelKey`](@ref)
(altitude bin × residual bin), plus the provenance metadata carried alongside them.

Each row is a distribution over the JOINT successor `(alt, residual)` — see
[`kernel_columns`](@ref) for the flattening.
"""
struct AltTables
    correct::Dict{KernelKey,Vector{Float64}}
    # ⚠️ TWO-LEVEL AS OF 2026-08-30: excurse[ACTION][key], one kernel PER EXCURSE
    # ACTION. It was previously `excurse[from_bin]` — a single kernel shared by every band.
    # See `alt_kernel` for why that was wrong and what it cost.
    excurse::Dict{Symbol,Dict{KernelKey,Vector{Float64}}}
    meta::Dict{String,Any}
end

"""
    alt_kernel(tables, action, key) -> Vector{Float64}
    alt_kernel(tables, action, alt, residual) -> Vector{Float64}

The measured joint successor distribution for `action` from `(alt, residual)`, over the
columns of [`kernel_columns`](@ref).

⚠️ ONE KERNEL PER EXCURSE ACTION (changed 2026-08-30). This previously returned a SINGLE
pooled `excurse[alt]` row for every band, on the exp-12 finding that excursion RISK does not
differ meaningfully by band — only ΔV cost does. That is true for the SAFETY question and
wrong for the SCIENCE question, because a kernel also encodes WHERE YOU LAND, and where you
land is the entire point of aiming at a band.

What the pooling cost, measured: the old `A34_44` EXCURSE row (from the limit cycle) read
1/3 `A20_27`, 1/3 `A27_34`, 1/3 `ABOVE_44` over n = 3 — which is not "an excursion lands
randomly" but the LOW walk landing in LOW, the MID walk in MID and the HIGH walk in HIGH,
one trial each, averaged together. So the model was told `EXCURSE_HIGH` and `EXCURSE_LOW`
have IDENTICAL successor distributions.

Why it went unnoticed: `action_dv_cost` gave the three actions different prices, so the
policy appeared to choose among three excursions while actually choosing among three
prices. At `fuel_weight = 0` the reward spread across `EXCURSE_{LOW,MID,HIGH}` is exactly
0.0, the action set is degenerate, and no action raises P(landing in a deep band) — which
makes a plume-gradient θ unable to change behaviour at all.

Falls back to a pooled row if `action` has no measured kernel, so a legacy artifact still
loads (`load_tables` handles the format bump).
"""
function alt_kernel(tables::AltTables, action::Symbol, key::KernelKey)
    action === :CORRECT && return tables.correct[key]
    haskey(tables.excurse, action) && return tables.excurse[action][key]
    # Legacy / unmeasured action: fall back to any available EXCURSE kernel rather than
    # throwing, but this is a DEGENERATE model — see the warning above.
    isempty(tables.excurse) && error("tables has no EXCURSE kernel for $action")
    return first(values(tables.excurse))[key]
end

alt_kernel(tables::AltTables, action::Symbol, alt::Symbol, residual::Symbol) =
    alt_kernel(tables, action, KernelKey(alt, residual))

# ── Serialization ─────────────────────────────────────────────────────────────
"""A `KernelKey` serializes as `"ALT|RESIDUAL"`, so the JSON stays eyeball-readable."""
_key_to_json(k::KernelKey) = string(k.alt, "|", k.residual)

function _key_from_json(s::AbstractString)
    parts = split(s, "|")
    length(parts) == 2 || error(
        "kernel row key \"$s\" is not \"ALT|RESIDUAL\". A single-part key is the " *
        "PRE-2026-08-31 format, whose rows are not conditioned on the residual " *
        "(orbit-damage) bin — re-measure with `experiments/calibrate.jl`.")
    return KernelKey(Symbol(parts[1]), Symbol(parts[2]))
end

_kernel_to_json(d::Dict{KernelKey,Vector{Float64}}) =
    Dict(_key_to_json(k) => v for (k, v) in d)

_kernel_from_json(d::AbstractDict) =
    Dict{KernelKey,Vector{Float64}}(_key_from_json(string(k)) => Float64.(v)
                                    for (k, v) in d)

"""Serialize/parse the two-level `excurse[action][key]` kernel."""
_excurse_to_json(d::Dict{Symbol,Dict{KernelKey,Vector{Float64}}}) =
    Dict(string(a) => _kernel_to_json(k) for (a, k) in d)

_excurse_from_json(d::AbstractDict) =
    Dict{Symbol,Dict{KernelKey,Vector{Float64}}}(
        Symbol(a) => _kernel_from_json(k) for (a, k) in d)

"""
    write_tables(tables; path = DEFAULT_TABLES_PATH)

Serialize measured kernels + provenance to JSON. Called by the calibration step, not by
the model.
"""
function write_tables(tables::AltTables; path::AbstractString = DEFAULT_TABLES_PATH)
    mkpath(dirname(path))
    payload = Dict(
        "alt_next" => string.(collect(ALT_ALL)),
        # The COLUMN convention, written out so a consumer never has to infer the
        # flattening: each entry is the joint successor "ALT|RESIDUAL".
        "columns"  => [string(a, "|", r) for (a, r) in kernel_columns()],
        "residual_bins"  => string.(collect(RESIDUAL_BINS)),
        "residual_edges" => collect(RESIDUAL_EDGES),
        "correct"  => _kernel_to_json(tables.correct),
        "excurse"  => _excurse_to_json(tables.excurse),
        # Format marker: 1 = pooled `excurse[bin]`, 2 = per-action `excurse[action][bin]`,
        # 3 = per-action AND residual-conditioned, joint (alt, residual) columns.
        "format"   => 3,
        "meta"     => tables.meta,
    )
    open(path, "w") do io
        JSON.print(io, payload, 2)
    end
    return path
end

"""
    load_tables(path = DEFAULT_TABLES_PATH) -> AltTables

Read measured kernels from JSON and validate them. Throws if the file's altitude ordering
disagrees with `ALT_ALL` (a silent reordering would corrupt every transition) or if any
row is not a normalized distribution.
"""
function load_tables(path::AbstractString = DEFAULT_TABLES_PATH)
    isfile(path) || error("measured tables not found at $path — run the calibration " *
                          "step (experiments/calibrate.jl) to generate them")
    raw = JSON.parsefile(path)

    haskey(raw, "alt_next") || error(
        "$path has no \"alt_next\" key. This is the PRE-2026-08-30 artifact, measured " *
        "against the old (dev, cov) state space, and it does not describe the current " *
        "(alt, visits) model. Re-measure with calibrate_tables rather than remapping it.")
    got = Symbol.(raw["alt_next"])
    got == collect(ALT_ALL) || error(
        "tables.json alt_next = $got disagrees with ALT_ALL = $(collect(ALT_ALL)); " *
        "the kernel columns would be misaligned")

    # ⚠️ FORMAT 2 IS REQUIRED (2026-08-30). Format 1 stored a SINGLE pooled `excurse[bin]`
    # row shared by every band, which makes EXCURSE_{LOW,MID,HIGH} mathematically identical
    # in T — a degenerate action set that silently defeats any science/altitude objective
    # (see `alt_kernel`). Detect it by shape (format 1's values are arrays of numbers,
    # format 2's are dicts keyed by action) and REJECT rather than remap: the per-band
    # aiming was never recorded in format 1, so it cannot be recovered from the file.
    fmt = get(raw, "format", 1)
    exc_raw = raw["excurse"]
    looks_pooled = !isempty(exc_raw) && first(values(exc_raw)) isa AbstractVector
    if fmt < 2 || looks_pooled
        error("$path is a FORMAT 1 artifact: `excurse` is a single pooled kernel shared " *
              "by every band, so EXCURSE_LOW/MID/HIGH would be identical in T and no " *
              "action would steer altitude. The per-band aiming is not recoverable from " *
              "this file — re-measure with `experiments/calibrate.jl`.")
    end

    # ⚠️ FORMAT 3 IS REQUIRED (2026-08-31). Format 2's rows are keyed on the altitude bin
    # ALONE and its columns are marginal over altitude, so it carries no information about
    # the orbit's DAMAGE — the conditioning that makes the killing transition visible at
    # all. A format-2 row is the AVERAGE of a fresh departure and a degraded one, which is
    # exactly the average that reports P(loss) = 0.0 for the transition that loses the
    # vehicle. The conditioning was never recorded, so it cannot be recovered by remapping.
    if fmt < 3
        error("$path is a FORMAT $fmt artifact: its rows are keyed on the altitude bin " *
              "alone, with no RESIDUAL (orbit-damage) conditioning, so every row pools " *
              "fresh and degraded departures into one number. That is the defect the " *
              "residual dimension exists to fix and it is not recoverable from this " *
              "file — re-measure with `experiments/calibrate.jl`.")
    end

    # The column convention must match, or every successor probability lands on the wrong
    # (alt, residual) pair. Checked explicitly rather than trusted, for the same reason
    # `alt_next` is: a silent reordering corrupts the whole kernel with no error.
    if haskey(raw, "columns")
        want_cols = [string(a, "|", r) for (a, r) in kernel_columns()]
        got_cols = String.(raw["columns"])
        got_cols == want_cols || error(
            "$path column convention disagrees with kernel_columns(); the joint " *
            "(alt, residual) successor probabilities would be misaligned.\n" *
            "  file : $got_cols\n  model: $want_cols")
    end
    if haskey(raw, "residual_bins")
        got_rb = Symbol.(raw["residual_bins"])
        got_rb == collect(RESIDUAL_BINS) || error(
            "$path residual_bins = $got_rb disagrees with " *
            "RESIDUAL_BINS = $(collect(RESIDUAL_BINS))")
    end
    # The EDGES are θ-adjacent: two artifacts with the same bin NAMES but different edges
    # describe different environments, and averaging them would be meaningless.
    if haskey(raw, "residual_edges")
        got_re = Float64.(raw["residual_edges"])
        got_re == collect(RESIDUAL_EDGES) || error(
            "$path was measured with residual_edges = $got_re, but this model uses " *
            "$(collect(RESIDUAL_EDGES)). The bin LABELS would match while meaning " *
            "different things — re-measure rather than reusing these kernels.")
    end

    tables = AltTables(
        _kernel_from_json(raw["correct"]),
        _excurse_from_json(exc_raw),
        Dict{String,Any}(get(raw, "meta", Dict())),
    )
    validate_tables(tables)
    return tables
end

"""
    validate_tables(tables)

Check every kernel row covers the live altitude bins and sums to 1. Cheap, and it catches
a hand-edited artifact before the error reaches the solver as a subtly wrong policy.
"""
function validate_tables(tables::AltTables)
    # Flatten the two-level excurse kernel into ("excurse/ACTION", rows) pairs so every
    # per-action row gets the same checks the pooled one used to get.
    groups = Pair{String,Dict{KernelKey,Vector{Float64}}}[("correct" => tables.correct)]
    for (a, k) in tables.excurse
        push!(groups, "excurse/$a" => k)
    end
    isempty(tables.excurse) && error("tables has no EXCURSE kernels at all")
    for (name, k) in groups
        # EVERY (alt × residual) conditioning must be present, not just every altitude.
        # A missing residual row is how the pooled-average defect would creep back in.
        for key in kernel_keys_all()
            haskey(k, key) || error("tables.$name is missing a row for $key")
            row = k[key]
            length(row) == N_KERNEL_COLS || error(
                "tables.$name[$key] has $(length(row)) entries, expected $N_KERNEL_COLS")
            any(<(0.0), row) && error("tables.$name[$key] has a negative probability")
            isapprox(sum(row), 1.0; atol = 1e-9) || error(
                "tables.$name[$key] sums to $(sum(row)), not 1")
        end
    end
    return true
end