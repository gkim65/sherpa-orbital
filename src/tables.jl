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

Rows are P(next alt | alt, action) over ALT_ALL = the five live altitude bins plus
(:CRASHED, :LOST).

⚠️ REKEYED 2026-08-30 from dev bins to ALTITUDE bins. The committed `artifacts/tables.json`
was measured against the old (dev, cov) state space and does NOT describe this model;
`load_tables` rejects it rather than silently misaligning the kernel columns. Re-measure
with `calibrate_tables` before quoting any result.
"""

const DEFAULT_TABLES_PATH =
    normpath(joinpath(@__DIR__, "..", "artifacts", "tables.json"))

"""
    AltTables

Measured altitude-transition kernels, keyed by action group then by current altitude bin,
plus the provenance metadata carried alongside them.
"""
struct AltTables
    correct::Dict{Symbol,Vector{Float64}}
    # ⚠️ TWO-LEVEL AS OF 2026-08-30: excurse[ACTION][from_bin], one kernel PER EXCURSE
    # ACTION. It was previously `excurse[from_bin]` — a single kernel shared by every band.
    # See `alt_kernel` for why that was wrong and what it cost.
    excurse::Dict{Symbol,Dict{Symbol,Vector{Float64}}}
    meta::Dict{String,Any}
end

"""
    alt_kernel(tables, action, alt) -> Vector{Float64}

The measured next-altitude distribution for `action` from bin `alt`, over `ALT_ALL`.

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
function alt_kernel(tables::AltTables, action::Symbol, alt::Symbol)
    action === :CORRECT && return tables.correct[alt]
    haskey(tables.excurse, action) && return tables.excurse[action][alt]
    # Legacy / unmeasured action: fall back to any available EXCURSE kernel rather than
    # throwing, but this is a DEGENERATE model — see the warning above.
    isempty(tables.excurse) && error("tables has no EXCURSE kernel for $action")
    return first(values(tables.excurse))[alt]
end

# ── Serialization ─────────────────────────────────────────────────────────────
_kernel_to_json(d::Dict{Symbol,Vector{Float64}}) =
    Dict(string(k) => v for (k, v) in d)

_kernel_from_json(d::AbstractDict) =
    Dict{Symbol,Vector{Float64}}(Symbol(k) => Float64.(v) for (k, v) in d)

"""Serialize/parse the two-level `excurse[action][from_bin]` kernel."""
_excurse_to_json(d::Dict{Symbol,Dict{Symbol,Vector{Float64}}}) =
    Dict(string(a) => _kernel_to_json(k) for (a, k) in d)

_excurse_from_json(d::AbstractDict) =
    Dict{Symbol,Dict{Symbol,Vector{Float64}}}(
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
        "correct"  => _kernel_to_json(tables.correct),
        "excurse"  => _excurse_to_json(tables.excurse),
        # Format marker: 1 = pooled `excurse[bin]`, 2 = per-action `excurse[action][bin]`.
        "format"   => 2,
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
    groups = Pair{String,Dict{Symbol,Vector{Float64}}}[("correct" => tables.correct)]
    for (a, k) in tables.excurse
        push!(groups, "excurse/$a" => k)
    end
    isempty(tables.excurse) && error("tables has no EXCURSE kernels at all")
    for (name, k) in groups
        for alt in ALT_BINS
            haskey(k, alt) || error("tables.$name is missing a row for alt=$alt")
            row = k[alt]
            length(row) == length(ALT_ALL) || error(
                "tables.$name[$alt] has $(length(row)) entries, expected $(length(ALT_ALL))")
            any(<(0.0), row) && error("tables.$name[$alt] has a negative probability")
            isapprox(sum(row), 1.0; atol = 1e-9) || error(
                "tables.$name[$alt] sums to $(sum(row)), not 1")
        end
    end
    return true
end