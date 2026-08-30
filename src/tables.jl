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
    excurse::Dict{Symbol,Vector{Float64}}
    meta::Dict{String,Any}
end

"""
    alt_kernel(tables, action, alt) -> Vector{Float64}

The measured next-altitude distribution for `action` from bin `alt`, over `ALT_ALL`. All
EXCURSE_* actions share one kernel: the measured safety cost of an excursion did not
differ meaningfully by band (exp 12), only its ΔV cost did. ⚠️ That finding predates the
band rebase to 20–50 km — re-check it when the kernels are re-measured.
"""
function alt_kernel(tables::AltTables, action::Symbol, alt::Symbol)
    action === :CORRECT && return tables.correct[alt]
    return tables.excurse[alt]
end

# ── Serialization ─────────────────────────────────────────────────────────────
_kernel_to_json(d::Dict{Symbol,Vector{Float64}}) =
    Dict(string(k) => v for (k, v) in d)

_kernel_from_json(d::AbstractDict) =
    Dict{Symbol,Vector{Float64}}(Symbol(k) => Float64.(v) for (k, v) in d)

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
        "excurse"  => _kernel_to_json(tables.excurse),
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

    tables = AltTables(
        _kernel_from_json(raw["correct"]),
        _kernel_from_json(raw["excurse"]),
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
    for (name, k) in (("correct", tables.correct), ("excurse", tables.excurse))
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