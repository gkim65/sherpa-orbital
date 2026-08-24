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

Rows are P(next dev | dev, action) over DEV_NEXT = (:OK,:DRIFT,:FAR,:LOST,:CRASHED).
"""

const DEFAULT_TABLES_PATH =
    normpath(joinpath(@__DIR__, "..", "artifacts", "tables.json"))

"""
    DevTables

Measured dev-transition kernels, keyed by action group then by current dev bin, plus the
provenance metadata carried alongside them.
"""
struct DevTables
    correct::Dict{Symbol,Vector{Float64}}
    observe::Dict{Symbol,Vector{Float64}}
    excurse::Dict{Symbol,Vector{Float64}}
    meta::Dict{String,Any}
end

"""
    dev_kernel(tables, action, dev) -> Vector{Float64}

The measured next-dev distribution for `action` from bin `dev`, over `DEV_NEXT`. All
EXCURSE_* actions share one kernel: the measured safety cost of an excursion did not
differ meaningfully by band (exp 12), only its ΔV cost did.
"""
function dev_kernel(tables::DevTables, action::Symbol, dev::Symbol)
    action === :CORRECT && return tables.correct[dev]
    action === :OBSERVE && return tables.observe[dev]
    return tables.excurse[dev]
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
function write_tables(tables::DevTables; path::AbstractString = DEFAULT_TABLES_PATH)
    mkpath(dirname(path))
    payload = Dict(
        "dev_next" => string.(collect(DEV_NEXT)),
        "correct"  => _kernel_to_json(tables.correct),
        "observe"  => _kernel_to_json(tables.observe),
        "excurse"  => _kernel_to_json(tables.excurse),
        "meta"     => tables.meta,
    )
    open(path, "w") do io
        JSON.print(io, payload, 2)
    end
    return path
end

"""
    load_tables(path = DEFAULT_TABLES_PATH) -> DevTables

Read measured kernels from JSON and validate them. Throws if the file's dev ordering
disagrees with `DEV_NEXT` (a silent reordering would corrupt every transition) or if any
row is not a normalized distribution.
"""
function load_tables(path::AbstractString = DEFAULT_TABLES_PATH)
    isfile(path) || error("measured tables not found at $path — run the calibration " *
                          "step (experiments/calibrate.jl) to generate them")
    raw = JSON.parsefile(path)

    got = Symbol.(raw["dev_next"])
    got == collect(DEV_NEXT) || error(
        "tables.json dev_next = $got disagrees with DEV_NEXT = $(collect(DEV_NEXT)); " *
        "the kernel columns would be misaligned")

    tables = DevTables(
        _kernel_from_json(raw["correct"]),
        _kernel_from_json(raw["observe"]),
        _kernel_from_json(raw["excurse"]),
        Dict{String,Any}(get(raw, "meta", Dict())),
    )
    validate_tables(tables)
    return tables
end

"""
    validate_tables(tables)

Check every kernel row covers the non-terminal dev bins and sums to 1. Cheap, and it
catches a hand-edited artifact before the error reaches the solver as a subtly wrong
policy.
"""
function validate_tables(tables::DevTables)
    for (name, k) in (("correct", tables.correct), ("observe", tables.observe),
                      ("excurse", tables.excurse))
        for dev in NONTERM_DEV
            haskey(k, dev) || error("tables.$name is missing a row for dev=$dev")
            row = k[dev]
            length(row) == length(DEV_NEXT) || error(
                "tables.$name[$dev] has $(length(row)) entries, expected $(length(DEV_NEXT))")
            any(<(0.0), row) && error("tables.$name[$dev] has a negative probability")
            isapprox(sum(row), 1.0; atol = 1e-9) || error(
                "tables.$name[$dev] sums to $(sum(row)), not 1")
        end
    end
    return true
end