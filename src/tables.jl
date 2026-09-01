"""
tables.jl — the measured transition kernels, as a versioned artifact.

Rows are `P(next alt, next residual | alt, residual, action)` — a joint distribution over
`ALT_ALL × RESIDUAL_BINS`, flattened with the residual varying fastest (see
[`kernel_entry_index`](@ref)). They live in `artifacts/tables.json` alongside the θ and
effort they were measured under, so a re-measurement is a reviewable diff.

JSON rather than a binary format on purpose: the artifact is committed, and eyeballing a
changed probability in review matters more than compactness. Swapping to JLD2 is a drop-in
behind `load_tables` / `write_tables`.

NOTE: `load_tables` rejects older artifact formats rather than remapping them. Neither the
per-action keying (format 2) nor the residual conditioning (format 3) is recoverable from
a file that never recorded it.
"""

const DEFAULT_TABLES_PATH =
    normpath(joinpath(@__DIR__, "..", "artifacts", "tables.json"))

"""
    KernelKey

The conditioning of one measured row: which altitude bin the pass departed from, and how
degraded the orbit was when it did.

  - `alt` — departure altitude bin, a member of `ALT_BINS`
  - `residual` — departure orbit-damage bin, a member of `RESIDUAL_BINS`

NOTE: keyed on `alt` alone, a row pools a fresh departure with one from an orbit already
far out of trim and reports their average — which is P(loss) = 0.0, because fresh
departures dominate the sample.
"""
struct KernelKey
    alt::Symbol
    residual::Symbol
end

Base.show(io::IO, k::KernelKey) = print(io, k.alt, "/", k.residual)

"""
    kernel_keys_all() -> Vector{KernelKey}

Every (live altitude bin x residual bin) conditioning, altitude varying slowest.

Returns `length(ALT_BINS) * length(RESIDUAL_BINS)` keys in the canonical order rows are
enumerated and reported in.
"""
kernel_keys_all() = [KernelKey(a, r) for a in ALT_BINS for r in RESIDUAL_BINS]

"""
    kernel_entry_index(alt, residual) -> Int
    kernel_columns() -> Vector{Tuple{Symbol,Symbol}}

The flattening convention for a kernel row. The successor is a joint `(alt, residual)` and
the columns run over `ALT_BINS × RESIDUAL_BINS` with the residual varying fastest, then the
two terminal altitudes last.

Row length is `|ALT_BINS| * |RESIDUAL_BINS| + |TERMINAL_ALT|` = 17.

NOTE: the terminal altitudes are carried once each, not once per residual bin — a lost
orbit ran no onboard solve, so it has no residual.
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
    # Two-level: excurse[ACTION][key], one kernel per EXCURSE action. A single pooled
    # kernel would make the excursions identical in T — see `alt_kernel`.
    excurse::Dict{Symbol,Dict{KernelKey,Vector{Float64}}}
    meta::Dict{String,Any}
end

"""
    alt_kernel(tables, action, key) -> Vector{Float64}
    alt_kernel(tables, action, alt, residual) -> Vector{Float64}

The measured joint successor distribution for `action` departing `(alt, residual)`.

  - `tables` — the loaded kernels
  - `action` — `:CORRECT` or an `:EXCURSE_<BAND>`
  - `key` / `alt`, `residual` — the departure conditioning

Returns a probability vector over [`kernel_columns`](@ref).

NOTE: one kernel per EXCURSE action, because a kernel encodes where you LAND and that is
the point of aiming at a band. Pooling them across bands makes the excursions identical in
T, so no action raises P(landing in a deep band) and a plume-gradient θ cannot change
behaviour at all. At `fuel_weight = 0` the reward spread across them is then exactly 0.0.

Falls back to any available EXCURSE kernel if `action` has none of its own.
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
        "kernel row key \"$s\" is not \"ALT|RESIDUAL\". A single-part key is an older " *
        "format whose rows carry no residual (orbit-damage) conditioning — re-measure " *
        "with `experiments/calibrate.jl`.")
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

Read measured kernels from JSON and validate them.

  - `path` — artifact to read; defaults to the packaged one

Returns an [`AltTables`](@ref). Throws if the artifact predates the current format, if its
altitude ordering, column convention or residual edges disagree with the model, or if any
row is not a normalized distribution — a silent mismatch would corrupt every transition.
"""
function load_tables(path::AbstractString = DEFAULT_TABLES_PATH)
    isfile(path) || error("measured tables not found at $path — run the calibration " *
                          "step (experiments/calibrate.jl) to generate them")
    raw = JSON.parsefile(path)

    haskey(raw, "alt_next") || error(
        "$path has no \"alt_next\" key, so it predates altitude-keyed kernels and does " *
        "not describe this model. Re-measure with calibrate_tables rather than remapping.")
    got = Symbol.(raw["alt_next"])
    got == collect(ALT_ALL) || error(
        "tables.json alt_next = $got disagrees with ALT_ALL = $(collect(ALT_ALL)); " *
        "the kernel columns would be misaligned")

    # Format 1 stored a single pooled `excurse[bin]` shared by every band, making the
    # excursions identical in T. Detected by shape — format 1's values are arrays, later
    # formats' are dicts keyed by action — and rejected, since the per-band aiming it never
    # recorded cannot be recovered.
    fmt = get(raw, "format", 1)
    exc_raw = raw["excurse"]
    looks_pooled = !isempty(exc_raw) && first(values(exc_raw)) isa AbstractVector
    if fmt < 2 || looks_pooled
        error("$path is a FORMAT 1 artifact: `excurse` is a single pooled kernel shared " *
              "by every band, so EXCURSE_LOW/MID/HIGH would be identical in T and no " *
              "action would steer altitude. The per-band aiming is not recoverable from " *
              "this file — re-measure with `experiments/calibrate.jl`.")
    end

    # Format 2's rows are keyed on the altitude bin alone, so they carry no orbit-damage
    # conditioning: each row averages a fresh departure with a degraded one and reports
    # P(loss) = 0.0 for the transition that loses the vehicle. Not recoverable by remapping.
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

Check every kernel row is present, correctly sized, non-negative and sums to 1.

  - `tables` — the kernels to check

Returns `true`, or throws naming the offending row. Cheap, and it catches a hand-edited
artifact before the error reaches the solver as a subtly wrong policy.
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