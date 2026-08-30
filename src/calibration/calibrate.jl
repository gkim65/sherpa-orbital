"""
calibration/calibrate.jl — MEASURE the dev-transition kernels from the Julia truth model.

WHAT THIS REPLACES
`artifacts/tables.json` has until now held HAND-TRANSCRIBED numbers, copied out of the
Python experiment logs (exp 11b for CORRECT/OBSERVE, exp 12 for EXCURSE) with the DRIFT/FAR
rows filled in by engineering judgement anchored on very few trials. That artifact's own
`meta.caveats` says so. This module measures them from the ported Julia truth model, so the
committed artifact becomes a machine-generated record with a reproducible provenance.

⚠️ THE MEASUREMENT IS THE PRODUCT — read the caveats before trusting a row.

  1. THE ORBIT IS THE KNOWN-DEFECTIVE ONE. The period-3 IC's periapsis sits at +87°
     latitude — the NORTH pole — not the south pole the science case wants (docs/todo.md
     "Step A′"). Every kernel here is conditional on that orbit. A row must NOT be
     presented as characterising the intended south-polar orbit.
  2. THE ORBIT IS VIOLENTLY UNSTABLE. It escapes in ~2 revs uncontrolled, so the OBSERVE
     rows are dominated by that instability rather than by model uncertainty (see the
     planner-model ablation: failure is instability-dominated).
  3. FAR IS BARELY REACHABLE UNDER A HOLDING CONTROLLER. The sustained CORRECT loop keeps
     dev small by construction, so the loop visits OK constantly, DRIFT occasionally, and
     FAR almost never. Rows are reported WITH THEIR TRIAL COUNTS and a row measured from
     too few trials is flagged, not silently smoothed.
  4. `mode = :position` (Strategy 3 proper, which the kernels are defined against) does not
     reliably converge on this orbit — measured in `scratch/compare/compare_simulate.jl`
     section [D]. A non-converged `solve_burn` returns ΔV = 0, i.e. CORRECT degenerates
     into OBSERVE. That is recorded per row as `n_nonconverged`.

METHOD
  CORRECT : run the sustained shell-cadence CORRECT loop and log dev_before → dev_after.
  OBSERVE : from the SAME visited shell states, coast one pass with NO burn — a
            counterfactual "what if we had observed instead", so both rows condition on the
            identical state distribution rather than on two different trajectories.
  EXCURSE : a one-pass excursion toward a band's scaled periapsis, then ONE recovery
            CORRECT, logging the dev bin after recovery. All EXCURSE_* share one kernel:
            exp 12 found excursion SAFETY did not differ meaningfully by band, only ΔV cost.

⚠️ TRUTH/ONBOARD SPLIT. `truth_eom!` is an argument and is integrated only in the coast
helpers; all planning goes through `solve_burn` on the onboard CR3BP.
"""

"""
    CalibrationRow

One measured kernel row: the counts over `ALT_ALL`, the normalized probabilities, and the
diagnostics needed to judge whether the row is trustworthy.
"""
struct CalibrationRow
    counts::Vector{Int}          # over ALT_ALL
    probs::Vector{Float64}       # counts ./ n, or a fallback if n == 0
    n::Int
    n_nonconverged::Int          # solve_burn failures folded into this row
    dv_ms::Vector{Float64}       # per-trial ΔV, for the action_dv_cost proxy
end

"""Representative altitude (km) for seeding a bin: its midpoint, or a sensible interior
point for the two open-ended outer bins."""
function _bin_rep_alt(bin::Symbol, e::NTuple{4,<:Real})
    # NOT the midpoint (10 km): the continued family floor is 17.565 km, so a 10 km seed
    # finds no member and the row silently stays unmeasured. Seed just inside the family.
    bin === :BELOW_20 && return 18.0
    bin === :A20_30   && return 0.5 * (e[1] + e[2])
    bin === :A30_40   && return 0.5 * (e[2] + e[3])
    bin === :A40_50   && return 0.5 * (e[3] + e[4])
    return e[4] + 0.5 * (e[4] - e[3])                   # 55 km
end

"""Minimum trials before a measured row is considered anything but indicative."""
const MIN_TRIALS_TRUSTED = 20

# ── Coast helpers (the ONLY place truth is integrated) ─────────────────────────
"""
    _coast(truth_eom!, state, horizon, make_primary, arg) -> (:ok|:crash|:escape|:none, u)

Propagate under truth to the primary event, watching for a crash or an escape. Shares the
crash/escape shells with the rollout harness so calibration and rollout cannot disagree
about what "lost" means.
"""
function _coast(truth_eom!, state::AbstractVector{<:Real}, horizon::Real,
                make_primary, arg)
    r = _coast_to(truth_eom!, state, horizon, make_primary, arg;
                  rtol = RTOL_TRUTH, atol = ATOL_TRUTH)
    return r.outcome, r.u
end

_to_shell(eom!, s, h) = _coast(eom!, s, h, _terminal_shell_callback, CONTROL_ALT_KM)
_to_peri(eom!, s, h)  = _coast(eom!, s, h, _terminal_periapsis_callback_arg, nothing)

"""Map a terminal coast outcome onto its `ALT_ALL` label."""
_terminal_dev(outcome::Symbol) = outcome === :crash ? :CRASHED : :LOST

"""
    _excurse_targets(target_km, r_peri_nom, r_apo_nom, apo_nom_alt) -> (r_peri, r_apo)

Apse POSITION targets for a one-pass excursion to `target_km`: the nominal periapsis vector
radially scaled to the target altitude, with apoapsis held nominal.

⚠️ A SCALED VECTOR IS NOT A REFERENCE ORBIT, and that is fine HERE but nowhere else. An
excursion is a single-impulse dip lasting one pass, so the target only has to be a reachable
waypoint, not a sustainable orbit.

⚠️ DO NOT "FIX" THIS BY TARGETING A REAL FAMILY MEMBER'S APSES FROM A DRIFTED STATE.
Tried and measured 2026-08-30, and the failure is PHASE, not altitude.

`retarget_to_altitude` returns a genuine periodic orbit (20/25/30/40/50 km met to 4
decimals). But `next_apse_positions(member.ic)` returns where THAT MEMBER'S periapsis sits
at ITS OWN epoch, which has no relation to where our spacecraft's next periapsis will be.
Measured from the 31 km limit cycle against the 30 km member: the two orbits' periapsis
ALTITUDES differ by ~22 km, but their periapsis POSITIONS are **585 km apart** and their
apoapsis positions **2319 km apart**. The `:position` residual at ΔV = 0 is **2392 km**, so
the solver burns 78–180 m/s trying to move the apse across the orbit and throws the vehicle
out:

  | commanded | family-member apses      | scaled waypoint    |
  |-----------|--------------------------|--------------------|
  | 20 km     | 177.6 m/s -> ESCAPE      | 2.39 m/s -> 37.9 km|
  | 30 km     |  78.7 m/s -> 593.8 km    |                    |
  | 40 km     | 179.8 m/s -> 395.5 km    | 1.51 m/s -> 42.8 km|
  | 50 km     |  73.0 m/s -> 562.0 km    | 1.23 m/s -> 45.3 km|

⚠️ THIS IS NOT A REGRESSION AND NOT A DEFECT IN `retarget_to_altitude`. Verified against the
pre-2026-08-30 planner: identical numbers. `MPCController.target_alt_km` works (50 -> 56.01,
70 -> 75.86 km, held 30 d) because it calls `next_apse_positions(member.ic)` at SETUP, from
the IC, so vehicle and reference start phase-aligned and stay so. Calling it mid-flight from
a drifted shell state breaks that alignment. It is the mirror image of the standing
"measure at a drifted state, never at the reference IC" trap.

⟹ THE OPEN FIX is PHASE MATCHING: take the member's periapsis RADIUS but place it where our
own trajectory's next periapsis actually is, rather than importing the member's absolute
position. `_scale_to_altitude` already does exactly that, which is why the "fake" waypoint
outperforms the "real" orbit by ~100x here. See docs/todo.md.

⚠️ Separately: single-impulse command authority is poor and pre-existing. Commanding 20 vs
50 km moves achieved periapsis only 37.9 -> 45.3 km — the known ~25% authority finding
(`docs/todo.md`), not a defect of this function.
"""
_excurse_targets(target_km::Real, r_peri_nom, r_apo_nom, apo_nom_alt) =
    (_scale_to_altitude(r_peri_nom, target_km),
     _scale_to_altitude(r_apo_nom, apo_nom_alt))

"""
    calibrate_tables(; truth_eom!, n_steps, horizon_s, alt_edges, band_target_km,
                     mode, verbose) -> (AltTables, diagnostics)

Measure all three kernels and return them alongside a per-row diagnostics dict.

  - `truth_eom!` — the truth model to calibrate against. Defaults to
    [`cr3bp_j2_eom!`](@ref) (CR3BP + Enceladus J2), which is the rung the committed
    hand-transcribed tables were measured on, so the diff is apples-to-apples.
  - `n_steps` — sustained-loop steps to log for the CORRECT/OBSERVE rows.
  - `mode` — targeting mode for `solve_burn`. `:position` is Strategy 3 proper and matches
    how the kernels are defined; see caveat 4 above about its convergence.
"""
function calibrate_tables(;
    truth_eom! = cr3bp_j2_eom!,
    truth_name::AbstractString = "CR3BP + Enceladus J2",
    ic::AbstractVector{<:Real} = nondim_to_cr3bp(collect(PERIOD1_NORTH_IC_ND)),
    period_s::Real = PERIOD1_TRIPLE_PERIOD_S,
    n_steps::Integer = 120,
    horizon_s::Real = 25 * 86400.0,
    alt_edges::NTuple{4,Float64} = (20.0, 30.0, 40.0, 50.0),
    band_names::NTuple{3,Symbol} = (:LOW, :MID, :HIGH),
    band_target_km::Dict{Symbol,Float64} =
        Dict(:LOW => 25.0, :MID => 35.0, :HIGH => 45.0),
    mode::Symbol = :position,
    excurse_trials::Integer = 8,
    family_table::Union{Nothing,Vector{NamedTuple}} = halo_family_table_cached(),
    seed_bins::Bool = true,
    seed_trials::Integer = 3,
    verbose::Bool = false,
)
    ic = collect(float.(ic))
    period_s = float(period_s)
    one_rev_s = period_s / 3          # 3 periapsis passes per period-3 orbit

    # Count-based apse search: `one_rev_s = T/3` is the inter-periapsis CONTROL cadence, not
    # an apse-to-apse revolution — as a search window it lands 0.136 s short of the first
    # apoapsis and yields a NaN target, which silently zeroes every CORRECT burn.
    r_peri_nom, r_apo_nom = next_apse_positions(ic; eom! = cr3bp_eom!)
    apo_nom_alt = norm(_enc_relative(r_apo_nom)) - R_ENCELADUS

    # Bin by achieved periapsis ALTITUDE. ⚠️ This must stay bit-identical to `alt_bin`
    # in states.jl, or the kernels are labelled with bins the model does not use.
    bin_of(h) = !isfinite(h) ? :LOST :
                h < alt_edges[1] ? :BELOW_20 :
                h < alt_edges[2] ? :A20_30 :
                h < alt_edges[3] ? :A30_40 :
                h < alt_edges[4] ? :A40_50 : :ABOVE_50
    alt_of(u) = norm(_enc_relative(u[1:3])) - R_ENCELADUS

    nxt = collect(ALT_ALL)
    # counts[action][alt_from] -> Vector{Int} over ALT_ALL
    counts = Dict(a => Dict(d => zeros(Int, length(nxt)) for d in ALT_BINS)
                  for a in (:CORRECT, :OBSERVE, :EXCURSE))
    nonconv = Dict(a => Dict(d => 0 for d in ALT_BINS)
                   for a in (:CORRECT, :OBSERVE, :EXCURSE))
    dvs = Dict(a => Dict(d => Float64[] for d in ALT_BINS)
               for a in (:CORRECT, :OBSERVE, :EXCURSE))
    # ΔV per band, for the action_dv_cost proxy.
    band_dv = Dict(b => Float64[] for b in band_names)
    band_achieved_alt = Dict(b => Float64[] for b in band_names)

    bump!(a, from, to) = counts[a][from][findfirst(==(to), nxt)] += 1

    # ── CORRECT + OBSERVE: the sustained loop plus its counterfactual ──────────
    # Get onto the orbit at a periapsis first, so step 1 starts in the same phase every
    # later step does.
    st, u = _to_peri(truth_eom!, ic, 4 * one_rev_s)
    state = st === :ok ? copy(u) : copy(ic)
    t = 0.0
    nlogged = 0

    while t < horizon_s && nlogged < n_steps
        from = bin_of(alt_of(state))
        from in ALT_BINS || break

        sh, sc = _to_shell(truth_eom!, state, horizon_s - t)
        if sh !== :ok
            bump!(:CORRECT, from, _terminal_dev(sh))
            break
        end

        # OBSERVE counterfactual from this shell state: coast one pass, no burn.
        po, uo = _to_peri(truth_eom!, sc, 4 * one_rev_s)
        bump!(:OBSERVE, from, po === :ok ? bin_of(alt_of(uo)) : _terminal_dev(po))
        push!(dvs[:OBSERVE][from], 0.0)

        # CORRECT: the branch the loop actually takes.
        b = solve_burn(sc, one_rev_s; eom! = cr3bp_eom!, mode = mode,
                       r_peri_nom = r_peri_nom, r_apo_nom = r_apo_nom)
        b.converged || (nonconv[:CORRECT][from] += 1)
        push!(dvs[:CORRECT][from], b.dv_mag_ms)

        spost = copy(sc)
        spost[4:6] .+= b.dv
        pc, uc = _to_peri(truth_eom!, spost, 4 * one_rev_s)
        bump!(:CORRECT, from, pc === :ok ? bin_of(alt_of(uc)) : _terminal_dev(pc))

        verbose && @printf("  step %3d  %-6s -> CORRECT ΔV=%6.3f m/s %s  -> %s\n",
                           nlogged + 1, from, b.dv_mag_ms,
                           b.converged ? "  " : "NC",
                           pc === :ok ? string(bin_of(alt_of(uc))) : string(_terminal_dev(pc)))

        pc === :ok || break
        state = copy(uc)
        t += one_rev_s
        nlogged += 1
    end

    # ── EXCURSE: one step of a PERSISTENT walk toward a band ───────────────────
    # ⚠️ CHANGED 2026-08-30, twice, and both changes matter.
    #
    # (1) The outcome is where the EXCURSION lands. This previously coasted through a
    #     recovery CORRECT and binned the state AFTER it, so every excursion recorded as
    #     landing back on the nominal orbit — a 100% self-transition telling the policy that
    #     excursions do nothing.
    # (2) The reference PERSISTS across passes, matching `SARSOPController.active_band`.
    #     An excursion is not a one-pass dip: single-impulse authority is only ~25%, so
    #     reaching a commanded altitude takes several passes holding the same reference.
    #     Each trial here is therefore ONE STEP of that walk, started from where the
    #     previous step left off — which is exactly the transition the policy reasons about
    #     when it decides whether to keep aiming.
    #
    # The kernel is measured from a REAL family member at the band altitude, because a
    # persistent reference has to be an orbit the vehicle can settle onto. (A scaled
    # waypoint is fine for a single dip and wrong for this; aiming ONE impulse at a family
    # member escapes — see `_excurse_targets`. The difference is that here the same target
    # is held for consecutive passes rather than jumped at once.)
    st0, u0 = _to_peri(truth_eom!, ic, 4 * one_rev_s)
    base = st0 === :ok ? copy(u0) : copy(ic)

    for band in band_names
        target = band_target_km[band]
        member = family_table === nothing ? nothing :
                 retarget_to_altitude(family_table, target)
        if member === nothing
            verbose && @info "excurse: no family member at $(target) km for $band — skipped"
            continue
        end
        # Phase-matched: the member gives the periapsis RADIUS, our nominal apses give the
        # direction and the apoapsis anchor. See `_excurse_targets` for why importing the
        # member's absolute apse positions escapes.
        rp = _scale_to_altitude(r_peri_nom, member.info.periapsis_alt_km)
        ra = collect(r_apo_nom)

        walker = copy(base)          # carried across trials: the walk accumulates
        for _ in 1:excurse_trials
            from = bin_of(alt_of(walker))
            from in ALT_BINS || break

            sh, sc = _to_shell(truth_eom!, walker, 4 * one_rev_s)
            if sh !== :ok
                bump!(:EXCURSE, from, _terminal_dev(sh))
                break
            end

            b = solve_burn(sc, one_rev_s; eom! = cr3bp_eom!, mode = :position,
                           r_peri_nom = rp, r_apo_nom = ra)
            b.converged || (nonconv[:EXCURSE][from] += 1)

            sp = copy(sc)
            sp[4:6] .+= b.dv
            pe, ue = _to_peri(truth_eom!, sp, 4 * one_rev_s)
            if pe !== :ok
                bump!(:EXCURSE, from, _terminal_dev(pe))
                push!(dvs[:EXCURSE][from], b.dv_mag_ms)
                push!(band_dv[band], b.dv_mag_ms)
                break
            end

            bump!(:EXCURSE, from, bin_of(alt_of(ue)))
            push!(dvs[:EXCURSE][from], b.dv_mag_ms)
            push!(band_dv[band], b.dv_mag_ms)
            push!(band_achieved_alt[band], altitude(ue))

            verbose && @printf("  excurse %-4s step from %-8s ΔV=%6.3f m/s -> alt %7.2f km (%s)\n",
                               band, from, b.dv_mag_ms, altitude(ue), bin_of(alt_of(ue)))
            walker = copy(ue)        # continue the walk from here
        end
    end

    # ── SEED the bins the natural loop never visits ────────────────────────────
    # ⚠️ WITHOUT THIS, THREE OF FIVE ROWS ARE EMPTY. Measured 2026-08-30: a 50-step
    # sustained loop puts 50/50 CORRECT trials in A30_40 and 1 in A40_50, and never enters
    # BELOW_20, A20_30 or ABOVE_50 at all — a working controller has no reason to go there.
    # Those rows are still needed, because the solver queries every state, and filling them
    # with a guess would teach the policy something we never measured (e.g. that a low
    # periapsis is unrecoverable). So we place the spacecraft ON a real halo-family member
    # at each bin's representative altitude and measure what the controller actually does
    # from there. A bin with no family member is left unmeasured rather than invented.
    if seed_bins && family_table !== nothing
        for bin in ALT_BINS
            rep = _bin_rep_alt(bin, alt_edges)
            m = retarget_to_altitude(family_table, rep)
            if m === nothing
                verbose && @info "seed: no family member near $(rep) km for $bin — row left unmeasured"
                continue
            end
            seed_ic = collect(float.(m.ic))
            ps, us = _to_peri(truth_eom!, seed_ic, 4 * one_rev_s)
            ps === :ok || continue
            # Hold the SEEDED orbit, not the nominal one: targeting a 31 km reference from
            # 18 km would measure a transfer, not stationkeeping in that bin.
            rp_s, ra_s = next_apse_positions(seed_ic; eom! = cr3bp_eom!)

            for _ in 1:seed_trials
                st_s = copy(us)
                from = bin_of(alt_of(st_s))
                from in ALT_BINS || continue

                # OBSERVE: coast one pass, no burn.
                po, uo = _to_peri(truth_eom!, st_s, 4 * one_rev_s)
                bump!(:OBSERVE, from, po === :ok ? bin_of(alt_of(uo)) : _terminal_dev(po))
                push!(dvs[:OBSERVE][from], 0.0)

                # CORRECT: burn at the shell toward the seeded orbit, then coast.
                shs, scs = _to_shell(truth_eom!, st_s, 4 * one_rev_s)
                if shs !== :ok
                    bump!(:CORRECT, from, _terminal_dev(shs))
                    continue
                end
                bs = solve_burn(scs, one_rev_s; eom! = cr3bp_eom!, mode = mode,
                                r_peri_nom = rp_s, r_apo_nom = ra_s)
                bs.converged || (nonconv[:CORRECT][from] += 1)
                sps = copy(scs); sps[4:6] .+= bs.dv
                pcs, ucs = _to_peri(truth_eom!, sps, 4 * one_rev_s)
                bump!(:CORRECT, from, pcs === :ok ? bin_of(alt_of(ucs)) : _terminal_dev(pcs))
                push!(dvs[:CORRECT][from], bs.dv_mag_ms)
                pcs === :ok && (us = ucs)
            end
        end
    end

    # ── Assemble ──────────────────────────────────────────────────────────────
    rows = Dict{Symbol,Dict{Symbol,CalibrationRow}}()
    for a in (:CORRECT, :OBSERVE, :EXCURSE)
        rows[a] = Dict{Symbol,CalibrationRow}()
        for d in ALT_BINS
            c = counts[a][d]
            n = sum(c)
            rows[a][d] = CalibrationRow(c, n == 0 ? Float64[] : c ./ n, n,
                                        nonconv[a][d], dvs[a][d])
        end
    end

    diagnostics = Dict{String,Any}(
        "n_loop_steps"       => nlogged,
        "rows"               => rows,
        "band_dv_ms"         => Dict(string(b) => band_dv[b] for b in band_names),
        "band_achieved_alt"  => Dict(string(b) => band_achieved_alt[b] for b in band_names),
        "truth_name"         => truth_name,
        "mode"               => string(mode),
        "alt_edges"          => collect(alt_edges),
        "min_trials_trusted" => MIN_TRIALS_TRUSTED,
    )
    return rows, diagnostics
end
"""
    tables_from_rows(rows, diagnostics) -> AltTables

Assemble measured [`CalibrationRow`](@ref)s into an [`AltTables`](@ref) ready for
[`write_tables`](@ref). This is the step that was missing while `artifacts/tables.json`
held hand-transcribed numbers: `calibrate_tables` measured the rows and nothing turned
them into the artifact, so the numbers were copied across by hand and could drift from the
run that produced them with nothing to detect it.

⚠️ A ROW WITH `n = 0` IS NOT MEASURED. `calibrate_tables(seed_bins = true)` seeds a trial
from every altitude bin precisely so this does not happen, but a bin with no reachable halo
family member still comes back empty. Such rows are filled with a self-transition ONLY so
the kernel is a valid stochastic matrix, and `meta.trials` records the count for every row
so the fill is auditable rather than invisible. A policy is no more trustworthy than its
least-visited row — check `meta.trials` against [`MIN_TRIALS_TRUSTED`](@ref) before
believing what the policy does in a rarely-visited bin.
"""
function tables_from_rows(rows::Dict{Symbol,Dict{Symbol,CalibrationRow}},
                          diagnostics::AbstractDict)
    nxt = collect(ALT_ALL)
    kernels = Dict{Symbol,Dict{Symbol,Vector{Float64}}}()
    trials  = Dict{String,Dict{String,Int}}()
    unmeasured = String[]

    for a in (:CORRECT, :OBSERVE, :EXCURSE)
        kernels[a] = Dict{Symbol,Vector{Float64}}()
        trials[string(a)] = Dict{String,Int}()
        for bin in ALT_BINS
            row = rows[a][bin]
            if row.n == 0
                push!(unmeasured, "$a/$bin")
                kernels[a][bin] = Float64[b === bin ? 1.0 : 0.0 for b in nxt]
            else
                kernels[a][bin] = collect(row.probs)
            end
            trials[string(a)][string(bin)] = row.n
        end
    end

    meta = Dict{String,Any}(
        "generated_by"       => "calibrate_tables + tables_from_rows",
        "truth_name"         => get(diagnostics, "truth_name", "unknown"),
        "mode"               => get(diagnostics, "mode", "unknown"),
        "alt_edges"          => get(diagnostics, "alt_edges", Float64[]),
        "n_loop_steps"       => get(diagnostics, "n_loop_steps", 0),
        "trials"             => trials,
        "unmeasured_rows"    => unmeasured,
        "min_trials_trusted" => MIN_TRIALS_TRUSTED,
        "caveats"            => [
            "Rows listed in unmeasured_rows have n = 0 and are FILLED with a " *
            "self-transition, not measured. They are not evidence.",
            "Rows with 0 < n < min_trials_trusted are indicative only.",
            "Noise-free unless the run was configured otherwise, so any survival number " *
            "derived from these kernels is an UPPER BOUND, not feasibility.",
        ],
    )
    return AltTables(kernels[:CORRECT], kernels[:OBSERVE], kernels[:EXCURSE], meta)
end
