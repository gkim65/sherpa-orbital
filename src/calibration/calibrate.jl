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
     Session-3 planner-model ablation: failure is instability-dominated).
  3. FAR IS BARELY REACHABLE UNDER A HOLDING CONTROLLER. The sustained CORRECT loop keeps
     dev small by construction, so the loop visits OK constantly, DRIFT occasionally, and
     FAR almost never. Rows are reported WITH THEIR TRIAL COUNTS and a row measured from
     too few trials is flagged, not silently smoothed.
  4. `mode = :position` (Strategy 3 proper, which the kernels are defined against) does not
     reliably converge on this orbit — measured in `scratch/compare/compare_simulate.jl`
     section [D]. A non-converged `solve_burn` returns ΔV = 0, i.e. CORRECT degenerates
     into OBSERVE. That is recorded per row as `n_nonconverged`.

METHOD (mirrors python-legacy/scripts/pomdp_experiments/11b and 12)
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

One measured kernel row: the counts over `DEV_NEXT`, the normalized probabilities, and the
diagnostics needed to judge whether the row is trustworthy.
"""
struct CalibrationRow
    counts::Vector{Int}          # over DEV_NEXT
    probs::Vector{Float64}       # counts ./ n, or a fallback if n == 0
    n::Int
    n_nonconverged::Int          # solve_burn failures folded into this row
    dv_ms::Vector{Float64}       # per-trial ΔV, for the action_dv_cost proxy
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

"""Map a terminal coast outcome onto its `DEV_NEXT` label."""
_terminal_dev(outcome::Symbol) = outcome === :crash ? :CRASHED : :LOST

"""
    calibrate_tables(; truth_eom!, n_steps, horizon_s, dev_edges, band_target_km,
                     n_revs, mode, verbose) -> (DevTables, diagnostics)

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
    ic::AbstractVector{<:Real} = nondim_to_cr3bp(collect(PERIOD3_IC_ND)),
    period_s::Real = PERIOD3_PERIOD_S,
    n_steps::Integer = 120,
    horizon_s::Real = 25 * 86400.0,
    dev_edges::NTuple{3,Float64} = (15.0, 60.0, 200.0),
    band_names::NTuple{3,Symbol} = (:LOW, :MID, :HIGH),
    band_target_km::Dict{Symbol,Float64} =
        Dict(:LOW => 40.0, :MID => 70.0, :HIGH => 120.0),
    n_revs::Integer = 3,
    mode::Symbol = :position,
    excurse_trials::Integer = 3,
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

    bin_of(d) = !isfinite(d) ? :LOST :
                d < dev_edges[1] ? :OK :
                d < dev_edges[2] ? :DRIFT :
                d < dev_edges[3] ? :FAR : :LOST
    dev_of(u) = norm(u[1:3] .- r_peri_nom)

    nxt = collect(DEV_NEXT)
    # counts[action][dev_from] -> Vector{Int} over DEV_NEXT
    counts = Dict(a => Dict(d => zeros(Int, length(nxt)) for d in NONTERM_DEV)
                  for a in (:CORRECT, :OBSERVE, :EXCURSE))
    nonconv = Dict(a => Dict(d => 0 for d in NONTERM_DEV)
                   for a in (:CORRECT, :OBSERVE, :EXCURSE))
    dvs = Dict(a => Dict(d => Float64[] for d in NONTERM_DEV)
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
        from = bin_of(dev_of(state))
        from in NONTERM_DEV || break

        sh, sc = _to_shell(truth_eom!, state, horizon_s - t)
        if sh !== :ok
            bump!(:CORRECT, from, _terminal_dev(sh))
            break
        end

        # OBSERVE counterfactual from this shell state: coast one pass, no burn.
        po, uo = _to_peri(truth_eom!, sc, 4 * one_rev_s)
        bump!(:OBSERVE, from, po === :ok ? bin_of(dev_of(uo)) : _terminal_dev(po))
        push!(dvs[:OBSERVE][from], 0.0)

        # CORRECT: the branch the loop actually takes.
        b = solve_burn(sc, one_rev_s; n_revs = n_revs, eom! = cr3bp_eom!, mode = mode,
                       r_peri_nom = r_peri_nom, r_apo_nom = r_apo_nom)
        b.converged || (nonconv[:CORRECT][from] += 1)
        push!(dvs[:CORRECT][from], b.dv_mag_ms)

        spost = copy(sc)
        spost[4:6] .+= b.dv
        pc, uc = _to_peri(truth_eom!, spost, 4 * one_rev_s)
        bump!(:CORRECT, from, pc === :ok ? bin_of(dev_of(uc)) : _terminal_dev(pc))

        verbose && @printf("  step %3d  %-6s -> CORRECT ΔV=%6.3f m/s %s  -> %s\n",
                           nlogged + 1, from, b.dv_mag_ms,
                           b.converged ? "  " : "NC",
                           pc === :ok ? string(bin_of(dev_of(uc))) : string(_terminal_dev(pc)))

        pc === :ok || break
        state = copy(uc)
        t += one_rev_s
        nlogged += 1
    end

    # ── EXCURSE: one-pass excursion + one recovery CORRECT ─────────────────────
    st0, u0 = _to_peri(truth_eom!, ic, 4 * one_rev_s)
    base = st0 === :ok ? copy(u0) : copy(ic)

    for band in band_names, _ in 1:excurse_trials
        state_e = copy(base)
        from = bin_of(dev_of(state_e))
        from in NONTERM_DEV || continue

        sh, sc = _to_shell(truth_eom!, state_e, 4 * one_rev_s)
        if sh !== :ok
            bump!(:EXCURSE, from, _terminal_dev(sh))
            continue
        end

        rp = _scale_to_altitude(r_peri_nom, band_target_km[band])
        ra = _scale_to_altitude(r_apo_nom, apo_nom_alt)
        b = solve_burn(sc, one_rev_s; n_revs = n_revs, eom! = cr3bp_eom!, mode = :position,
                       r_peri_nom = rp, r_apo_nom = ra)
        b.converged || (nonconv[:EXCURSE][from] += 1)
        total_dv = b.dv_mag_ms

        sp = copy(sc)
        sp[4:6] .+= b.dv
        pe, ue = _to_peri(truth_eom!, sp, 4 * one_rev_s)
        if pe !== :ok
            bump!(:EXCURSE, from, _terminal_dev(pe))
            push!(dvs[:EXCURSE][from], total_dv)
            push!(band_dv[band], total_dv)
            continue
        end
        push!(band_achieved_alt[band], altitude(ue))

        # Recovery: one nominal CORRECT at the next shell.
        sh2, sc2 = _to_shell(truth_eom!, ue, 4 * one_rev_s)
        if sh2 !== :ok
            bump!(:EXCURSE, from, _terminal_dev(sh2))
            push!(dvs[:EXCURSE][from], total_dv); push!(band_dv[band], total_dv)
            continue
        end
        b2 = solve_burn(sc2, one_rev_s; n_revs = n_revs, eom! = cr3bp_eom!, mode = mode,
                        r_peri_nom = r_peri_nom, r_apo_nom = r_apo_nom)
        total_dv += b2.dv_mag_ms
        sp2 = copy(sc2)
        sp2[4:6] .+= b2.dv
        pr2, ur2 = _to_peri(truth_eom!, sp2, 4 * one_rev_s)
        bump!(:EXCURSE, from, pr2 === :ok ? bin_of(dev_of(ur2)) : _terminal_dev(pr2))
        push!(dvs[:EXCURSE][from], total_dv)
        push!(band_dv[band], total_dv)

        verbose && @printf("  excurse %-4s from %-6s ΔV=%6.2f m/s -> alt %.1f km -> %s\n",
                           band, from, total_dv, altitude(ue),
                           pr2 === :ok ? string(bin_of(dev_of(ur2))) : string(_terminal_dev(pr2)))
    end

    # ── Assemble ──────────────────────────────────────────────────────────────
    rows = Dict{Symbol,Dict{Symbol,CalibrationRow}}()
    for a in (:CORRECT, :OBSERVE, :EXCURSE)
        rows[a] = Dict{Symbol,CalibrationRow}()
        for d in NONTERM_DEV
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
        "dev_edges"          => collect(dev_edges),
        "min_trials_trusted" => MIN_TRIALS_TRUSTED,
    )
    return rows, diagnostics
end