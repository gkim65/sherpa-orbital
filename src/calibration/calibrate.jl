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
  2. THE ORBIT IS VIOLENTLY UNSTABLE — it escapes in ~2 revs uncontrolled. This is why
     there is no longer an OBSERVE action to calibrate: a no-burn pass is not a decision on
     this orbit, it is a step toward losing the vehicle (see `actions`).
  3. EVERY ROW IS REPORTED WITH ITS TRIAL COUNT and a row below `MIN_TRIALS_TRUSTED` is
     flagged, not silently smoothed. A row with n = 0 is FILLED with a self-transition so
     the kernel stays a valid stochastic matrix, and named in `meta.unmeasured_rows`.
  4. `converged` IS THE WRONG SUCCESS TEST FOR AN EXCURSION. `mode = :altitude_position`
     leaves the apoapsis-POSITION block in the residual, which never clears
     `TARGET_TOL_KM` from a drifted state, so `converged` reads false on passes that hit
     the commanded altitude to ~0.05 km. The EXCURSE rows count a trial as
     non-converged on `peri_err_km`, the periapsis-altitude error alone.

METHOD
  CORRECT : run the sustained shell-cadence CORRECT loop and log alt_before → alt_after,
            then seed one short loop per altitude bin from a real halo-family member there,
            so the bins a working controller never visits are measured rather than invented.
  EXCURSE : one step of a PERSISTENT walk toward a band's commanded altitude, carried across
            trials so the walk accumulates — single-pass authority is not the quantity the
            policy reasons about, the settling walk is. Seeded from every (bin × band) pair
            for the same reason as CORRECT. All EXCURSE_* share one kernel: exp 12 found
            excursion SAFETY did not differ meaningfully by band, only ΔV cost.
            ⚠️ That finding predates the band rebase to 25/35/45 km — re-check it.

⚠️ SEEDING IS WHAT MAKES THE ROWS MEAN ANYTHING, AND IT HAS A KNOWN BIAS. A seeded state is
a freshly-placed PERIODIC family member, which is the most stable state in the system; a row
measured only from seeds therefore understates how fast the orbit diverges. That bias is what
made the old OBSERVE rows come back as 100% self-transitions and taught the solved policy
that coasting was free — it chose OBSERVE twice and escaped at 3.78 d (2026-08-30). The
EXCURSE walks mitigate it by carrying the walker forward across trials rather than re-seeding
each time, so later trials condition on drifted states. Prefer a row with sustained-loop
trials in it over one that is purely seeded.

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
    bin === :A20_27   && return 0.5 * (e[1] + e[2])
    bin === :A27_34   && return 0.5 * (e[2] + e[3])
    bin === :A34_44   && return 0.5 * (e[3] + e[4])
    # ABOVE_44 is open-ended, so pick a seed that is HOLDABLE rather than the geometric
    # continuation. Measured 2026-08-30: a family member holds its own altitude at 45 km but
    # ESCAPES by pass 3 at 55 and 60 km, so seeding at 55 would measure a doomed orbit and
    # report it as this bin's behaviour. 46 km is just inside the band and survivable.
    return 46.0
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

# ── Why the EXCURSE target is an ALTITUDE, not a position vector ───────────────
# Superseded 2026-08-30 by `mode = :altitude_position`; the helper this replaces
# (`_excurse_targets`, a radially-scaled waypoint) is gone. The measurements that forced
# the change are worth keeping:
#
# Aiming ONE impulse at a real family member's absolute apse POSITIONS from a drifted state
# fails on PHASE, not altitude. `next_apse_positions(member.ic)` returns where THAT member's
# periapsis sits at ITS OWN epoch. From the 31 km limit cycle vs the 30 km member, periapsis
# ALTITUDES differ ~22 km but periapsis POSITIONS are 585 km apart and apoapsis positions
# 2319 km apart, so the `:position` residual at ΔV = 0 is 2392 km:
#
#   | commanded | family-member apses | scaled waypoint     |
#   |-----------|---------------------|---------------------|
#   | 20 km     | 177.6 m/s -> ESCAPE | 2.39 m/s -> 37.9 km |
#   | 40 km     | 179.8 m/s -> 395 km | 1.51 m/s -> 42.8 km |
#   | 50 km     |  73.0 m/s -> 562 km | 1.23 m/s -> 45.3 km |
#
# Both columns are bad: the "real orbit" throws the vehicle out, and the waypoint only moves
# achieved periapsis 37.9 -> 45.3 km across a 30 km command range (~25% authority).
# `:altitude_position` constrains the periapsis ALTITUDE directly and delivers the command to
# ~0.2 km, which is what made these rows measurable at all.

"""
    calibrate_tables(; truth_eom!, n_steps, horizon_s, alt_edges, band_target_km,
                     mode, verbose) -> (AltTables, diagnostics)

Measure all three kernels and return them alongside a per-row diagnostics dict.

  - `truth_eom!` — the truth model to calibrate against. Defaults to
    [`cr3bp_j2_eom!`](@ref) (CR3BP + Enceladus J2), which is the rung the committed
    hand-transcribed tables were measured on, so the diff is apples-to-apples.
  - `n_steps` — sustained-loop steps to log for the CORRECT rows.
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
    # ⚠️ SET SO EVERY ROW CLEARS `MIN_TRIALS_TRUSTED`, not by feel. A row measured from 3
    # trials reads as a probability distribution but cannot distinguish a real one-third from
    # a coin flip that landed that way, and the solver will exploit the difference as if it
    # were signal. 8 CORRECT trials per bin plus the sustained loop's own visits, and
    # 8 EXCURSE trials per (bin × band), is what puts all five rows above 20.
    seed_trials::Integer = 8,
    # EXCURSE seeding runs per (bin × band), so it multiplies out faster than the
    # CORRECT seeding and gets its own knob rather than sharing one.
    excurse_seed_trials::Integer = 8,
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
                h < alt_edges[2] ? :A20_27 :
                h < alt_edges[3] ? :A27_34 :
                h < alt_edges[4] ? :A34_44 : :ABOVE_44
    alt_of(u) = norm(_enc_relative(u[1:3])) - R_ENCELADUS

    nxt = collect(ALT_ALL)
    # counts[action][alt_from] -> Vector{Int} over ALT_ALL
    counts = Dict(a => Dict(d => zeros(Int, length(nxt)) for d in ALT_BINS)
                  for a in (:CORRECT, :EXCURSE))
    nonconv = Dict(a => Dict(d => 0 for d in ALT_BINS)
                   for a in (:CORRECT, :EXCURSE))
    dvs = Dict(a => Dict(d => Float64[] for d in ALT_BINS)
               for a in (:CORRECT, :EXCURSE))
    # ΔV per band, for the action_dv_cost proxy.
    band_dv = Dict(b => Float64[] for b in band_names)
    band_achieved_alt = Dict(b => Float64[] for b in band_names)

    bump!(a, from, to) = counts[a][from][findfirst(==(to), nxt)] += 1

    # ── CORRECT: the sustained shell-cadence loop ──────────────────────────────
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

        # CORRECT: the branch the loop actually takes.
        b = solve_burn(sc, one_rev_s; eom! = cr3bp_eom!, mode = mode,
                       r_peri_nom = r_peri_nom, r_apo_nom = r_apo_nom)
        b.converged || (nonconv[:CORRECT][from] += 1)
        push!(dvs[:CORRECT][from], b.dv_mag_ms)

        # A lost apse pair is a LOSS charged to this bin — same reasoning as the EXCURSE
        # walk below, where the full argument and the measurements live. ⚠️ Note this is NOT
        # the same as `converged == false`: the ~8 km `:position` residual floor means a
        # healthy CORRECT burn reports `converged = false` on almost every pass (57 of 58 in
        # the sustained loop) while burning a real ~1.3 m/s. Only a NON-FINITE residual, or a
        # literally zero ΔV, means the planner had nothing to aim at.
        if !isfinite(b.residual_km) || b.dv_mag_ms == 0.0
            bump!(:CORRECT, from, :LOST)
            verbose && @printf("  step %3d  %-6s -> CORRECT NO APSE PAIR (ΔV=0) -> LOST\n",
                               nlogged + 1, from)
            break
        end

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
    # member escapes — see the phase note above. The difference is that here the same target
    # is held for consecutive passes rather than jumped at once.)
    st0, u0 = _to_peri(truth_eom!, ic, 4 * one_rev_s)
    base = st0 === :ok ? copy(u0) : copy(ic)

    # One persistent-reference walk, logged into the EXCURSE kernel. Factored out so the
    # bin-seeded walks below run the IDENTICAL measurement as the primary ones — a
    # hand-copied second loop is how the two would silently diverge.
    function excurse_walk!(band, peri_tgt, ra, start_state, n_trials)
        walker = copy(start_state)      # carried across trials: the walk accumulates
        for _ in 1:n_trials
            from = bin_of(alt_of(walker))
            from in ALT_BINS || break

            sh, sc = _to_shell(truth_eom!, walker, 4 * one_rev_s)
            if sh !== :ok
                bump!(:EXCURSE, from, _terminal_dev(sh))
                break
            end

            b = solve_burn(sc, one_rev_s; eom! = cr3bp_eom!, mode = :altitude_position,
                           peri_target_km = peri_tgt, r_apo_nom = ra)
            # In `:altitude_position` the total residual is dominated by the apoapsis
            # POSITION block and never clears `TARGET_TOL_KM` from a drifted state, so
            # `converged` would flag every excursion as failed. The delivery question is
            # whether the COMMANDED ALTITUDE was met, which is `peri_err_km`.
            isfinite(b.peri_err_km) && b.peri_err_km < TARGET_TOL_KM ||
                (nonconv[:EXCURSE][from] += 1)

            # ⚠️ A LOST APSE PAIR IS A LOSS, AND IT MUST BE CHARGED TO *THIS* BIN.
            #
            # When the onboard prediction cannot find the next apse pair, `solve_burn`
            # returns a non-finite residual and ΔV = 0 — the pass flies UNCONTROLLED. The
            # spacecraft is not yet outside the escape shell, so the coast below still
            # reaches a periapsis and the old code recorded that as an ordinary transition
            # (typically to ABOVE_50), then charged the escape on the FOLLOWING pass to
            # whatever bin the vehicle had fled to.
            #
            # That misattribution is not cosmetic — it is what made the kernel dangerous.
            # Measured 2026-08-30, holding the commanded 25 km from the drifted limit cycle:
            # passes 1 and 2 deliver 25.07 and 25.21 km, pass 3 loses the apse pair, and the
            # vehicle is gone by pass 4 — EVERY time. Yet `EXCURSE/A20_30` came back with
            # P(LOST) = 0.0 and a 71% self-transition, because the deaths were booked against
            # ABOVE_50 (whose 0.667 LOST mass is largely these misattributions). The solved
            # policy then chose EXCURSE_LOW three passes running and escaped at 5.3 d — it
            # was behaving correctly given a kernel that told it low excursions are free.
            #
            # A control step with no control is a loss of the orbit at the bin where the
            # decision was made, so record it there and stop the walk.
            if !isfinite(b.residual_km) || b.dv_mag_ms == 0.0
                bump!(:EXCURSE, from, :LOST)
                push!(dvs[:EXCURSE][from], 0.0)
                push!(band_dv[band], 0.0)
                verbose && @printf("  excurse %-4s step from %-8s NO APSE PAIR (ΔV=0) -> LOST\n",
                                   band, from)
                break
            end

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
            walker = copy(ue)          # continue the walk from here
        end
        return nothing
    end

    for band in band_names
        target = band_target_km[band]
        member = family_table === nothing ? nothing :
                 retarget_to_altitude(family_table, target)
        if member === nothing
            verbose && @info "excurse: no family member at $(target) km for $band — skipped"
            continue
        end
        # `:altitude_position` (2026-08-30): the periapsis is commanded as an ALTITUDE, so
        # the family member's role is only to confirm the band is a realisable orbit — the
        # radius no longer has to be smuggled in through a scaled position vector, and the
        # phase-matching problem noted at the top of this file does not arise for the
        # periapsis half at all. The apoapsis anchor stays our own nominal.
        peri_tgt = member.info.periapsis_alt_km
        ra = collect(r_apo_nom)

        excurse_walk!(band, peri_tgt, ra, copy(base), excurse_trials)
    end

    # ── SEED the EXCURSE walk from each altitude bin ───────────────────────────
    # ⚠️ WITHOUT THIS, THE EXCURSE KERNEL DESCRIBES ONE BIN. Measured 2026-08-30: the walks
    # above all start from the IC limit cycle (~37 km, `A30_40`) and, once
    # `:altitude_position` made the aim accurate, they SETTLE — 8 passes commanded 35 km go
    # 34.97, 34.95, 34.94, 34.94 … and never leave the bin they started in. So 11 of 20
    # trials land in `A30_40` and three rows stay empty. Accurate targeting made this WORSE
    # than the old sloppy aim, which at least overshot into `A40_50` twice by accident.
    #
    # The rows are keyed by the bin the vehicle STARTS a pass in, so covering them means
    # starting passes in them. This mirrors the CORRECT seeding below, which exists
    # for exactly this reason; the EXCURSE loop simply never used it.
    #
    # ⚠️ AND THE LOW BINS ARE REACHABLE — the earlier read that they were not was wrong.
    # Commanding 20 or 25 km from the 37 km limit cycle escapes, but that is the TRANSFER
    # failing, not the destination: placed ON the 18/20/22/25/28 km family members and told
    # to hold their own altitude, `:altitude_position` holds every one of them to ~0.09 km
    # for 12 passes with real burns. `ABOVE_50` is the genuinely hard bin — 55 and 60 km
    # escape by the third pass.
    if seed_bins && family_table !== nothing
        for bin in ALT_BINS, band in band_names
            rep = _bin_rep_alt(bin, alt_edges)
            m = retarget_to_altitude(family_table, rep)
            m === nothing && continue
            seed_ic = collect(float.(m.ic))
            ps, us = _to_peri(truth_eom!, seed_ic, 4 * one_rev_s)
            ps === :ok || continue

            tgt_m = retarget_to_altitude(family_table, band_target_km[band])
            tgt_m === nothing && continue
            # Apoapsis anchor from the SEEDED orbit, not the nominal one: from an 18 km
            # halo the nominal 37 km apoapsis is not the apse this vehicle is holding, and
            # targeting it would measure a transfer rather than an excursion in that bin.
            _, ra_s = next_apse_positions(seed_ic; eom! = cr3bp_eom!)
            excurse_walk!(band, tgt_m.info.periapsis_alt_km, ra_s, copy(us),
                          excurse_seed_trials)
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

                # CORRECT: burn at the shell toward the seeded orbit, then coast.
                shs, scs = _to_shell(truth_eom!, st_s, 4 * one_rev_s)
                if shs !== :ok
                    bump!(:CORRECT, from, _terminal_dev(shs))
                    continue
                end
                bs = solve_burn(scs, one_rev_s; eom! = cr3bp_eom!, mode = mode,
                                r_peri_nom = rp_s, r_apo_nom = ra_s)
                bs.converged || (nonconv[:CORRECT][from] += 1)
                # Lost apse pair = LOST, charged here. See the sustained loop above.
                if !isfinite(bs.residual_km) || bs.dv_mag_ms == 0.0
                    bump!(:CORRECT, from, :LOST)
                    push!(dvs[:CORRECT][from], 0.0)
                    continue
                end
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
    for a in (:CORRECT, :EXCURSE)
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

    for a in (:CORRECT, :EXCURSE)
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
    return AltTables(kernels[:CORRECT], kernels[:EXCURSE], meta)
end
