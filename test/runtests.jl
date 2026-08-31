using Test
using SherpaOrbital
using LinearAlgebra: norm, Diagonal
using JSON

const CFG = StationkeepingPOMDP()

# ── Testset filtering ────────────────────────────────────────────────────────
# The orbit-geometry, family-continuation and rollout testsets take minutes (the
# SARSOPController one alone is ~1.5 min), which makes the full suite too slow to use as an
# edit-loop check on the MODEL layer. Pass substrings to run only matching testsets:
#
#     julia --project=. -e 'using Pkg; Pkg.test(test_args=["plume","rewards"])'
#     julia --project=test test/runtests.jl plume rewards
#
# No arguments runs everything, so CI and `Pkg.test()` are unchanged.
const FILTERS = lowercase.(ARGS)
_want(name) = isempty(FILTERS) || any(f -> occursin(f, lowercase(name)), FILTERS)

"""
    @maybe_testset "name" begin ... end

`@testset` that is SKIPPED unless its name matches one of the CLI filters. The body is
wrapped in a thunk rather than spliced, so the filter check happens before the body's
`model_tables` calls run — a filtered-out testset costs nothing.
"""
macro maybe_testset(name, body)
    esc(:(_want($name) && @testset $name $body))
end

@testset "SherpaOrbital" begin

    # MODEL-LAYER INVARIANTS ONLY.
    #
    # The pinning tests that used to live here (|S| == 26, size(T) == (26,5,26), an exact
    # action-list equality) were deliberately DROPPED on 2026-08-30. They asserted the
    # current shape, so they broke on every redesign while catching nothing — the state
    # space changing is the intended edit, not a regression. What remains are properties
    # that stay true across redesigns and whose violation is SILENT: a malformed kernel
    # produces a confidently wrong policy with no error anywhere.
    @maybe_testset "T and O are valid distributions" begin
        T, O = model_tables(CFG)
        S = SherpaOrbital.states(CFG)
        @test size(T) == (length(S), SherpaOrbital.n_actions(CFG), length(S))
        @test size(O) == (length(S), SherpaOrbital.n_observations(CFG))
        @test all(isapprox.(sum(T, dims = 3), 1.0; atol = 1e-9))
        @test all(isapprox.(sum(O, dims = 2), 1.0; atol = 1e-9))
        @test all(T .>= 0.0)
        @test all(O .>= 0.0)
        @test allunique(S)
        idx = SherpaOrbital.state_index(CFG)
        @test all(idx[s] == i for (i, s) in enumerate(S))
    end

    @maybe_testset "terminal states absorb and self-announce" begin
        T, O = model_tables(CFG)
        idx  = SherpaOrbital.state_index(CFG)
        oidx = SherpaOrbital.observation_index(CFG)
        zv   = ntuple(_ -> 0, SherpaOrbital.n_bands(CFG))
        for term in SherpaOrbital.TERMINAL_ALT
            i = idx[SKState(term, zv)]
            @test all(T[i, a, i] == 1.0 for a in 1:SherpaOrbital.n_actions(CFG))
            @test O[i, oidx[term]] == 1.0          # a crash or escape is unambiguous
            @test SherpaOrbital.isterminal_state(SKState(term, zv))
        end
    end

    # Every action must be priced or reward_function throws at solve time — a failure that
    # surfaces deep inside the solver rather than at configuration.
    @maybe_testset "every action is priced" begin
        @test all(haskey(CFG.action_dv_cost, a) for a in SherpaOrbital.actions(CFG))
    end

    # Half-open [lo, hi): an edge belongs to the OUTER bin. This must stay identical to
    # policy_alt_bin in simulate.jl, or the belief filter is fed labels the model does not
    # use.
    @maybe_testset "altitude binning" begin
        @test alt_bin(CFG, 19.9) == :BELOW_20
        @test alt_bin(CFG, 20.0) == :A20_27
        @test alt_bin(CFG, 26.9) == :A20_27
        @test alt_bin(CFG, 27.0) == :A27_34
        @test alt_bin(CFG, 33.9) == :A27_34
        @test alt_bin(CFG, 34.0) == :A34_44
        @test alt_bin(CFG, 43.9) == :A34_44
        @test alt_bin(CFG, 44.0) == :ABOVE_44

        # The measured CORRECT limit cycle (37.17 km noise-free) must land in the
        # NON-science bin, or a science band is banked by doing nothing — the defect these
        # edges were rebased to fix. See `alt_edges`.
        @test alt_bin(CFG, 37.17) == :A34_44
        @test alt_bin(CFG, 37.17) ∉ CFG.band_bins
        @test alt_bin(CFG, NaN)  == :LOST
        @test alt_bin(CFG, Inf)  == :LOST
    end

    # Visit counts must saturate, or the state space is unbounded.
    @maybe_testset "visit counts saturate" begin
        v0 = ntuple(_ -> 0, SherpaOrbital.n_bands(CFG))
        v  = v0
        for _ in 1:(CFG.visit_cap + 5)
            v = visit_inc(v, 1, CFG.visit_cap)
        end
        @test v[1] == CFG.visit_cap
        @test visit_total(v) == CFG.visit_cap        # other bands untouched
        @test length(visit_tuples(CFG)) == SherpaOrbital.n_visit_combos(CFG)
    end

    # Science is now EXPECTED over T and keyed on the SUCCESSOR's visit count, not on the
    # action. This is the subtlest change in the 2026-08-30 redesign: get it wrong and the
    # policy is quietly miscalibrated rather than broken.
    @maybe_testset "rewards" begin
        T, _ = model_tables(CFG)
        r    = SherpaOrbital.reward_function(CFG, T)
        zv   = ntuple(_ -> 0, SherpaOrbital.n_bands(CFG))

        # Terminal states earn nothing; the loss was charged on entry, not on occupancy.
        for term in SherpaOrbital.TERMINAL_ALT
            @test r(SKState(term, zv), :CORRECT) == 0.0
        end

        # A band at the visit cap can bank no more, so it earns strictly less than a fresh
        # one from the same altitude under the same action.
        capped = ntuple(i -> i == 2 ? CFG.visit_cap : 0, SherpaOrbital.n_bands(CFG))
        @test r(SKState(:A27_34, zv), :CORRECT) > r(SKState(:A27_34, capped), :CORRECT)

        # Fuel is charged WHEN THE WEIGHT IS NONZERO: every action burns (there is no free
        # OBSERVE to be indifferent), so a positive weight must strictly lower every
        # action's reward relative to a zero weight.
        #
        # ⚠️ BOTH WEIGHTS ARE NOW EXPLICIT. `fuel_weight` defaults to 0.0 as of 2026-08-30
        # (the study is science yield, not fuel feasibility), so comparing against `CFG`
        # would compare zero to zero and assert `>` on four equal numbers. The fuel
        # machinery is intact and this test is what proves it — it must not silently pass
        # by both sides being unweighted.
        cfg0 = StationkeepingPOMDP(; fuel_weight = 0.0)
        cfgF = StationkeepingPOMDP(; fuel_weight = 1.0)
        r0   = SherpaOrbital.reward_function(cfg0, model_tables(cfg0)[1])
        rF   = SherpaOrbital.reward_function(cfgF, model_tables(cfgF)[1])
        for a in SherpaOrbital.actions(CFG)
            @test r0(SKState(:A27_34, zv), a) > rF(SKState(:A27_34, zv), a)
        end
    end

    # The plume gradient is the THIRD SWEEP AXIS, and its θ = 0 behaviour is a hard gate:
    # if θ = 0 does not make every band identical, every downstream regret number is
    # meaningless. These assert the parameterization's structural properties, which is
    # cheaper and stricter than inspecting a solved policy.
    @maybe_testset "plume gradient" begin
        # θ = 0 ⟹ EVERY band gets the IDENTICAL uniform distribution. This is the gate.
        flat = StationkeepingPOMDP(; plume_gradient = 0.0, plume_levels = 3)
        d1 = plume_intensity_dist(flat, 1)
        @test all(isapprox.(d1, 1 / 3; atol = 1e-12))
        for b in 2:SherpaOrbital.n_bands(flat)
            @test plume_intensity_dist(flat, b) ≈ d1
        end

        # Rising θ tilts the DEEPEST band toward high intensity and the shallowest away
        # from it. Bands are (LOW, MID, HIGH) so band 1 is deepest, band 3 shallowest.
        steep = StationkeepingPOMDP(; plume_gradient = 3.0, plume_levels = 3)
        @test plume_band_depth(steep, 1) ≈ 1.0        # LOW is the deepest
        @test plume_band_depth(steep, 3) ≈ 0.0        # HIGH is the shallowest
        top = steep.plume_levels
        @test plume_intensity_dist(steep, 1)[top] > d1[top]     # LOW gains high intensity
        @test plume_intensity_dist(steep, 3) ≈ d1               # depth 0 ⟹ still uniform

        # Monotone in θ: a steeper gradient must put strictly more mass on high intensity
        # in the deepest band. A form that saturates would break the sweep (two θ, one POMDP).
        for (lo, hi) in ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0))
            a = plume_intensity_dist(StationkeepingPOMDP(; plume_gradient = lo), 1)[end]
            b = plume_intensity_dist(StationkeepingPOMDP(; plume_gradient = hi), 1)[end]
            @test b > a
        end

        # Every distribution is valid for every θ, with no clipping.
        for θ in (0.0, 0.5, 2.0, 10.0, 50.0), k in (1, 3, 5)
            cfg = StationkeepingPOMDP(; plume_gradient = θ, plume_levels = k)
            for b in 1:SherpaOrbital.n_bands(cfg)
                p = plume_intensity_dist(cfg, b)
                @test length(p) == k
                @test all(p .>= 0.0)
                @test sum(p) ≈ 1.0
            end
        end

        # Intensity values span [0, 1] and are ordered, so a higher level pays more.
        k5 = StationkeepingPOMDP(; plume_levels = 5)
        @test plume_intensity_value(k5, 1) == 0.0
        @test plume_intensity_value(k5, 5) == 1.0
        @test issorted([plume_intensity_value(k5, l) for l in 1:5])
        # k = 1 disables the dimension: |S| must match the pre-intensity model exactly.
        k1 = StationkeepingPOMDP(; plume_levels = 1)
        @test plume_intensity_value(k1, 1) == 1.0
        @test SherpaOrbital.n_states(k1) ==
              length(ALT_BINS) * SherpaOrbital.n_visit_combos(k1) + 2
        # |S| is LINEAR in k, and the intensity dimension actually enumerates.
        @test SherpaOrbital.n_states(StationkeepingPOMDP(; plume_levels = 3)) ==
              3 * (SherpaOrbital.n_states(k1) - 2) + 2
        @test length(SherpaOrbital.states(k5)) == SherpaOrbital.n_states(k5)
    end

    # θ must reach the SOLVABLE MODEL, not just the helper functions. A gradient that
    # changes `plume_intensity_dist` but leaves T identical would pass every test above and
    # make the whole sweep a no-op.
    @maybe_testset "plume gradient reaches T" begin
        flat  = StationkeepingPOMDP(; plume_gradient = 0.0, plume_levels = 3)
        steep = StationkeepingPOMDP(; plume_gradient = 4.0, plume_levels = 3)
        Tf, _ = model_tables(flat)
        Ts, _ = model_tables(steep)
        @test size(Tf) == size(Ts)
        @test !(Tf ≈ Ts)                        # θ changed the transition kernel
        @test all(isapprox.(sum(Ts, dims = 3), 1.0; atol = 1e-9))   # still stochastic

        # And it must reach the REWARD, via the realized intensity. Under a steep gradient
        # the deepest band's expected science strictly exceeds the flat case.
        rf = SherpaOrbital.reward_function(flat, Tf)
        rs = SherpaOrbital.reward_function(steep, Ts)
        zv = ntuple(_ -> 0, SherpaOrbital.n_bands(flat))
        @test rs(SKState(:A20_27, zv, 1), :EXCURSE_LOW) >
              rf(SKState(:A20_27, zv, 1), :EXCURSE_LOW)
    end

    # `repeat_factor` replaced a HARD ZERO past the visit cap. The failure it fixes is
    # silent: with all bands capped the policy becomes indifferent between sampling and
    # not, which reads as a converged policy rather than a broken reward.
    @maybe_testset "repeat_factor softens the visit cap" begin
        allcap = ntuple(_ -> CFG.visit_cap, SherpaOrbital.n_bands(CFG))
        hard = StationkeepingPOMDP(; repeat_factor = 0.0)
        soft = StationkeepingPOMDP(; repeat_factor = 0.5)
        r_h = SherpaOrbital.reward_function(hard, model_tables(hard)[1])
        r_s = SherpaOrbital.reward_function(soft, model_tables(soft)[1])
        # From a fully saturated state a band pass still pays under `repeat_factor`.
        @test r_s(SKState(:A27_34, allcap, 1), :CORRECT) >
              r_h(SKState(:A27_34, allcap, 1), :CORRECT)
        # And it must stay a DISCOUNT, never a bonus: capped < uncapped at equal θ.
        zv = ntuple(_ -> 0, SherpaOrbital.n_bands(CFG))
        @test r_s(SKState(:A27_34, zv, 1), :CORRECT) >
              r_s(SKState(:A27_34, allcap, 1), :CORRECT)
    end

    # OBSERVE was REMOVED as an action on 2026-08-30 (a no-burn coast loses this orbit —
    # see `actions`). Assert it is gone everywhere, because a stale reference would not
    # error: `alt_kernel` would fall through to the EXCURSE kernel and quietly mis-price a
    # coast as an excursion.
    @maybe_testset "OBSERVE is not an action" begin
        @test :OBSERVE ∉ SherpaOrbital.actions(CFG)
        @test SherpaOrbital.n_actions(CFG) == length(SherpaOrbital.actions(CFG))
        @test !haskey(CFG.action_dv_cost, :OBSERVE)
        @test all(a -> a === :CORRECT || SherpaOrbital.is_excursion(CFG, a),
                  SherpaOrbital.actions(CFG))
        @test !hasfield(SherpaOrbital.AltTables, :observe)
    end

    # Hyperparameters must actually flow through the model rather than being shadowed.
    @maybe_testset "config is the scenario" begin
        loud = StationkeepingPOMDP(; r_science = 100.0)
        base = StationkeepingPOMDP()
        zv   = ntuple(_ -> 0, SherpaOrbital.n_bands(base))
        r_l  = SherpaOrbital.reward_function(loud, model_tables(loud)[1])
        r_b  = SherpaOrbital.reward_function(base, model_tables(base)[1])
        @test r_l(SKState(:A27_34, zv), :CORRECT) > r_b(SKState(:A27_34, zv), :CORRECT)

        # Nav sigma must move the observation model, or the POMDP is secretly an MDP.
        sharp = StationkeepingPOMDP(; sigma_nav_km = 0.1)
        blurry = StationkeepingPOMDP(; sigma_nav_km = 8.0)
        i = SherpaOrbital.state_index(sharp)[SKState(:A27_34, zv)]
        oi = SherpaOrbital.observation_index(sharp)[:A27_34]
        @test model_tables(sharp)[2][i, oi] > model_tables(blurry)[2][i, oi]
    end

    # A reordered or non-normalized artifact must be REJECTED, not silently accepted:
    # misaligned kernel columns corrupt every transition with no visible error.
    @maybe_testset "measured tables validate" begin
        tables = load_tables()
        @test validate_tables(tables)

        # ⚠️ INVERTED 2026-08-30, AND THE OLD ASSERTION WAS PINNING A DEFECT. It used to
        # require alt_kernel(:EXCURSE_LOW) == alt_kernel(:EXCURSE_HIGH), i.e. that aiming
        # at a band does not change where you land — which makes the three EXCURSE actions
        # identical in T and leaves nothing in the model able to steer altitude. Aiming at
        # different bands MUST give different successor distributions.
        @test SherpaOrbital.alt_kernel(tables, :EXCURSE_LOW, :A34_44) !=
              SherpaOrbital.alt_kernel(tables, :EXCURSE_HIGH, :A34_44)

        # And the aiming must point the right way: from the limit cycle, aiming LOW must put
        # strictly more mass in the LOW band's bin than aiming HIGH does, and vice versa.
        # This is the property the whole plume-gradient θ axis rests on.
        oi = Dict(b => i for (i, b) in enumerate(ALT_ALL))
        k_lo = SherpaOrbital.alt_kernel(tables, :EXCURSE_LOW,  :A34_44)
        k_hi = SherpaOrbital.alt_kernel(tables, :EXCURSE_HIGH, :A34_44)
        @test k_lo[oi[:A20_27]]   > k_hi[oi[:A20_27]]
        @test k_hi[oi[:ABOVE_44]] > k_lo[oi[:ABOVE_44]]

        # Every EXCURSE action must have its own kernel, or `alt_kernel` silently falls
        # back to a pooled row and the degeneracy returns unnoticed.
        for a in SherpaOrbital.actions(CFG)
            a === :CORRECT && continue
            @test haskey(tables.excurse, a)
        end

        @test_throws ErrorException load_tables(tempname() * ".json")
    end

    @maybe_testset "builds" begin
        @test build_pomdp(CFG) !== nothing
    end

    # Orbit geometry. These pin the 2026-08-29 findings: the shipped IC is period-1 (not
    # period-3) and NORTH-polar, and the z-mirror of it is the south-polar science orbit.
    # Without the latitude assertions an IC in the wrong hemisphere is invisible — which is
    # exactly how the north-polar orbit survived several sessions of measurement.
    @maybe_testset "orbit geometry and hemisphere" begin
        ic_n = nondim_to_cr3bp(collect(PERIOD1_NORTH_IC_ND))
        ic_s = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND))

        # The true period is HALO_PERIOD_S; PERIOD1_TRIPLE_PERIOD_S is 3x it (three traversals).
        @test HALO_PERIOD_S ≈ PERIOD1_TRIPLE_PERIOD_S / 3
        ch_n = characterise_orbit(ic_n, HALO_PERIOD_S; verbose = false)
        # Closes at T/3 to well under the 0.028 km it reaches at the mislabelled 3T,
        # which is what makes it period-1 rather than period-3.
        @test ch_n.closure_km < 0.02

        # The shipped IC is NORTH-polar despite z0 < 0 — the sign of the out-of-plane
        # amplitude is opposite to the periapsis hemisphere.
        @test PERIOD1_NORTH_IC_ND[3] < 0
        @test ch_n.periapsis_lat_deg > 85.0

        # The mirror is south-polar and otherwise identical.
        ch_s = characterise_orbit(ic_s, HALO_PERIOD_S; verbose = false)
        @test ch_s.periapsis_lat_deg < -85.0
        @test ch_s.periapsis_lat_deg ≈ -ch_n.periapsis_lat_deg
        @test ch_s.periapsis_alt_km  ≈ ch_n.periapsis_alt_km
        @test ch_s.apoapsis_alt_km   ≈ ch_n.apoapsis_alt_km
        @test ch_s.jacobi            ≈ ch_n.jacobi

        # mirror_z is an involution and flips only z, vz.
        @test mirror_z(mirror_z(PERIOD1_NORTH_IC_ND)) ≈ collect(PERIOD1_NORTH_IC_ND)

        # Linear stability: all Floquet multipliers on the unit circle in the CR3BP, so
        # the orbit has NO hyperbolic instability and no finite e-folding time. The
        # divergence seen in rollouts comes from the truth model, not from this.
        Mn = monodromy_matrix(PERIOD1_NORTH_IC_ND, HALO_PERIOD_S / SherpaOrbital.T_STAR)
        @test stability_index(Mn) ≈ 1.0 atol = 1e-6
        @test !isfinite(e_folding_time_s(Mn, HALO_PERIOD_S))
        # Mirrored orbit has identical stability.
        @test stability_index(monodromy_matrix(PERIOD1_SOUTH_IC_ND,
                                               HALO_PERIOD_S / SherpaOrbital.T_STAR)) ≈ 1.0 atol = 1e-6
    end

    @maybe_testset "family continuation and retargeting" begin
        # A short table keeps the suite fast. `dx` is passed EXPLICITLY: it is the sign that
        # sets the direction (dx < 0 raises periapsis, dx > 0 lowers it), so a test that
        # relies on the default is really asserting the default, not the behaviour.
        tbl = halo_family_table(; dx = -5.0e-6, n_steps = 14)
        @test length(tbl) == 15

        # Every member must be a genuine periodic orbit, in the SOUTH hemisphere. Closure is
        # the property a radially-scaled "reference" does not have, and the whole point of
        # using the corrector rather than _scale_to_altitude.
        @test all(m -> m.info.closure_km < 1e-3, tbl)
        @test all(m -> m.info.periapsis_lat_deg < 0, tbl)

        # Periapsis altitude must be strictly monotone in the family parameter, otherwise
        # retarget_to_altitude's secant has no bracket to work with. dx < 0 -> ascending.
        alts = [m.info.periapsis_alt_km for m in tbl]
        @test issorted(alts)
        @test alts[1] ≈ 30.9753 atol = 1e-2

        # The other direction lowers periapsis below the nominal, which is what makes a
        # sub-31 km band reachable at all. Fine steps are required: dx = -5e-6 mirrored
        # (+5e-6) stalls near 26.8 km, which previously read as "the family ends here".
        dn = halo_family_table(; dx = 1.0e-6, n_steps = 40)
        dn_alts = [m.info.periapsis_alt_km for m in dn]
        @test issorted(dn_alts; rev = true)
        @test minimum(dn_alts) < 30.0
        @test all(m -> m.info.closure_km < 1e-3, dn)

        # halo_family_span brackets the seed on BOTH sides and returns ascending altitude,
        # so a commanded altitude below the nominal resolves instead of reading as absent.
        sp = halo_family_span(; dx = 1.0e-6, n_up = 12, n_down = 12)
        sp_alts = [m.info.periapsis_alt_km for m in sp]
        @test issorted(sp_alts)
        @test minimum(sp_alts) < 30.9753 < maximum(sp_alts)

        # The first member reproduces the shipped IC (x0 step k = 0).
        @test tbl[1].ic_nd ≈ collect(PERIOD1_SOUTH_IC_ND) atol = 1e-9

        # Retargeting hits a commanded altitude inside the span, to far better than the
        # ~1 km targeting tolerance. Measured 2026-08-29: better than 0.0003 km.
        target = 0.5 * (alts[1] + alts[end])
        m = retarget_to_altitude(tbl, target)
        @test m !== nothing
        @test m.info.periapsis_alt_km ≈ target atol = 1e-2
        @test m.info.closure_km < 1e-3
        @test m.info.periapsis_lat_deg < 0

        # Outside the continued family is `nothing` — a real answer ("no orbit there"), not
        # an error and not a silently-wrong nearest member.
        @test retarget_to_altitude(tbl, 5000.0) === nothing
        @test retarget_to_altitude(tbl, 1.0) === nothing

        # The x0-parameterised corrector reproduces the shipped member from its own seed.
        r = corrector_free_z0(PERIOD1_SOUTH_IC_ND[1], PERIOD1_SOUTH_IC_ND[3],
                              PERIOD1_SOUTH_IC_ND[5])
        @test r.converged
        @test r.residual < 1e-10
    end

    @maybe_testset "MPCController retarget toggle" begin
        ic = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND))
        # Small explicit table (spans ~31-45 km) so the suite does not pay the ~2 min cost of
        # continuing the full 181-member default family three times over.
        tbl = halo_family_table(; n_steps = 14)
        mid = 0.5 * (tbl[1].info.periapsis_alt_km + tbl[end].info.periapsis_alt_km)

        # DEFAULT IS PINNED. target_alt_km = nothing must leave the pre-2026-08-29 behaviour
        # untouched: targets come from ref_ic and no family member is looked up. Every
        # measurement before that date depends on this staying true.
        c_pinned = MPCController(; ref_ic = collect(ic), mode = :position)
        @test c_pinned.target_alt_km === nothing
        SherpaOrbital.controller_setup!(c_pinned, ic, PERIOD1_TRIPLE_PERIOD_S)
        @test c_pinned.ref_member === nothing
        @test c_pinned.r_peri_nom !== nothing

        # With the toggle set, the targets come from a real family member at that altitude.
        c_rt = MPCController(; ref_ic = collect(ic), mode = :position,
                             target_alt_km = mid, family_table = tbl)
        SherpaOrbital.controller_setup!(c_rt, ic, PERIOD1_TRIPLE_PERIOD_S)
        @test c_rt.ref_member !== nothing
        @test c_rt.ref_member.info.periapsis_alt_km ≈ mid atol = 1e-2
        @test c_rt.ref_member.info.periapsis_lat_deg < 0

        # The two must actually differ — a toggle that changes nothing is the failure mode
        # this whole session existed to find (the pinned reference ignored the command).
        @test norm(c_rt.r_peri_nom - c_pinned.r_peri_nom) > 1.0

        # An unreachable altitude is a SETUP error and must throw, not quietly fall back to
        # the pinned reference and report a run that ignored the command.
        c_bad = MPCController(; ref_ic = collect(ic), mode = :position,
                              target_alt_km = 5000.0, family_table = tbl)
        @test_throws ErrorException SherpaOrbital.controller_setup!(
            c_bad, ic, PERIOD1_TRIPLE_PERIOD_S)
    end

    @maybe_testset "altitude_position targeting mode" begin
        ic = nondim_to_cr3bp(collect(PERIOD1_NORTH_IC_ND))
        one_rev = PERIOD1_TRIPLE_PERIOD_S / 3
        _, ra = SherpaOrbital.next_apse_positions(ic; eom! = SherpaOrbital.cr3bp_eom!)

        # The residual is 4-D: 1 periapsis altitude + 3 apoapsis position components.
        r = SherpaOrbital.apse_residual(ic, zeros(3), one_rev, SherpaOrbital.cr3bp_eom!;
                                        mode = :altitude_position, peri_target_km = 30.0,
                                        r_apo_nom = ra)
        @test length(r) == 4

        # The mode must DELIVER a commanded altitude from a DRIFTED state — measuring at the
        # reference IC proves nothing (the residual is 0 there whatever the mode).
        s = copy(ic)
        for _ in 1:6
            sh, sc = SherpaOrbital._to_shell(SherpaOrbital.cr3bp_j2_eom!, s, 4 * one_rev)
            sh === :ok || break
            b = SherpaOrbital.solve_burn(sc, one_rev; mode = :altitude_position,
                                         peri_target_km = 35.0, r_apo_nom = ra)
            sp = copy(sc); sp[4:6] .+= b.dv
            pc, uc = SherpaOrbital._to_peri(SherpaOrbital.cr3bp_j2_eom!, sp, 4 * one_rev)
            pc === :ok || break
            s = copy(uc)
        end
        # Delivered to well under a km, where the unweighted 6-on-3 :position mode lands
        # ~2 km away and cannot be commanded at all.
        @test SherpaOrbital.altitude(s) ≈ 35.0 atol = 1.0

        # ⚠️ `converged` is NOT the delivery test here: the apoapsis POSITION block dominates
        # the total residual and does not clear TARGET_TOL_KM from a drifted state. That is
        # what `peri_err_km` exists to report, and conflating the two would read a working
        # excursion as a dead ΔV = 0 solve.
        sh, sc = SherpaOrbital._to_shell(SherpaOrbital.cr3bp_j2_eom!, s, 4 * one_rev)
        if sh === :ok
            b = SherpaOrbital.solve_burn(sc, one_rev; mode = :altitude_position,
                                         peri_target_km = 35.0, r_apo_nom = ra)
            @test isfinite(b.peri_err_km)
            @test b.peri_err_km < b.residual_km
        end

        # Weighting must be the IDENTITY for every other mode, or every pre-existing
        # baseline number silently changes.
        J = [1.0 0.0 0.0; 0.0 100.0 0.0; 0.0 0.0 1.0; 1.0 1.0 1.0]
        @test SherpaOrbital._residual_weights(:position, J) == Diagonal(ones(4))
        @test SherpaOrbital._residual_weights(:altitude, J) == Diagonal(ones(4))
        @test SherpaOrbital._residual_weights(:altitude_position, J)[1, 1] > 1.0

        # A commanded periapsis below the crash shell is a SETUP error, not a target.
        @test_throws ErrorException SherpaOrbital.solve_burn(
            ic, one_rev; mode = :altitude_position, peri_target_km = 1.0, r_apo_nom = ra)
        # And the apoapsis target is required.
        @test_throws ArgumentError SherpaOrbital.solve_burn(
            ic, one_rev; mode = :altitude_position, peri_target_km = 35.0)
    end

    @maybe_testset "SARSOPController retarget_bands toggle" begin
        # Skip cleanly if no COMPATIBLE policy has been exported — the toggle is structural
        # and should not make the suite depend on a solved artifact being present.
        # ⚠️ The `state_alt` check is not redundant with `isfile`: the committed policy may
        # predate the 2026-08-30 (alt, visits) redesign, in which case it parses fine and
        # then fails deep inside the constructor on a missing key. A stale artifact should
        # skip this test, not error it.
        _pol_ok = isfile(SherpaOrbital.DEFAULT_POLICY_PATH) &&
                  haskey(JSON.parsefile(SherpaOrbital.DEFAULT_POLICY_PATH), "state_alt")
        if !_pol_ok
            @test_skip "no compatible exported policy at $(SherpaOrbital.DEFAULT_POLICY_PATH)"
        else
            ic  = nondim_to_cr3bp(collect(PERIOD1_SOUTH_IC_ND))
            pol = load_policy()

            # DEFAULT IS THE WAYPOINT BEHAVIOUR. Every pre-2026-08-29 SARSOP number depends
            # on this staying the default.
            c_wp = SARSOPController(pol; ref_ic = collect(ic))
            @test c_wp.retarget_bands == false
            SherpaOrbital.controller_setup!(c_wp, ic, PERIOD1_TRIPLE_PERIOD_S)

            # Waypoint targets are the SCALED vector, so the periapsis target altitude equals
            # the band altitude exactly — that is what makes it a waypoint and not an orbit.
            rp_wp, _ = SherpaOrbital._sarsop_band_targets(c_wp, "MID")
            alt_wp = norm(SherpaOrbital._enc_relative(rp_wp)) - SherpaOrbital.R_ENCELADUS
            @test alt_wp ≈ c_wp.band_target_km["MID"] atol = 1e-6

            # With the toggle on, a band's targets come from a REAL family member instead.
            # The band is overridden into the default family span (~19-63 km): the shipped
            # MID = 70 km is OUTSIDE it, and the toggle throws for an unreachable band rather
            # than silently substituting a waypoint. That throw is asserted below.
            c_rt = SARSOPController(pol; ref_ic = collect(ic),
                                    retarget_bands = true)
            c_rt.band_target_km["MID"] = 40.0
            SherpaOrbital.controller_setup!(c_rt, ic, PERIOD1_TRIPLE_PERIOD_S)
            rp_rt, _ = SherpaOrbital._sarsop_band_targets(c_rt, "MID")
            @test haskey(c_rt.band_targets, "MID")          # cached, not re-continued
            @test norm(rp_rt - rp_wp) > 1.0                 # genuinely different target

            # CORRECT is UNAFFECTED by the toggle: it always returns to the original
            # reference orbit. Only EXCURSE_* retargets.
            @test c_rt.r_peri_nom == c_wp.r_peri_nom

            # A band outside the family span is a configuration error and must THROW, not
            # fall back to a scaled waypoint — a band that is not an orbit cannot be held.
            c_far = SARSOPController(pol; ref_ic = collect(ic),
                                     retarget_bands = true)
            c_far.band_target_km["MID"] = 5000.0
            SherpaOrbital.controller_setup!(c_far, ic, PERIOD1_TRIPLE_PERIOD_S)
            @test_throws ErrorException SherpaOrbital._sarsop_band_targets(c_far, "MID")
        end
    end

end