using Test
using SherpaOrbital

const CFG = StationkeepingPOMDP()

@testset "SherpaOrbital" begin

    @testset "state space" begin
        S = SherpaOrbital.states(CFG)
        # 3 non-terminal dev bins x 8 coverage masks, plus LOST and CRASHED.
        @test length(S) == 26
        @test length(S) == SherpaOrbital.n_states(CFG)
        @test S[1] == SKState(:OK, 0)
        @test S[end - 1] == SKState(:LOST, 0)
        @test S[end] == SKState(:CRASHED, 0)
        @test allunique(S)

        idx = SherpaOrbital.state_index(CFG)
        @test idx[SKState(:OK, 0)] == 1
        @test all(idx[s] == i for (i, s) in enumerate(S))

        @test SherpaOrbital.isterminal_dev(:LOST)
        @test SherpaOrbital.isterminal_dev(:CRASHED)
        @test !SherpaOrbital.isterminal_dev(:FAR)
    end


    @testset "dev binning" begin
        # Half-open [lo, hi): an edge belongs to the OUTER bin.
        @test dev_bin(CFG, 0.0) == :OK
        @test dev_bin(CFG, 14.9) == :OK
        @test dev_bin(CFG, 15.0) == :DRIFT
        @test dev_bin(CFG, 59.9) == :DRIFT
        @test dev_bin(CFG, 60.0) == :FAR
        @test dev_bin(CFG, 199.9) == :FAR
        @test dev_bin(CFG, 200.0) == :LOST
        @test dev_bin(CFG, Inf) == :LOST
        @test dev_bin(CFG, NaN) == :LOST
    end

    @testset "actions" begin
        A = SherpaOrbital.actions(CFG)
        @test A == [:OBSERVE, :CORRECT, :EXCURSE_LOW, :EXCURSE_MID, :EXCURSE_HIGH]
        @test length(A) == SherpaOrbital.n_actions(CFG)
        # Every action must be priced, or the reward silently throws.
        @test all(haskey(CFG.action_dv_cost, a) for a in A)
        @test CFG.action_dv_cost[:OBSERVE] == 0.0

        eb = SherpaOrbital.excurse_band(CFG)
        @test eb[:EXCURSE_LOW] == 1
        @test eb[:EXCURSE_HIGH] == 3
        @test !SherpaOrbital.is_excursion(CFG, :CORRECT)
        @test SherpaOrbital.is_excursion(CFG, :EXCURSE_MID)
    end

    @testset "measured tables" begin
        tables = load_tables()
        @test validate_tables(tables)
        # All EXCURSE_* share one kernel (exp 12: safety did not differ by band).
        @test SherpaOrbital.dev_kernel(tables, :EXCURSE_LOW, :OK) ==
              SherpaOrbital.dev_kernel(tables, :EXCURSE_HIGH, :OK)
        # CORRECT from OK is the measured 0.98 hold.
        @test SherpaOrbital.dev_kernel(tables, :CORRECT, :OK)[1] ≈ 0.98

        # A misaligned dev_next ordering must be rejected, not silently accepted:
        # reordered columns would corrupt every transition with no visible error.
        raw = read(SherpaOrbital.DEFAULT_TABLES_PATH, String)
        tmp = tempname() * ".json"
        write(tmp, replace(raw, "\"OK\",\n    \"DRIFT\"" => "\"DRIFT\",\n    \"OK\""))
        @test_throws ErrorException load_tables(tmp)
        rm(tmp; force = true)

        # A non-normalized row must be rejected too.
        tmp2 = tempname() * ".json"
        write(tmp2, replace(raw, "\"OK\": [0.98, 0.01, 0.0, 0.01, 0.0]" =>
                                 "\"OK\": [0.98, 0.01, 0.0, 0.5, 0.0]"))
        @test_throws ErrorException load_tables(tmp2)
        rm(tmp2; force = true)

        # A missing file must fail loudly, not fall back to a default.
        @test_throws ErrorException load_tables(tempname() * ".json")
    end

    @testset "transition matrix" begin
        T, O = model_tables(CFG)
        @test size(T) == (26, 5, 26)
        @test all(isapprox.(sum(T, dims = 3), 1.0; atol = 1e-9))
        @test all(T .>= 0.0)

        idx = SherpaOrbital.state_index(CFG)
        # Terminal states absorb under every action.
        for term in (SKState(:LOST, 0), SKState(:CRASHED, 0))
            i = idx[term]
            @test all(T[i, a, i] == 1.0 for a in 1:5)
        end

        # An excursion banks its band when the pass is survived.
        i_from = idx[SKState(:OK, 0)]
        a_low  = SherpaOrbital.action_index(CFG)[:EXCURSE_LOW]
        @test T[i_from, a_low, idx[SKState(:OK, cov_set(0, 1))]] > 0.0
        # ...and never banks it into a NON-excursed coverage.
        @test T[i_from, a_low, idx[SKState(:OK, 0)]] == 0.0
        # CORRECT never changes coverage.
        a_cor = SherpaOrbital.action_index(CFG)[:CORRECT]
        @test T[i_from, a_cor, idx[SKState(:OK, 0)]] ≈ 0.98
    end

    @testset "observation matrix" begin
        _, O = model_tables(CFG)
        @test size(O) == (26, 5)
        @test all(isapprox.(sum(O, dims = 2), 1.0; atol = 1e-9))
        @test all(O .>= 0.0)

        idx = SherpaOrbital.state_index(CFG)
        oidx = SherpaOrbital.observation_index(CFG)
        # Mission loss is self-announcing.
        @test O[idx[SKState(:LOST, 0)], oidx[:LOST]] == 1.0
        @test O[idx[SKState(:CRASHED, 0)], oidx[:CRASHED]] == 1.0
        # A truly-OK state most likely reads OK (nav sigma 2 km << 15 km edge).
        row = O[idx[SKState(:OK, 0)], :]
        @test argmax(row) == oidx[:OK]
        # Coverage does not affect the observation — only dev is measured.
        @test O[idx[SKState(:OK, 0)], :] == O[idx[SKState(:OK, 7)], :]
    end

    @testset "rewards" begin
        T, _ = model_tables(CFG)
        r = SherpaOrbital.reward_function(CFG, T)

        # Terminal states earn nothing; the loss was charged on entry.
        @test r(SKState(:LOST, 0), :OBSERVE) == 0.0
        @test r(SKState(:CRASHED, 0), :CORRECT) == 0.0

        # Science pays out only the first time a band is sampled.
        fresh = r(SKState(:OK, 0), :EXCURSE_LOW)
        repeat = r(SKState(:OK, cov_set(0, 1)), :EXCURSE_LOW)
        @test fresh - repeat ≈ CFG.r_science

        # Fuel is a cost: CORRECT is charged r_fuel, OBSERVE is not. Asserted on the
        # FUEL TERM directly rather than on the two totals.
        #
        # ⚠️ The totals do NOT order the intuitive way, and the original `- 50` fudge
        # here was hiding that rather than smoothing a rounding error. Measured:
        # OBSERVE from (:OK, 7) = -5.5 vs CORRECT = -2.8, i.e. the free action is WORSE.
        # Cause: from :OK, OBSERVE carries transition risk that CORRECT removes, and that
        # risk outweighs CORRECT's fuel cost. That is the model working as intended --
        # stationkeeping is worth paying for -- so the old comment ("the free action beats
        # an equally-safe paid one") was simply false: the two actions are NOT equally safe.
        cfg0 = StationkeepingPOMDP(; fuel_weight = 0.0)
        no_fuel = SherpaOrbital.reward_function(cfg0, model_tables(cfg0)[1])
        @test no_fuel(SKState(:OK, 7), :CORRECT) > r(SKState(:OK, 7), :CORRECT)
        @test no_fuel(SKState(:OK, 7), :OBSERVE) == r(SKState(:OK, 7), :OBSERVE)

        # Risk is priced: OBSERVE from FAR is far worse than CORRECT from FAR.
        @test r(SKState(:FAR, 0), :OBSERVE) < r(SKState(:FAR, 0), :CORRECT)
    end

    @testset "config is the scenario" begin
        # Hyperparameters must actually flow through the model, not be shadowed
        # by a constant somewhere.
        loud = StationkeepingPOMDP(; r_science = 100.0)
        T, _ = model_tables(loud)
        r_loud = SherpaOrbital.reward_function(loud, T)
        gain = r_loud(SKState(:OK, 0), :EXCURSE_LOW) -
               r_loud(SKState(:OK, cov_set(0, 1)), :EXCURSE_LOW)
        @test gain ≈ 100.0

        tight = StationkeepingPOMDP(; dev_edges = (10.0, 50.0, 150.0))
        @test dev_bin(tight, 12.0) == :DRIFT      # would be :OK under the default
        @test dev_bin(CFG, 12.0) == :OK
    end

    # The POMDPs.jl interface is exercised end-to-end by experiments/example.jl, which
    # actually solves the model — a broken interface fails loudly there, not silently.
    @testset "builds" begin
        @test build_pomdp(CFG) !== nothing
    end

end