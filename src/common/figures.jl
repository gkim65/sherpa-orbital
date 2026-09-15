"""
common/figures.jl — baseline-comparison figures from rollout traces.

Two panels, deliberately separate, because a single scalar return conflates what a
controller EARNED with what it RISKED: a run that survives all 30 days can still score
badly for expected loss it never incurred.

  - cumulative science earned over the horizon
  - orbit damage carried over the horizon, with a marker where a run was lost

Plotting is loaded by [`plot_baseline_comparison`](@ref) at call time, so the package
declares no plotting dependency — the caller supplies CairoMakie.

    using SherpaOrbital, CairoMakie
    plot_baseline_comparison(runs, config; path = "figures/baselines")

NOTE: science here is the reward's science term ONLY, with no terminal-risk contribution.
The risk lives in the second panel, as a state, not as a cost.
"""

"""
    science_trace(result, config; gamma = config.discount) -> (t_days, cumulative)

Per-pass cumulative discounted science earned along a rollout.

  - `result` — a [`run_rollout`](@ref) result
  - `config` — the scenario supplying the reward parameters
  - `gamma` — discount factor; pass `1.0` for undiscounted science

Returns `(t_days, cumulative)`: elapsed time per pass (days) and the running sum of
`r_science × f × u × y` in reward units.

NOTE: excludes `r_step_ok` and the terminal-risk term. This is the science axis alone —
see [`damage_trace`](@ref) for the risk the run was carrying while earning it.
"""
function science_trace(result, config::StationkeepingPOMDP;
                       gamma::Real = config.discount)
    T, _   = model_tables(config)
    S      = states(config)
    idx    = state_index(config)
    aidx   = action_index(config)
    zero_v = _zero_visits(config)

    t_days = Float64[]
    cum    = Float64[]
    total  = 0.0
    visits, alt, res = zero_v, config.correct_bin, :R_OK

    for (t, step) in enumerate(result.steps)
        s = SKState(alt, visits, 1, res)
        haskey(idx, s) || break

        # The science term of `reward_function`, isolated: expected over the successor
        # distribution, paid for the realized intensity and the orbit's damage.
        row = @view T[idx[s], aidx[Symbol(step.action)], :]
        sci = 0.0
        for (spi, sp) in enumerate(S)
            p = row[spi]
            (p == 0.0 || isterminal_state(sp)) && continue
            val = plume_intensity_value(config, sp.intensity) *
                  config.damage_yield[residual_index(sp.residual)]
            gained = visit_total(sp.visits) - visit_total(s.visits)
            sci += gained > 0 ? p * config.r_science * gained * val :
                                p * config.r_science * config.repeat_factor * val
        end
        total += gamma^(t - 1) * sci
        push!(t_days, step.t_s / 86400)
        push!(cum, total)

        nxt = isfinite(step.peri_alt_km) ? alt_bin(config, step.peri_alt_km) : :LOST
        isterminal_alt(nxt) && break
        b = band_of_alt(config, step.peri_alt_km)
        visits = b === nothing ? visits : visit_inc(visits, b, config.visit_cap)
        res = residual_bin(step.residual_km)
        alt = nxt
    end
    return (t_days = t_days, cumulative = cum)
end

"""
    damage_trace(result) -> (t_days, residual_km, level, lost_day)

Per-pass orbit damage along a rollout.

  - `result` — a [`run_rollout`](@ref) result

Returns the elapsed time per pass (days), the raw onboard solve residual (km), its damage
level as a 1-based index into `RESIDUAL_BINS`, the running count and running FRACTION of
passes flown at `R_DEGRADED` or worse, and the day the vehicle was lost (`nothing` if it
survived).

`frac_degraded` is the plotted quantity. The instantaneous `level` oscillates every pass and
several traces overlaid on one axis are unreadable; the cumulative count instead rewards a
controller for simply stopping. The fraction is the rate at which the orbit is being run
hard, and is comparable across runs of different length.

NOTE: a non-finite residual is the lost-apse-pair case and reports as the worst level, to
match [`residual_bin`](@ref).
"""
function damage_trace(result)
    t_days = [s.t_s / 86400 for s in result.steps]
    resid  = [s.residual_km for s in result.steps]
    level  = [residual_index(residual_bin(r)) for r in resid]
    n_deg  = isempty(level) ? Int[] : cumsum(level .>= 2)
    # Running FRACTION of passes flown degraded. Normalizes out episode length, so a run
    # lost on day 5 is comparable with one that survived 30, and a controller that stops
    # excursing is not rewarded merely for having stopped accumulating.
    frac   = isempty(n_deg) ? Float64[] : n_deg ./ (1:length(n_deg))
    lost   = result.outcome in (:crash, :escape) && !isempty(t_days) ? last(t_days) : nothing
    return (t_days = t_days, residual_km = resid, level = level,
            n_degraded = n_deg, frac_degraded = frac, lost_day = lost)
end

"""
    delivery_trace(result, config) -> (t_days, commanded_km, achieved_km, err_km, band)

Commanded-versus-achieved periapsis altitude for every excursion in a rollout.

  - `result` — a [`run_rollout`](@ref) result
  - `config` — the scenario supplying `band_target_km`

Returns, per EXCURSE step, the elapsed time (days), the commanded band altitude (km), the
altitude the pass actually reached (km), the signed error (achieved − commanded, km), and
the band name.

Delivery error is distinct from the observation misbin rate: nav noise mis-ATTRIBUTES a
pass, while this is the pass not going where it was commanded. Both cost coverage, and a
science total is only interpretable with each reported.

NOTE: an excursion reference PERSISTS until a `CORRECT` clears it, so consecutive
`EXCURSE_*` steps are one multi-pass approach settling onto the band, not independent
attempts. Early passes in a run of them are expected to show large error.
"""
function delivery_trace(result, config::StationkeepingPOMDP)
    t_days = Float64[]; cmd = Float64[]; ach = Float64[]
    err = Float64[]; bands = String[]
    for step in result.steps
        a = String(step.action)
        startswith(a, "EXCURSE_") || continue
        band = replace(a, "EXCURSE_" => "")
        haskey(config.band_target_km, Symbol(band)) || continue
        isfinite(step.peri_alt_km) || continue
        c = config.band_target_km[Symbol(band)]
        push!(t_days, step.t_s / 86400); push!(cmd, c)
        push!(ach, step.peri_alt_km); push!(err, step.peri_alt_km - c)
        push!(bands, band)
    end
    return (t_days = t_days, commanded_km = cmd, achieved_km = ach,
            err_km = err, band = bands)
end

"""
    plot_baseline_comparison(runs, config; path, theme, gamma, size, horizon_days)

Two-panel baseline comparison: cumulative science earned, and orbit damage carried.

  - `runs` — `Vector{Pair{String,Any}}` of `label => rollout result`, in legend order
  - `config` — the scenario the runs were scored under
  - `path` — output path WITHOUT extension; writes `.pdf`, `.svg` and `.png`
  - `theme` — `:light` or `:dark`; saved transparent so either background works
  - `gamma` — discount for the science panel; `1.0` plots raw science earned
  - `horizon_days` — x-axis limit; `nothing` uses the longest run

Returns the Makie `Figure`.

Requires CairoMakie to be loaded by the caller. A run that ended in a crash or escape stops
at its loss time and is marked with a vertical dotted line.

    using SherpaOrbital, CairoMakie
    plot_baseline_comparison(["MPC" => r1, "POMDP" => r2], cfg; path = "figures/baselines")
"""
function plot_baseline_comparison(runs, config::StationkeepingPOMDP;
                                  path::AbstractString = "figures/baseline_comparison",
                                  theme::Symbol = :light,
                                  gamma::Real = config.discount,
                                  horizon_days::Union{Nothing,Real} = nothing,
                                  size::Tuple{Int,Int} = (700, 620))
    Makie = get(Base.loaded_modules,
                Base.PkgId(Base.UUID("13f3f980-e62b-5c42-98c6-ff1f3baf88f0"), "CairoMakie"),
                nothing)
    Makie === nothing && error(
        "plot_baseline_comparison needs CairoMakie loaded by the caller: " *
        "`using CairoMakie` before calling. The library declares no plotting dependency.")

    fg = theme === :dark ? Makie.RGBf(0.92, 0.92, 0.92) : Makie.RGBf(0.10, 0.10, 0.10)
    # Wong palette: colour-blind safe, and distinguishable in greyscale print.
    palette = [Makie.RGBf(0.00, 0.45, 0.70), Makie.RGBf(0.90, 0.62, 0.00),
               Makie.RGBf(0.00, 0.62, 0.45), Makie.RGBf(0.80, 0.47, 0.65),
               Makie.RGBf(0.84, 0.37, 0.00), Makie.RGBf(0.35, 0.35, 0.35),
               Makie.RGBf(0.58, 0.44, 0.86), Makie.RGBf(0.00, 0.62, 0.79)]

    fig = Makie.Figure(; size = size, backgroundcolor = :transparent,
                       fonts = (; regular = "CMU Serif", bold = "CMU Serif Bold"))

    ax1 = Makie.Axis(fig[1, 1]; ylabel = "Cumulative science reward",
                     title = "Science earned and risk carried",
                     backgroundcolor = :transparent,
                     xgridvisible = false, ygridvisible = true,
                     titlecolor = fg, ylabelcolor = fg,
                     xticklabelcolor = fg, yticklabelcolor = fg,
                     leftspinecolor = fg, bottomspinecolor = fg,
                     topspinevisible = false, rightspinevisible = false,
                     xtickcolor = fg, ytickcolor = fg)
    ax2 = Makie.Axis(fig[2, 1]; xlabel = "Time (days)",
                     ylabel = "Fraction of passes\ndegraded or worse",
                     backgroundcolor = :transparent,
                     xgridvisible = false, ygridvisible = true,
                     xlabelcolor = fg, ylabelcolor = fg,
                     xticklabelcolor = fg, yticklabelcolor = fg,
                     leftspinecolor = fg, bottomspinecolor = fg,
                     topspinevisible = false, rightspinevisible = false,
                     xtickcolor = fg, ytickcolor = fg)
    Makie.linkxaxes!(ax1, ax2)
    Makie.rowsize!(fig.layout, 2, Makie.Relative(0.32))

    tmax = 0.0
    for (i, pair) in enumerate(runs)
        label, res = first(pair), last(pair)
        col = palette[mod1(i, length(palette))]
        sci = science_trace(res, config; gamma = gamma)
        dmg = damage_trace(res)
        isempty(sci.t_days) && continue
        tmax = max(tmax, maximum(sci.t_days))

        sty = occursin("MPC", label) ? :dash : :solid
        Makie.lines!(ax1, sci.t_days, sci.cumulative; color = col, linewidth = 2,
                     linestyle = sty, label = label)
        # A rate, not a count: see `damage_trace`. MPC hold sits at zero, so it is dashed
        # in both panels to stay visible on the axis and to mark it as the no-science arm.
        Makie.lines!(ax2, dmg.t_days, dmg.frac_degraded; color = col, linewidth = 2,
                     linestyle = sty)

        # A lost run stops where it died; the marker says the curve ENDED rather than
        # flattened, which a truncated line alone does not distinguish.
        if dmg.lost_day !== nothing
            for ax in (ax1, ax2)
                Makie.vlines!(ax, [dmg.lost_day]; color = col, linestyle = :dot,
                              linewidth = 1.5)
            end
            Makie.scatter!(ax1, [dmg.lost_day], [last(sci.cumulative)]; color = col,
                           marker = :xcross, markersize = 11)
            Makie.scatter!(ax2, [dmg.lost_day], [last(dmg.frac_degraded)];
                           color = col, marker = :xcross, markersize = 11)
        end
    end

    xhi = horizon_days === nothing ? tmax * 1.02 : float(horizon_days)
    Makie.xlims!(ax1, 0, xhi); Makie.xlims!(ax2, 0, xhi)
    # Padded below zero: an arm that never degrades sits flat at 0 and would otherwise be
    # drawn on top of the axis line and read as missing.
    Makie.ylims!(ax2, -0.04, 1.0)
    Makie.hidexdecorations!(ax1; grid = false)

    Makie.Legend(fig[1, 2], ax1; framevisible = false, labelcolor = fg,
                 labelsize = 11, patchsize = (18.0f0, 10.0f0))
    Makie.colsize!(fig.layout, 2, Makie.Auto(false))

    mkpath(dirname(path))
    for ext in ("pdf", "svg", "png")
        Makie.save("$path.$ext", fig; backgroundcolor = :transparent)
    end
    return fig
end