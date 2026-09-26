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

NOTE: one impulse does not deliver a band, so consecutive `EXCURSE_*` steps read as one
multi-pass approach settling onto it — each pass re-solves from the state the previous burn
produced. Early passes in a run of them are expected to show large error.
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
                       # Tight: the default 16pt margin is visible whitespace once the
                       # figure is placed in a document that already has margins.
                       figure_padding = (2, 4, 2, 2),
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
                 labelsize = 13, patchsize = (20.0f0, 10.0f0))
    Makie.colsize!(fig.layout, 2, Makie.Auto(false))

    mkpath(dirname(path))
    for ext in ("pdf", "svg", "png")
        Makie.save("$path.$ext", fig; backgroundcolor = :transparent)
    end
    return fig
end

"""
    sweep_summary(rows, key) -> Vector{NamedTuple}

Aggregate sweep checkpoints into one row per (arm, swept value).

  - `rows` — checkpoints from [`load_sweep`](@ref)
  - `key` — the swept field name, e.g. `:sigma_nav_km`

Returns `(arm, value, n, science, science_sd, frac_degraded, survival, misbin, dv,
samples)`, sorted by arm then value. `survival` is the fraction of seeds that neither
crashed nor escaped; `misbin` is the measured observed-versus-true bin disagreement rate,
which is the mechanism a nav sweep is testing.

NOTE: reads only fields `save_rollout` writes, so a sweep does not need re-flying to add a
metric here.
"""
function sweep_summary(rows, key::Symbol)
    k = String(key)
    cells = Dict{Tuple{String,Float64},Vector{Dict{String,Any}}}()
    for r in rows
        haskey(r, k) || continue
        push!(get!(cells, (r["arm"], Float64(r[k])), Dict{String,Any}[]), r)
    end
    out = NamedTuple[]
    for ((arm, v), rs) in cells
        sci = [Float64(r["science"]) for r in rs]
        # Misbin is per pass, pooled over the cell's rollouts rather than averaged per
        # rollout, so a short (lost) run does not weigh as much as a full one.
        mis = 0; nb = 0
        for r in rs, (t, o) in zip(get(r, "true_bins", String[]), get(r, "obs_bins", String[]))
            (isempty(t) || isempty(o)) && continue
            nb += 1; t == o || (mis += 1)
        end
        # Survival time, three ways, because one number cannot carry a bimodal
        # distribution: runs either fail within days or reach the horizon.
        #   `days_mean`   over ALL runs, censored ones counted at the horizon. Lands in the
        #                 gap between the two modes, so it describes no actual run.
        #   `days_median` same population, robust to the split but still a mixture.
        #   `days_failed` median over the runs that DIED — "when it fails, how fast".
        dys = [Float64(r["survival_days"]) for r in rs]
        fail = [Float64(r["survival_days"]) for r in rs if !Bool(r["survived"])]
        # Science among the runs that SURVIVED. Pooling survivors with failures averages two
        # populations: a run that dies on day 3 banks almost nothing, so the mixed mean
        # tracks the survival rate rather than the science a working controller collects.
        # Measured at sigma = 0.3: Threshold pools to 54.7 +/- 31.0, but its three survivors
        # are 92.4 +/- 3.9 against the policy's 100.2 +/- 4.6 — an 8% gap, not a 2x one.
        # NaN when nothing survived; there is no science to report, which a gap in the plot
        # states more honestly than a zero.
        srv = [Float64(r["science"]) for r in rs if Bool(r["survived"])]
        push!(out, (arm = arm, value = v, n = length(rs),
                    days_mean = mean(dys),
                    days_sd = length(dys) < 2 ? 0.0 : std(dys),
                    days_median = median(dys),
                    days_failed = isempty(fail) ? NaN : median(fail),
                    # Mean over the runs that DIED. More sensitive to a long tail than the
                    # median, so an arm with a few late failures reads higher; with one
                    # failure in a cell the two coincide.
                    days_failed_mean = isempty(fail) ? NaN : mean(fail),
                    n_failed = length(fail),
                    science = mean(sci),
                    science_sd = length(sci) < 2 ? 0.0 : std(sci),
                    science_survivors = isempty(srv) ? NaN : mean(srv),
                    science_survivors_sd = length(srv) < 2 ? 0.0 : std(srv),
                    n_survivors = length(srv),
                    frac_degraded = mean(Float64(r["frac_degraded"]) for r in rs),
                    survival = mean(Bool(r["survived"]) for r in rs),
                    misbin = nb == 0 ? NaN : mis / nb,
                    dv = mean(Float64(r["total_dv_ms"]) for r in rs),
                    samples = mean(Float64(r["n_samples"]) for r in rs)))
    end
    return sort(out, by = r -> (r.arm, r.value))
end

"""
    plot_sweep(rows, key; path, theme, xlabel, panels, size)

Plot a parameter sweep: one line per controller, the swept value on x.

  - `rows` — checkpoints from [`load_sweep`](@ref)
  - `key` — the swept field, e.g. `:sigma_nav_km`
  - `path` — output path WITHOUT extension; writes `.pdf`, `.svg` and `.png`
  - `theme` — `:light` or `:dark`; saved transparent so either background works
  - `xlabel` — x-axis label; defaults to the field name
  - `arms` — arms to draw, in legend order; `nothing` uses every arm present
  - `legend_below` — one horizontal row under the panels, for a single-column figure
  - `bar_panels` — panels to draw as grouped bars rather than lines. Use it for a metric
    that is flat across the swept axis, where a line invites reading a trend that is not
    there, or whose sample size varies per point,
    instead of a column beside them
  - `panels` — which metrics to stack, from `(:science, :science_survivors,
    :frac_degraded, :survival, :dv, :misbin, :samples)`

NOTE: `:misbin` is near zero at realistic navigation error — the altitude regions are
7-10 km wide, so a sub-kilometre read almost never lands in the wrong one. It is
informative only over a much wider sigma range; the failure mechanism at these levels is
PLANNING error, not misattribution.

Returns the Makie `Figure`. Requires CairoMakie to be loaded by the caller.

Science carries a ±1sd band across seeds. This answers whether a ranking measured at one
theta survives the sweep, which a single-run time series cannot.

NOTE: on a nav sweep, read the flat lines carefully. `MPCController` consumes no
observation and `CyclicController` uses one only to bank coverage, so neither is really
being tested by `sigma_nav_km` — they are open-loop reference floors, not robust
controllers. The informative comparison is the policy against the closed-loop baselines
(greedy, threshold), which choose actions from what they observed.

    using SherpaOrbital, CairoMakie
    plot_sweep(load_sweep("artifacts/sweeps/sigma_nav_km"), :sigma_nav_km;
               path = "figures/nav_sweep", xlabel = "Navigation error sigma (km)")
"""
function plot_sweep(rows, key::Symbol;
                    path::AbstractString = "figures/sweep",
                    theme::Symbol = :light,
                    xlabel::AbstractString = String(key),
                    arms = nothing,
                    legend_below::Bool = false,
                    bar_panels = (),
                    panels = (:science, :frac_degraded, :survival, :dv),
                    size::Tuple{Int,Int} = (700, 200 * length(panels) + 60))
    Makie = get(Base.loaded_modules,
                Base.PkgId(Base.UUID("13f3f980-e62b-5c42-98c6-ff1f3baf88f0"), "CairoMakie"),
                nothing)
    Makie === nothing && error(
        "plot_sweep needs CairoMakie loaded by the caller: `using CairoMakie` before " *
        "calling. The library declares no plotting dependency.")

    summ = sweep_summary(rows, key)
    isempty(summ) && error("no checkpoints carrying the field $key")

    fg = theme === :dark ? Makie.RGBf(0.92, 0.92, 0.92) : Makie.RGBf(0.10, 0.10, 0.10)
    palette = [Makie.RGBf(0.00, 0.45, 0.70), Makie.RGBf(0.90, 0.62, 0.00),
               Makie.RGBf(0.00, 0.62, 0.45), Makie.RGBf(0.80, 0.47, 0.65),
               Makie.RGBf(0.84, 0.37, 0.00), Makie.RGBf(0.35, 0.35, 0.35),
               Makie.RGBf(0.58, 0.44, 0.86), Makie.RGBf(0.00, 0.62, 0.79)]
    LABEL = Dict(:science => "Science reward, all\nruns (mean ± 1 sd)",
                 :frac_degraded => "Fraction of passes\ndegraded",
                 :survival => "Survival rate", :misbin => "Region misbin rate",
                 :dv => "Total dV (m/s)", :samples => "Samples banked",
                 :days_mean => "Mean survival\n(days)",
                 :days_median => "Median survival\n(days)",
                 :days_failed => "Days to failure\n(median, failed runs)",
                 :days_failed_mean => "Days to failure\n(mean, failed runs)",
                 :science_survivors => "Science reward, surviving\nruns (mean ± 1 sd)")

    fig = Makie.Figure(; size = size, backgroundcolor = :transparent,
                       # Tight: the default 16pt margin is visible whitespace once the
                       # figure is placed in a document that already has margins.
                       figure_padding = (2, 4, 2, 2),
                       fonts = (; regular = "CMU Serif", bold = "CMU Serif Bold"))
    # Caller order, so the legend can lead with the arm the figure is about; `sweep_summary`
    # sorts alphabetically, which buries it.
    present = unique(r.arm for r in summ)
    arms = arms === nothing ? present : [a for a in arms if a in present]
    axes = Makie.Axis[]

    xall = sort(unique(r.value for r in summ))
    for (pi, p) in enumerate(panels)
        last_panel = pi == length(panels)
        ax = Makie.Axis(fig[pi, 1];
                        xlabel = last_panel ? xlabel : "",
                        ylabel = get(LABEL, p, String(p)),
                        backgroundcolor = :transparent,
                        xgridvisible = false, ygridvisible = true,
                        xlabelcolor = fg, ylabelcolor = fg,
                        xticklabelcolor = fg, yticklabelcolor = fg,
                        leftspinecolor = fg, bottomspinecolor = fg,
                        topspinevisible = false, rightspinevisible = false,
                        xtickcolor = fg, ytickcolor = fg)
        push!(axes, ax)
        for (ai, arm) in enumerate(arms)
            rs = filter(r -> r.arm == arm, summ)
            isempty(rs) && continue
            col = palette[mod1(ai, length(palette))]
            # MPC hold collects no science, so it is dashed to mark it as the reference
            # floor rather than a competitor.
            sty = occursin("MPC", arm) ? :dash : :solid
            x = [r.value for r in rs]
            y = [Float64(getproperty(r, p)) for r in rs]
            # `days_failed` is NaN where nothing failed, and `science_survivors` NaN where
            # nothing survived. Drop those points: a gap says "not defined here", a zero
            # would claim a measurement.
            if any(!isfinite, y)
                ok = isfinite.(y)
                x, y = x[ok], y[ok]
                sdv_keep = ok
            else
                sdv_keep = trues(length(y))
            end
            isempty(x) && continue
            # Band in the ARM's colour: a shared grey pools into an unreadable smudge once
            # several arms overlap.
            sdv = p === :science           ? [r.science_sd for r in rs] :
                  p === :science_survivors ? [r.science_survivors_sd for r in rs] : nothing
            # Per-arm colour at low alpha. A shared grey pools into one smudge, and the
            # pooled-science sd is genuinely wide — it averages runs that died on day 3
            # with runs that flew the full horizon — so the band cannot be made small
            # without changing what is plotted. Use `:science_survivors` for a tight one.
            if sdv !== nothing
                lo, hi = y .- sdv[sdv_keep], y .+ sdv[sdv_keep]
                Makie.band!(ax, x, lo, hi; color = (col, 0.10))
                Makie.lines!(ax, x, lo; color = (col, 0.35), linewidth = 0.5)
                Makie.lines!(ax, x, hi; color = (col, 0.35), linewidth = 0.5)
            end
            if p in bar_panels
                # CATEGORICAL x: the swept values are unevenly spaced, so a bar sized in
                # data units collapses to the smallest gap between them. Each value gets
                # one integer slot and the arms are offset inside it.
                slot = [Float64(findfirst(==(v), xall)) for v in x]
                w    = 0.8 / max(length(arms), 1)
                xs = slot .+ (ai - (length(arms) + 1) / 2) * w
                Makie.barplot!(ax, xs, y;
                               width = w * 0.9, color = (col, 0.85),
                               strokecolor = col, strokewidth = 0.4,
                               label = pi == 1 ? arm : nothing)
                # Spread across rollouts, where the metric has one. CLAMPED TO THE
                # SUPPORT: a survival time lives in [0, horizon], so an unclamped
                # mean ± sd whisker claims runs lasted longer than the run did. The
                # distribution is bimodal — a run either fails in days or reaches the
                # horizon — so the sd is wide and the clamp bites on most arms.
                bsd = p === :days_mean ? [r.days_sd for r in rs][sdv_keep] : nothing
                if bsd !== nothing
                    hi = maximum(r.days_mean for r in summ)   # the horizon, as flown
                    Makie.rangebars!(ax, xs, max.(y .- bsd, 0.0), min.(y .+ bsd, hi);
                                     color = fg, whiskerwidth = 5, linewidth = 1.1)
                end
            else
                Makie.lines!(ax, x, y; color = col, linewidth = 2, linestyle = sty,
                             label = pi == 1 ? arm : nothing)
                Makie.scatter!(ax, x, y; color = col, markersize = 7)
            end
        end
        if p in bar_panels
            ax.xticks = (collect(eachindex(xall)), string.(xall))
            Makie.xlims!(ax, 0.4, length(xall) + 0.6)
        end
        p in (:survival, :frac_degraded, :misbin) && Makie.ylims!(ax, -0.04, 1.04)
        p in (:days_mean, :days_median, :days_failed, :days_failed_mean) &&
            Makie.ylims!(ax, low = 0)
        (last_panel || p in bar_panels) || Makie.hidexdecorations!(ax; grid = false)
    end
    # Only the line panels share the numeric x axis; a bar panel is categorical.
    linkable = [ax for (ax, p) in zip(axes, panels) if !(p in bar_panels)]
    length(linkable) > 1 && Makie.linkxaxes!(linkable...)

    if legend_below
        # One row under the panels: a side legend narrows them enough that the x axis
        # crowds in a single-column figure.
        #
        # NOTE: this centres on the PLOT AREA, not the figure, so it sits slightly right
        # of centre — the axis column is inset by the width of the y tick labels and the
        # legend inherits that geometry. Cosmetic; left as is.
        Makie.Legend(fig[length(panels) + 1, 1], first(axes); framevisible = false,
                     labelcolor = fg, labelsize = 13, patchsize = (22.0f0, 10.0f0),
                     orientation = :horizontal, nbanks = 2, colgap = 14, rowgap = 2,
                     halign = :center, tellheight = true, tellwidth = false)
    else
        Makie.Legend(fig[1, 2], first(axes); framevisible = false, labelcolor = fg,
                     labelsize = 13, patchsize = (20.0f0, 10.0f0))
        Makie.colsize!(fig.layout, 2, Makie.Auto(false))
    end

    mkpath(dirname(path))
    for ext in ("pdf", "svg", "png")
        Makie.save("$path.$ext", fig; backgroundcolor = :transparent)
    end
    return fig
end

"""
    plot_sweep_timelines(rows, key; path, theme, xlabel, arms, seed, size)

Cumulative science over the horizon, one panel per swept value.

  - `rows` — checkpoints from [`load_sweep`](@ref)
  - `key` — the swept field, e.g. `:sigma_nav_km`
  - `path` — output path WITHOUT extension; writes `.pdf`, `.svg` and `.png`
  - `theme` — `:light` or `:dark`; saved transparent so either background works
  - `xlabel` — x-axis label for the shared time axis
  - `arms` — which arms to draw, in legend order; `nothing` uses every arm present
  - `seed` — which seed's trace to draw per cell; `nothing` draws the median-science seed

Returns the Makie `Figure`. Requires CairoMakie to be loaded by the caller.

Shows WHEN a controller diverges rather than only where it ended: a run lost mid-horizon
stops at its loss time and is marked, so a panel distinguishes "earned less" from "died
trying". Reads the stored `science_cum` trace, so it needs no re-flying.
"""
function plot_sweep_timelines(rows, key::Symbol;
                              path::AbstractString = "figures/sweep_timelines",
                              theme::Symbol = :light,
                              xlabel::AbstractString = "Time (days)",
                              arms = nothing,
                              seed::Union{Nothing,Integer} = nothing,
                              size::Union{Nothing,Tuple{Int,Int}} = nothing)
    Makie = get(Base.loaded_modules,
                Base.PkgId(Base.UUID("13f3f980-e62b-5c42-98c6-ff1f3baf88f0"), "CairoMakie"),
                nothing)
    Makie === nothing && error(
        "plot_sweep_timelines needs CairoMakie loaded by the caller: `using CairoMakie` " *
        "before calling. The library declares no plotting dependency.")

    k = String(key)
    vals = sort(unique(Float64(r[k]) for r in rows if haskey(r, k)))
    isempty(vals) && error("no checkpoints carrying the field $key")
    all_arms = unique(r["arm"] for r in rows)
    draw = arms === nothing ? all_arms : [a for a in arms if a in all_arms]

    fg = theme === :dark ? Makie.RGBf(0.92, 0.92, 0.92) : Makie.RGBf(0.10, 0.10, 0.10)
    palette = [Makie.RGBf(0.00, 0.45, 0.70), Makie.RGBf(0.90, 0.62, 0.00),
               Makie.RGBf(0.00, 0.62, 0.45), Makie.RGBf(0.80, 0.47, 0.65),
               Makie.RGBf(0.84, 0.37, 0.00), Makie.RGBf(0.35, 0.35, 0.35),
               Makie.RGBf(0.58, 0.44, 0.86), Makie.RGBf(0.00, 0.62, 0.79)]

    figsize = size === nothing ? (260 * length(vals) + 150, 300) : size
    fig = Makie.Figure(; size = figsize, backgroundcolor = :transparent,
                       figure_padding = (2, 4, 2, 2),
                       fonts = (; regular = "CMU Serif", bold = "CMU Serif Bold"))

    # Shared y limit so panels are comparable by eye rather than each self-scaling.
    ymax = maximum(maximum(Float64.(r["science_cum"]); init = 0.0)
                   for r in rows if r["arm"] in draw && !isempty(r["science_cum"]))
    axes = Makie.Axis[]

    for (vi, v) in enumerate(vals)
        ax = Makie.Axis(fig[1, vi];
                        xlabel = xlabel,
                        ylabel = vi == 1 ? "Cumulative science reward" : "",
                        title = "$k = $v",
                        backgroundcolor = :transparent,
                        xgridvisible = false, ygridvisible = true,
                        titlecolor = fg, xlabelcolor = fg, ylabelcolor = fg,
                        xticklabelcolor = fg, yticklabelcolor = fg,
                        leftspinecolor = fg, bottomspinecolor = fg,
                        topspinevisible = false, rightspinevisible = false,
                        xtickcolor = fg, ytickcolor = fg)
        push!(axes, ax)

        for (ai, arm) in enumerate(draw)
            cell = filter(r -> r["arm"] == arm && haskey(r, k) && Float64(r[k]) == v, rows)
            isempty(cell) && continue
            # One trace per panel: the requested seed, else the median-science seed, so a
            # panel shows a REAL run rather than an average of runs that ended at
            # different times.
            pick = if seed === nothing
                cell[sortperm([Float64(r["science"]) for r in cell])[cld(length(cell), 2)]]
            else
                idx = findfirst(r -> Int(r["seed"]) == seed, cell)
                idx === nothing ? first(cell) : cell[idx]
            end
            t, y = Float64.(pick["t_days"]), Float64.(pick["science_cum"])
            (isempty(t) || isempty(y)) && continue
            n = min(length(t), length(y))
            col = palette[mod1(ai, length(palette))]
            sty = occursin("MPC", arm) ? :dash : :solid
            Makie.lines!(ax, t[1:n], y[1:n]; color = col, linewidth = 2, linestyle = sty,
                         label = vi == 1 ? arm : nothing)
            # A lost run ends where it died; the marker says the curve STOPPED rather than
            # flattened, which a truncated line alone does not distinguish.
            if !Bool(pick["survived"])
                Makie.scatter!(ax, [t[n]], [y[n]]; color = col, marker = :xcross,
                               markersize = 11)
            end
        end
        Makie.ylims!(ax, 0, ymax * 1.05)
        vi == 1 || Makie.hideydecorations!(ax; grid = false)
    end
    Makie.linkaxes!(axes...)

    Makie.Legend(fig[1, length(vals) + 1], first(axes); framevisible = false,
                 labelcolor = fg, labelsize = 13, patchsize = (20.0f0, 10.0f0))
    Makie.colsize!(fig.layout, length(vals) + 1, Makie.Auto(false))

    mkpath(dirname(path))
    for ext in ("pdf", "svg", "png")
        Makie.save("$path.$ext", fig; backgroundcolor = :transparent)
    end
    return fig
end

"""
    survival_days(rows, key, value; arms = nothing) -> Vector{NamedTuple}

Per-rollout survival time for one sweep cell, grouped by arm.

  - `rows` — checkpoints from [`load_sweep`](@ref)
  - `key` — the swept field, e.g. `:sigma_nav_km`
  - `value` — the swept value to select
  - `arms` — arms to include, in order; `nothing` uses every arm present

Returns `(arm, days, censored)` per arm: `days` is the survival time of each rollout and
`censored` marks the ones that reached the horizon without a crash or an escape.

NOTE: a rollout that survives reports the HORIZON, not a time of death. Those entries are
right-censored — the run would have gone on — so a mean over `days` understates nothing but
also means nothing. Plot the distribution, and mark the censored fraction.
"""
function survival_days(rows, key::Symbol, value::Real; arms = nothing)
    k = String(key)
    present = unique(r["arm"] for r in rows)
    draw = arms === nothing ? present : [a for a in arms if a in present]
    out = NamedTuple[]
    for arm in draw
        cell = filter(r -> r["arm"] == arm && haskey(r, k) && Float64(r[k]) == Float64(value),
                      rows)
        isempty(cell) && continue
        push!(out, (arm = arm,
                    days = [Float64(r["survival_days"]) for r in cell],
                    censored = [Bool(r["survived"]) for r in cell]))
    end
    return out
end

"""
    plot_survival_box(rows, key; path, theme, xlabel, arms, size)

Survival time per rollout as a box plot, one group per swept value.

  - `rows` — checkpoints from [`load_sweep`](@ref)
  - `key` — the swept field, e.g. `:sigma_nav_km`
  - `path` — output path WITHOUT extension; writes `.pdf`, `.svg` and `.png`
  - `theme` — `:light` or `:dark`; saved transparent so either background works
  - `xlabel` — x-axis label
  - `arms` — arms to draw, in order; `nothing` uses every arm present

Returns the Makie `Figure`. Requires CairoMakie to be loaded by the caller.

An arm that never died draws a bar at the horizon rather than a box, so perfect survival
does not read as missing data; a partly-censored arm is annotated with how many of its
rollouts reached the horizon.

Replaces a mean-and-spread summary of survival, which is misleading here: the distribution
is bimodal — a run either fails within days or reaches the horizon — so a mean sits in a gap
where no rollout landed. Censored rollouts (reached the horizon) are drawn as open markers
at the top.
"""
function plot_survival_box(rows, key::Symbol;
                           path::AbstractString = "figures/survival",
                           theme::Symbol = :light,
                           xlabel::AbstractString = String(key),
                           arms = nothing,
                           size::Tuple{Int,Int} = (760, 380))
    Makie = get(Base.loaded_modules,
                Base.PkgId(Base.UUID("13f3f980-e62b-5c42-98c6-ff1f3baf88f0"), "CairoMakie"),
                nothing)
    Makie === nothing && error(
        "plot_survival_box needs CairoMakie loaded by the caller: `using CairoMakie` " *
        "before calling. The library declares no plotting dependency.")

    k = String(key)
    vals = sort(unique(Float64(r[k]) for r in rows if haskey(r, k)))
    present = unique(r["arm"] for r in rows)
    draw = arms === nothing ? present : [a for a in arms if a in present]

    fg = theme === :dark ? Makie.RGBf(0.92, 0.92, 0.92) : Makie.RGBf(0.10, 0.10, 0.10)
    palette = [Makie.RGBf(0.00, 0.45, 0.70), Makie.RGBf(0.90, 0.62, 0.00),
               Makie.RGBf(0.00, 0.62, 0.45), Makie.RGBf(0.80, 0.47, 0.65),
               Makie.RGBf(0.84, 0.37, 0.00), Makie.RGBf(0.35, 0.35, 0.35),
               Makie.RGBf(0.58, 0.44, 0.86), Makie.RGBf(0.00, 0.62, 0.79)]

    fig = Makie.Figure(; size = size, backgroundcolor = :transparent,
                       # Tight: the default 16pt margin is visible whitespace once the
                       # figure is placed in a document that already has margins.
                       figure_padding = (2, 4, 2, 2),
                       fonts = (; regular = "CMU Serif", bold = "CMU Serif Bold"))
    # Arms are clustered within each swept value, so a group reads as "at this sigma, who
    # survives" rather than making the reader hop between panels.
    n_arm = length(draw)
    centres = collect(eachindex(vals)) .* (n_arm + 2.5)
    ax = Makie.Axis(fig[1, 1]; xlabel = xlabel, ylabel = "Survival time (days)",
                    xticks = (centres, string.(vals)),
                    backgroundcolor = :transparent,
                    xgridvisible = false, ygridvisible = true,
                    xlabelcolor = fg, ylabelcolor = fg,
                    xticklabelcolor = fg, yticklabelcolor = fg,
                    leftspinecolor = fg, bottomspinecolor = fg,
                    topspinevisible = false, rightspinevisible = false,
                    xtickcolor = fg, ytickcolor = fg)

    horizon = maximum(Float64(r["survival_days"]) for r in rows)
    for (vi, v) in enumerate(vals)
        groups = survival_days(rows, key, v; arms = draw)
        for (ai, g) in enumerate(groups)
            col = palette[mod1(findfirst(==(g.arm), draw), length(palette))]
            x = centres[vi] - (n_arm + 1) / 2 + ai
            nc = count(g.censored)
            # An arm that never died has no box to draw, and an empty slot reads as missing
            # data rather than as perfect survival. Mark it with a filled bar at the
            # horizon instead, and keep the legend entry on this path too.
            if nc == length(g.days)
                Makie.scatter!(ax, [x], [horizon]; color = col, marker = :hline,
                               markersize = 16, strokewidth = 0,
                               label = vi == 1 ? g.arm : nothing)
            else
                Makie.boxplot!(ax, fill(x, length(g.days)), g.days;
                               width = 0.62, color = (col, 0.55), strokecolor = col,
                               strokewidth = 1, markersize = 0,
                               label = vi == 1 ? g.arm : nothing)
            end
            # Censored runs reached the horizon; an open marker says the run did not end
            # there, the experiment did. The count is annotated so a partly-censored box is
            # readable without counting whiskers.
            0 < nc < length(g.days) &&
                Makie.text!(ax, x, horizon * 1.03; text = string(nc), color = col,
                            fontsize = 9, align = (:center, :bottom))
        end
    end
    Makie.hlines!(ax, [horizon]; color = (fg, 0.25), linestyle = :dot, linewidth = 1)
    Makie.ylims!(ax, 0, horizon * 1.15)

    Makie.Legend(fig[1, 2], ax; framevisible = false, labelcolor = fg, labelsize = 13,
                 patchsize = (20.0f0, 10.0f0), merge = true)
    Makie.colsize!(fig.layout, 2, Makie.Auto(false))

    mkpath(dirname(path))
    for ext in ("pdf", "svg", "png")
        Makie.save("$path.$ext", fig; backgroundcolor = :transparent)
    end
    return fig
end

"""
    plot_sweep_bars(rows, key; path, theme, xlabel, arms, panels, size)

Grouped bars per swept value: one bar per controller, one panel per metric.

  - `rows` — checkpoints from [`load_sweep`](@ref)
  - `key` — the swept field, e.g. `:sigma_nav_km`
  - `path` — output path WITHOUT extension; writes `.pdf`, `.svg` and `.png`
  - `theme` — `:light` or `:dark`; saved transparent so either background works
  - `xlabel` — x-axis label
  - `arms` — arms to draw, in order; `nothing` uses every arm present
  - `panels` — metrics to stack, default survivors-only science then survival rate

Returns the Makie `Figure`. Requires CairoMakie to be loaded by the caller.

Bars rather than lines because the swept values are treated as separate conditions to
compare across, not as a continuum to interpolate along — and because a controller with no
survivors has no science to plot, which a missing bar states and a broken line does not.

Defaults to `:science_survivors`: pooling survivors with failures averages two populations,
so the pooled mean tracks the survival rate instead of the science a working controller
collects. The survival panel carries the risk.
"""
function plot_sweep_bars(rows, key::Symbol;
                         path::AbstractString = "figures/sweep_bars",
                         theme::Symbol = :light,
                         xlabel::AbstractString = String(key),
                         arms = nothing,
                         panels = (:science_survivors, :survival),
                         size::Union{Nothing,Tuple{Int,Int}} = nothing)
    Makie = get(Base.loaded_modules,
                Base.PkgId(Base.UUID("13f3f980-e62b-5c42-98c6-ff1f3baf88f0"), "CairoMakie"),
                nothing)
    Makie === nothing && error(
        "plot_sweep_bars needs CairoMakie loaded by the caller: `using CairoMakie` before " *
        "calling. The library declares no plotting dependency.")

    summ = sweep_summary(rows, key)
    isempty(summ) && error("no checkpoints carrying the field $key")
    vals = sort(unique(r.value for r in summ))
    present = unique(r.arm for r in summ)
    draw = arms === nothing ? present : [a for a in arms if a in present]

    fg = theme === :dark ? Makie.RGBf(0.92, 0.92, 0.92) : Makie.RGBf(0.10, 0.10, 0.10)
    palette = [Makie.RGBf(0.00, 0.45, 0.70), Makie.RGBf(0.90, 0.62, 0.00),
               Makie.RGBf(0.00, 0.62, 0.45), Makie.RGBf(0.80, 0.47, 0.65),
               Makie.RGBf(0.84, 0.37, 0.00), Makie.RGBf(0.35, 0.35, 0.35),
               Makie.RGBf(0.58, 0.44, 0.86), Makie.RGBf(0.00, 0.62, 0.79)]
    LABEL = Dict(:science_survivors => "Science reward\n(surviving runs)",
                 :science => "Science reward", :survival => "Survival rate",
                 :frac_degraded => "Fraction of passes\ndegraded",
                 :dv => "Total dV (m/s)", :days_mean => "Mean survival\n(days)")

    n_arm = length(draw)
    figsize = size === nothing ? (170 * length(vals) + 210, 200 * length(panels) + 60) : size
    fig = Makie.Figure(; size = figsize, backgroundcolor = :transparent,
                       figure_padding = (2, 4, 2, 2),
                       fonts = (; regular = "CMU Serif", bold = "CMU Serif Bold"))
    centres = collect(eachindex(vals)) .* (n_arm + 2.0)
    axes = Makie.Axis[]

    xall = sort(unique(r.value for r in summ))
    for (pi, p) in enumerate(panels)
        last_panel = pi == length(panels)
        ax = Makie.Axis(fig[pi, 1];
                        xlabel = last_panel ? xlabel : "",
                        ylabel = get(LABEL, p, String(p)),
                        xticks = (centres, string.(vals)),
                        backgroundcolor = :transparent,
                        xgridvisible = false, ygridvisible = true,
                        xlabelcolor = fg, ylabelcolor = fg,
                        xticklabelcolor = fg, yticklabelcolor = fg,
                        leftspinecolor = fg, bottomspinecolor = fg,
                        topspinevisible = false, rightspinevisible = false,
                        xtickcolor = fg, ytickcolor = fg)
        push!(axes, ax)

        for (ai, arm) in enumerate(draw)
            col = palette[mod1(ai, length(palette))]
            xs = Float64[]; ys = Float64[]; los = Float64[]; his = Float64[]
            for (vi, v) in enumerate(vals)
                row = findfirst(r -> r.arm == arm && r.value == v, summ)
                row === nothing && continue
                y = Float64(getproperty(summ[row], p))
                # No survivors means no science to report; skip the bar rather than
                # drawing a zero, which would read as "collected nothing" instead of
                # "never got there".
                isfinite(y) || continue
                push!(xs, centres[vi] - (n_arm + 1) / 2 + ai); push!(ys, y)
                if p === :science_survivors
                    sd = summ[row].science_survivors_sd
                    push!(los, y - sd); push!(his, y + sd)
                end
            end
            isempty(xs) && continue
            Makie.barplot!(ax, xs, ys; width = 0.85, color = (col, 0.85),
                           strokecolor = col, strokewidth = 0.5,
                           label = pi == 1 ? arm : nothing)
            isempty(los) ||
                Makie.rangebars!(ax, xs, los, his; color = fg, whiskerwidth = 5,
                                 linewidth = 0.9)
        end
        p === :survival && Makie.ylims!(ax, 0, 1.05)
        (last_panel || p in bar_panels) || Makie.hidexdecorations!(ax; grid = false)
    end

    Makie.Legend(fig[1, 2], first(axes); framevisible = false, labelcolor = fg,
                 labelsize = 13, patchsize = (18.0f0, 10.0f0))
    Makie.colsize!(fig.layout, 2, Makie.Auto(false))

    mkpath(dirname(path))
    for ext in ("pdf", "svg", "png")
        Makie.save("$path.$ext", fig; backgroundcolor = :transparent)
    end
    return fig
end

"""
    action_bands(rows, key, value; arm = "POMDP", actions = nothing) -> (M, n, nroll)

Fraction of rollouts choosing each action at each pass, for one sweep cell.

  - `rows` — checkpoints from [`load_sweep`](@ref)
  - `key` — the swept field, e.g. `:sigma_nav_km`
  - `value` — the swept value to select
  - `arm` — which controller's traces to measure
  - `actions` — action order, top to bottom; `nothing` orders by commanded altitude

Returns `M[action, pass]`, the number of passes `n`, the number of rollouts `nroll`, the
count still flying at each pass, and the median elapsed time per pass.

`n` is the LONGEST trace, and each column is normalised by how many rollouts were still
flying at that pass. Truncating to the shortest trace instead would let a single early
death collapse the whole cell: at 200 seeds with one rollout lost at pass 7, a
shortest-trace strip shows 7 passes of a 60-pass episode and reads as a clean short
pattern.

`alive[p]` is returned alongside, since a column late in the horizon may rest on very few
rollouts.

NOTE: a policy is a function of belief, not of pass number, so a column being unanimous
means every rollout arrived at the same belief by that pass — not that the policy is
open-loop. Unanimity decaying with pass is the belief distribution spreading.
"""
function action_bands(rows, key::Symbol, value::Real;
                      arm::AbstractString = "POMDP", actions = nothing)
    k = String(key)
    cell = filter(r -> r["arm"] == arm && haskey(r, k) &&
                       Float64(r[k]) == Float64(value), rows)
    isempty(cell) && return (M = zeros(0, 0), n = 0, nroll = 0)
    acts = actions === nothing ?
        ["EXCURSE_HIGH", "CORRECT", "EXCURSE_MID", "EXCURSE_LOW"] : collect(actions)
    n = maximum(length(r["actions"]) for r in cell)
    n == 0 && return (M = zeros(length(acts), 0), n = 0, nroll = length(cell),
                      alive = Int[])
    alive = [count(r -> length(r["actions"]) >= p, cell) for p in 1:n]
    # Median elapsed time at each pass, so the strip can be drawn against days. The passes
    # are near-periodic, so the spread across rollouts is small; the median keeps one run
    # that coasted long from stretching the axis.
    t_days = [median(Float64(r["t_days"][p]) for r in cell if length(r["t_days"]) >= p)
              for p in 1:n]
    M = [alive[p] == 0 ? 0.0 :
         count(r -> length(r["actions"]) >= p && r["actions"][p] == a, cell) / alive[p]
         for a in acts, p in 1:n]
    return (M = M, n = n, nroll = length(cell), alive = alive, t_days = t_days)
end

"""
    plot_action_bands(rows, key; path, theme, values, arm, size)

Action chosen per pass, one row of strips per swept value.

  - `rows` — checkpoints from [`load_sweep`](@ref)
  - `key` — the swept field, e.g. `:sigma_nav_km`
  - `path` — output path WITHOUT extension; writes `.pdf`, `.svg` and `.png`
  - `theme` — `:light` or `:dark`; saved transparent so either background works
  - `values` — swept values to draw, top to bottom; `nothing` uses all present
  - `arm` — which controller to measure
  - `label` — how to name the swept quantity in each row title
  - `unit` — unit appended to the swept value in each row title
  - `xaxis` — `:days` for elapsed mission time, `:pass` for the periapsis index

Returns the Makie `Figure`. Requires CairoMakie to be loaded by the caller.

One heatmap strip per action, shaded in that action's own colour with opacity carrying the
fraction of rollouts that chose it. Rows are ordered by commanded altitude, so the vertical
position of the shading reads directly as "how low is it flying".

Stacking one row per swept value shows how the policy's behaviour changes with the
uncertainty it was solved for — the campaign structure at low noise against whatever
replaces it at high noise.
"""
function plot_action_bands(rows, key::Symbol;
                           path::AbstractString = "figures/action_bands",
                           theme::Symbol = :light,
                           values = nothing,
                           arm::AbstractString = "POMDP",
                           label::AbstractString = String(key),
                           unit::AbstractString = "",
                           xaxis::Symbol = :days,
                           size::Union{Nothing,Tuple{Int,Int}} = nothing)
    Makie = get(Base.loaded_modules,
                Base.PkgId(Base.UUID("13f3f980-e62b-5c42-98c6-ff1f3baf88f0"), "CairoMakie"),
                nothing)
    Makie === nothing && error(
        "plot_action_bands needs CairoMakie loaded by the caller: `using CairoMakie` " *
        "before calling. The library declares no plotting dependency.")

    k = String(key)
    vals = values === nothing ?
        sort(unique(Float64(r[k]) for r in rows if haskey(r, k))) : collect(values)
    # Top to bottom by commanded altitude, so the shading's height reads as altitude.
    ACTS = ["EXCURSE_HIGH", "CORRECT", "EXCURSE_MID", "EXCURSE_LOW"]
    # Tick labels drop the EXCURSE_ prefix and carry the commanded altitude on ONE line —
    # a two-line label collides with its neighbour at this row height.
    TICKS = ["HIGH (46 km)", "CORRECT (37 km)", "MID (30 km)", "LOW (24 km)"]
    fg = theme === :dark ? Makie.RGBf(0.92, 0.92, 0.92) : Makie.RGBf(0.10, 0.10, 0.10)
    ACOL = [Makie.RGBf(0.93, 0.47, 0.20), Makie.RGBf(0.62, 0.62, 0.62),
            Makie.RGBf(0.13, 0.53, 0.20), Makie.RGBf(0.27, 0.47, 0.67)]

    rowtitle(v) = isempty(unit) ? "$label = $v" : "$label = $v $unit"
    cells = [(v, action_bands(rows, key, v; arm = arm, actions = ACTS)) for v in vals]
    cells = [(v, c) for (v, c) in cells if c.n > 0]
    isempty(cells) && error("no traces for $arm at any requested value of $key")

    nr = length(cells)
    figsize = size === nothing ? (520, 108 * nr + 58) : size
    fig = Makie.Figure(; size = figsize, backgroundcolor = :transparent,
                       figure_padding = (2, 4, 2, 2),
                       fonts = (; regular = "CMU Serif", bold = "CMU Serif Bold"))

    for (r, (v, c)) in enumerate(cells)
        ax = Makie.Axis(fig[r, 1];
                        yticks = (1:4, TICKS),
                        xlabel = r == nr ?
                            (xaxis === :days ? "Mission time (days)" : "Periapsis pass") : "",
                        xticklabelsvisible = r == nr,
                        ylabel = "", yticklabelsize = 8,
                        title = rowtitle(v),
                        titlealign = :left, titlesize = 10, titlecolor = fg,
                        backgroundcolor = :transparent,
                        xgridvisible = false, ygridvisible = false,
                        xlabelcolor = fg, xticklabelcolor = fg, yticklabelcolor = fg,
                        leftspinecolor = fg, bottomspinecolor = fg,
                        topspinevisible = false, rightspinevisible = false,
                        xtickcolor = fg, ytickcolor = fg)
        # Strip edges: in days, a pass spans the interval between its neighbours' midpoints
        # rather than one unit, so the cells are placed at their actual times.
        xs = xaxis === :days ? c.t_days : collect(1.0:c.n)
        for a in 1:length(ACTS)
            # A transparent-to-solid ramp in the action's own colour. `cgrad` on
            # (colour, alpha) tuples reads them as colourscheme stops and throws, so the
            # stops are built as RGBA directly.
            Makie.heatmap!(ax, xs, [a - 0.5, a + 0.5], reshape(c.M[a, :], :, 1);
                           colormap = Makie.cgrad([
                               Makie.RGBAf(Makie.red(ACOL[a]), Makie.green(ACOL[a]),
                                           Makie.blue(ACOL[a]), 0.0f0),
                               Makie.RGBAf(Makie.red(ACOL[a]), Makie.green(ACOL[a]),
                                           Makie.blue(ACOL[a]), 1.0f0)]),
                           colorrange = (0, 1))
        end
        Makie.hlines!(ax, [1.5, 2.5, 3.5]; color = (fg, 0.25), linewidth = 0.6)
        # REVERSED: ACTS runs top-to-bottom by altitude but Makie counts y upward, so
        # without this EXCURSE_HIGH would land at the bottom and the altitude ordering
        # would read upside down.
        Makie.ylims!(ax, 4.5, 0.5)
        if xaxis === :days
            Makie.xlims!(ax, 0.0, maximum(xs))
        else
            Makie.xlims!(ax, 0.5, c.n + 0.5)
        end
    end

    mkpath(dirname(path))
    for ext in ("pdf", "svg", "png")
        Makie.save("$path.$ext", fig; backgroundcolor = :transparent)
    end
    return fig
end

"""
    band_orbits(config; eom! = cr3bp_j2_eom!, n_rev = 1, family_table = nothing)

Propagate one revolution of the nominal orbit and of each science band's halo member.

  - `config` — the scenario, supplying `band_target_km` and `correct_bin`
  - `eom!` — dynamics to propagate under; the truth model by default
  - `n_rev` — revolutions to trace
  - `family_table` — prebuilt family; `nothing` continues one, which costs ~80 s cold

Returns a Vector of `(label, alt_km, xyz)` in Enceladus-centred km, nominal orbit first.

Each band is a genuine member of the continued L1 halo family at that periapsis altitude,
not the nominal orbit scaled — a radially scaled apse vector is not a solution of the
dynamics and would draw an orbit the spacecraft could not fly.
"""
function band_orbits(config::StationkeepingPOMDP = StationkeepingPOMDP();
                     eom! = cr3bp_j2_eom!, n_rev::Real = 1,
                     family_table = nothing)
    table = family_table === nothing ? halo_family_table_cached() : family_table
    nominal_alt = config.alt_rep_km[config.correct_bin]
    want = [("Nominal", nominal_alt)]
    for b in config.band_names
        push!(want, (string(b), config.band_target_km[b]))
    end

    out = NamedTuple[]
    for (lbl, alt) in want
        m = retarget_to_altitude(table, alt)
        m === nothing && continue
        sol = propagate(eom!, m.ic, (0.0, n_rev * PERIOD1_TRIPLE_PERIOD_S);
                        saveat = range(0, n_rev * PERIOD1_TRIPLE_PERIOD_S; length = 600))
        # Enceladus-centred, which is the frame the altitudes are quoted in.
        xyz = reduce(hcat, [_enc_relative(u[1:3]) for u in sol.u])
        push!(out, (label = lbl, alt_km = alt, xyz = xyz))
    end
    return out
end

"""
    plot_orbit_geometry(config; path, theme, texture, elevation, azimuth, size)

The halo orbits the mission chooses between, around Enceladus with its south-polar plume.

  - `config` — the scenario
  - `path` — output path WITHOUT extension; writes `.pdf`, `.svg` and `.png`
  - `theme` — `:light` or `:dark`; saved transparent so either background works
  - `texture` — image to wrap the moon in; `nothing` draws a plain sphere
  - `elevation`, `azimuth` — camera angles (radians). Negative elevation looks up at the
    south pole, which is where the plume and every periapsis are
  - `inset` — add a periapsis-region panel. The bands differ by ~23 km on a ~1200 km orbit,
    so at a scale that shows the orbit they overlap and the inset is what makes the
    altitude separation visible

Returns the Makie `Figure`. Requires CairoMakie to be loaded by the caller.

Shows why the decision problem exists: the science bands are separate orbits at different
periapsis altitudes over the south pole, the plume is densest low down, and the nominal
stationkeeping orbit sits outside every band, so sampling requires leaving it.
"""
function plot_orbit_geometry(config::StationkeepingPOMDP = StationkeepingPOMDP();
                             path::AbstractString = "figures/orbit_geometry",
                             theme::Symbol = :light,
                             texture = nothing,
                             elevation::Real = 0.12,
                             azimuth::Real = 1.15,
                             inset::Bool = true,
                             orbits = nothing,
                             size::Tuple{Int,Int} = (560, 480))
    Makie = get(Base.loaded_modules,
                Base.PkgId(Base.UUID("13f3f980-e62b-5c42-98c6-ff1f3baf88f0"), "CairoMakie"),
                nothing)
    Makie === nothing && error(
        "plot_orbit_geometry needs CairoMakie loaded by the caller: `using CairoMakie` " *
        "before calling. The library declares no plotting dependency.")

    orb = orbits === nothing ? band_orbits(config) : orbits
    fg = theme === :dark ? Makie.RGBf(0.92, 0.92, 0.92) : Makie.RGBf(0.10, 0.10, 0.10)
    # Nominal in grey, bands warm-to-cool by depth so the lowest reads as the "deepest".
    COL = Dict("Nominal" => Makie.RGBf(0.45, 0.45, 0.45),
               "HIGH"    => Makie.RGBf(0.93, 0.47, 0.20),
               "MID"     => Makie.RGBf(0.13, 0.53, 0.20),
               "LOW"     => Makie.RGBf(0.27, 0.47, 0.67))

    fig = Makie.Figure(; size = size, backgroundcolor = :transparent,
                       figure_padding = (2, 4, 2, 2),
                       fonts = (; regular = "CMU Serif", bold = "CMU Serif Bold"))
    ax = Makie.Axis3(fig[1, 1]; aspect = :data,
                     xlabel = "x (km)", ylabel = "y (km)", zlabel = "z (km)",
                     elevation = elevation, azimuth = azimuth,
                     backgroundcolor = :transparent,
                     xlabelcolor = fg, ylabelcolor = fg, zlabelcolor = fg,
                     xticklabelcolor = fg, yticklabelcolor = fg, zticklabelcolor = fg,
                     xgridcolor = (fg, 0.12), ygridcolor = (fg, 0.12),
                     zgridcolor = (fg, 0.12),
                     xspinecolor_1 = (fg, 0.3), yspinecolor_1 = (fg, 0.3),
                     zspinecolor_1 = (fg, 0.3))

    # Enceladus. A textured sphere when an image is supplied; the UV sphere is built
    # explicitly because the default `Sphere` primitive carries no texture coordinates.
    nu, nv = 72, 36
    θs = range(0, 2pi; length = nu)
    φs = range(0, pi; length = nv)
    X = [R_ENCELADUS * cos(θ) * sin(φ) for θ in θs, φ in φs]
    Y = [R_ENCELADUS * sin(θ) * sin(φ) for θ in θs, φ in φs]
    Z = [R_ENCELADUS * cos(φ) for θ in θs, φ in φs]
    if texture === nothing
        Makie.surface!(ax, X, Y, Z; color = fill(Makie.RGBAf(0.72, 0.74, 0.78, 0.92),
                                                 nu, nv), shading = Makie.NoShading)
    else
        Makie.surface!(ax, X, Y, Z; color = texture, shading = Makie.NoShading,
                       transparency = true, alpha = 0.95)
    end

    # South-polar plume, as a translucent cone. Illustrative geometry, not a measured
    # density profile — Cassini transits do not resolve yield against altitude here.
    pu, pv = 40, 24
    ph = range(0, 120.0; length = pv)
    pθ = range(0, 2pi; length = pu)
    PX = [0.45 * h * cos(t) for t in pθ, h in ph]
    PY = [0.45 * h * sin(t) for t in pθ, h in ph]
    PZ = [-R_ENCELADUS - h for t in pθ, h in ph]
    Makie.surface!(ax, PX, PY, PZ;
                   color = [Makie.RGBAf(0.55, 0.75, 0.95, 0.30 * (1 - h / 130))
                            for t in pθ, h in ph],
                   shading = Makie.NoShading, transparency = true)

    for o in orb
        c = get(COL, o.label, fg)
        Makie.lines!(ax, o.xyz[1, :], o.xyz[2, :], o.xyz[3, :];
                     color = c, linewidth = o.label == "Nominal" ? 2.4 : 1.8,
                     linestyle = o.label == "Nominal" ? :dash : :solid,
                     label = "$(o.label) ($(round(o.alt_km; digits = 1)) km)")
    end

    # Periapsis zoom. The whole point of the science bands is a ~23 km spread on a
    # ~1200 km orbit, which is invisible at the scale that shows the orbit.
    if inset
        ax2 = Makie.Axis(fig[1, 2];
                         xlabel = "Cross-track (km)", ylabel = "Altitude (km)",
                         title = "Periapsis, south pole", titlesize = 10,
                         backgroundcolor = :transparent,
                         titlecolor = fg, xlabelcolor = fg, ylabelcolor = fg,
                         xticklabelcolor = fg, yticklabelcolor = fg,
                         leftspinecolor = fg, bottomspinecolor = fg,
                         topspinevisible = false, rightspinevisible = false,
                         xtickcolor = fg, ytickcolor = fg, xgridvisible = false)
        # Surface, then each band's periapsis altitude as a level.
        Makie.hlines!(ax2, [0.0]; color = (fg, 0.7), linewidth = 1.5)
        Makie.band!(ax2, [-60.0, 60.0], [-14.0, -14.0], [0.0, 0.0];
                    color = (Makie.RGBf(0.72, 0.74, 0.78), 0.9))
        Makie.text!(ax2, 0.0, -8.0; text = "Enceladus", color = fg, fontsize = 8,
                    align = (:center, :center))
        for o in orb
            c = get(COL, o.label, fg)
            Makie.hlines!(ax2, [o.alt_km]; color = c, linewidth = 2,
                          linestyle = o.label == "Nominal" ? :dash : :solid)
            Makie.text!(ax2, -56.0, o.alt_km + 1.2; text = o.label, color = c,
                        fontsize = 9, align = (:left, :bottom))
        end
        # Plume density falling off with altitude, as a shaded wedge.
        Makie.band!(ax2, [-60.0, 60.0], [0.0, 0.0], [55.0, 55.0];
                    color = (Makie.RGBf(0.55, 0.75, 0.95), 0.10))
        Makie.xlims!(ax2, -60, 60); Makie.ylims!(ax2, -16, 56)
        Makie.colsize!(fig.layout, 2, Makie.Relative(0.34))
    end

    Makie.Legend(fig[2, 1:(inset ? 2 : 1)], ax; framevisible = false, labelcolor = fg,
                 labelsize = 11, patchsize = (18.0f0, 8.0f0),
                 orientation = :horizontal, nbanks = 2,
                 tellheight = true, tellwidth = false)

    mkpath(dirname(path))
    for ext in ("pdf", "svg", "png")
        Makie.save("$path.$ext", fig; backgroundcolor = :transparent)
    end
    return fig
end
