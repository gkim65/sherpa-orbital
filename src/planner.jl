"""
Onboard burn planner — the apse-targeting solver shared by every stationkeeping baseline.

This is the ONBOARD half of MacKenzie et al. 2020 §B.2.3 "Strategy 3". Given a state at a
control point, it solves for the impulsive ΔV that re-targets the next periapsis and
apoapsis, predicting by propagating the onboard model to the next apse pair.

⚠️ TRUTH/ONBOARD SPLIT (CLAUDE.md rule — do NOT collapse). Everything in this file plans
with the ONBOARD model, [`cr3bp_eom!`](@ref) (CR3BP only, no perturbations), at the loose
onboard tolerances. No function here may reach for a truth EOM. The gap between what the
planner believes and what the world does is the model uncertainty the POMDP absorbs; a
truth EOM appearing in this file would silently erase the project's central quantity.

The `eom` keyword exists ONLY so an ablation study can deliberately hand the planner a
perfect model and measure what that is worth. Measured: a perfect onboard model does not
extend survival — the failure is instability-dominated. It defaults to the onboard model.

Burn solver
-----------
  - Decision variable: full 3-D impulsive ΔV (km/s) added to the velocity at the control
    point.
  - Targets: either the two apse ALTITUDES (`mode = :altitude`) or the two apse POSITION
    VECTORS against a nominal orbit (`mode = :position`, Strategy 3 proper).
  - 3 controls vs 2 (or 6) residuals → minimum-norm Gauss-Newton via the pseudo-inverse,
    which picks the smallest-ΔV correction each iteration and keeps the under- or
    over-determined system well posed.

References
  MacKenzie, S. M. et al. (2020). Enceladus Orbilander: A Flagship Mission Concept for
    Astrobiology. §B.2.3 (stationkeeping Strategy 3), §3.5 (propulsion).
"""

# ── Control configuration ─────────────────────────────────────────────────────
# Controller/feasibility parameters, not physical constants — those live in constants.jl.

const CONTROL_ALT_KM = 600.0   # Strategy 3 trigger shell, km above the Enceladus surface

const PERIAPSIS_ALT_TARGET = 0.5 * (PERIAPSIS_ALT_MIN + PERIAPSIS_ALT_MAX)  # 42.05 km
const APOAPSIS_ALT_TARGET  = 0.5 * (APOAPSIS_ALT_MIN  + APOAPSIS_ALT_MAX)   # 1055 km

const TARGET_TOL_KM = 1.0      # apse-targeting tolerance (MacKenzie ≤ 1 km)

# Apse-search horizon, as a multiple of the single-revolution period estimate. Beyond this
# the prediction returns NaN and solve_burn stops.
#
# ⚠️ NOT AN INERT PARAMETER, and not a physically-derived one. This is the surviving numeric
# content of the deleted `n_revs = 3`: on-orbit the value provably does not matter (1/2/3/5
# all predict periapsis 40.407873 km, identical to six decimals), which is why `n_revs` was
# removed. But NEAR ESCAPE it matters a great deal, and that regime was never measured until
# 2026-08-29. An escaping spacecraft stays bound to SATURN, so its apses still exist tens of
# thousands of km out; an unbounded search finds them and solve_burn chases them. Measured on
# the run_mpc :altitude parity row (CR3BP + Enceladus J2, 120 hr): unbounded turns the final
# burn from 21.9 m/s into ~182 m/s and moves escape from 77.96 hr to 79.52 hr.
#
# Pinned at 3 to hold every existing baseline bit-for-bit (user decision, 2026-08-29). The
# value itself is UNJUSTIFIED — it was inherited from a parameter chosen to match MacKenzie's
# Nm = 3, which is a maneuver count, not a horizon. Treat it as an open controller parameter
# needing a measured choice, not as a settled constant. See docs/todo.md.
const APSE_SEARCH_REVS = 3

# Escape threshold: above this altitude the spacecraft has left the ~1065-km-apoapsis
# science orbit and the controller can no longer recover it (the 600-km descending control
# shell never re-arms). ~5× the nominal apoapsis altitude.
const ESCAPE_ALT_KM = 5.0 * APOAPSIS_ALT_MAX   # 5550 km

"""
    escape_callback(altitude_km = ESCAPE_ALT_KM) -> ContinuousCallback

Terminal event fired when the spacecraft ASCENDS through `altitude_km` above the Enceladus
surface — it has left the science orbit and is escaping.

Direction `+1` (ascending) so it does not fire on the normal inbound leg of the apoapsis
arc. This complements [`crash_callback`](@ref), the other terminal outcome.
"""
function escape_callback(altitude_km::Real = ESCAPE_ALT_KM)
    target_r = R_ENCELADUS + altitude_km
    g(u, t, integrator) = r_enceladus(u) - target_r
    return _directed_callback(g, +1, terminate!)
end

# ── Apse prediction (onboard model) ───────────────────────────────────────────

"""
    predict_apse_states(state0, period_s; eom!, rtol, atol) -> (peri_state, apo_state)

Predict the NEXT periapsis and NEXT apoapsis 6-states reached under the ONBOARD model.
Barycentre frame, km / km/s.

Using the FIRST apse of each type — rather than the min/max over the span — keeps the
residual a smooth function of ΔV, which the Gauss-Newton Jacobian needs.

The search asks [`next_apses`](@ref) for the apse pair BY COUNT rather than taking whatever
apses fall inside a window, so the two apse-prediction paths in this file now search the same
way and a missing apse is a named failure rather than a silent empty result.

⚠️ THE SEARCH IS DELIBERATELY BOUNDED at `max_horizon_s` — see [`APSE_SEARCH_REVS`](@ref) for
why, and why the bound is an open parameter rather than a settled one. Beyond the bound this
returns `NaN`, the sentinel `solve_burn` stops on, which is what keeps a near-escape burn
finite.
"""
function predict_apse_states(
    state0::AbstractVector{<:Real},
    period_s::Real;
    eom! = cr3bp_eom!,
    rtol::Real = RTOL_ONBOARD,
    atol::Real = ATOL_ONBOARD,
    max_horizon_s::Real = APSE_SEARCH_REVS * float(period_s),
)
    nan6 = fill(NaN, 6)
    peri, apo = try
        next_apses(state0, 1, 1; eom! = eom!, rtol = rtol, atol = atol,
                   t_guess = float(max_horizon_s), max_expansions = 0)
    catch
        return nan6, nan6
    end
    return peri[1][2], apo[1][2]
end

"""
    predict_apses(state0, period_s; eom!, rtol, atol) -> (peri_alt_km, apo_alt_km)

Predict the ALTITUDE (km) of the next periapsis and next apoapsis under the onboard model.
`NaN` for an apse the trajectory never reaches.

The altitude-only counterpart of [`predict_apse_states`](@ref), used by `mode = :altitude`.
"""
function predict_apses(
    state0::AbstractVector{<:Real},
    period_s::Real;
    eom! = cr3bp_eom!,
    rtol::Real = RTOL_ONBOARD,
    atol::Real = ATOL_ONBOARD,
)
    peri_state, apo_state = predict_apse_states(state0, period_s;
                                                eom! = eom!, rtol = rtol, atol = atol)
    peri_alt = isnan(peri_state[1]) ? NaN : altitude(peri_state)
    apo_alt  = isnan(apo_state[1])  ? NaN : altitude(apo_state)
    return peri_alt, apo_alt
end


"""
    next_apses(state0, n_peri, n_apo; eom!, rtol, atol, t_guess, max_expansions)
        -> (peri, apo)

Collect the next `n_peri` periapsis and `n_apo` apoapsis passages ahead of `state0` under
the onboard model, as two vectors of `(t, state)` pairs. Barycentre frame, km / km/s.

Asks for a COUNT, not a time window: it propagates in expanding chunks until it has the
apses requested, and throws if they cannot be found within the safety horizon. There is no
empty-return path to mistake for a valid answer.

A window-based search (ask for a time span, take whatever apses fall inside) is the trap this
replaces: on this orbit the period-3 halo has 3 periapses but only 2 apoapses, so the `T/3`
control interval falls 0.136 s short of the first apoapsis and a window search comes back
empty.

  - `t_guess` — first chunk length (s); a hint, not a bound. Later chunks double.
  - `max_expansions` — doubling limit before erroring. Horizon searched is
    `t_guess × (2^max_expansions − 1)`.

Escape is the expected failure: a departing spacecraft has no next apoapsis, and the error
names the horizon searched.
"""
function next_apses(
    state0::AbstractVector{<:Real},
    n_peri::Integer = 1,
    n_apo::Integer = 1;
    eom! = cr3bp_eom!,
    rtol::Real = RTOL_ONBOARD,
    atol::Real = ATOL_ONBOARD,
    t_guess::Real = PERIOD1_TRIPLE_PERIOD_S,
    max_expansions::Integer = 6,
)
    (n_peri >= 0 && n_apo >= 0) ||
        throw(ArgumentError("apse counts must be non-negative, got n_peri=$n_peri, n_apo=$n_apo"))
    t_guess > 0 || throw(ArgumentError("t_guess must be positive, got $t_guess"))

    # Re-propagating from `state0` over a growing horizon (rather than resuming) keeps the
    # apse times referenced to t = 0 and the integration identical to a single long call,
    # so a caller's answer never depends on where the chunk boundaries happened to land.
    horizon = float(t_guess)
    for _ in 0:max_expansions
        peri, apo = collect_apses(eom!, state0, horizon; rtol = rtol, atol = atol)
        if length(peri) >= n_peri && length(apo) >= n_apo
            return peri[1:n_peri], apo[1:n_apo]
        end
        horizon *= 2
    end

    searched = float(t_guess) * (2^(max_expansions + 1) - 1)
    error("next_apses: could not find $n_peri periapsis / $n_apo apoapsis passages " *
          "within $(searched) s (~$(round(searched / 3600, digits = 2)) hr) of the initial " *
          "state. The trajectory most likely escaped or crashed. " *
          "state0 = $(collect(float.(state0)))")
end

"""
    next_apse_positions(state0, n_peri, n_apo; kwargs...) -> (r_peri, r_apo)

POSITION vectors (3-vectors, km, barycentre frame) of the next periapsis and next apoapsis
ahead of `state0`, via [`next_apses`](@ref).

Count-based: it takes no time
window, so it cannot return a `NaN` target from a too-short horizon. Both returned vectors
are guaranteed finite — `next_apses` throws rather than reporting a missing apse.
"""
function next_apse_positions(
    state0::AbstractVector{<:Real};
    eom! = cr3bp_eom!,
    rtol::Real = RTOL_ONBOARD,
    atol::Real = ATOL_ONBOARD,
    t_guess::Real = PERIOD1_TRIPLE_PERIOD_S,
    max_expansions::Integer = 6,
)
    peri, apo = next_apses(state0, 1, 1; eom! = eom!, rtol = rtol, atol = atol,
                           t_guess = t_guess, max_expansions = max_expansions)
    return copy(peri[1][2][1:3]), copy(apo[1][2][1:3])
end

# ── Target validation ─────────────────────────────────────────────────────────

"""
    validate_apse_targets(state0, r_peri_nom, r_apo_nom; max_alt_km, phantom_tol_km)

Throw if the `mode = :position` nominal apse targets are not a usable pair of targets.
Returns `nothing` on success. Called by [`solve_burn`](@ref) BEFORE any solving.

⚠️ WHY A BAD TARGET MUST THROW RATHER THAN BE SOLVED. A `:position` solve against a bad
target does not fail visibly — it returns ΔV = 0, which is indistinguishable from a
deliberate decision not to burn, so a CORRECT action silently degrades into an OBSERVE. A
whole calibration run can come back with every burn quietly disabled and still look like
data. Bad targets are a SETUP error, so they are raised where they are introduced.

Three rejections, one per way a target has actually gone wrong in this project:

 1. **Non-finite** — a window-based apse search returns `NaN` when
    its window is too short to contain an apse (the `T/3` defect). `NaN` poisons its half of
    the residual and `solve_burn` cannot converge.

 2. **Equal to `state0`'s own position** (within `phantom_tol_km`) — the signature of the
    Python reference's phantom `t = 0` root. scipy reports an apse at `t = 0` for a state
    that starts at an apse, so the "target" is the current position: the residual compares a
    phantom target against a phantom prediction, scores exactly 0.0, and reports
    `converged = true` having planned nothing. A `converged` flag alone does NOT catch this,
    which is why this check exists as well as the flag.

 3. **Implausibly high altitude** (above `max_alt_km`) — a count-based apse search
    ([`next_apses`](@ref)) always finds *an* apse, because a spacecraft that has left the
    science orbit is still bound to Saturn and its Enceladus-relative radius keeps
    oscillating. Those apses are real apses of the trajectory but not of the orbit we are
    holding: a 1 km/s kick yields an "apoapsis" at ~420,000 km altitude, ~7500× nominal.
    Guaranteeing a finite target is not the same as guaranteeing a meaningful one, so the
    count-based fix needs this bound to avoid trading a loud `NaN` for a quiet absurdity.

`phantom_tol_km` defaults to `EVENT_TOL`-scale rather than 0: a correctly-found apse very
near the current state (the period-3 apoapses are near-degenerate in radius) sits ~1e-06 km
away, whereas a true phantom is at *exactly* 0.0. The default separates those by orders of
magnitude while staying far below `TARGET_TOL_KM`.
"""
function validate_apse_targets(
    state0::AbstractVector{<:Real},
    r_peri_nom::Union{AbstractVector{<:Real},Nothing},
    r_apo_nom::Union{AbstractVector{<:Real},Nothing};
    max_alt_km::Real = ESCAPE_ALT_KM,
    phantom_tol_km::Real = 1.0e-9,
)
    (r_peri_nom === nothing || r_apo_nom === nothing) &&
        throw(ArgumentError("mode = :position requires r_peri_nom and r_apo_nom"))

    r0 = collect(float.(state0[1:3]))
    for (name, r) in (("r_peri_nom", r_peri_nom), ("r_apo_nom", r_apo_nom))
        length(r) >= 3 || throw(ArgumentError("$name must have at least 3 components"))
        rr = collect(float.(r[1:3]))

        all(isfinite, rr) || error(
            "validate_apse_targets: $name = $rr is not finite. A window-based apse search " *
            "found no apse inside its horizon — use next_apse_positions, which takes an " *
            "apse COUNT and cannot come back empty.")

        d0 = norm(rr .- r0)
        d0 > phantom_tol_km || error(
            "validate_apse_targets: $name is $(d0) km from state0's own position " *
            "(tol $(phantom_tol_km) km) — this is the phantom t = 0 apse signature, not a " *
            "target. Solving against it yields a circular zero residual and a false " *
            "converged = true.")

        alt = norm(_planner_enc_relative(rr)) - R_ENCELADUS
        alt <= max_alt_km || error(
            "validate_apse_targets: $name is at altitude $(alt) km, above the " *
            "max_alt_km = $(max_alt_km) km bound. This apse belongs to a trajectory that " *
            "has left the science orbit, not to the orbit being held.")
    end
    return nothing
end

"""Enceladus-relative position from a barycentre-frame position vector (km)."""
function _planner_enc_relative(r::AbstractVector{<:Real})
    out = collect(float.(r[1:3]))
    out[1] -= X_ENCELADUS
    return out
end

# ── Residual ──────────────────────────────────────────────────────────────────

"""
    apse_residual(state0, dv, period_s, eom!; ...) -> Vector{Float64}

The residual `r(ΔV)` driven to zero by [`solve_burn`](@ref). `dv` (km/s) is added to the
velocity of `state0` and the apses are predicted with the ONBOARD model.

Two targeting modes:

  - `mode = :altitude` (default; MacKenzie-like Strategy 1/2) — the 2-vector
    `[peri_alt − peri_target_km, apo_alt − apo_target_km]` (km). The targets default to
    the centre of the MacKenzie period-3 bands; pass an orbit's own apse altitudes to hold
    that orbit instead.
  - `mode = :position` (Strategy 3 proper) — the 6-vector
    `[r_peri − r_peri_nom, r_apo − r_apo_nom]` (km), bounding the full apse position
    vectors against the nominal orbit. `r_peri_nom` and `r_apo_nom` are required
"""
function apse_residual(
    state0::AbstractVector{<:Real},
    dv::AbstractVector{<:Real},
    period_s::Real,
    eom!;
    peri_target_km::Real = PERIAPSIS_ALT_TARGET,
    apo_target_km::Real = APOAPSIS_ALT_TARGET,
    mode::Symbol = :altitude,
    r_peri_nom::Union{AbstractVector{<:Real},Nothing} = nothing,
    r_apo_nom::Union{AbstractVector{<:Real},Nothing} = nothing,
)
    s = collect(float.(state0))
    s[4:6] .+= dv

    if mode === :position
        (r_peri_nom === nothing || r_apo_nom === nothing) &&
            throw(ArgumentError("mode = :position requires r_peri_nom and r_apo_nom"))
        peri_state, apo_state = predict_apse_states(s, period_s; eom! = eom!)
        return vcat(peri_state[1:3] .- r_peri_nom, apo_state[1:3] .- r_apo_nom)
    end

    peri_alt, apo_alt = predict_apses(s, period_s; eom! = eom!)
    return [peri_alt - peri_target_km, apo_alt - apo_target_km]
end

# ── Burn solver ───────────────────────────────────────────────────────────────

"""
    solve_burn(state0, period_s; eom!, max_iter, fd_step, damp, tol_km,
               peri_target_km, apo_target_km, mode, r_peri_nom, r_apo_nom) -> NamedTuple

Solve for the impulsive ΔV that re-targets the next periapsis/apoapsis.

Minimum-norm Gauss-Newton on a 3-control problem (ΔV ∈ ℝ³). The residual is 2-D in
`mode = :altitude` or 6-D in `mode = :position` (see [`apse_residual`](@ref)). The Jacobian
`J = ∂r/∂ΔV` (m×3) is built by forward finite differences with step `fd_step`, and the
pseudo-inverse step `ΔV ← ΔV − damp · J⁺ r` selects the minimum-norm correction — so the
solver prefers cheap burns and stays well posed whichever way the system is determined.

Iteration stops on `‖r‖ < tol_km`, on a non-finite residual (the prediction lost an apse
over the horizon), or at `max_iter`.

⚠️ Planning uses the ONBOARD model. `eom!` defaults to [`cr3bp_eom!`](@ref) and exists as a
keyword only for the deliberate perfect-model ablation; the caller is responsible for never
passing a truth EOM in normal operation.

  - `state0` — barycentre-frame state at the control point, PRE-burn (km, km/s)
  - `period_s` — single-revolution period estimate (s), the first chunk of the count-based
    apse search. The planner targets the NEXT apse pair; it has no multi-revolution horizon.

Returns `(dv, dv_mag_ms, converged, residual_km, iterations)`, where `dv` is the ΔV in km/s
and `dv_mag_ms` its magnitude in m/s.
"""
function solve_burn(
    state0::AbstractVector{<:Real},
    period_s::Real;
    eom! = cr3bp_eom!,
    max_iter::Integer = 20,
    fd_step::Real = 1e-6,
    damp::Real = 0.8,
    tol_km::Real = TARGET_TOL_KM,
    peri_target_km::Real = PERIAPSIS_ALT_TARGET,
    apo_target_km::Real = APOAPSIS_ALT_TARGET,
    mode::Symbol = :altitude,
    r_peri_nom::Union{AbstractVector{<:Real},Nothing} = nothing,
    r_apo_nom::Union{AbstractVector{<:Real},Nothing} = nothing,
    validate_targets::Bool = true,
)
    # Bad :position targets are a SETUP error, not a burn to attempt — see
    # validate_apse_targets. Checked before any propagation so the failure surfaces where it
    # was introduced rather than as a silent ΔV = 0.
    mode === :position && validate_targets &&
        validate_apse_targets(state0, r_peri_nom, r_apo_nom)

    resid(dv) = apse_residual(state0, dv, period_s, eom!;
                              peri_target_km = peri_target_km,
                              apo_target_km = apo_target_km,
                              mode = mode,
                              r_peri_nom = r_peri_nom, r_apo_nom = r_apo_nom)

    dv = zeros(3)
    r = resid(dv)

    iters = 0
    for i in 1:max_iter
        iters = i
        all(isfinite, r) || break
        norm(r) < tol_km && break

        # Forward-difference Jacobian J (m×3): columns = ∂r/∂ΔV_i.
        J = zeros(length(r), 3)
        ok = true
        for j in 1:3
            dv_p = copy(dv)
            dv_p[j] += fd_step
            r_p = resid(dv_p)
            if !all(isfinite, r_p)
                ok = false
                break
            end
            J[:, j] .= (r_p .- r) ./ fd_step
        end
        ok || break

        # Minimum-norm Gauss-Newton step: ΔV -= damp · J⁺ r.
        dv = dv .- damp .* (pinv(J) * r)
        r = resid(dv)
    end

    residual = all(isfinite, r) ? norm(r) : Inf
    return (
        dv          = dv,
        dv_mag_ms   = norm(dv) * 1.0e3,   # km/s → m/s
        converged   = residual < tol_km,
        residual_km = residual,
        iterations  = iters,
    )
end