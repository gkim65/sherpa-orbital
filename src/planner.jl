"""
Onboard burn planner — the apse-targeting solver shared by every stationkeeping baseline.

This is the ONBOARD half of MacKenzie et al. 2020 §B.2.3 "Strategy 3". Given a state at a
control point, it solves for the impulsive ΔV that re-targets the next periapsis and
apoapsis, predicting by propagating the onboard model to the next apse pair.

NOTE: truth/onboard split — do not collapse it. Everything here plans with the onboard
model, [`cr3bp_eom!`](@ref) (CR3BP only, no perturbations), at the loose onboard
tolerances. The gap between what the planner believes and what the world does is the model
uncertainty the POMDP absorbs, so a truth EOM appearing in this file would silently erase
the project's central quantity.

The `eom!` keyword exists only so an ablation can hand the planner a perfect model and
measure what that is worth; it defaults to the onboard model.

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
# NOTE: not an inert parameter. On-orbit the value does not matter — 1, 2, 3 and 5 all
# predict the same periapsis to six decimals — but NEAR ESCAPE it matters a great deal. An
# escaping spacecraft stays bound to SATURN, so its apses still exist tens of thousands of
# km out; an unbounded search finds them and `solve_burn` chases them, turning a ~22 m/s
# final burn into ~180 m/s.
#
# NOTE: the value 3 is unjustified. It was inherited from a parameter chosen to match
# MacKenzie's Nm = 3, which is a maneuver count rather than a search horizon, and is pinned
# here only to hold existing baselines bit-for-bit. Treat it as an open controller
# parameter needing a measured choice.
const APSE_SEARCH_REVS = 3

# Escape threshold: above this altitude the spacecraft has left the ~1065-km-apoapsis
# science orbit and the controller can no longer recover it (the 600-km descending control
# shell never re-arms). ~5× the nominal apoapsis altitude.
const ESCAPE_ALT_KM = 5.0 * APOAPSIS_ALT_MAX   # 5550 km

"""
    escape_callback(altitude_km = ESCAPE_ALT_KM) -> ContinuousCallback

Terminal event for escape: the spacecraft ascending through a shell above the surface.

  - `altitude_km` — escape shell altitude (km)

Returns a `ContinuousCallback`. Ascending only, so it does not fire on the normal outbound
leg of the apoapsis arc. Complements [`crash_callback`](@ref), the other terminal outcome.
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

  - `state0` — barycentre-frame state at the control point (km, km/s)
  - `period_s` — single-revolution period estimate (s)
  - `eom!` — onboard model
  - `rtol`, `atol` — integration tolerances
  - `max_horizon_s` — search bound; beyond it both states come back `NaN`

Returns `(peri_state, apo_state)`, both 6-element (km, km/s), or `NaN`-filled if no apse
pair was found.

The search asks [`next_apses`](@ref) for the pair BY COUNT rather than taking whatever
apses fall inside a window, so a missing apse is a named failure rather than a silent empty
result.

NOTE: the search is bounded on purpose — see [`APSE_SEARCH_REVS`](@ref). The `NaN` beyond
the bound is the sentinel `solve_burn` stops on, and is what keeps a near-escape burn
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

Predict the altitude of the next periapsis and next apoapsis under the onboard model.

  - `state0` — barycentre-frame state (km, km/s)
  - `period_s` — single-revolution period estimate (s)
  - `eom!`, `rtol`, `atol` — forwarded to [`predict_apse_states`](@ref)

Returns `(peri_alt_km, apo_alt_km)`, `NaN` for an apse the trajectory never reaches. The
altitude-only counterpart of [`predict_apse_states`](@ref), used by `mode = :altitude`.
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

Collect the next apse passages ahead of `state0` under the onboard model.

  - `state0` — barycentre-frame state (km, km/s)
  - `n_peri`, `n_apo` — how many of each to collect
  - `eom!` — onboard model
  - `rtol`, `atol` — integration tolerances
  - `t_guess` — first chunk length (s); a hint, not a bound, since later chunks double
  - `max_expansions` — doubling limit before erroring; the horizon searched is
    `t_guess * (2^max_expansions - 1)`

Returns `(peri, apo)`, each a vector of `(t, state)` pairs in seconds and (km, km/s).
Throws if the requested apses are not found — escape is the expected failure, since a
departing spacecraft has no next apoapsis, and the error names the horizon searched.

NOTE: asks for a COUNT, not a time window. A window search finds no apoapsis inside the
control interval and silently returns empty.
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
    next_apse_positions(state0; kwargs...) -> (r_peri, r_apo)

Position vectors of the next periapsis and next apoapsis ahead of `state0`, via
[`next_apses`](@ref).

  - `state0` — barycentre-frame state (km, km/s)
  - `eom!`, `rtol`, `atol`, `t_guess`, `max_expansions` — forwarded to [`next_apses`](@ref)

Returns two 3-vectors in km, barycentre frame. Both are guaranteed finite: the search is
count-based, so it cannot return a `NaN` target from a too-short horizon — `next_apses`
throws rather than reporting a missing apse.
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

  - `state0` — barycentre-frame state at the control point (km, km/s)
  - `r_peri_nom`, `r_apo_nom` — nominal apse position targets (km); `r_peri_nom` is unused
    in `:altitude_position`
  - `max_alt_km` — reject a target above this altitude (km)
  - `phantom_tol_km` — how close to `state0` counts as a phantom target (km)
  - `mode` — `:position` or `:altitude_position`
  - `peri_target_km` — commanded periapsis altitude (km), required in `:altitude_position`

Returns `nothing`, or throws naming the offending target.

NOTE: a bad target must throw rather than be solved. A `:position` solve against one does
not fail visibly — it returns ΔV = 0, indistinguishable from a deliberate decision not to
burn, so a whole calibration run can come back with every burn quietly disabled and still
look like data.

Three rejections, one per way a target has gone wrong here:

 1. **Non-finite** — a window-based apse search returns `NaN` when its window is too short
    to contain an apse. `NaN` poisons its half of the residual and the solve cannot
    converge.

 2. **Equal to `state0`'s own position**, within `phantom_tol_km` — a phantom `t = 0` apse.
    The residual then compares a phantom target against a phantom prediction, scores
    exactly 0.0, and reports `converged = true` having planned nothing, so the `converged`
    flag alone does not catch it. The tolerance is `EVENT_TOL`-scale rather than 0 because
    a correctly-found apse very near the current state sits ~1e-06 km away while a true
    phantom is at exactly 0.0.

 3. **Implausibly high altitude** — a count-based search always finds *an* apse, because a
    spacecraft that has left the science orbit is still bound to Saturn. Those are real
    apses of the trajectory but not of the orbit being held: a 1 km/s kick yields an
    "apoapsis" thousands of times the nominal altitude. A finite target is not the same as
    a meaningful one.

In `:altitude_position` the periapsis target is a scalar altitude, so the phantom check
does not apply to it and `r_peri_nom` is not required. It is checked instead for being
finite, below `max_alt_km`, and above the crash shell — a commanded periapsis inside
Enceladus is a setup error the solver would happily converge on.
"""
function validate_apse_targets(
    state0::AbstractVector{<:Real},
    r_peri_nom::Union{AbstractVector{<:Real},Nothing},
    r_apo_nom::Union{AbstractVector{<:Real},Nothing};
    max_alt_km::Real = ESCAPE_ALT_KM,
    phantom_tol_km::Real = 1.0e-9,
    mode::Symbol = :position,
    peri_target_km::Union{Nothing,Real} = nothing,
)
    if mode === :altitude_position
        r_apo_nom === nothing &&
            throw(ArgumentError("mode = :altitude_position requires r_apo_nom"))
        peri_target_km === nothing &&
            throw(ArgumentError("mode = :altitude_position requires peri_target_km"))
        isfinite(peri_target_km) || error(
            "validate_apse_targets: peri_target_km = $(peri_target_km) is not finite.")
        PERIAPSIS_CRASH_ALT < peri_target_km <= max_alt_km || error(
            "validate_apse_targets: peri_target_km = $(peri_target_km) km is outside " *
            "($(PERIAPSIS_CRASH_ALT), $(max_alt_km)] km — a commanded periapsis below the " *
            "crash shell or outside the science orbit is a setup error, not a target.")
        _validate_apse_position(state0, "r_apo_nom", r_apo_nom, max_alt_km, phantom_tol_km)
        return nothing
    end

    (r_peri_nom === nothing || r_apo_nom === nothing) &&
        throw(ArgumentError("mode = :position requires r_peri_nom and r_apo_nom"))

    for (name, r) in (("r_peri_nom", r_peri_nom), ("r_apo_nom", r_apo_nom))
        _validate_apse_position(state0, name, r, max_alt_km, phantom_tol_km)
    end
    return nothing
end

"""
    _validate_apse_position(state0, name, r, max_alt_km, phantom_tol_km) -> Nothing

Check one apse POSITION target against the three rejections in
[`validate_apse_targets`](@ref).

  - `state0` — barycentre-frame state at the control point (km, km/s)
  - `name` — the target's name, for the error message
  - `r` — the apse position target (km)
  - `max_alt_km` — reject above this altitude (km)
  - `phantom_tol_km` — how close to `state0` counts as a phantom (km)

Returns `nothing`, or throws naming the target and which check failed.
"""
function _validate_apse_position(state0::AbstractVector{<:Real}, name::AbstractString,
                                 r::AbstractVector{<:Real}, max_alt_km::Real,
                                 phantom_tol_km::Real)
    r0 = collect(float.(state0[1:3]))
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

The residual `r(ΔV)` driven to zero by [`solve_burn`](@ref).

  - `state0` — barycentre-frame state at the control point, pre-burn (km, km/s)
  - `dv` — candidate ΔV (km/s), added to `state0`'s velocity
  - `period_s` — single-revolution period estimate (s)
  - `eom!` — onboard model used to predict the apses
  - `peri_target_km`, `apo_target_km` — commanded apse altitudes (km)
  - `mode` — which residual to build, below
  - `r_peri_nom`, `r_apo_nom` — nominal apse position targets (km)

Returns the residual vector in km. Three modes:

  - `:altitude` — the 2-vector `[peri_alt - peri_target_km, apo_alt - apo_target_km]`.
    Targets default to the centre of the MacKenzie bands; pass an orbit's own apse
    altitudes to hold that orbit instead.
  - `:position` — the 6-vector `[r_peri - r_peri_nom, r_apo - r_apo_nom]`, bounding the
    full apse position vectors against a nominal orbit. Both nominals required.
  - `:altitude_position` — the 4-vector `[peri_alt - peri_target_km, r_apo - r_apo_nom]`:
    periapsis by ALTITUDE, the quantity actually commanded, and apoapsis by full POSITION,
    which is what pins the orbit's orientation. `r_apo_nom` required, `r_peri_nom` unused.

NOTE: do not invert `:altitude_position` into constraining apoapsis altitude and periapsis
position. That was tried: apoapsis position drifts and the vehicle escapes within days.
Freeing the periapsis DIRECTION is what absorbs the unsatisfiable part of the `:position`
residual, as a radial bias, while the three apoapsis-position constraints keep the
orientation pinned. 4 residuals on 3 controls is still over-determined, so that pinning is
not given up wholesale.
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

    if mode === :altitude_position
        r_apo_nom === nothing &&
            throw(ArgumentError("mode = :altitude_position requires r_apo_nom"))
        peri_state, apo_state = predict_apse_states(s, period_s; eom! = eom!)
        # `altitude` on a NaN state would return NaN anyway, but going through the same
        # isnan guard as `predict_apses` keeps the sentinel explicit for solve_burn.
        peri_alt = isnan(peri_state[1]) ? NaN : altitude(peri_state)
        return vcat(peri_alt - peri_target_km, apo_state[1:3] .- r_apo_nom)
    end

    peri_alt, apo_alt = predict_apses(s, period_s; eom! = eom!)
    return [peri_alt - peri_target_km, apo_alt - apo_target_km]
end

# ── Burn solver ───────────────────────────────────────────────────────────────

"""
    _residual_weights(mode, J) -> Diagonal

Row weights `W` for the Gauss-Newton step, so `solve_burn` minimizes `‖W r‖` rather than
`‖r‖`.

  - `mode` — targeting mode; only `:altitude_position` is weighted
  - `J` — the current Jacobian, `∂r/∂ΔV`

Returns a `Diagonal`; identity for every mode except `:altitude_position`, so the other
baselines stay bit-for-bit reproducible.

`:altitude_position` needs it because its 4 residuals are one periapsis ALTITUDE and three
apoapsis POSITION components: same units, but their sensitivity to ΔV differs by more than
an order of magnitude. Least squares weights by the square of that, so an unweighted solve
spends all 3 controls nulling apoapsis position and abandons the commanded altitude —
delivering a fraction of the commanded change. Equilibrating by the block Jacobian norms
makes the two physical requirements carry equal weight instead of weighting them by how
strongly each responds to a velocity kick. Standard row equilibration, no tuned constant.

Recomputed each iteration from the current `J`, so it tracks the operating point rather
than freezing a scaling measured at the first step.
"""
function _residual_weights(mode::Symbol, J::AbstractMatrix)
    mode === :altitude_position || return Diagonal(ones(size(J, 1)))
    n_peri = norm(@view J[1, :])
    n_apo  = norm(@view J[2:end, :])
    w = (n_peri > 0 && n_apo > 0) ? n_apo / n_peri : 1.0
    return Diagonal([w; ones(size(J, 1) - 1)])
end

"""
    solve_burn(state0, period_s; eom!, max_iter, fd_step, damp, tol_km,
               peri_target_km, apo_target_km, mode, r_peri_nom, r_apo_nom) -> NamedTuple

Solve for the impulsive ΔV that re-targets the next periapsis/apoapsis.

Minimum-norm Gauss-Newton on a 3-control problem (ΔV ∈ ℝ³).

  - `state0` — barycentre-frame state at the control point, PRE-burn (km, km/s)
  - `period_s` — single-revolution period estimate (s), the first chunk of the count-based
    apse search. The planner targets the NEXT apse pair; it has no multi-revolution horizon
  - `eom!` — onboard model
  - `max_iter` — iteration cap
  - `fd_step` — forward-difference step for the Jacobian (km/s)
  - `damp` — step scaling in `ΔV ← ΔV − damp · J⁺ r`
  - `tol_km` — convergence threshold on `‖r‖`
  - `peri_target_km`, `apo_target_km`, `mode`, `r_peri_nom`, `r_apo_nom` — forwarded to
    [`apse_residual`](@ref)
  - `validate_targets` — run [`validate_apse_targets`](@ref) before solving

Returns `(dv, dv_mag_ms, converged, residual_km, peri_err_km, iterations)`: the ΔV in km/s,
its magnitude in m/s, and the diagnostics.

The Jacobian is built by forward finite differences and the pseudo-inverse step selects the
minimum-norm correction, so the solver prefers cheap burns and stays well posed whichever
way the system is determined. Iteration stops on `‖r‖ < tol_km`, on a non-finite residual
(the prediction lost an apse over the horizon), or at `max_iter`.

NOTE: in `:altitude_position`, read `peri_err_km`, not `converged`. The total residual
there is dominated by the apoapsis-position block, which is held against a nominal orbit
and is not driven to zero from a drifted state, so `converged` reads false on passes that
hit the commanded altitude to well under a km. `peri_err_km` is the periapsis-altitude
error alone and is `NaN` in the other modes.

NOTE: planning uses the ONBOARD model. `eom!` is a keyword only for the perfect-model
ablation; never pass a truth EOM in normal operation.
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
    mode in (:position, :altitude_position) && validate_targets &&
        validate_apse_targets(state0, r_peri_nom, r_apo_nom;
                              mode = mode, peri_target_km = peri_target_km)

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

        # Minimum-norm Gauss-Newton step: ΔV -= damp · (WJ)⁺ (Wr).
        W = _residual_weights(mode, J)
        dv = dv .- damp .* (pinv(W * J) * (W * r))
        r = resid(dv)
    end

    residual = all(isfinite, r) ? norm(r) : Inf

    # Periapsis-altitude error, broken out separately. In `:altitude_position` the total
    # residual is dominated by the apoapsis POSITION block, which is a 3-vector held against
    # a nominal orbit and is never driven to zero at a drifted state — so `converged` reads
    # false even on a pass that hit the commanded altitude to 0.08 km. That distinction is
    # load-bearing: `n_failed_solves` is how every rollout result is qualified, and without
    # this field a working excursion is indistinguishable from a dead ΔV = 0 solve.
    # `NaN` in the other modes, where there is no single altitude residual to report.
    peri_err = mode === :altitude_position && all(isfinite, r) ? abs(r[1]) : NaN

    return (
        dv          = dv,
        dv_mag_ms   = norm(dv) * 1.0e3,   # km/s → m/s
        converged   = residual < tol_km,
        residual_km = residual,
        peri_err_km = peri_err,
        iterations  = iters,
    )
end