"""
Onboard burn planner — the apse-targeting solver shared by every stationkeeping baseline.

This is the ONBOARD half of MacKenzie et al. 2020 §B.2.3 "Strategy 3". Given a state at a
control point, it solves for the impulsive ΔV that re-targets the next periapsis and
apoapsis, predicting with a multiple-shooting propagation over `n_revs` revolutions.

⚠️ TRUTH/ONBOARD SPLIT (CLAUDE.md rule — do NOT collapse). Everything in this file plans
with the ONBOARD model, [`cr3bp_eom!`](@ref) (CR3BP only, no perturbations), at the loose
onboard tolerances. No function here may reach for a truth EOM. The gap between what the
planner believes and what the world does is the model uncertainty the POMDP absorbs; a
truth EOM appearing in this file would silently erase the project's central quantity.

The `eom` keyword exists ONLY so an ablation study can deliberately hand the planner a
perfect model and measure what that is worth (`scripts/mpc_planner_ablation.py` in the
Python reference measured this: a perfect onboard model does not extend survival —
the failure is instability-dominated). It defaults to the onboard model.

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
    predict_apse_states(state0, n_revs, period_s; eom!, rtol, atol)
        -> (peri_state, apo_state)

Predict the FIRST periapsis and FIRST apoapsis 6-states over the next `n_revs` revolutions
of the ONBOARD model. Barycentre frame, km / km/s.

This is the multiple-shooting prediction behind [`solve_burn`](@ref): it propagates
`n_revs × period_s` seconds and records every apse passage (non-terminal), then returns the
first of each type. An apse not reached within the horizon comes back as a vector of `NaN`.

Using the FIRST apse of each type — rather than the min/max over the whole span — keeps the
residual a smooth function of ΔV, which the Gauss-Newton Jacobian needs. The multi-rev
horizon exists so the solver still sees several apses and stays well-conditioned when the
first one falls close to the burn.
"""
function predict_apse_states(
    state0::AbstractVector{<:Real},
    n_revs::Integer,
    period_s::Real;
    eom! = cr3bp_eom!,
    rtol::Real = RTOL_ONBOARD,
    atol::Real = ATOL_ONBOARD,
)
    peri, apo = collect_apses(eom!, state0, n_revs * float(period_s);
                              rtol = rtol, atol = atol)
    nan6 = fill(NaN, 6)
    peri_state = isempty(peri) ? nan6 : peri[1][2]
    apo_state  = isempty(apo)  ? nan6 : apo[1][2]
    return peri_state, apo_state
end

"""
    predict_apses(state0, n_revs, period_s; eom!, rtol, atol) -> (peri_alt_km, apo_alt_km)

Predict the ALTITUDE (km) of the first periapsis and first apoapsis over the next `n_revs`
revolutions of the onboard model. `NaN` for an apse not reached within the horizon.

The altitude-only counterpart of [`predict_apse_states`](@ref), used by `mode = :altitude`.
"""
function predict_apses(
    state0::AbstractVector{<:Real},
    n_revs::Integer,
    period_s::Real;
    eom! = cr3bp_eom!,
    rtol::Real = RTOL_ONBOARD,
    atol::Real = ATOL_ONBOARD,
)
    peri_state, apo_state = predict_apse_states(state0, n_revs, period_s;
                                                eom! = eom!, rtol = rtol, atol = atol)
    peri_alt = isnan(peri_state[1]) ? NaN : altitude(peri_state)
    apo_alt  = isnan(apo_state[1])  ? NaN : altitude(apo_state)
    return peri_alt, apo_alt
end

"""
    nominal_apse_positions(ref_ic, period_s; eom!, rtol, atol) -> (r_peri_nom, r_apo_nom)

Periapsis and apoapsis POSITION vectors (3-vectors, km, barycentre frame) of a nominal
reference orbit.

Propagates the uncontrolled reference IC for one revolution under the onboard model and
records its first periapsis/apoapsis positions. These are the `r_apse,nominal` targets for
MacKenzie Strategy 3 apse-position bounding — see `mode = :position` in
[`solve_burn`](@ref).
"""
function nominal_apse_positions(
    ref_ic::AbstractVector{<:Real},
    period_s::Real;
    eom! = cr3bp_eom!,
    rtol::Real = RTOL_ONBOARD,
    atol::Real = ATOL_ONBOARD,
)
    peri_state, apo_state = predict_apse_states(ref_ic, 1, period_s;
                                                eom! = eom!, rtol = rtol, atol = atol)
    return copy(peri_state[1:3]), copy(apo_state[1:3])
end

# ── Residual ──────────────────────────────────────────────────────────────────

"""
    apse_residual(state0, dv, n_revs, period_s, eom!; ...) -> Vector{Float64}

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
    (see [`nominal_apse_positions`](@ref)).
"""
function apse_residual(
    state0::AbstractVector{<:Real},
    dv::AbstractVector{<:Real},
    n_revs::Integer,
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
        peri_state, apo_state = predict_apse_states(s, n_revs, period_s; eom! = eom!)
        return vcat(peri_state[1:3] .- r_peri_nom, apo_state[1:3] .- r_apo_nom)
    end

    peri_alt, apo_alt = predict_apses(s, n_revs, period_s; eom! = eom!)
    return [peri_alt - peri_target_km, apo_alt - apo_target_km]
end

# ── Burn solver ───────────────────────────────────────────────────────────────

"""
    solve_burn(state0, period_s; n_revs, eom!, max_iter, fd_step, damp, tol_km,
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
  - `period_s` — single-revolution period estimate (s)
  - `n_revs` — multiple-shooting horizon in revolutions (MacKenzie `N_m` = 2–3)

Returns `(dv, dv_mag_ms, converged, residual_km, iterations)`, where `dv` is the ΔV in km/s
and `dv_mag_ms` its magnitude in m/s.
"""
function solve_burn(
    state0::AbstractVector{<:Real},
    period_s::Real;
    n_revs::Integer = 3,
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
)
    resid(dv) = apse_residual(state0, dv, n_revs, period_s, eom!;
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