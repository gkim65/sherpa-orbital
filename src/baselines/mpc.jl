"""
Deterministic MPC stationkeeping baseline (Strategy 3, MacKenzie et al. 2020 §B.2.3).

The feasibility controller for the Enceladus Orbilander period-3 L1 halo orbit. It answers
"can we even stationkeep this orbit?" before further investment in propagator fidelity
(external review 2026-06-22, Decision D3), and it is re-run at each fidelity rung to find
the rung where a deterministic controller fails and the POMDP earns its keep.

Control concept
---------------
The spacecraft coasts under the TRUTH dynamics. At each descending crossing of a fixed
altitude shell ([`CONTROL_ALT_KM`](@ref) = 600 km above the surface) the controller solves
for an impulsive ΔV re-targeting the next periapsis and apoapsis. The burn is planned by
[`solve_burn`](@ref) with a multiple-shooting prediction over `n_revs` revolutions of the
ONBOARD model; the world then propagates the post-burn state forward under the truth model
until the next trigger.

⚠️ TRUTH/ONBOARD SPLIT (CLAUDE.md rule). [`run_mpc`](@ref) is the ONLY place `truth_eom!` is
integrated. All burn planning happens inside `solve_burn`, which sees the onboard CR3BP
model exclusively. `truth_eom!` is a positional ARGUMENT, deliberately: the same controller
must run unchanged against CR3BP+EncJ2, +Saturn J2, and later the SPICE inertial truth. Do
not close over a specific truth model here.

Fuel metric
-----------
Reported as ΔV (m/s). `ISP_MR106E` and `M_SPACECRAFT_WET` are in `constants.jl` but are not
yet wired: the ΔV → propellant-mass conversion via the rocket equation is still a TODO,
carried over from the Python reference.

References
  MacKenzie, S. M. et al. (2020). Enceladus Orbilander: A Flagship Mission Concept for
    Astrobiology. §B.2.3 (stationkeeping Strategy 3), §3.5 (propulsion).
"""

"""
    run_mpc(state0, truth_eom!, period_s, horizon_s; n_revs, rtol_truth, atol_truth,
            verbose, peri_target_km, apo_target_km, mode, ref_ic, max_burns) -> NamedTuple

Run the event-driven MPC stationkeeping loop against a TRUTH model.

The world propagates under `truth_eom!`; at each descending 600-km altitude crossing the
controller plans a burn with the onboard model ([`solve_burn`](@ref)) and applies the
impulsive ΔV. After each burn the spacecraft coasts under truth dynamics to the next
periapsis before the (descending) control shell re-arms — otherwise the event re-fires at
the same point and the loop makes no time progress. That sets the cadence to one burn per
periapsis approach, as Strategy 3 specifies.

The loop ends when the horizon is reached, the spacecraft crashes
(`PERIAPSIS_CRASH_ALT` = 5 km), or it escapes ([`ESCAPE_ALT_KM`](@ref)).

  - `state0` — initial barycentre-frame state (km, km/s), typically the period-3 IC
  - `truth_eom!` — world dynamics: [`cr3bp_j2_eom!`](@ref),
    [`cr3bp_saturn_enc_j2_eom!`](@ref), or later the SPICE inertial model. Passed in so the
    SAME controller runs at every fidelity rung.
  - `period_s` — single-revolution period estimate (s)
  - `horizon_s` — total mission horizon to simulate (s)
  - `n_revs` — multiple-shooting horizon for the burn solver (`N_m`)
  - `peri_target_km`, `apo_target_km` — apse-altitude targets used when
    `mode = :altitude`; default to the centre of the MacKenzie bands (42.05 / 1055 km).
    Set them to another orbit's own apse altitudes to stationkeep that orbit without
    touching the module constants.
  - `mode` — `:altitude` (default, apse-altitude targeting) or `:position` (Strategy 3
    proper: bound the apse POSITION vectors against a nominal orbit). `:position` uses
    `ref_ic`, defaulting to `state0`.
  - `max_burns` — hard cap on executed burns (default 2000), a safety guard so the loop can
    never run unbounded. A normal 30-day run uses ~60 burns; hitting the cap returns
    outcome `:max_burns` and means something is wrong upstream.

Returns a NamedTuple:

  - `survived` — reached the horizon without crashing or escaping
  - `outcome` — `:held` (actively stationkept to the horizon), `:idle` (no crash, but the
    controller stopped triggering), `:crash`, `:escape`, or `:max_burns`
  - `survival_time_s` — time of loss, or `horizon_s`
  - `controller_active_until_s` — last time the controller was still triggering
  - `n_burns`, `total_dv_ms` — burn count and cumulative ΔV (m/s)
  - `burns` — per-burn records `(t_s, alt_km, dv_ms, converged, residual_km)`
  - `min_peri_alt_km` — smallest periapsis altitude seen (km)
"""
function run_mpc(
    state0::AbstractVector{<:Real},
    truth_eom!,
    period_s::Real,
    horizon_s::Real;
    n_revs::Integer = 3,
    rtol_truth::Real = RTOL_TRUTH,
    atol_truth::Real = ATOL_TRUTH,
    verbose::Bool = false,
    peri_target_km::Real = PERIAPSIS_ALT_TARGET,
    apo_target_km::Real = APOAPSIS_ALT_TARGET,
    mode::Symbol = :altitude,
    ref_ic::Union{AbstractVector{<:Real},Nothing} = nothing,
    max_burns::Integer = 2000,
)
    state = collect(float.(state0))
    horizon_s = float(horizon_s)
    t_now = 0.0
    total_dv_ms = 0.0
    burns = NamedTuple[]
    min_peri_alt = Inf

    # Strategy 3: compute the nominal apse position targets once, from the reference orbit
    # (defaults to the initial state). Planning uses the onboard model so the targets live
    # in the same model the burn solver predicts in.
    r_peri_nom = nothing
    r_apo_nom = nothing
    if mode === :position
        ref = ref_ic === nothing ? state0 : ref_ic
        # Count-based apse search. `period_s` is the CONTROL cadence (T/3 for the period-3
        # orbit) and is NOT a valid apse-search window: the orbit has 3 periapses but only 2
        # apoapses, so T/3 lands 0.136 s short of the first apoapsis and the old
        # window-based nominal_apse_positions returned a NaN apoapsis target.
        # The `mode = :altitude` default never reached this branch, which is why the
        # Session-3 run_mpc reference signature is unaffected.
        r_peri_nom, r_apo_nom = next_apse_positions(ref; eom! = cr3bp_eom!)
    end

    # Assembled once per outcome so every early return reports the same fields.
    finish(outcome, t_loss, survived, peri) = (
        survived                  = survived,
        outcome                   = outcome,
        survival_time_s           = t_loss,
        controller_active_until_s = t_now,
        n_burns                   = length(burns),
        total_dv_ms               = total_dv_ms,
        burns                     = burns,
        min_peri_alt_km           = isfinite(peri) ? peri : NaN,
    )

    while t_now < horizon_s
        # Safety guard: t_now advances each iteration by construction, but cap the burn
        # count so a pathological no-progress state can never spin forever.
        if length(burns) >= max_burns
            return finish(:max_burns, t_now, false, min_peri_alt)
        end

        # Leg 1: coast under TRUTH to the next control trigger, watching for a crash or an
        # escape and recording every periapsis en route.
        peri_seen = Tuple{Float64,Vector{Float64}}[]
        ctrl = _EventRecord()
        crash = _EventRecord()
        esc = _EventRecord()

        # The solution itself is not needed: every branch below reads an event record.
        propagate(
            truth_eom!, state, (0.0, horizon_s - t_now);
            callback = CallbackSet(
                _terminal_shell_callback(CONTROL_ALT_KM, ctrl),
                _terminal_shell_callback(PERIAPSIS_CRASH_ALT, crash),
                _record_periapsis_callback(peri_seen),
                _terminal_escape_callback(ESCAPE_ALT_KM, esc),
            ),
            rtol = rtol_truth, atol = atol_truth,
        )

        for (_, u) in peri_seen
            min_peri_alt = min(min_peri_alt, altitude(u))
        end

        # Escape — orbit lost, the controller cannot recover it.
        if esc.fired
            t_esc = t_now + esc.t
            verbose && @printf("  ESCAPE (alt > %.0f km) at t=%.2f hr after %d burns\n",
                               ESCAPE_ALT_KM, t_esc / 3600, length(burns))
            return finish(:escape, t_esc, false, min_peri_alt)
        end

        # Crash — terminal failure.
        if crash.fired
            t_crash = t_now + crash.t
            min_peri_alt = min(min_peri_alt, PERIAPSIS_CRASH_ALT)
            verbose && @printf("  CRASH at t=%.2f hr after %d burns\n",
                               t_crash / 3600, length(burns))
            return finish(:crash, t_crash, false, min_peri_alt)
        end

        # No 600-km descending crossing before the horizon. Either the orbit stayed above
        # the shell the whole time, or it never came back inbound — no crash either way, so
        # it survived, but the loop exits and the outcome is flagged `:idle` below.
        ctrl.fired || break

        state_ctrl = copy(ctrl.u)
        t_now += ctrl.t

        burn = solve_burn(state_ctrl, period_s;
                          n_revs = n_revs, eom! = cr3bp_eom!,
                          peri_target_km = peri_target_km,
                          apo_target_km = apo_target_km,
                          mode = mode,
                          r_peri_nom = r_peri_nom, r_apo_nom = r_apo_nom)
        state_post = copy(state_ctrl)
        state_post[4:6] .+= burn.dv
        total_dv_ms += burn.dv_mag_ms

        # Leg 2: coast past the control shell under TRUTH to the next periapsis, so the
        # descending 600-km event re-arms rather than re-triggering in place.
        peri2 = _EventRecord()
        crash2 = _EventRecord()
        esc2 = _EventRecord()
        coast = propagate(
            truth_eom!, state_post, (0.0, horizon_s - t_now);
            callback = CallbackSet(
                _terminal_periapsis_callback(peri2),
                _terminal_shell_callback(PERIAPSIS_CRASH_ALT, crash2),
                _terminal_escape_callback(ESCAPE_ALT_KM, esc2),
            ),
            rtol = rtol_truth, atol = atol_truth,
        )

        # Record the burn before any terminal-outcome return, so a fatal leg still reports it.
        push!(burns, (t_s = t_now, alt_km = altitude(state_ctrl),
                      dv_ms = burn.dv_mag_ms, converged = burn.converged,
                      residual_km = burn.residual_km))
        verbose && @printf("  burn %2d @ t=%7.2f hr  ΔV=%7.3f m/s  res=%.3f km  %s\n",
                           length(burns), t_now / 3600, burn.dv_mag_ms,
                           burn.residual_km, burn.converged ? "ok" : "NO-CONV")

        if esc2.fired   # escape during the post-burn coast
            t_esc = t_now + esc2.t
            verbose && @printf("  ESCAPE (post-burn coast) at t=%.2f hr\n", t_esc / 3600)
            return finish(:escape, t_esc, false, min_peri_alt)
        end
        if crash2.fired   # crash during the inbound coast
            t_crash = t_now + crash2.t
            min_peri_alt = min(min_peri_alt, PERIAPSIS_CRASH_ALT)
            verbose && @printf("  CRASH (post-burn coast) at t=%.2f hr after %d burns\n",
                               t_crash / 3600, length(burns))
            return finish(:crash, t_crash, false, min_peri_alt)
        end

        if peri2.fired
            state = copy(peri2.u)
            min_peri_alt = min(min_peri_alt, altitude(state))
            t_now += peri2.t
        else
            # No periapsis before the horizon — advance to the horizon and finish.
            state = copy(coast.u[end])
            t_now = horizon_s
        end
    end

    # Reached the horizon with no crash and no escape: survived. `:idle` flags that the
    # controller stopped triggering (the orbit drifted off the 600-km shell) before the
    # horizon — survived in the no-crash sense, but no longer actively held.
    return finish(t_now < horizon_s ? :idle : :held, horizon_s, true, min_peri_alt)
end

# ── Event bookkeeping ─────────────────────────────────────────────────────────
# scipy reports every event through `sol.t_events` / `sol.y_events` after the fact, so the
# Python reference could pass four events to one `solve_ivp` call and inspect which fired.
# `ContinuousCallback` has no equivalent, so each condition writes into its own record and
# the caller reads the flags — reproducing the same "which of these four ended the leg?"
# dispatch. Recording (t, u) explicitly also avoids having to infer the cause from
# `sol.t[end]`, which is ambiguous when two conditions are close together.

mutable struct _EventRecord
    fired::Bool
    t::Float64
    u::Vector{Float64}
end
_EventRecord() = _EventRecord(false, NaN, Float64[])

function _record!(rec::_EventRecord, integrator)
    # First crossing wins, matching scipy's `t_events[i][0]` indexing.
    if !rec.fired
        rec.fired = true
        rec.t = integrator.t
        rec.u = copy(integrator.u)
    end
    return nothing
end

# Descending through an altitude shell (scipy direction −1): the control trigger and the
# crash test. Terminal.
function _terminal_shell_callback(altitude_km::Real, rec::_EventRecord)
    target_r = R_ENCELADUS + altitude_km
    g(u, t, integrator) = r_enceladus(u) - target_r
    return ContinuousCallback(g, nothing,
                              integrator -> (_record!(rec, integrator); terminate!(integrator));
                              abstol = EVENT_TOL, reltol = 0.0)
end

# Ascending through the escape shell (scipy direction +1). Terminal.
function _terminal_escape_callback(altitude_km::Real, rec::_EventRecord)
    target_r = R_ENCELADUS + altitude_km
    g(u, t, integrator) = r_enceladus(u) - target_r
    return ContinuousCallback(g,
                              integrator -> (_record!(rec, integrator); terminate!(integrator)),
                              nothing; abstol = EVENT_TOL, reltol = 0.0)
end

# Periapsis: ṙ_enc upcrossing (scipy direction +1).
function _terminal_periapsis_callback(rec::_EventRecord)
    g(u, t, integrator) = rdot_enceladus(u)
    return ContinuousCallback(g,
                              integrator -> (_record!(rec, integrator); terminate!(integrator)),
                              nothing; abstol = EVENT_TOL, reltol = 0.0)
end

function _record_periapsis_callback(store::Vector{Tuple{Float64,Vector{Float64}}})
    g(u, t, integrator) = rdot_enceladus(u)
    return ContinuousCallback(g,
                              integrator -> (push!(store, (integrator.t, copy(integrator.u))); nothing),
                              nothing; abstol = EVENT_TOL, reltol = 0.0,
                              save_positions = (false, false))
end