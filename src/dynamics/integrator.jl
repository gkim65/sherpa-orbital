"""
Propagation and event detection for CR3BP / CR3BP+J2 trajectories.

Provides:
  - `propagate`             general-purpose IVP integration with optional events
  - `propagate_to_apoapsis` propagate until the next apoapsis passage
  - `propagate_to_periapsis` propagate until the next periapsis passage
  - `propagate_n_orbits`    propagate for a fixed number of orbital periods

Event detection uses the radial-velocity sign-change condition:

  - Apoapsis:  d/dt(r_enc) = 0 with ṙ going from + to −
  - Periapsis: d/dt(r_enc) = 0 with ṙ going from − to +

`r_enc` is the distance from Enceladus, NOT from the barycentre.

Integrator choice and event semantics
------------------------------------
The Python reference used `scipy.solve_ivp(method="RK45")`, i.e. Dormand-Prince 5(4).
`Vern9` is used here instead: at the truth tolerances (rtol 1e-10 / atol 1e-12) a 5th-order
method is working far below its efficient range, and the higher-order method reaches the
same tolerance with a much better error constant. Trajectory agreement against the Python
reference is recorded in the session log.

Event *semantics* differ between the two libraries and the difference matters, because these
events classify crash vs. escape:

  - scipy: one event function per condition with a `direction` flag on the sign change of
    `g`, root-found by Brent's method on a dense interpolant.
  - Julia: `ContinuousCallback(g, affect!, affect_neg!)`, where `affect!` fires on an
    UPcrossing (`g`: − → +) and `affect_neg!` on a DOWNcrossing. Passing `nothing` for one
    of them suppresses that direction, which is how `direction` is reproduced here:

        direction = +1  →  ContinuousCallback(g, terminate!, nothing)
        direction = -1  →  ContinuousCallback(g, nothing, terminate!)

    Root-finding tolerances are tightened below scipy's defaults (`abstol`/`reltol` on the
    callback) so the recovered event time is not the limiting error source.

⚠️ Behaviour change vs. the Python reference — events at t0
-----------------------------------------------------------
`solve_ivp` evaluates the event function at `t0` and reports a root if it is already zero
there. The period-3 IC starts EXACTLY at apoapsis (`ṙ(0) = 0.0` identically), so the Python
`propagate_to_apoapsis` returns the initial state itself — a zero-length "propagation" — and
the non-terminal apoapsis event list carries a phantom `t = 0` entry.

`ContinuousCallback` requires a sign CHANGE across a step, so it does not fire at `t0`, and
these functions return the FIRST DOWNSTREAM apoapsis. That is the intended meaning of
"propagate to the next apoapsis", so the Julia behaviour is kept rather than bug-compatible.

Consequence for callers: anything that consumed the Python apoapsis list by index is off by
one at `t = 0`. Validated (2026-08-24): after dropping scipy's phantom entry, apoapsis counts
match exactly and event times agree to ≤1.1e-3 s over 3 orbits; `propagate_to_apoapsis`
matches Python's *second* apoapsis to 5.7e-7 s. Periapsis is unaffected (no root at `t0`).
"""

# ── Integration tolerances ────────────────────────────────────────────────────
const RTOL_TRUTH   = 1e-10   # truth model (CR3BP+J2)
const ATOL_TRUTH   = 1e-12
const RTOL_ONBOARD = 1e-8    # onboard model (CR3BP only)
const ATOL_ONBOARD = 1e-10

# Root-finding tolerance for event localisation. Tighter than the ODE tolerance so the
# event time is limited by the trajectory, not by the root solve.
const EVENT_TOL = 1e-14

"""
    r_enceladus(u) -> Float64

Distance from Enceladus' centre (km).
"""
r_enceladus(u::AbstractVector{<:Real}) =
    sqrt((u[1] - X_ENCELADUS)^2 + u[2]^2 + u[3]^2)

"""
    rdot_enceladus(u) -> Float64

Radial velocity relative to Enceladus (km/s). Positive = moving away.
"""
function rdot_enceladus(u::AbstractVector{<:Real})
    dx = u[1] - X_ENCELADUS
    r = sqrt(dx^2 + u[2]^2 + u[3]^2)
    return (dx * u[4] + u[2] * u[5] + u[3] * u[6]) / r
end

"""
    altitude(u) -> Float64

Altitude above the Enceladus reference surface (km).
"""
altitude(u::AbstractVector{<:Real}) = r_enceladus(u) - R_ENCELADUS

# ── Event constructors ────────────────────────────────────────────────────────
# Each returns a ContinuousCallback. `terminal` selects whether the crossing stops the
# integration (`terminate!`) or is only recorded via a saved-values callback by the caller.
# The direction convention is documented in the module docstring.

_directed_callback(g, direction::Int, affect) =
    direction > 0 ? ContinuousCallback(g, affect, nothing;
                                       abstol = EVENT_TOL, reltol = 0.0,
                                       save_positions = (true, true)) :
                    ContinuousCallback(g, nothing, affect;
                                       abstol = EVENT_TOL, reltol = 0.0,
                                       save_positions = (true, true))

"""
    apoapsis_callback(; terminal = true) -> ContinuousCallback

Event at apoapsis: `ṙ_enc = 0` crossing from + to − (scipy `direction = -1`).
"""
function apoapsis_callback(; terminal::Bool = true)
    g(u, t, integrator) = rdot_enceladus(u)
    return _directed_callback(g, -1, terminal ? terminate! : (integrator -> nothing))
end

"""
    periapsis_callback(; terminal = true) -> ContinuousCallback

Event at periapsis: `ṙ_enc = 0` crossing from − to + (scipy `direction = +1`).
"""
function periapsis_callback(; terminal::Bool = true)
    g(u, t, integrator) = rdot_enceladus(u)
    return _directed_callback(g, +1, terminal ? terminate! : (integrator -> nothing))
end

"""
    altitude_callback(altitude_km; terminal = false) -> ContinuousCallback

Event when the altitude above Enceladus descends through `altitude_km` (scipy
`direction = -1`). Used for stationkeeping: Strategy 3 fires at the 600-km altitude
crossing (MacKenzie 2020 §B.2.3).
"""
function altitude_callback(altitude_km::Real; terminal::Bool = false)
    target_r = R_ENCELADUS + altitude_km
    g(u, t, integrator) = r_enceladus(u) - target_r
    return _directed_callback(g, -1, terminal ? terminate! : (integrator -> nothing))
end

"""
    crash_callback(crash_altitude_km = PERIAPSIS_CRASH_ALT) -> ContinuousCallback

Terminal event fired when the altitude descends below `crash_altitude_km` — surface
impact. Always terminal.
"""
function crash_callback(crash_altitude_km::Real = PERIAPSIS_CRASH_ALT)
    crash_r = R_ENCELADUS + crash_altitude_km
    g(u, t, integrator) = r_enceladus(u) - crash_r
    return _directed_callback(g, -1, terminate!)
end

# ── Propagation ───────────────────────────────────────────────────────────────

"""
    propagate(eom!, state0, tspan; callback = nothing, rtol, atol, saveat = Float64[],
              max_step = Inf) -> ODESolution

Integrate `eom!` (an in-place `f(du, u, p, t)`) from `tspan[1]` to `tspan[2]` (s).

  - `state0`: `[x, y, z, vx, vy, vz]` in km / km/s
  - `callback`: a `ContinuousCallback` / `CallbackSet` for event detection
  - `rtol`, `atol`: integration tolerances (default: truth model)
  - `saveat`: request output at specific times (s); empty = solver's own steps
  - `max_step`: maximum allowed step size (s)

Returns the `ODESolution`; `sol.t[end]` and `sol.u[end]` are the terminating time and
state, which for a terminal callback is the event itself. The solution is dense
(interpolable) unless `saveat` is given.
"""
function propagate(
    eom!,
    state0::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real};
    callback = nothing,
    rtol::Real = RTOL_TRUTH,
    atol::Real = ATOL_TRUTH,
    saveat = Float64[],
    max_step::Real = Inf,
)
    prob = ODEProblem(eom!, collect(float.(state0)), float.(tspan))
    # Qualified: both POMDPs and OrdinaryDiffEq export `solve`.
    return OrdinaryDiffEq.solve(
        prob, Vern9();
        reltol = rtol,
        abstol = atol,
        callback = callback,
        saveat = saveat,
        dtmax = max_step,
        maxiters = 10_000_000,
    )
end

"""
    propagate_to_apoapsis(eom!, state0, t_max; rtol, atol) -> (state_apo, t_apo)

Propagate until the next apoapsis (`ṙ_enc = 0`, decelerating), within `t_max` seconds.
Returns the state (km, km/s) and time (s) at apoapsis. Throws if no apoapsis is found
within `t_max`.
"""
function propagate_to_apoapsis(
    eom!,
    state0::AbstractVector{<:Real},
    t_max::Real;
    rtol::Real = RTOL_TRUTH,
    atol::Real = ATOL_TRUTH,
)
    sol = propagate(eom!, state0, (0.0, float(t_max));
                    callback = apoapsis_callback(), rtol = rtol, atol = atol)
    _check_event_found(sol, t_max, "Apoapsis")
    return sol.u[end], sol.t[end]
end

"""
    propagate_to_periapsis(eom!, state0, t_max; rtol, atol) -> (state_peri, t_peri)

Propagate until the next periapsis (`ṙ_enc = 0`, accelerating), within `t_max` seconds.
Returns the state (km, km/s) and time (s) at periapsis. Throws if no periapsis is found
within `t_max`.
"""
function propagate_to_periapsis(
    eom!,
    state0::AbstractVector{<:Real},
    t_max::Real;
    rtol::Real = RTOL_TRUTH,
    atol::Real = ATOL_TRUTH,
)
    sol = propagate(eom!, state0, (0.0, float(t_max));
                    callback = periapsis_callback(), rtol = rtol, atol = atol)
    _check_event_found(sol, t_max, "Periapsis")
    return sol.u[end], sol.t[end]
end

"""
    propagate_n_orbits(eom!, state0, n_orbits, orbit_period_s; rtol, atol, saveat)
        -> ODESolution

Propagate for `n_orbits` complete orbital periods of `orbit_period_s` seconds, with a
terminal crash event. The returned solution ends early (at `sol.t[end] < n*T`) if the
spacecraft impacts.
"""
function propagate_n_orbits(
    eom!,
    state0::AbstractVector{<:Real},
    n_orbits::Integer,
    orbit_period_s::Real;
    rtol::Real = RTOL_TRUTH,
    atol::Real = ATOL_TRUTH,
    saveat = Float64[],
)
    t_end = n_orbits * float(orbit_period_s)
    return propagate(eom!, state0, (0.0, t_end);
                     callback = crash_callback(), rtol = rtol, atol = atol,
                     saveat = saveat)
end

"""
    collect_apses(eom!, state0, t_max; rtol, atol) -> (peri, apo)

Record every periapsis and apoapsis passage in `[0, t_max]` without terminating, as
two vectors of `(t, state)` pairs. This is the non-terminal counterpart to
[`propagate_to_periapsis`](@ref); it replaces scipy's "non-terminal event with
`t_events`" pattern, which has no direct `ContinuousCallback` equivalent.
"""
function collect_apses(
    eom!,
    state0::AbstractVector{<:Real},
    t_max::Real;
    rtol::Real = RTOL_TRUTH,
    atol::Real = ATOL_TRUTH,
)
    peri = Tuple{Float64,Vector{Float64}}[]
    apo  = Tuple{Float64,Vector{Float64}}[]

    g(u, t, integrator) = rdot_enceladus(u)
    record!(store) = integrator -> push!(store, (integrator.t, copy(integrator.u)))

    # Upcrossing of ṙ (− → +) is periapsis; downcrossing is apoapsis.
    cb = ContinuousCallback(g, record!(peri), record!(apo);
                            abstol = EVENT_TOL, reltol = 0.0,
                            save_positions = (false, false))

    propagate(eom!, state0, (0.0, float(t_max)); callback = cb, rtol = rtol, atol = atol)
    return peri, apo
end

# Terminal-callback solutions stop before t_max; if they ran to t_max the event never fired.
function _check_event_found(sol, t_max::Real, what::AbstractString)
    if sol.t[end] >= float(t_max) * (1 - 1e-12)
        error("$what not found within t_max=$(t_max) s. " *
              "Final r_enc=$(r_enceladus(sol.u[end])) km")
    end
    return nothing
end