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

`r_enc` is the distance from Enceladus, not from the barycentre.

Integration is by `Vern9`: at the truth tolerances (rtol 1e-10 / atol 1e-12) a 5th-order
method works far below its efficient range, and the higher-order method reaches the same
tolerance with a much better error constant.

Event direction is encoded by which `ContinuousCallback` slot is filled — `affect!` fires
on an upcrossing of `g`, `affect_neg!` on a downcrossing, and `nothing` suppresses that
direction:

    direction = +1  ->  ContinuousCallback(g, terminate!, nothing)
    direction = -1  ->  ContinuousCallback(g, nothing, terminate!)

Root-finding tolerances are tightened below the defaults so the recovered event time is
not the limiting error source.

NOTE: `ContinuousCallback` needs a sign CHANGE across a step, so an event never fires at
`t0`. The IC starts exactly at apoapsis (`ṙ(0) = 0` identically), so
`propagate_to_apoapsis` returns the first DOWNSTREAM apoapsis rather than the initial
state. Periapsis is unaffected.
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

Distance from Enceladus' centre.

  - `u` — barycentre-frame state (km, km/s); only the position is read

Returns the distance in km.
"""
r_enceladus(u::AbstractVector{<:Real}) =
    sqrt((u[1] - X_ENCELADUS)^2 + u[2]^2 + u[3]^2)

"""
    rdot_enceladus(u) -> Float64

Radial velocity relative to Enceladus.

  - `u` — barycentre-frame state (km, km/s)

Returns the radial rate in km/s; positive is moving away. Zero at an apse, which is the
event condition the apse callbacks watch.
"""
function rdot_enceladus(u::AbstractVector{<:Real})
    dx = u[1] - X_ENCELADUS
    r = sqrt(dx^2 + u[2]^2 + u[3]^2)
    return (dx * u[4] + u[2] * u[5] + u[3] * u[6]) / r
end

"""
    altitude(u) -> Float64

Altitude above the Enceladus reference surface.

  - `u` — barycentre-frame state (km, km/s)

Returns `r_enceladus(u) - R_ENCELADUS` in km.
"""
altitude(u::AbstractVector{<:Real}) = r_enceladus(u) - R_ENCELADUS

# ── Event constructors ────────────────────────────────────────────────────────
# Each returns a ContinuousCallback. `terminal` selects whether the crossing stops the
# integration or is only recorded. Direction convention is in the module docstring.

_directed_callback(g, direction::Int, affect) =
    direction > 0 ? ContinuousCallback(g, affect, nothing;
                                       abstol = EVENT_TOL, reltol = 0.0,
                                       save_positions = (true, true)) :
                    ContinuousCallback(g, nothing, affect;
                                       abstol = EVENT_TOL, reltol = 0.0,
                                       save_positions = (true, true))

"""
    apoapsis_callback(; terminal = true) -> ContinuousCallback

Event at apoapsis: `ṙ_enc = 0` crossing from + to −.

  - `terminal` — stop the integration at the crossing, rather than only recording it

Returns a `ContinuousCallback`.
"""
function apoapsis_callback(; terminal::Bool = true)
    g(u, t, integrator) = rdot_enceladus(u)
    return _directed_callback(g, -1, terminal ? terminate! : (integrator -> nothing))
end

"""
    periapsis_callback(; terminal = true) -> ContinuousCallback

Event at periapsis: `ṙ_enc = 0` crossing from − to +.

  - `terminal` — stop the integration at the crossing, rather than only recording it

Returns a `ContinuousCallback`.
"""
function periapsis_callback(; terminal::Bool = true)
    g(u, t, integrator) = rdot_enceladus(u)
    return _directed_callback(g, +1, terminal ? terminate! : (integrator -> nothing))
end

"""
    altitude_callback(altitude_km; terminal = false) -> ContinuousCallback

Event when the altitude above Enceladus descends through a shell.

  - `altitude_km` — shell altitude above the surface (km)
  - `terminal` — stop the integration at the crossing, rather than only recording it

Returns a `ContinuousCallback`. Descending only, so it does not fire on the outbound leg.
Strategy 3 triggers control at the `CONTROL_ALT_KM` = 600 km crossing
(MacKenzie 2020 §B.2.3).
"""
function altitude_callback(altitude_km::Real; terminal::Bool = false)
    target_r = R_ENCELADUS + altitude_km
    g(u, t, integrator) = r_enceladus(u) - target_r
    return _directed_callback(g, -1, terminal ? terminate! : (integrator -> nothing))
end

"""
    crash_callback(crash_altitude_km = PERIAPSIS_CRASH_ALT) -> ContinuousCallback

Terminal event for surface impact: the altitude descending below a floor.

  - `crash_altitude_km` — impact altitude (km); defaults to `PERIAPSIS_CRASH_ALT`

Returns a `ContinuousCallback`. Always terminal.
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

Integrate `eom!` from `tspan[1]` to `tspan[2]`.

  - `eom!` — in-place equations of motion, `f(du, u, p, t)`
  - `state0` — initial state `[x, y, z, vx, vy, vz]` (km, km/s)
  - `tspan` — `(t0, tf)` in seconds
  - `callback` — a `ContinuousCallback` or `CallbackSet` for event detection
  - `rtol`, `atol` — integration tolerances; default to the truth-model values
  - `saveat` — output times (s); empty uses the solver's own steps
  - `max_step` — maximum step size (s)

Returns the `ODESolution`. `sol.t[end]` and `sol.u[end]` are the terminating time and
state, which for a terminal callback is the event itself. Dense (interpolable) unless
`saveat` is given.
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

Propagate until the next apoapsis (`ṙ_enc = 0`, receding).

  - `eom!` — equations of motion
  - `state0` — initial state (km, km/s)
  - `t_max` — search horizon (s)
  - `rtol`, `atol` — integration tolerances

Returns `(state_apo, t_apo)` in (km, km/s) and seconds. Throws if no apoapsis is found
within `t_max`. Never returns `t = 0` even if `state0` is at an apoapsis — see the module
docstring.
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

Propagate until the next periapsis (`ṙ_enc = 0`, approaching).

  - `eom!` — equations of motion
  - `state0` — initial state (km, km/s)
  - `t_max` — search horizon (s)
  - `rtol`, `atol` — integration tolerances

Returns `(state_peri, t_peri)` in (km, km/s) and seconds. Throws if no periapsis is found
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

Propagate for a whole number of orbital periods, with a terminal crash event.

  - `eom!` — equations of motion
  - `state0` — initial state (km, km/s)
  - `n_orbits` — how many periods to cover
  - `orbit_period_s` — one period (s)
  - `rtol`, `atol` — integration tolerances
  - `saveat` — output times (s)

Returns the `ODESolution`, ending early at `sol.t[end] < n_orbits * orbit_period_s` if the
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

Record every apse passage in `[0, t_max]` without terminating.

  - `eom!` — equations of motion
  - `state0` — initial state (km, km/s)
  - `t_max` — how long to propagate (s)
  - `rtol`, `atol` — integration tolerances

Returns `(peri, apo)`, each a vector of `(t, state)` pairs in seconds and (km, km/s). The
non-terminal counterpart to [`propagate_to_periapsis`](@ref).
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