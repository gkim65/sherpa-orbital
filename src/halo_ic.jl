"""
Differential corrector for the L1 halo orbit initial condition.

Finds a closed halo orbit in the Saturn-Enceladus CR3BP matching the Enceladus Orbilander
science orbit (MacKenzie et al. 2020, §B.2.3): periapsis altitude 19.8–64.3 km, apoapsis
altitude 1000–1110 km. The operational orbit is the period-3 member of the family
(`n_crossings = 3`), whose full period is ~36 hr with three periapsis passes.

Algorithm
---------
1. [`richardson_ic`](@ref) — Richardson (1980) 3rd-order analytic approximation, which puts
   the seed on (or very near) the halo family, so no blind scan is needed.
2. [`differential_corrector`](@ref) — single-shooting corrector (Howell 1984):
     * IC constrained to the x-z plane: `[x0, 0, z0, 0, vy0, 0]`
     * free variables `x0`, `vy0`; `z0` is held as the family parameter
     * target `vx_f = 0` and `vz_f = 0` at the n-th `y = 0` crossing
     * 2×2 Newton step from the STM, damped to prevent divergence
3. [`characterise_orbit`](@ref) — propagate one full period in physical units and report
   periapsis/apoapsis altitude, period, Jacobi constant and closure.

⚠️ UNITS. Unlike `src/dynamics/`, this module works internally in NON-DIMENSIONAL CR3BP
units (`L_STAR`, `T_STAR`, `V_STAR`; `MU` as the mass ratio, frame rate 1). Only
[`characterise_orbit`](@ref) and the `ic` / `period_s` fields of the corrector result are
physical.

⚠️ EVENT SEMANTICS. The half-period propagator does NOT rely on the library's apse events;
it detects `y = 0` crossings with its own callback, and brackets each one with an explicit
`t_min` coast. That coast is what steps over the degenerate `y = 0` at `t = 0` (the IC lies
in the symmetry plane by construction) and over each crossing just found, so the port does
not inherit the scipy-vs-`ContinuousCallback` "event at `t0`" difference documented in
`dynamics/integrator.jl`. Do not "simplify" the coasts away — they are the crossing-counting
logic.

⚠️ KNOWN DEFECT, carried over from the Python reference. The converged period-3 IC has
`z0 < 0` (southern out-of-plane amplitude), but the closest approach of this symmetric
period-3 bifurcation orbit falls at ~+87° latitude — the NORTH pole — not the south pole
the science case targets. Placing periapsis over the south pole needs the asymmetric
6-variable corrector ("Step A′", `docs/todo.md`); this IC is the symmetric degenerate case.

References
  Richardson, D. L. (1980). "Analytic Construction of Periodic Orbits about the Collinear
    Points." Celestial Mechanics 22(3), 241–253.
  Howell, K. C. (1984). "Three-Dimensional, Periodic, 'Halo' Orbits."
    Celestial Mechanics 32(1), 53–71.
  Koon, Lo, Marsden & Ross (2011). "Dynamical Systems, the Three-Body Problem and Space
    Mission Design", Ch. 2 (Richardson coefficient notation).
  MacKenzie et al. (2020). Enceladus Orbilander Mission Concept Study, §B.2.3.
"""

# ── Verified period-3 L1 halo orbit IC (southern-amplitude family) ────────────
# Source: JPL Three-Body Periodic Orbits Catalog (DE440), corrected in our CR3BP.
# MacKenzie et al. 2020 §B.2.3 science orbit — periapsis ~31 km altitude, apoapsis
# ~1065 km, period 35.99 hr, 3 periapsis passes per revolution. Converged with
# `differential_corrector(...; n_crossings = 3)`.
#
# Physical: x0 = 237911.1 km, z0 = −1162.8 km, vy0 = 0.06895 km/s, T = 35.99 hr.
const PERIOD3_IC_ND = [
     9.974083488926582e-01,   # x0  (nondim)
     0.0,                     # y0
    -4.874948479304304e-03,   # z0  (nondim; negative = southern amplitude)
     0.0,                     # vx0
     5.466912598254453e-03,   # vy0 (nondim)
     0.0,                     # vz0
]

const PERIOD3_PERIOD_S  = 35.988 * 3600.0                      # s
const PERIOD3_PERIOD_ND = PERIOD3_PERIOD_S / 18913.2798604104   # nondim (TU)

# Non-dimensional positions of the primaries
const _X_SAT_ND = -MU
const _X_ENC_ND = 1.0 - MU

# ── Non-dimensional CR3BP EOM + STM ──────────────────────────────────────────

"""
    cr3bp_nd(s) -> Vector{Float64}

Non-dimensional CR3BP equations of motion (no STM). `s = [x, y, z, vx, vy, vz]` in
non-dimensional units; returns `ds/dt`.
"""
function cr3bp_nd(s::AbstractVector{<:Real})
    x, y, z, vx, vy, vz = s
    r1 = sqrt((x + MU)^2 + y^2 + z^2)
    r2 = sqrt((x - 1.0 + MU)^2 + y^2 + z^2)
    r1_3, r2_3 = r1^3, r2^3
    ax =  2.0 * vy + x - (1 - MU) * (x + MU) / r1_3 - MU * (x - 1 + MU) / r2_3
    ay = -2.0 * vx + y - (1 - MU) * y / r1_3       - MU * y / r2_3
    az =                -(1 - MU) * z / r1_3       - MU * z / r2_3
    return [vx, vy, vz, ax, ay, az]
end

"""
    cr3bp_stm_nd!(daug, aug, p, t)

Augmented non-dimensional EOM: 6 state components followed by the 36 flattened STM
components (42 total), with `dΦ/dt = A(t) Φ` where `A` is the Jacobian of
[`cr3bp_nd`](@ref).

The STM is stored column-major (Julia `reshape` order); the Python reference stores it
row-major. The two agree because `A Φ` is computed from the reshaped matrix in both, but
any code reading raw elements out of the flat vector must use the matching order.
"""
function cr3bp_stm_nd!(daug, aug, p, t)
    s = @view aug[1:6]
    phi = reshape(@view(aug[7:42]), 6, 6)

    x, y, z = s[1], s[2], s[3]
    r1 = sqrt((x + MU)^2 + y^2 + z^2)
    r2 = sqrt((x - 1.0 + MU)^2 + y^2 + z^2)
    r1_3, r2_3 = r1^3, r2^3
    r1_5, r2_5 = r1^5, r2^5
    dx1, dx2 = x + MU, x - 1.0 + MU

    Uxx = (1.0
           - (1 - MU) / r1_3 + 3 * (1 - MU) * dx1^2 / r1_5
           - MU / r2_3       + 3 * MU * dx2^2 / r2_5)
    Uyy = (1.0
           - (1 - MU) / r1_3 + 3 * (1 - MU) * y^2 / r1_5
           - MU / r2_3       + 3 * MU * y^2 / r2_5)
    Uzz = (- (1 - MU) / r1_3 + 3 * (1 - MU) * z^2 / r1_5
           - MU / r2_3       + 3 * MU * z^2 / r2_5)
    Uxy = 3 * (1 - MU) * dx1 * y / r1_5 + 3 * MU * dx2 * y / r2_5
    Uxz = 3 * (1 - MU) * dx1 * z / r1_5 + 3 * MU * dx2 * z / r2_5
    Uyz = 3 * (1 - MU) * y * z / r1_5   + 3 * MU * y * z / r2_5

    A = [
        0.0  0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0  0.0  1.0  0.0
        0.0  0.0  0.0  0.0  0.0  1.0
        Uxx  Uxy  Uxz  0.0  2.0  0.0
        Uxy  Uyy  Uyz -2.0  0.0  0.0
        Uxz  Uyz  Uzz  0.0  0.0  0.0
    ]

    daug[1:6] .= cr3bp_nd(s)
    reshape(@view(daug[7:42]), 6, 6) .= A * phi
    return nothing
end

"""
    cr3bp_stm_jacobian_nd(s) -> Matrix{Float64}

The 6×6 Jacobian `A` of the non-dimensional CR3BP EOM at state `s`. Exposed separately so
the STM propagation can be validated pointwise against the reference.
"""
function cr3bp_stm_jacobian_nd(s::AbstractVector{<:Real})
    daug = zeros(42)
    aug = vcat(collect(float.(s)), vec(Matrix{Float64}(I, 6, 6)))
    cr3bp_stm_nd!(daug, aug, nothing, 0.0)
    return reshape(daug[7:42], 6, 6)
end

# ── Half-period propagator ────────────────────────────────────────────────────

"""
    propagate_half_period_nd(x0, z0, vy0; t_min, t_half_max, n_crossings, rtol, atol)
        -> (state_f, phi, t_cross)

Propagate `[x0, 0, z0, 0, vy0, 0]` (non-dimensional) to the `n_crossings`-th `y = 0`
crossing, accumulating the STM from `t = 0` all the way to that crossing.

For a period-1 halo, `n_crossings = 1` gives the half-period `T/2`. For the period-3 orbit,
`n_crossings = 3` gives half of the full three-revolution period.

Only crossings in ONE direction are counted: `y` descending for `vy0 > 0`, ascending
otherwise. That is the direction of the symmetry-plane half-period for a `z0 < 0`,
`vy0 > 0` southern halo.

`t_min` is a fixed coast used twice: once at `t = 0` to step over the degenerate `y = 0` at
the IC, and once after each crossing so the event does not immediately re-fire on the
crossing just found. It must stay small enough not to overshoot the next real crossing.

Throws if a crossing is not found before `t_half_max`.
"""
function propagate_half_period_nd(
    x0::Real,
    z0::Real,
    vy0::Real;
    t_min::Real = 0.3,
    t_half_max::Real = 15.0,
    n_crossings::Integer = 1,
    rtol::Real = 1e-10,
    atol::Real = 1e-12,
)
    aug0 = vcat([float(x0), 0.0, float(z0), 0.0, float(vy0), 0.0],
                vec(Matrix{Float64}(I, 6, 6)))

    # Segment 1: coast past the degenerate y = 0 at t = 0.
    sol1 = OrdinaryDiffEq.solve(
        ODEProblem(cr3bp_stm_nd!, aug0, (0.0, float(t_min))), Vern9();
        reltol = rtol, abstol = atol,
    )
    aug_cur = copy(sol1.u[end])
    t_cur = float(t_min)

    # Count crossings of one direction only: descending for vy0 > 0.
    direction = vy0 > 0 ? -1 : +1

    for crossing in 1:n_crossings
        g(u, t, integrator) = u[2]
        cb = direction > 0 ?
             ContinuousCallback(g, terminate!, nothing; abstol = 1e-14, reltol = 0.0) :
             ContinuousCallback(g, nothing, terminate!; abstol = 1e-14, reltol = 0.0)

        sol2 = OrdinaryDiffEq.solve(
            ODEProblem(cr3bp_stm_nd!, aug_cur, (t_cur, float(t_half_max))), Vern9();
            reltol = rtol, abstol = atol, callback = cb,
        )

        if sol2.t[end] >= float(t_half_max) * (1 - 1e-12)
            error("y=0 crossing #$(crossing)/$(n_crossings) not found " *
                  "(x0=$(x0), z0=$(z0), vy0=$(vy0)). " *
                  "Reached t=$(sol2.t[end]), final y=$(sol2.u[end][2])")
        end

        aug_cur = copy(sol2.u[end])
        t_cur = sol2.t[end]

        # Coast briefly past y = 0 so the next event fires on the SUBSEQUENT crossing.
        if crossing < n_crossings
            coast = OrdinaryDiffEq.solve(
                ODEProblem(cr3bp_stm_nd!, aug_cur, (t_cur, t_cur + float(t_min))), Vern9();
                reltol = rtol, abstol = atol,
            )
            aug_cur = copy(coast.u[end])
            t_cur += float(t_min)
        end
    end

    return aug_cur[1:6], reshape(aug_cur[7:42], 6, 6), t_cur
end

# ── Richardson (1980) 3rd-order approximation ─────────────────────────────────

"""
    richardson_ic(Az_km; northern = false) -> (x0_nd, z0_nd, vy0_nd, t_half_nd)

Third-order Lindstedt-Poincaré approximation for an L1 halo initial condition, in
non-dimensional units, suitable for seeding [`differential_corrector`](@ref).

`Az_km` is the out-of-plane amplitude (km) and is the family parameter: larger `Az` gives a
larger orbit and a different periapsis altitude. For the mission orbit (periapsis
~20–70 km) try `Az_km` in 200–800 km.

`northern = false` (default) returns the southern halo, obtained by negating `z0` and `vy0`.

Richardson (1980) §3–4; coefficient notation follows Koon et al. (2011) Ch. 2.
"""
function richardson_ic(Az_km::Real; northern::Bool = false)
    mu = MU

    # ── L1 location (nondim) ─────────────────────────────────────────────────
    xL1_km, _, _ = libration_points_x()
    xL1 = xL1_km / L_STAR
    # gamma1 = distance from Enceladus to L1 (nondim); L1 is Saturn-side of Enceladus.
    gamma1 = (1.0 - mu) - xL1

    # ── Legendre-expansion coefficients c_n ──────────────────────────────────
    cn(n::Integer) = (1.0 / gamma1^3) *
        (mu + (-1)^n * (1.0 - mu) * (gamma1^(n + 1)) / (1.0 - gamma1)^(n + 1))

    c2 = cn(2)
    c3 = cn(3)
    c4 = cn(4)

    # In-plane frequency lambda (lp) and out-of-plane frequency kappa (kp).
    lp2 = 0.5 * ((c2 - 2) + sqrt(9 * c2^2 - 8 * c2))
    lp = sqrt(lp2)
    kp = sqrt(c2)

    # ── Amplitude ratios ─────────────────────────────────────────────────────
    k  = (lp^2 + 1 + 2 * c2) / (2 * lp)      # Ax/Az
    d1 = (3 * lp^2 / k) * (k * (6 * lp^2 - 1) - 2 * lp)
    d2 = (8 * lp^2 / k) * (k * (11 * lp^2 - 1) - 2 * lp)

    a21 = (3 * c3 * (k^2 - 2)) / (4 * (1 + 2 * c2))
    a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = -(3 * c3 * lp / (4 * k * d1)) * (3 * k^3 * lp - 6 * k * (lp^2 - c2) + 4)
    a24 = -(3 * c3 * lp / (4 * k * d1)) * (2 + 3 * k * lp)
    b21 = -(3 * c3 * lp / (2 * d1)) * (3 * k * lp - 4)
    b22 = 3 * c3 * lp / d1
    d21 = -c3 / (2 * lp^2)

    a31 = -(9 * lp / (4 * d2)) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k^2)) +
          (9 * lp^2 + 1 - c2) / (2 * d2) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k^2))
    a32 = -(1 / d2) * (9 * lp * (4 * c3 * (k * a24 - b22) + k * c4) / 4.0 +
                       3.0 / 2.0 * (9 * lp^2 + 1 - c2) * (c3 * (k * b22 + d21 - 2 * a24) - c4))
    b31 = (3 / (8 * d2)) * (8 * lp * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k^2)) +
                            (9 * lp^2 + 1 + 2 * c2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k^2)))
    b32 = (1.0 / d2) * (
        9 * lp * (c3 * (k * b22 + d21 - 2 * a24) - c4) +
        (3.0 / 8.0) * (9 * lp^2 + 1 + 2 * c2) *
          (8 * lp * (c3 * (k * a24 - b22) + c4) +
           (9 * lp^2 + 1 - c2) * (c3 * (k * b22 + d21 - 2 * a24) - c4))
    )

    d31 = 3 / (64 * lp^2) * (4 * c3 * a24 + c4)
    d32 = 3 / (64 * lp^2) * (4 * c3 * (a23 - d21) + c4 * (4 + k^2))

    # ── Frequency correction Δ = Δ1·Az² + Δ2·Ax² (Koon et al. 2011 eq. 2.30) ──
    Az = Az_km / L_STAR
    Ax = k * Az

    delta1 = (3.0 / 2.0) * c3 * (2 * a21 * (k^2 - 2) - a23 * (k^2 + 2) - 2 * k * b21) -
             (3.0 / 8.0) * c4 * (3 * k^4 - 8 * k^2 + 8)
    delta2 = (3.0 / 2.0) * c3 * (k * a24 + b21 - 2 * a22) + (9.0 / 8.0) * c4 * k
    delta_freq = delta1 * Az^2 + delta2 * Ax^2

    omega_p = lp + delta_freq / (2 * lp)

    # ── Phase angle: start on the x-z plane (y = 0) ──────────────────────────
    psi = pi / 2.0

    # ── 3rd-order position and velocity at t = 0 (Richardson eqs. 22–24) ─────
    dx = (a21 * Ax^2 + a22 * Az^2
          - Ax * cos(psi)
          + (a23 * Ax^2 - a24 * Az^2) * cos(2 * psi)
          + (a31 * Ax^3 - a32 * Ax * Az^2) * cos(3 * psi))

    dz = (Az * cos(psi)
          + d21 * Ax * Az * (cos(2 * psi) - 3)
          + (d32 * Az * Ax^2 - d31 * Az^3) * cos(3 * psi))

    # At psi = pi/2: sin(psi) = 1, sin(2psi) = 0, sin(3psi) = -1.
    dvy = (lp * Ax * 1.0
           + 2 * omega_p * (a24 * Az^2 - a23 * Ax^2) * 0.0
           + 3 * omega_p * (a32 * Ax * Az^2 - a31 * Ax^3) * (-1.0))

    x0_nd  = xL1 + dx
    z0_nd  = dz    # positive for the northern halo
    vy0_nd = dvy   # positive for the northern halo (y goes positive first)

    # Southern halo: mirror in the x-z plane (z → −z, vy → −vy).
    if !northern
        z0_nd  = -z0_nd
        vy0_nd = -vy0_nd
    end

    t_half_nd = pi / omega_p

    return x0_nd, z0_nd, vy0_nd, t_half_nd
end

# ── Seed scan ─────────────────────────────────────────────────────────────────

"""
    seed_scan(x0_nd, z0_nd, vy0_range; n_vy, n_crossings, t_half_max_nd, verbose)
        -> Union{Float64, Nothing}

Scan `vy0` at fixed `(x0_nd, z0_nd)` for the value minimising `|vx_f| + |vz_f|` at the
`n_crossings`-th `y = 0` crossing. Returns the best `vy0` (non-dimensional), or `nothing`
if no value produced the required number of crossings.

`vy0_range` is `(lo, hi)`, sampled at `n_vy` equally spaced points.
"""
function seed_scan(
    x0_nd::Real,
    z0_nd::Real,
    vy0_range::Tuple{<:Real,<:Real};
    n_vy::Integer = 30,
    n_crossings::Integer = 3,
    t_half_max_nd::Real = 12.0,
    verbose::Bool = true,
)
    best_res = Inf
    best_vy0 = nothing

    for vy0 in range(float(vy0_range[1]), float(vy0_range[2]); length = n_vy)
        try
            sf, _, _ = propagate_half_period_nd(
                x0_nd, z0_nd, vy0;
                n_crossings = n_crossings, t_half_max = t_half_max_nd,
            )
            res = abs(sf[4]) + abs(sf[6])
            if res < best_res
                best_res = res
                best_vy0 = vy0
            end
        catch e
            e isa ErrorException || rethrow()
        end
    end

    if verbose && best_vy0 !== nothing
        @printf("Seed scan: best vy0=%.6f nondim (%.5f km/s), residual=%.4e\n",
                best_vy0, best_vy0 * V_STAR, best_res)
    end
    return best_vy0
end

# ── Differential corrector ────────────────────────────────────────────────────

"""
    differential_corrector(x0_nd, z0_nd, vy0_nd; t_half_max_nd, n_crossings, tol,
                           max_iter, damp, verbose) -> NamedTuple

Single-shooting differential corrector for CR3BP halo orbits (Howell 1984).

Free variables `x0` and `vy0` (non-dimensional); `z0` is fixed as the family parameter.
Targets `vx_f = 0` and `vz_f = 0` at the `n_crossings`-th `y = 0` crossing: `n_crossings = 1`
is a simple halo, `n_crossings = 3` the period-3 orbit of MacKenzie §B.2.3.

!!! warning "period_nd / period_s are NaN when n_crossings > 1"
    The full period is `2 × t_half` only for a single crossing. For more crossings
    `t_half` also contains the intermediate `t_min` coasts inserted by
    [`propagate_half_period_nd`](@ref), so doubling it overshoots — at `n_crossings = 3`
    by exactly 5/3. Rather than return that plausible-but-wrong number (as the Python
    reference does), both period fields are `NaN` for `n_crossings > 1`, so a caller gets
    a loud failure instead of silently propagating 1.67 revolutions as one orbit. Use
    `PERIOD3_PERIOD_S` for the period-3 science orbit, or `n × 2 × t_cross(n_crossings = 1)`.
    `t_half_nd` is always returned if you need to compute it yourself.

The Newton step uses the 2×2 STM sub-block (rows `vx = 4`, `vz = 6`; columns `x = 1`,
`vy = 5`) and is scaled by `damp ∈ (0, 1]` to prevent overshoot. Iteration stops when
`|vx_f| + |vz_f| < tol`, when the sub-block becomes singular (`cond > 1e12`), or at
`max_iter`.

Returns a NamedTuple with `ic_nd`, `ic` (physical, km/km/s), `period_nd`, `period_s`,
`t_half_nd`, `n_crossings`, `converged`, `iterations`, `residual`, and `error` (a message
string, empty unless propagation failed).
"""
function differential_corrector(
    x0_nd::Real,
    z0_nd::Real,
    vy0_nd::Real;
    t_half_max_nd::Real = 15.0,
    n_crossings::Integer = 1,
    tol::Real = 1e-10,
    max_iter::Integer = 50,
    damp::Real = 0.6,
    verbose::Bool = true,
)
    x0 = float(x0_nd)
    vy0 = float(vy0_nd)
    residual = Inf

    for i in 0:(max_iter - 1)
        state_f, phi, t_half = try
            propagate_half_period_nd(
                x0, z0_nd, vy0;
                n_crossings = n_crossings, t_half_max = t_half_max_nd,
            )
        catch e
            e isa ErrorException || rethrow()
            return _corrector_failure(x0, z0_nd, vy0, i, residual, n_crossings, sprint(showerror, e))
        end

        vx_f = state_f[4]
        vz_f = state_f[6]
        residual = abs(vx_f) + abs(vz_f)

        if verbose
            @printf("  iter %2d: x0=%.7f, vy0=%.7f  vx_f=%+.3e, vz_f=%+.3e, res=%.3e\n",
                    i, x0, vy0, vx_f, vz_f, residual)
        end

        if residual < tol
            ic_nd = [x0, 0.0, z0_nd, 0.0, vy0, 0.0]
            # `2 × t_half` is the true period ONLY for n_crossings == 1. For more
            # crossings, t_half includes the (n_crossings - 1) intermediate `t_min`
            # coasts that `propagate_half_period_nd` inserts to avoid re-detecting the
            # crossing it starts on, so `2 × t_half` overshoots (measured: exactly 5/3
            # too large at n_crossings = 3). The Python reference returns that inflated
            # value; we return NaN instead so a caller cannot silently propagate 1.67
            # revolutions and call it one orbit. Deliberate divergence from the
            # reference — see the 2026-08-24 halo-IC session log §5.
            # Use PERIOD3_PERIOD_S, or n × 2 × t_cross(n_crossings = 1), until the
            # general relation is derived and regression-tested.
            multi = n_crossings > 1
            return (
                ic_nd       = ic_nd,
                ic          = nondim_to_cr3bp(ic_nd),
                period_nd   = multi ? NaN : 2.0 * t_half,
                period_s    = multi ? NaN : 2.0 * t_half * T_STAR,
                t_half_nd   = t_half,
                n_crossings = Int(n_crossings),
                converged   = true,
                iterations  = i,
                residual    = residual,
                error       = "",
            )
        end

        # 2×2 Newton: free variables x0, vy0; targets vx_f = vz_f = 0.
        M = [phi[4, 1]  phi[4, 5]
             phi[6, 1]  phi[6, 5]]

        c = cond(M)
        if c > 1e12
            verbose && @printf("  WARNING: M singular (cond=%.1e), stopping.\n", c)
            break
        end

        delta = try
            -(M \ [vx_f, vz_f])
        catch e
            e isa LinearAlgebra.SingularException || rethrow()
            break
        end

        x0  += damp * delta[1]
        vy0 += damp * delta[2]
    end

    return _corrector_failure(x0, z0_nd, vy0, max_iter, residual, n_crossings, "")
end

function _corrector_failure(x0, z0, vy0, iters, residual, n_crossings, error_msg)
    ic_nd = [float(x0), 0.0, float(z0), 0.0, float(vy0), 0.0]
    return (
        ic_nd       = ic_nd,
        ic          = nondim_to_cr3bp(ic_nd),
        period_nd   = NaN,
        period_s    = NaN,
        t_half_nd   = NaN,
        n_crossings = Int(n_crossings),
        converged   = false,
        iterations  = Int(iters),
        residual    = residual,
        error       = error_msg,
    )
end

# ── Orbit characterisation ────────────────────────────────────────────────────

"""
    characterise_orbit(ic, period_s; verbose = true) -> NamedTuple

Propagate one full period from `ic` (PHYSICAL units, km / km/s) under the onboard CR3BP
model and extract the orbit's properties.

Returns `(periapsis_alt_km, apoapsis_alt_km, period_s, period_hr, jacobi, closure_km,
closure_kms)`. `closure_km` is `‖r(T) − r(0)‖` — how nearly the orbit closes — and is the
primary quality metric for a corrected IC.
"""
function characterise_orbit(ic::AbstractVector{<:Real}, period_s::Real; verbose::Bool = true)
    ic = collect(float.(ic))
    # Sample densely enough that the min/max radius over the orbit is resolved; the
    # Python reference read the solver's own step sequence, which is equivalent in the
    # limit and differs here only at the tolerance level.
    ts = range(0.0, float(period_s); length = 20_001)
    sol = propagate(cr3bp_eom!, ic, (0.0, float(period_s));
                    rtol = RTOL_ONBOARD, atol = ATOL_ONBOARD, saveat = ts)

    r_enc = [r_enceladus(u) for u in sol.u]
    peri_alt = minimum(r_enc) - R_ENCELADUS
    apo_alt  = maximum(r_enc) - R_ENCELADUS

    state_f = sol.u[end]
    closure_r = norm(state_f[1:3] - ic[1:3])
    closure_v = norm(state_f[4:6] - ic[4:6])
    jc = jacobi_constant(ic)

    if verbose
        println("\nOrbit characterisation:")
        @printf("  Periapsis alt  : %.1f km  (target: %.1f–%.1f km)\n",
                peri_alt, PERIAPSIS_ALT_MIN, PERIAPSIS_ALT_MAX)
        @printf("  Apoapsis alt   : %.1f km  (target: %.0f–%.0f km)\n",
                apo_alt, APOAPSIS_ALT_MIN, APOAPSIS_ALT_MAX)
        @printf("  Period         : %.3f hr\n", period_s / 3600.0)
        @printf("  Jacobi const   : %.6f km²/s²\n", jc)
        @printf("  Closure (pos)  : %.4f km\n", closure_r)
        @printf("  Closure (vel)  : %.2e km/s\n", closure_v)
        peri_ok = PERIAPSIS_ALT_MIN <= peri_alt <= PERIAPSIS_ALT_MAX
        apo_ok  = APOAPSIS_ALT_MIN  <= apo_alt  <= APOAPSIS_ALT_MAX
        println("  Periapsis OK   : ", peri_ok ? "✓" : "✗")
        println("  Apoapsis OK    : ", apo_ok  ? "✓" : "✗")
    end

    return (
        periapsis_alt_km = peri_alt,
        apoapsis_alt_km  = apo_alt,
        period_s         = float(period_s),
        period_hr        = period_s / 3600.0,
        jacobi           = jc,
        closure_km       = closure_r,
        closure_kms      = closure_v,
    )
end

# ── Full pipeline ─────────────────────────────────────────────────────────────

"""
    find_halo_ic(; x0_km, z0_km, vy0_km_s, northern, n_crossings, tol, max_iter, damp,
                 t_half_max_hr, verbose) -> NamedTuple

Full pipeline: seed scan → [`differential_corrector`](@ref) → [`characterise_orbit`](@ref).

Targets the period-3 L1 halo orbit for the Enceladus Orbilander (MacKenzie et al. 2020
§B.2.3): with `n_crossings = 3` the corrector propagates to the third `y = 0` crossing,
i.e. half of the ~36-hr period-3 orbit.

  - `x0_km` — IC x-coordinate (km from the barycentre). `nothing` → `x_L1 − 100 km`.
  - `z0_km` — IC z-coordinate (km); negative = southern amplitude. The family parameter.
  - `vy0_km_s` — IC y-velocity (km/s). `nothing` → found by [`seed_scan`](@ref).
  - `t_half_max_hr` — propagation ceiling in hours; 60 hr covers three loops.

Returns the corrector NamedTuple, merged with the characterisation fields when it converged.
"""
function find_halo_ic(;
    x0_km::Union{Real,Nothing} = nothing,
    z0_km::Real = -280.0,
    vy0_km_s::Union{Real,Nothing} = nothing,
    northern::Bool = false,
    n_crossings::Integer = 3,
    tol::Real = 1e-10,
    max_iter::Integer = 50,
    damp::Real = 0.7,
    t_half_max_hr::Real = 60.0,
    verbose::Bool = true,
)
    xL1_km, _, _ = libration_points_x()
    x0_km = x0_km === nothing ? xL1_km - 100.0 : float(x0_km)
    z0_km = northern ? abs(float(z0_km)) : float(z0_km)

    x0_nd = x0_km / L_STAR
    z0_nd = z0_km / L_STAR
    t_half_max_nd = t_half_max_hr * 3600.0 / T_STAR

    if verbose
        hemi = northern ? "northern" : "southern (south-polar)"
        println("=== Period-$(n_crossings) $(hemi) halo IC finder ===")
        @printf("  x0 = %.1f km (L1 offset: %+.1f km)\n", x0_km, x0_km - xL1_km)
        @printf("  z0 = %.1f km\n", z0_km)
    end

    local vy0_nd
    if vy0_km_s === nothing
        verbose && println("\n--- Seed scan over vy0 ---")
        # Southern: vy0 > 0 (orbit goes +y first); northern: vy0 < 0.
        lo = northern ? -0.3 / V_STAR : 0.001 / V_STAR
        hi = northern ? -0.001 / V_STAR : 0.3 / V_STAR
        scanned = seed_scan(
            x0_nd, z0_nd, (lo, hi);
            n_vy = 40, n_crossings = n_crossings,
            t_half_max_nd = t_half_max_nd, verbose = verbose,
        )
        if scanned === nothing
            return (converged = false, error = "Seed scan found no viable vy0")
        end
        vy0_nd = scanned
    else
        vy0_nd = float(vy0_km_s) / V_STAR
    end

    if verbose
        println("\n--- Differential corrector (Howell 1984, n_crossings=$(n_crossings)) ---")
        @printf("  Starting: vy0 = %.5f km/s\n", vy0_nd * V_STAR)
    end

    result = differential_corrector(
        x0_nd, z0_nd, vy0_nd;
        t_half_max_nd = t_half_max_nd, n_crossings = n_crossings,
        tol = tol, max_iter = max_iter, damp = damp, verbose = verbose,
    )

    if result.converged
        # `result.period_s` is NaN for n_crossings > 1 (its `2 × t_half` would include the
        # intermediate coasts — see the differential_corrector docstring), so recover the
        # true period from a single-crossing solve of the SAME converged IC and scale by
        # the crossing count. Verified at n_crossings = 3: this reproduces
        # PERIOD3_PERIOD_S (35.98811 hr vs the 35.988 hr constant).
        period_s = result.period_s
        if !isfinite(period_s)
            single = differential_corrector(
                result.ic_nd[1], result.ic_nd[3], result.ic_nd[5];
                t_half_max_nd = t_half_max_nd, n_crossings = 1,
                tol = tol, max_iter = max_iter, damp = damp, verbose = false,
            )
            single.converged || return merge(result, (
                error = "converged at n_crossings=$(n_crossings) but the single-crossing " *
                        "solve needed for the true period did not converge; period unknown",
            ))
            period_s = n_crossings * single.period_s
        end
        # `info` carries the period it was characterised over, so merging it last
        # deliberately overrides the corrector's NaN with the true period. `find_halo_ic`
        # therefore always reports a usable `period_s`; only the lower-level
        # `differential_corrector` returns NaN.
        info = characterise_orbit(result.ic, period_s; verbose = verbose)
        return merge(result, (period_s = period_s, period_nd = period_s / T_STAR), info)
    end
    return result
end