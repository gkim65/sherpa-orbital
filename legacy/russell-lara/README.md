# Russell & Lara (2009) Fig. 9 reproduction — FROZEN

**Status: frozen, not maintained.** Standalone alternative-dynamics study. It is *not* part
of the POMDP stationkeeping pipeline and was deliberately **not** ported to Julia.

> Russell, R. P., & Lara, M. (2009). "On the design of an Enceladus science orbit."
> *Acta Astronautica* **65**, 27–39.

## What this is

An implementation of Russell & Lara's **Hill's problem + non-spherical gravity** model
(J2/J3 zonal harmonics), used to reproduce their Fig. 9 — the south-pole-grazing L1 halo —
and to check their claimed altitude band and orbital-element history.

This is a *different dynamical model* from the project's own CR3BP / CR3BP+J2: Hill's
problem is the Enceladus-centred limit, whereas the project works in the barycentric
Saturn–Enceladus rotating frame. Per `CLAUDE.md` the two are kept separate on purpose.

## What it showed

Two results, and the second is the one that mattered to the project.

**1. The Hill model holds the altitude band.** Reproduced here:

```
[Hill]   1-period closure altitude band: 23.9..1053.2 km
[Hill]   osc a 1165..1283 km, e 0.124..0.775, i 87.1..118.5 deg
```

Osculating elements match Russell-Lara's Table 2 (a = 1195.7 km, e = 0.756) under the
rotating-frame velocity convention.

**2. Converting their IC into our CR3BP+EncJ2 truth model, the orbit impacts.** Last
recorded output of the comparison panel, before it was removed (see below):

```
[CR3BP]  impacts within 3 periods (~36 hr): True
```

This echoes Russell-Lara's own decay finding and **fed the project's
"instability-dominated" conclusion** — that the failure mode on this orbit family is
dynamical instability rather than the truth/onboard model gap. See
`docs/session-log/2026-06-22*.md`.

## How to run

Needs `numpy`, `scipy`, `matplotlib`. No project dependency; this directory is
self-contained.

```sh
cd legacy/russell-lara
python reproduce_rl_fig9.py     # writes rl_fig9.png alongside the script
```

Figure panels: (a) x–z trajectory, (b) x–y top-down, (c) osculating elements over one
period, (d) Hill altitude history over 3 periods.

## Known limits — read before quoting any number from this

- **One-period closure is only ~17 km,** not sub-km. The orbit's *character* (period,
  altitude band, shape) reproduces; its precise closure does not.
- **C22 is switched off.** Russell-Lara's C22 phase convention is underspecified in the
  paper and could not be matched to sub-km closure. Only the dominant, unambiguous J2+J3
  zonal field is used. This is the main reason for the 17 km figure.
- Elements use the **rotating-frame** velocity convention. That is what reproduces their
  Table 2; an inertial-velocity convention gives different numbers.
- The 3-period altitude history dips **below zero** (`-33.0 km` minimum), i.e. the Hill
  orbit also impacts by 3 periods. Only the 1-period band is the reproduced claim.

## What changed when this was frozen (Session 5)

`python-legacy/` was deleted when the pipeline became 100% Julia. Consequences:

- **The CR3BP-vs-Hill comparison panel was removed** from `reproduce_rl_fig9.py`. It
  imported `src.constants`, `src.dynamics.cr3bp`, and `src.dynamics.cr3bp_j2`, all of which
  are gone. The `[CR3BP] impacts within 3 periods: True` line quoted above is the **last
  recorded output** and is not regenerable from this directory.
- Panel (d) is now a Hill-only altitude history.
- `hill_nonspherical.py` and `constants_russell_lara.py` moved unchanged except for their
  import paths (`from src.constants_russell_lara` → `from constants_russell_lara`).
- The removed `rl_to_barycentric_cr3bp()` converter went with the panel. It is recoverable
  from git history if the comparison is ever wanted again — it would need re-porting against
  the Julia `cr3bp_j2_eom!`.

Verified running after the trim; output reproduces the numbers above.
