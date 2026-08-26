# Figure porting references — Python originals, DO NOT RUN

**These scripts do not run.** They import `python-legacy/`, which was deleted in Session 5.
They are kept as the **specification** for a Makie port: the plotting logic, panel layout,
and data pipeline are the thing worth preserving, not the matplotlib calls.

Deferred to a dedicated figures session, by user decision, sequenced **after**
`:position_altitude` lands — so results figures get rendered once with correct numbers
rather than twice.

## The four to port (Category 1 — data-driven physics figures)

| script | figure | why it's worth porting |
|---|---|---|
| `plot_orbit.py` (350 ln) | `trajectory_rotating_frame`, `figures/rollout_3d.gif` | The orbit itself — the figure every talk needs |
| `plot_rollout_3d.py` | 3-D rollout animation | Same data, 3-D view |
| `divergence_saturn_j2.py` (105 ln) | `j2_divergence`, `divergence_saturn_j2` | Truth-vs-onboard model gap; central to the project premise |
| `plot_burn_geometry.py` (133 ln) | `burn_geometry` | The cross-track burn geometry finding |
| `plot_meeting_figures.py` (198 ln) | `meeting_results`, `meeting_pipeline` (+ dark variants) | The measured results |
| `plot_pomdp_findings.py` (213 ln) | `pomdp_findings` (+ dark) | POMDP findings summary |

`plot_orbit.py` and `divergence_saturn_j2.py` depend only on propagation, so they port
independently of any formulation decision. `plot_meeting_figures.py` and
`plot_pomdp_findings.py` render measured results and should be ported **last**.

## Deliberately NOT kept

- **`plot_policy_table.py`, `plot_reward_figure.py`, `plot_pomdp_deck_diagrams.py`** — these
  render `artifacts/tables.json`, the reward function, and the state machine. Session 7
  regenerates `tables.json` and revisits band values and `cov` gating, so porting them now
  means porting them twice. Recoverable from git history.
- **`mpc_feasibility.py`, `mpc_planner_ablation.py`, `pomdp_rollout_feasibility.py`,
  `plot_stationkeeping.py`, `plot_two_burn_damping.py`, `plot_rollout_trace.py`** —
  exploratory, and several predate findings that overturned their conclusions.

## Already-rendered output is safe

`figures/` still holds the rendered PDF + SVG + PNG for all of the above from prior
sessions. Deleting `python-legacy/` removed the ability to *re-render*, not the figures.
`figures/` and `*.png` / `*.svg` are gitignored, so those live only in the working tree.

## When porting, follow the project figure conventions

From the global `CLAUDE.md`:

- **Sentence case** labels ("Miss distance at TCA"), never title case; never show raw
  underscore field names.
- Final figures get **PDF + SVG** as well as PNG.
- Fit an **8.5×11 page**; fonts should read ~11 pt at placed size.
- **No overlapping elements** — legends must not sit on data.
- **Light/dark background toggle** via a theme flag plus `transparent` save; do not bake in
  foreground colors that only read on one background.
- **Computer Modern serif** typography.