# Origin-only finite-window risk certificates on saved population trajectories

Date: 2026-10-21. Scope: a feasibility calculation, not a new training campaign or a novelty claim.

## What was checked

The existing `experiments/results/theory/mechanism_profiles.csv` contains the three matched-function square-model trajectories with scale parameters c = 1/2, 1, 3/2. Their common effective initial map is A = (0.02432, 0.02432); their initial excess risk for beta = e1 is 1/2. The unscaled state is

\[
w=(1/25,9/25),\quad
B=\begin{pmatrix}8/25&4/125\\4/125&8/125\end{pmatrix},\quad g=(1,2).
\]

At scale c the initial parameters are w/c^2 and c^2 B. In particular **c = 1 is the same center as the primary fixed-gap certificate**, rather than an unrelated initialization. The noise power is eta = 1/5, so s = 36/25. Both actual parameter learning rates are one in the scaled coordinates.

The new directly runnable script is `experiments/code/theory/task_motion_check.py`. It imports the existing 180-bit outward-rounded interval operations and Taylor-jet arithmetic, computes every prediction from the rational initial state, then opens the saved CSV only for validation. It does not integrate an ODE, train a model, download data, or inspect future derivatives. All numeric results are in `experiments/results/theory/task_motion_check.json`.

The calculation is a retrospective check of a forward certificate. The trajectories and scale choices existed before this analysis; this is not a newly preregistered experiment.

## A usable sufficient condition for persistence and amplitude

Write x = (w1,w2,B11,B12,B21,B22,g1,g2). For a = B^T w, V = ||a||^2,

\[
K=B^TB+\operatorname{diag}\!\left(4g_k\sum_cB_{ck}w_c^2\right),
\qquad H=a^TKa,
\]

and Gamma = s gamma(sV) - (s-1), with gamma(q) = (q+2)/(2(q+1)^2), the physical-time vector field is

\[
\dot w_c=2\Gamma(Ba)_c,\qquad
\dot B_{ck}=8\Gamma g_k B_{ck}w_ca_k.
\]

Also \(\dot\ell=4\Gamma H/V\), for \(\ell=\log(V/V_0)\). The label direction enters only the risk \(R=a_2^2/V\); the vector field is unsupervised.

For a positive initial state x0 choose a forward box D whose six moving coordinates range from x0_i to (1+r)x0_i, leaving g fixed. Suppose interval arithmetic proves F_i(D) > 0, with upper endpoints M_i. Then

\[
\tau_{\rm box}=0.9\min_{i\le6}
\frac{D_i^{\rm upper}-x_{0,i}^{\rm upper}}{M_i}
\]

is a valid lower bound on the first exit time: before that time no upper face can be reached; the strictly positive field excludes an exit through a lower face. Positivity of the rational denominators on the entire box ensures local smoothness, and boundedness on the closed box gives continuation to that time. Fixed g coordinates need no strict face slack.

Using Taylor jets with x in D gives an interval [q_minus,q_plus] containing the **Lie second derivative** \(D^2_FR(x)=D(DR\,F)(x)F(x)\), rather than a Hessian-only expression. If the initial derivative lies in [d_minus,d_plus], then throughout this box-valid time,

\[
d_-t+\tfrac12 q_-t^2\le R(t)-R(0)
\le d_+t+\tfrac12q_+t^2.
\]

Let m be the distance of the initial derivative interval from zero and M = max(|q_minus|,|q_plus|). The script takes

\[
\tau=\min\{\tau_{\rm box},m/(2M)\},
\]

rounded downward to a rational multiple of 10^-12 and then rechecked by exact interval inequalities. On [0,tau] the risk derivative retains its initial sign with magnitude at least m/2. Thus this calculation supplies both a time window and a nonzero amplitude, with all constants fixed at the origin. It is the usual second-order certificate made concrete and nonempty here; the Taylor inequality itself is not a new theorem.

The finite list of relative radii {1/100,1/20,1/10,1/5} is declared before reading the validation curves. The radius maximizing certified duration is selected from origin-only constants, not future fit. There is no sweep over observed future risk or reoptimization against an endpoint.

## Physical-time windows and frozen future samples

| Scale c | Selected radius | Initial dR/dt | Guaranteed window tau | Guaranteed change at tau | Saved future samples in window |
|---|---:|---:|---:|---:|---:|
| 1/2 | 1/20 | +0.18568384 | 0.06629667 | [+0.01011456,+0.01481762] | 2 |
| 1 | 1/20 | -0.04999180 | 0.09112778 | [-0.00539548,-0.00343713] | 1 |
| 3/2 | 1/10 | -0.47452536 | 0.03072008 | [-0.01661182,-0.01093309] | 2 |

Every saved future sample in these windows lies inside its amplitude envelope. These five records are deterministic samples from three trajectories, not five independent trials. For example, the c = 1 trajectory at its first stored nonzero section (physical t = 0.04581698) has predicted change [-0.00250277,-0.00200773] and observed change -0.00228001. This is a finite future interval, not a sign check after an arbitrarily small numerical step.

Bounds displayed here are conservatively rounded for prose; the script performs its actual checks with outward-rounded intervals. The existing CSV records are floating-point RK4 approximations; agreement with them is numerical validation, not a new validated integration of the entire curve.

## Equal-energy first sections: a speed-independent finite split

On the same physical box the script also bounds \(0<v_-\le\dot\ell\le v_+\). Therefore the first hit of section ell_star = 1/50 occurs by ell_star/v_minus, provided this is at most tau. Its first hitting time lies in [ell_star/v_plus,ell_star/v_minus]. All three physical boxes pass this check:

| c | Bound on d ell/dt | Bound on first physical hit of ell = 0.02 | Predicted change at first hit | Saved future change |
|---|---:|---:|---:|---:|
| 1/2 | [0.57305115,0.98029632] | [0.02040199,0.03490090] | [+0.00317983,+0.00717542] | +0.00532639 |
| 1 | [0.35785826,0.59508955] | [0.03360838,0.05588805] | [-0.00310983,-0.00125944] | -0.00228001 |
| 3/2 | [0.96667022,2.52784258] | [0.00791188,0.02068958] | [-0.01074049,-0.00210136] | -0.00664743 |

Each saved first-section time and risk change lies inside its prediction. The section and every bound are computed before opening the CSV. Since the initial function is shared and the energy section is shared, opposite nonzero changes cannot be accounted for by a positive time reparameterization. The c = 1/2 versus c = 1 pair has a guaranteed separation greater than 0.004439 in excess risk at that first equal-energy section.

Computing second derivatives directly in the log-energy clock is much more conservative: the certified durations are only 0.01289038, 0.01080879, and 0.02101264. Only the last directly covers the CSV spacing of 0.02. The first-section result above uses the physical boxes and rigorous lower bounds on d ell/dt; it does not hide the failure of those direct log-energy bounds.

## Common normalized learning-rate budget

The scaling identity maps the unit-rate scaled state c to the same unscaled initial parameters with rates (eta_h,eta_W) = (c^-2,c^4). Normalize their sum to one and let b denote elapsed time under that convention. Then the corresponding scaled physical time is b/(c^-2+c^4). At the common budget b = 1/10 all three times remain inside their certified windows:

| c | Normalized eta_h | Normalized eta_W | Certified risk change at b = 0.1 |
|---|---:|---:|---:|
| 1/2 | 64/65 | 1/65 | [+0.00426799,+0.00491635] |
| 1 | 1/2 | 1/2 | [-0.00275243,-0.00216286] |
| 3/2 | 64/793 | 729/793 | [-0.00932768,-0.00734348] |

These are the same initial parameters, same initial function, same sum of rates, and the same elapsed time; they give a finite-amplitude rate-ratio split. The first versus second row has guaranteed risk separation exceeding 0.0064308. A sum-of-rates convention is **not** a parameter path-length budget or a bound on total gradient norm, so this table must not be described as such. The first-section comparison above avoids any reliance on that convention.

## What this does and does not establish

This feasibility test adds a usable, origin-only finite-window and finite-amplitude statement for a previously stored construction family; it connects the local rate-ratio sign mechanism to future states on the primary certificate center. The opposite signs persist on nonzero certified intervals and are confirmed against saved later records. It removes the concern that this check only differentiates a risk formula and checks an infinitesimal step.

It does **not** by itself establish the desired general structural contribution. It is still a two-dimensional, single-output positive construction. The amplitude windows are modest and short compared with the complete rebound trajectory (the c = 1 primary path reaches ell = 6 at physical time about 5.6). No claim about finite-sample OLS generalization, stochastic gradient perturbations, or natural-image failure follows. The code has not used CIFAR or DTD at all. Extension to general moment features needs a separate structural theorem and separately measurable bounds; it cannot be inferred from this table.

The next defensible paper use is a compact nonvacuity example supporting a general motion theorem, with the full interval calculation in an appendix. Promoting it alone to the central contribution would repeat the existing special-case limitation.

Reproduce from the workspace root:

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/theory/task_motion_check.py
```
