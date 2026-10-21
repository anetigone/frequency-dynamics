# Explicit stability to six independent initialization perturbations

Working derivation, 2026-10-21. This result fixes isotropic Gaussian patches, nonzero augmentation power `eta=0.1`, and unit encoder/head learning rates. It supplies a full-dimensional neighborhood of the solvable two-filter family, rather than assuming that filter reflection and equal head weights persist. It makes no claim about arbitrary covariance, augmentation, or learning-rate perturbations.

## Statement

Let `x ~ N(0,I_2)`, use independent view noises `xi_j ~ N(0,I_2/10)`, centered square features and population GUPA, and train the actual Euclidean parameters with `eta_h=eta_W=1`. In the fixed basis `(e_+,e_-)`, let

\[
\bar q_1=(1/10,1/\sqrt{200}),\qquad
\bar q_2=(1/10,-1/\sqrt{200}),\qquad
\bar w_1=\bar w_2=1/10.
\]

Since the input covariance is the identity, `q_i=h_i`. Set

\[
e=1/125000=8\times10^{-6}.
\]

Allow **each of the six real coordinates to vary independently**:

\[
|q_{ij}(0)-\bar q_{ij}|\le e,\qquad
|w_i(0)-1/10|\le e.
\tag{A1}
\]

This is an explicit cube with nonempty six-dimensional interior. It has half-width about `0.008%` relative to the `0.1` coordinates, and about `0.0113%` relative to the smaller filter coordinates. It remains a local result; it is not a high-probability random-initialization theorem.

Write

\[
\theta_0=\arctan(1/2),\quad
\theta_1=3\theta_0/4,\quad \theta_2=\theta_0/6,
\qquad d_j=\arctan\sqrt{\tan\theta_j}.
\]

For each perturbed trajectory, let `T_j` be its **first crossing** of

\[
{\alpha_1-\alpha_2\over2}=d_j,
\]

where `alpha_i` are the continuously unwrapped input-space filter angles. Both crossings exist, `0<T_1<T_2<900`, and encoder/projection ranks remain `2/1` up to `T_2`. These are common geometric sections, not common physical times.

For every original task in the paper's cone,

\[
\beta_{\psi,z}=\sqrt{1-z^2}(\cos\psi E_++\sin\psi E_-)+zE_\times,
\quad 2\theta_0/3\le\psi\le5\theta_0/6,\quad |z|\le1/2,
\]

both optimal population encoder risk and optimal population projection risk satisfy the uniform bounds

\[
R(T_1)\le R(0)-0.001958,
\qquad
R(T_2)\ge R(0)+0.014234.
\tag{A2}
\]

In particular these exceed the requested retained half-gaps `0.001675` and `0.010905`. The pretraining loss strictly decreases throughout this interval. All comparisons use the task risk at the perturbed initialization itself.

## 1. Exact angular and imbalance equations

Write

\[
q_i=\sqrt{v_i}(\cos\alpha_i,\sin\alpha_i),\quad
a={\alpha_1+\alpha_2\over2},\quad d={\alpha_1-\alpha_2\over2},\quad D=2d,
\]

\[
m={v_1+v_2\over2},\quad \gamma={v_1-v_2\over v_1+v_2},\qquad
u={w_1+w_2\over2},\quad \zeta={w_1-w_2\over w_1+w_2}.
\]

Define the physical quadratic map and its energy by

\[
F=w_1q_1q_1^\top+w_2q_2q_2^\top,
\quad V=2\|F\|_F^2,\quad
\kappa=2\Phi'(V)=-2\Gamma_{s_v}(s_vV),\qquad s_v=121/100.
\]

The actual parameter flow is

\[
\dot q_i=-4\kappa w_iFq_i,
\qquad \dot w_i=-2\kappa q_i^\top Fq_i.
\]

Until a possible first loss of positive gain, use the increasing time change `ds/dt=-4 kappa`. The energy bound below excludes that loss and closes continuation. Then

\[
{dq_i\over ds}=w_iFq_i,\qquad
{dw_i\over ds}=\tfrac12q_i^\top Fq_i.
\tag{A3}
\]

In particular the two independent channel quantities

\[
I_i=v_i-2w_i^2
\tag{A4}
\]

are exactly constant. Direct angular differentiation gives

\[
{dd\over ds}=-u^2(1-\zeta^2)m\sin D\cos D,
\qquad
{da\over ds}=u^2(1-\zeta^2)m\gamma\sin D\cos D.
\tag{A5}
\]

For `0<D<pi/2`, introduce a progress clock

\[
\sigma={1\over2}\log{\tan D(0)\over\tan D},
\qquad {d\sigma\over ds}=u^2m(1-\zeta^2).
\tag{A6}
\]

This clock tracks directional progress, not elapsed physical time. It obeys

\[
{dd\over d\sigma}=-\sin D\cos D,
\qquad {da\over d\sigma}=\gamma\sin D\cos D.
\tag{A7}
\]

The exact imbalance equations are

\[
{d\gamma\over d\sigma}
={2(1-\gamma^2)\over1-\zeta^2}
\left[2\zeta+\gamma\{\sin^2D+\zeta^2(1+\cos^2D)\}\right],
\tag{A8}
\]

\[
{d\zeta\over d\sigma}
={m\over u^2}
\left[\gamma-\zeta{(1-\gamma^2)\cos^2D\over1-\zeta^2}\right].
\tag{A9}
\]

No reflection symmetry, equal norm, or equal head entry is imposed in these equations. In particular a nonzero `gamma` rotates the average filter angle `a`; this is the symmetry-breaking motion that the proof controls.

## 2. Source bounds and a short progress interval

Let `v_*=3/200=0.015` and `L=1/10+1/sqrt(200)<0.171`. The coordinate cube implies

\[
v_*-2Le+2e^2\le v_i(0)\le v_*+2Le+2e^2,
\]

and therefore

\[
|\gamma(0)|\le {2Le\over v_*+2e^2}<23e,
\quad |\zeta(0)|\le10e,
\quad N_0:=|\gamma(0)|+|\zeta(0)|<33e.
\tag{A10}
\]

An angle perturbation is at most

\[
\arcsin\left({\sqrt2e\over\sqrt{0.015}}\right)<12e.
\]

Put `d_0=atan(1/sqrt(2))`, the unperturbed half-angle. Then

\[
|a(0)|<12e,\qquad |d(0)-d_0|<12e.
\tag{A11}
\]

Also

\[
-0.005006<I_i<-0.004994<0.
\tag{A12}
\]

For example, `(A12)` follows from
`-0.005-(2L+0.4)e <= I_i <= -0.005+(2L+0.4)e`.

The unperturbed progress to the two sections is exactly

\[
\bar\sigma_j={1\over2}\log
\left[{\sqrt2(1-\tan\theta_j)\over\sqrt{\tan\theta_j}}\right].
\]

These values lie near `0.201921161342210` and `0.772590621801085`. The initial `D` perturbation is at most `24e`. On that initial angular interval,

\[
\left|{d\over dD}\tfrac12\log\tan D\right|
={1\over\sin2D}<{8\over5},
\]

because `sin(2 D_0)=4 sqrt(2)/9` and its possible decrease is at most `48e`. Consequently the actual section clocks satisfy

\[
\sigma_1<0.203,\qquad \sigma_2<0.774.
\tag{A13}
\]

The exact-rational checker described below verifies the displayed transcendental comparisons with Taylor remainder enclosures. The constants do not come from integrating a perturbed trajectory.

## 3. Transverse control without a physical-time Gronwall factor

Bootstrap `|gamma|,|zeta|<=0.01`. Averaging the two invariants yields

\[
\bar I=m-2u^2(1+\zeta^2)<0,
\quad {m\over u^2}<2(1+\zeta^2).
\tag{A14}
\]

Let `N=|gamma|+|zeta|`. Retain the stabilizing term in `(A9)` when differentiating `|zeta|`. The upper right derivative satisfies

\[
D^+|\zeta|\le2(1+10^{-4})|\gamma|.
\]

Equation `(A8)` gives

\[
D^+|\gamma|
\le {2(1+2\times10^{-4})\over1-10^{-4}}|\gamma|
+{4\over1-10^{-4}}|\zeta|.
\]

Thus

\[
D^+N\le4.002N,\qquad N(\sigma)\le N_0e^{4.002\sigma}.
\tag{A15}
\]

The maximal coefficient obtained directly is below `4.000801`; `4.002` is conservative. In view of `(A13)`,

\[
E_1=e^{4.002(0.203)}<2.254,\qquad
E_2=e^{4.002(0.774)}<22.144,
\]

so `N<33e E_2<0.005847<0.01`. This closes the transverse bootstrap with room to spare. The amplification is below `22.144`; no exponential in the hundreds of units of physical time appears.

Furthermore `(A7)` and `|sin D cos D|<=1/2` imply the useful integrated rotation bound

\[
|a(\sigma)|
\le12e+{33e\over8.004}\big(e^{4.002\sigma}-1\big).
\tag{A16}
\]

At the two sections the source-fixed estimates are therefore

\[
\begin{array}{c|cc}
 & \sigma_1 & \sigma_2\\ \hline
N/e & <74.382 & <730.752\\
|a|/e & <17.2 & <99.2.
\end{array}
\tag{A17}
\]

## 4. Encoder and projection geometry at the common sections

Let `R_a` denote the two-dimensional rotation by `a`. Independently of channel lengths or head weights, the encoder row space is the rotation by `R_a` of the span of

\[
\begin{pmatrix}\cos^2d&0\\0&\sin^2d\end{pmatrix},
\qquad
\begin{pmatrix}0&1\\1&0\end{pmatrix}.
\]

The encoder's unit normal in symmetric-matrix space is the normalized rotation of `diag(sin^2 d,-cos^2 d)`. Rotation has infinitesimal norm at most `2`, so at a fixed `d`,

\[
\|P_{\rm enc}(a,d)-P_{\rm enc}(0,d)\|_{\rm op}\le2|a|.
\tag{A18}
\]

The physical projection matrix, up to a strictly positive scale, is

\[
R_a\begin{pmatrix}
\cos^2d&\delta_{\rm mix}\sin d\cos d\\
\delta_{\rm mix}\sin d\cos d&\sin^2d
\end{pmatrix}R_a^\top,
\qquad \delta_{\rm mix}={\gamma+\zeta\over1+\gamma\zeta}.
\tag{A19}
\]

The off-diagonal perturbation in `(A19)` is Frobenius orthogonal to its diagonal part. Its relative norm is at most `|delta_mix|`, so their one-dimensional projector distance is at most `|delta_mix|`. Rotating the diagonal part changes its normalized vector at speed at most `sqrt(2)`. Therefore

\[
\|P_{\rm proj}(a,d,\delta_{\rm mix})-P_{\rm proj}(0,d,0)\|_{\rm op}
\le\sqrt2|a|+|\delta_{\rm mix}|.
\tag{A20}
\]

Throughout the bootstrap, `|delta_mix|<=N/(1-10^-4)<1.001N`.

At the source the perturbed `d(0)` must also be compared with `d_0`; it cannot be omitted. The diagonal direction angle is `theta(d)=atan(tan^2 d)` and has derivative at most `2` on `0<d<pi/4`. Equations `(A10)--(A11)` consequently give

\[
\|P_{\rm enc}(0)-\bar P_{\rm enc}(0)\|_{\rm op}<48e,
\qquad
\|P_{\rm proj}(0)-\bar P_{\rm proj}(0)\|_{\rm op}<75e.
\tag{A21}
\]

At the sections `d=d_j` exactly, so no section-angle error is present. From `(A17)--(A20)`,

\[
\begin{array}{c|cc}
 & T_1 & T_2\\ \hline
\|P_{\rm enc}-\bar P_{\rm enc}\|_{\rm op}/e & <34.4 & <198.4\\
\|P_{\rm proj}-\bar P_{\rm proj}\|_{\rm op}/e & <99 & <872.
\end{array}
\tag{A22}
\]

For a unit scalar task, `R(P)=1-beta^T P beta`, so each projector bound is also a task-risk bound, uniformly over all unit tasks. Subtracting the source comparison from the section comparison gives errors in the **risk changes**, for either representation, at most

\[
174e=0.001392\quad\text{at }T_1,
\qquad947e=0.007576\quad\text{at }T_2.
\tag{A23}
\]

The unperturbed construction guarantees improvement greater than `0.00335` and excess above initialization greater than `0.02181` on the complete original task cone. Subtract `(A23)` to obtain `(A2)`. Encoder-specific comparison errors are considerably smaller; the common bounds use the projection, which is the more sensitive object.

## 5. Bounded parameters, rank preservation, and physical reachability

The previous argument must not assume that the trajectory already reaches the sections. The required continuation follows from a first-exit argument in the progress clock.

The positive head weights increase under `(A3)` because `F` is PSD and each `q_i` contributes a positive term to `q_i^T F q_i`. The two squared filter norms increase as well. Direct differentiation yields

\[
{d\log m\over d\sigma}
=\frac{2[(1+\zeta^2)(1+\gamma^2)+4\gamma\zeta]}{1-\zeta^2}
+2(1-\gamma^2)\cos^2D
<4.002
\tag{A24}
\]

on the same bootstrap region. Hence, before the second section,

\[
m\le(0.015+0.342e+2e^2)E_2<0.333,
\]

\[
u^2={m-\bar I\over2(1+\zeta^2)}<0.17.
\tag{A25}
\]

All six parameters remain bounded, and their positive norms stay bounded away from zero. The transverse bound `(A15)` prevents exit through `|gamma|=0.01` or `|zeta|=0.01`. Equation `(A6)` gives the explicit monotone angle relation `tan D=tan D(0) exp(-2 sigma)`. Thus continuation reaches each prescribed geometric section in finite progress time and no earlier escape invalidates the bounds.

While `0<D<pi/2`, the filters are linearly independent. Their two outer products are therefore linearly independent, giving encoder rank two. Positive heads make `F` nonzero, giving projection rank one. This proves the asserted ranks throughout the comparison interval.

The energy bound follows directly from the positive matrix `F` and its trace:

\[
\operatorname{tr}F=2um(1+\gamma\zeta),\qquad
V=2\|F\|_F^2\le8u^2m^2(1+\gamma\zeta)^2
<0.151<0.16.
\tag{A26}
\]

This uses `u^2`, not an individual-head upper bound. At the fixed view factor `s_v=121/100`, the energy bound implies `Gamma_{s_v}(s_vV)>29/40` (the rational lower bound exceeds `0.7340497`). Thus positive gain cannot be lost. Since `-4 kappa=8 Gamma_{s_v}(s_vV)>5.8` and `u,m` increase,

\[
\begin{split}
{d\sigma\over dt}
&=(-4\kappa)u^2m(1-\zeta^2)\\
&>5.8(0.1-e)^2(0.015-0.342e+2e^2)(1-10^{-4})
>0.000869.
\end{split}
\tag{A27}
\]

Consequently `T_2<0.774/0.000869<900` (the slightly sharper checked value is below `890.049`). Finally, every head gradient `2 kappa q_i^T F q_i` is nonzero. Unit-rate Euclidean gradient flow therefore satisfies `dL/dt=-||grad L||^2<0` throughout.

## 6. Reproduction and evidential status

Run

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/theory/asymmetric_stability_constants.py
```

The script uses exact `Fraction` interval endpoints with outward rational rounding and explicit Taylor remainder bounds for the few transcendental constants. It checks the source section clocks, amplification bounds, `75/99/872` projector constants, retained gaps, energy bound, and physical-time bound. It performs **no ODE integration, random sampling, data download, or training**. Floating-point conversion is confined to display. Its interval endpoints and assertions support the constant arithmetic in the proof; they do not replace the algebraic derivation.

An independent numerical gradient/polar/projector implementation check is being maintained separately in `asymmetric_geometry_check.py`. Such a floating-point check can detect implementation and factor mistakes but is not the stability proof.

The substantive upgrade is an explicit neighborhood allowing filter directions, filter lengths, and head imbalance to break symmetry independently while retaining fixed encoder and projection risk gaps. The neighborhood is still local and the distribution/augmentation/rates remain fixed. This argument does not show that changing learning-rate ratio changes the directional path off the symmetric manifold, and it should not be used to claim that stronger rate-bifurcation conclusion.
