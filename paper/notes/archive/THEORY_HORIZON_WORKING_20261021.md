# Task cones, finite update budgets, and usable prediction windows

Working derivation, 2026-10-21. These statements concern population linear-probe square risk. They do not establish novelty, finite-sample OLS generalization, or a useful natural-image prediction window. The constants below are explicit but can be conservative. A numerically positive window alone is not a substantive success criterion.

## 1. Setting and notation

Let the centered local patch second moment be vectorized by an isometric symmetric vectorization, and whiten on the support of its covariance. Write the resulting feature as `u`, with `E[u u^T]=I`. A local square encoder followed by a linear head has the exact form

\[
B_c(h)=\operatorname{svec}(h_ch_c^\top)^\top\Sigma^{1/2},\qquad A(\theta)=WB(h),\qquad\theta=(h,W).
\]

Here covariance support coordinates are implicit. No independent-frequency or Gaussian-input approximation is used. For centered, possibly multi-output targets, let

\[
C=\mathbb E[uY^\top],\qquad T=CC^\top\succeq0.
\]

The population risk is `R(A)=tr Cov(Y)-tr(T P_A)`. Throughout the differential statements, `A` has full row rank. Constant-rank variants are possible, but an arbitrary perturbation of a rank-deficient matrix need not stay on its rank stratum. A singular-value lower bound is therefore an actual hypothesis, not a numerical bookkeeping detail.

Put `P=P_A`, `Q=I-P`. For an effective update `Z`, define its projector tangent

\[
\mathcal D_A[Z]=A^\dagger ZQ+QZ^\top(A^\dagger)^\top.
\]

Only the row-space-normal part `ZQ` enters. In particular, a large `Z` can have zero task effect. If `U,N` are orthonormal bases of the row and normal spaces, every such tangent has the form

\[
D=UK^\top N^\top+NKU^\top,\qquad K=N^\top DU.
\]

The instantaneous risk contribution is `-tr(TD)`.

## 2. A structural, task-uniform competition criterion

**Proposition 1 (complete criterion for two blocks).** Let `D_1,D_2` be two symmetric projector tangents at the same `P`. There is no rank-one task `T=beta beta^T` for which their two risk contributions have strictly opposite signs if and only if either one tangent is zero, or

\[
D_2=\lambda D_1\quad\text{for some }\lambda>0.
\]

Consequently, when both tangents are nonzero and fail to be positively proportional, an open family of linear tasks exhibits competition. Conversely, positive proportionality excludes competition for every PSD multi-output task matrix as well.

**Proof.** Write the tangents with cross blocks `K_1,K_2`. For `beta=Ux+Ny`,

\[
\beta^\top D_g\beta=2y^\top K_gx.
\]

No competition is equivalent to

\[
(y^\top K_1x)(y^\top K_2x)\ge0\quad\text{for every }x,y.
\]

Fix `x` with `K_1x != 0`. Two linear functionals of `y` have a nonnegative product everywhere only when their coefficient vectors are nonnegatively collinear. Thus `K_2x=lambda(x) K_1x`, `lambda(x)>=0`. If `z` lies in the kernel of `K_1`, apply this observation to `x+t z` for every real `t`. The vector `K_2x+t K_2z` must remain on one fixed nonnegative ray, which forces `K_2z=0`. Hence `K_2=F K_1` for a well-defined linear map `F` on `im K_1`. Every vector of this image is an eigenvector of `F` with nonnegative eigenvalue. Applying linearity to sums of two independent vectors shows that all eigenvalues agree; the one-dimensional case is immediate. Therefore `K_2=lambda K_1`, `lambda>=0`. If both tangents are nonzero, `lambda>0`. The reverse implication is immediate. A strict inequality persists under small perturbations of the task. QED.

This is a statement about the geometry of the actual update tangents, not merely the existence of two coefficient signs after choosing a task. It does not claim that a given class-label task belongs to the competing family.

**Explicit task neighborhood.** Suppose `||beta_0||=1`, `q_g(beta)=-beta^T D_g beta`, and the two signs at `beta_0` are opposite with `min_g |q_g(beta_0)|=m>0`. Set `d=max_g ||D_g||_op`. Every unit task satisfying

\[
\|\beta-\beta_0\|\le {m\over4d}
\]

retains both signs with margin at least `m/2`, because

\[
|\beta^\top D_g\beta-\beta_0^\top D_g\beta_0|\le2d\|\beta-\beta_0\|.
\]

This supplies an explicit task family. Its probability under a chosen initialization/task distribution is a separate calculation.

## 3. A finite-amplitude bound tied to task geometry

**Proposition 2 (exact leakage decomposition).** Let `P_1` have the same rank as `P`. Put

\[
s={1\over2}\|P_1-P\|_F^2,
\quad F=P(I-P_1)P,\quad E=QP_1Q,\quad K=QP_1P.
\]

Then `F,E` are PSD, `tr F=tr E=s`, `||K||_F<=sqrt(s)`, and

\[
R(P_1)-R(P)
=\operatorname{tr}(FPTP)-\operatorname{tr}(EQTQ)
-2\operatorname{tr}(KPTQ).
\]

Consequently,

\[
|R(P_1)-R(P)|\le2\|QTP\|_F\sqrt{s}+\|T\|_{\rm op}s.\tag{1}
\]

**Proof.** Decompose `P_1-P` into its `P/Q` blocks. Equal ranks give the equal traces. Principal angles give `s=sum_i sin^2 theta_i` and `||K||_F^2=sum_i sin^2 theta_i cos^2 theta_i<=s`. The two diagonal trace terms are nonnegative and individually bounded by `||T||_op s`, so the absolute value of their difference is bounded by the same quantity, rather than twice that quantity. Cauchy--Schwarz bounds the cross term. QED.

There are two distinct reasons a moving representation may barely change a task: little subspace motion (`s` small), and little task mass across the current row/normal split (`||QTP||` small). Feature variance concentration alone controls neither quantity.

If `QTP=0`, every fixed-rank risk change is second order in subspace distance. This is not exact risk conservation at finite distance. More precisely, if

\[
\lambda_{\min}(PTP|_{\operatorname{row}P})
-\lambda_{\max}(QTQ|_{\operatorname{row}Q})\ge\delta>0,
\]

then every equal-rank departure satisfies `R(P_1)-R(P)>=delta s`. Reversing the separated spectral ordering gives `R(P_1)-R(P)<=-delta s`. These follow directly from the exact decomposition. Thus a zero first derivative can coexist with a predictable, one-sided quadratic effect.

**Corollary (fixed parameter update budget).** Suppose a parameter path remains in a region where

\[
\sigma_{\min}(A)\ge\sigma>0,\qquad \|DA(\theta)[v]\|_F\le J\|v\|.
\]

Let its cumulative block path budget be

\[
b=\int\big(\|\dot h(t)\|_F+\|\dot W(t)\|_F\big)dt.
\]

Then, with all task blocks evaluated at the initial state,

\[
|R(b)-R(0)|
\le 2\|Q_0TP_0\|_F {Jb\over\sigma}
+\|T\|_{\rm op}\left({Jb\over\sigma}\right)^2.\tag{2}
\]

Indeed, `||dot P||_F=sqrt(2)||A^dagger dot A Q||_F`, so `sqrt(s)<=Jb/sigma`. This bound is valid for every layerwise rate schedule within the same tube and total path budget. It is a priori if the tube bounds and task moment are fixed before observing future updates. It becomes an a posteriori envelope if those constants are estimated from the future path.

## 4. Finite-budget comparison and persistence of competing signs

It is important to define the budget correctly. Holding the sum of two learning rates fixed does not hold actual parameter displacement fixed.

For genuine block path budget, define normalized negative gradient directions

\[
u_h(\theta)=(-g_h/\|g_h\|,0),\qquad
u_W(\theta)=(0,-g_W/\|g_W\|),
\]

and allocate a fraction `alpha` to the first block:

\[
{d\theta\over db}=F_\alpha(\theta)
=\alpha u_h(\theta)+(1-\alpha)u_W(\theta),\quad0\le\alpha\le1.
\]

The sum of the two block speeds is exactly one. Ordinary gradient flow at physical rates `eta_h,eta_W` has this form after the clock change

\[
db=(\eta_h\|g_h\|+\eta_W\|g_W\|)dt,\qquad
\alpha(\theta)={\eta_h\|g_h\|\over\eta_h\|g_h\|+\eta_W\|g_W\|}.
\]

A constant `alpha` therefore generally corresponds to state-dependent physical rate ratios. A statement comparing constant physical ratios must allow this induced allocation to vary or instead use the learning-rate clock described below.

Assume on a convex ball of radius `rho` around `theta_0` that

\[
\|\nabla R\|\le G_R,\qquad \|\nabla^2R\|\le H_R,\qquad
\operatorname{Lip}(u_g)\le K.
\]

Put

\[
q_g=\langle\nabla R(\theta_0),u_g(\theta_0)\rangle,
\qquad L_q=H_R+G_RK.
\]

**Proposition 3 (a priori budget envelope).** Every measurable allocation schedule that stays in the ball satisfies, for `b<=rho`,

\[
b\min(q_h,q_W)-\tfrac12L_qb^2
\le R(b)-R(0)
\le b\max(q_h,q_W)+\tfrac12L_qb^2.\tag{3}
\]

For a constant allocation, `q_alpha=alpha q_h+(1-alpha)q_W`, and

\[
|R_\alpha(b)-R(0)-bq_\alpha|\le\tfrac12L_qb^2.\tag{4}
\]

In particular, any two schedules have risk difference bounded by

\[
|R_1(b)-R_2(b)|\le b|q_h-q_W|+L_qb^2.\tag{5}
\]

For two constant allocations the tighter center is

\[
\left|R_\alpha(b)-R_{\alpha'}(b)
-b(\alpha-\alpha')(q_h-q_W)\right|\le L_qb^2.\tag{6}
\]

**Proof.** The block-orthogonal normalized directions have `||F_alpha||<=1`, so `||theta(s)-theta_0||<=s`. The function `theta -> <grad R,u_g>` is `L_q`-Lipschitz by the product rule. Its change is at most `L_q s`; integrate the convex combination of the two contributions. The two-trajectory bounds follow by subtraction. QED.

**Corollary (nonzero sustained bifurcation).** Suppose two fixed interior allocations satisfy `q_alpha_plus>=m>0` and `q_alpha_minus<=-m`. For

\[
0<b\le\min\{\rho,m/(2L_q)\},
\]

the instantaneous signs persist with margin at least `m/2`. Each branch changes risk by at least `mb/2` in its predicted direction, and their separation is at least `mb`. If `q_h=-a<0<q_W=c`, an explicit pair is

\[
\alpha_*={c\over a+c},\qquad
\alpha_+=\alpha_*/2,\qquad
\alpha_-=(1+\alpha_*)/2,
\]

with margin `m=min(a,c)/2`. Both blocks have strictly positive allocations. This is a finite-amplitude statement on one common initialization and one common budget. Whether `mb/2` is scientifically nontrivial must be checked numerically.

### Constant physical learning-rate ratio

For fixed rates, use `d tau=(eta_h+eta_W)dt`, `alpha=eta_h/(eta_h+eta_W)`, and raw block fields `f_h=(-g_h,0)`, `f_W=(0,-g_W)`. Suppose `||f_g||<=M`, `Lip(f_g)<=K_f`. Replace `q_g` by `<grad R,f_g>` and `L_q` by `H_R M+G_R K_f`. Equations (3)--(6) hold with quadratic remainder `L_q M tau^2/2` for a single branch, and require `M tau<=rho`. This is a fixed accumulated learning-rate budget; it must not be described as equal actual parameter displacement.

## 5. Gradient estimation and discrete steps consume the margin

Consider normalized-clock Euler updates

\[
\theta_{j+1}=\theta_j+\delta_j\{F_\alpha(\theta_j)+e_j\},\qquad
\|e_j\|\le\varepsilon,
\]

with fixed allocation, `b=sum delta_j`, and `delta_max=max delta_j`. The perturbed process uses the nominal budget clock; its actual block path budget can differ. Assume `(1+epsilon)b<=rho`. Taylor expansion and the preceding Lipschitz bound give

\[
\begin{split}
|R_N-R_0-bq_\alpha|
\le{}&\tfrac12L_q(1+\varepsilon)b^2
+G_R\varepsilon b\\
&+\tfrac12 H_R(1+\varepsilon)^2\sum_j\delta_j^2.\tag{7}
\end{split}
\]

The first term is state drift; the second is gradient-direction error; the third is the finite-step Taylor remainder. This is a deterministic statement conditional on a uniform error bound, not a claim that one observed batch supplies such a bound.

If `q_alpha>=m`, define

\[
m_{\rm eff}=m-G_R\varepsilon
-\tfrac12H_R(1+\varepsilon)^2\delta_{\max}.
\]

Provided `m_eff>0` and

\[
0<b\le\min\left\{{\rho\over1+\varepsilon},
{m_{\rm eff}\over L_q(1+\varepsilon)}\right\},\tag{8}
\]

the cumulative risk change is at least `m_eff b/2`. The negative-sign statement is symmetric. A stricter half-window makes every individual step retain a positive lower bound in its predicted direction. Thus a positive local slope gives no useful guarantee when stochastic error or step curvature already exceeds its margin at the source state.

For raw block gradients estimated with errors `||g_hat_g-g_g||<=delta_g`, and true block norms at least `mu_g>0`, normalized-direction error is bounded by `2 delta_g/mu_g`. Hence one may use

\[
\varepsilon\le
2\alpha\delta_h/\mu_h+2(1-\alpha)\delta_W/\mu_W.
\]

This exposes a failure near a block-gradient zero: normalization and rate-allocation inference become ill-conditioned even if the raw dynamics are regular.

## 6. Explicit tube constants for the local square model and GUPA

The preceding constants need not be unspecified existence constants. Let `kappa=||Sigma^(1/2)||_op`, and on the parameter ball bound `||h||_F<=H`, `||W||_op<=W_*` (using `||W||_F` is also valid). Then

\[
\|B\|_F\le\kappa H^2,\quad
\|DB[v_h]\|_F\le2\kappa H\|v_h\|,\quad
\|D^2B[v_h,w_h]\|_F\le2\kappa\|v_h\|\|w_h\|.
\]

Valid derivative bounds for `A=WB` are

\[
J=\kappa\sqrt{H^4+4W_*^2H^2},\qquad
H_A=2\kappa(H+W_*),\qquad
a=\kappa W_*H^2\ge\|A\|_F.
\]

Choose `H=||h_0||_F+rho`, `W_*=||W_0||_op+rho`. Then

\[
\sigma=\sigma_{\min}(A_0)-J\rho>0
\]

is a sufficient rank condition. It is valid because the entire ball is convex and `J` is a bound throughout it.

The risk derivatives obey

\[
G_R={2\|T\|_FJ\over\sigma},\qquad
H_R={6\|T\|_FJ^2\over\sigma^2}
+{2\|T\|_FH_A\over\sigma}.\tag{9}
\]

To verify the constants, use `grad_A R=-2(A^dagger)^T TQ`,

\[
D A^\dagger[Z]=-A^\dagger Z A^\dagger
+QZ^\top(AA^\top)^{-1},
\]

so `||D A^dagger[Z]||_op<=2||Z||_F/sigma^2`. Because `DP[Z]` is off diagonal, `||DP[Z]||_op<=||Z||_F/sigma`. Differentiating the risk gradient gives `||D grad_A R[Z]||_F<=6||T||_F||Z||_F/sigma^2`; composition gives (9).

For the GUPA objective with PSD clean/view covariance moment `S` and view-difference moment `D`, write

\[
G(A)=AD-U(A)AS,\quad
U=(I+ASA^\top)^{-1}+(I+ASA^\top)^{-2}.
\]

Put `s=||S||_op`, `d=||D||_op`, `v=d+2s`. Since the inverse has norm at most one,

\[
\|G(A)\|_F\le av,\qquad
\|DG(A)[Z]\|_F\le(v+6s^2a^2)\|Z\|_F.
\]

A valid parameter-loss Hessian bound is therefore

\[
H_L=(v+6s^2a^2)J^2+avH_A.\tag{10}
\]

Take `mu_g=||g_g(theta_0)||-H_L rho`. If both are positive, then

\[
K\le H_L/\min(\mu_h,\mu_W).
\]

Equations (8)--(10) give a fully explicit, source-state-and-moment-based sufficient window. They may be far too small because they discard task alignment, use global operator bounds, and normalize by the smaller block gradient. An empty or practically negligible bound is a failed prediction certificate, even if the observed derivative signs happen to match. Tighter directional or interval bounds can replace these constants without changing the theorem, but cannot be selected after seeing a successful future curve and then presented as a priori predictions.

## 7. What this does and does not accomplish

The strongest structural pieces are Proposition 1, which characterizes the task family admitting competition, and the leakage decomposition (1), which identifies task cross-block mass and subspace displacement as separate amplitude bottlenecks. The finite-window result gives a precise way to reject a purported intervention when its margin is below state drift, gradient uncertainty, or discrete curvature. Its generic Taylor component should not be advertised as a novel dynamical mechanism by itself.

A meaningful empirical check should fix `T`, tube constants, budgets, and branch allocations using source/calibration information, then compare the resulting risk intervals and window to future states or a separate update stream. A population texture calculation is a clean first check. Replacing `T` and the gradients by estimates from the same labeled batch that is used to assess the next infinitesimal step primarily checks differentiation. Natural-image claims require a further error analysis for the estimated task moment and fitted OLS risk; the present results do not supply that analysis.
