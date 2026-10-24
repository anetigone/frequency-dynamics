# Task-directed stochastic errors and finite-prefix abstention — 2026-10-21

This working note gives conditional theorems, explicit calibration recipes, and the assumptions needed to turn them into strict probability statements. No new training or numerical validation is claimed. Empirical quantiles, observed maxima, and ordinary sample variances alone do **not** establish the bounds used below.

The probability inequality is classical. The potentially useful structural content is that the stochastic quantity is the task contraction of a normal projector increment: task coupling controls the first-order noise, whereas off-subspace leakage controls a separate second-order noise and mean bias. This distinction can improve both variance and abstention rules. The existing implementation already projects its error into each task direction; this note does not introduce that projection for the first time.

## 0. Relationship to the existing predictor and the eight-stream evidence

Direct source inspection of `experiments/code/empirical/texture_motion_prediction.py` establishes that its calibration already saves `risk_slopes(D_batch-D_fit,tasks).abs()` as the task error. `make_predictions` uses the **task** coefficients in its radius

\[
 H q_{\rm noise}+\tfrac12H^2q_{\rm drift}+Hq_{\rm euler}.
\]

The separate Frobenius geometry error is used for geometric/displacement forecasts; it is not substituted for task-projected noise in those task-risk intervals. Therefore it would be incorrect to describe the proposed improvement as the first use of task direction or to attribute current interval width simply to projecting a full-norm error bound.

The change proposed here is to separate (i) source mean-gradient estimation uncertainty, (ii) centered task-scalar random variance, (iii) state drift, and (iv) finite-step leakage/curvature. A martingale boundary accumulates **centered** uncertainty according to its predictable variance; an absolute task-error quantile multiplied by H does not make that separation. A nonzero source mean error must still accumulate linearly and cannot be replaced by a square-root rule. Nor does a square-root variance boundary alone fix mean drift or discrete bias.

The saved eight-future-stream decomposition is relevant but descriptive. For a fixed source/task/rate/horizon cell, with frozen prediction p, future values y_i, mean m, and n=8,

\[
 \operatorname{MSE}=(m-p)^2+\frac18\sum_i(y_i-m)^2.
\]

The raw mean-offset square includes sampling variance of the eight-stream mean. If s^2 uses denominator seven, its sample-corrected squared-mean-bias estimate is (m-p)^2-s^2/8, and negative values must remain negative. The saved analysis reports both terms and does not identify their causal source.

In the supported H=8 square-model predefined-band summaries, the corrected mean term is approximately 30.6%–52.0% of MSE across encoder/projection and checkpoints; the within-source, across-future-stream population-variance term is approximately 42%–61%. These values show that neither the mean-offset contribution nor stream dispersion can be dismissed. They are estimates from eight paired streams, not known population bias/variance parameters. At H=1 the across-stream variance term is approximately 81.7%–92.8%, with noisy corrected mean estimates sometimes negative. H=64 lies outside the original eight-step calibration probe window.

The source radius's named `noise` term accounts for roughly 92%–98% of its H=8 width, but that inflated quantile budget is neither a standard deviation nor an identified variance share. The records do not retain signed per-batch task-error samples sufficient to reconstruct a strict centered-variance confidence bound. The theory below has therefore **not** established a nonempty strict stochastic window on those saved trajectories. Evidence files are `paper/notes/TEXTURE_MOTION_ERROR_DECOMPOSITION_20261021.md` and `experiments/results/empirical/task_motion_error_decomposition/summary.json`; tasks, ratios, checkpoints, and paired future streams remain repeated measurements within seed.

## 1. Fixed source, task, algorithm, and finite horizon

Condition on a source state theta_0 and all past training history. Fix the task matrix T, the two-block rate ratio r, a deterministic maximum horizon H, and a deterministic step schedule delta_k before observing future updates. The representation map A(theta) is the clean, whitened effective map of either the encoder or projection. Its row rank stays k on a specified parameter neighborhood. The task risk is

\[
 R(\theta)=R_* -\operatorname{tr}(T P(\theta)),\qquad
 P(\theta)=A(\theta)^\dagger A(\theta).
\]

The additive constant cancels in every risk-change statement. For general labels, T=CC^T with C=E[uY^T]. This is the population best-linear-probe risk, not validation error after fitting a finite-sample OLS decoder.

Let zeta_k denote a complete future random batch: data indices, augmentations, and any other randomness used by the training step. Define its **actual batch-gradient field**

\[
 F_r(\theta,\zeta)=-D_r\widehat g(\theta,\zeta),\qquad
 \theta_{k+1}=\theta_k+\delta_kF_r(\theta_k,\zeta_k), \tag{1}
\]

where D_r is the fixed positive block-rate multiplier. No individual-image gradient decomposition is assumed. For sample-centered GUPA, the batch matrix inverse and batch moments belong inside widehat g.

Write F_k for the sigma-field before batch k. In the basic calibration theorem, future whole batches are fresh independent draws from the same batch law as the independent source calibration batches. Finite-data sampling **with replacement** is included. Sampling without replacement changes the conditional batch law as an epoch is consumed; source calibration then needs an additional justified law-drift bound or an appropriate finite-population argument. A fixed empirical stream is not automatically covered by an iid-batch theorem.

In particular,

\[
 \overline F_r(\theta)=E[F_r(\theta,\zeta)]
\]

need not equal the negative gradient of population GUPA. The expectation of a nonlinear function of sample moments is not obtained by substituting population moments. The results below predict the actual finite-batch algorithm, including its mean finite-batch bias.

Source selection may use the past. Future batches must remain fresh relative to that selection and to calibration. If a rate ratio or task is selected after looking at source-calibration data, either use new calibration data after selection or construct simultaneous calibration coverage for the entire finite candidate family.

## 2. Exact one-step geometry: first-order task noise and second-order leakage

This section has no small-step assumption. For one rank-preserving random step, put P=P_k, Q=I-P, P'=P_{k+1}, and

\[
 X=P(I-P')P,\quad Y=QP'Q,\quad K=QP'P,\quad Z=QTP,
\]

\[
 s^2=\tfrac12\|P'-P\|_F^2.
\]

The matrices X,Y are positive semidefinite, tr X=tr Y=s^2, and ||K||_F<=s. Exact block expansion gives

\[
 D_k:=R_{k+1}-R_k
 =\underbrace{-2\langle Z,K\rangle_F}_{L_k}
 +\underbrace{\operatorname{tr}(T_{PP}X)
              -\operatorname{tr}(T_{QQ}Y)}_{B_k^{(2)}}. \tag{2}
\]

Let omega=lambda_max(T)-lambda_min(T). Since subtracting a scalar multiple of I leaves both terms unchanged,

\[
 |B_k^{(2)}|\leq\omega s^2,\qquad
 |D_k|\leq2\|Z\|_Fs+\omega s^2. \tag{3}
\]

All quantities other than P' are predictable at step k. The first term is an off-diagonal task contraction. The second is genuine risk bias/noise caused by leaving the current subspace; it cannot be inferred from the sign of the first derivative.

### Conditional mean and exact scalar martingale

Define

\[
 \mu_k=E[D_k\mid F_k],\qquad
 \xi_k=D_k-\mu_k,\qquad
 M_n=\sum_{k<n}\xi_k. \tag{4}
\]

Then R_n-R_0=sum_{k<n}mu_k+M_n exactly. Its conditional variance is the scalar quantity

\[
 v_k=\operatorname{Var}(D_k\mid F_k)
 =\operatorname{Var}\{\operatorname{tr}[T(P'-P)]\mid F_k\}. \tag{5}
\]

There is no ambient-dimension prefactor in this variance. Estimating a covariance matrix of all effective features is unnecessary if only this fixed task and rate ratio are being certified.

Let

\[
 v_k^{\rm cross}=\operatorname{Var}(\langle Z,K\rangle_F\mid F_k),
 \qquad m_{4,k}=E[s^4\mid F_k].
\]

Minkowski's inequality for centered L2 variables and (3) give

\[
 \sqrt{v_k}\leq2\sqrt{v_k^{\rm cross}}+\omega\sqrt{m_{4,k}},\quad
 v_k\leq(2\sqrt{v_k^{\rm cross}}+\omega\sqrt{m_{4,k}})^2. \tag{6}
\]

The cross variance equals vec(Z)^T Cov(vec K|F_k)vec(Z). It can be much smaller than ||Z||_F^2 times the operator norm or trace of the whole covariance. Cross/leakage covariance is retained by (5); inequality (6) bounds it rather than assuming independence.

If s<=s_max almost surely, the second term contributes at most omega*s_max^2 to the conditional standard deviation and at most omega*s_max^2 to the conditional mean in absolute value. For a regular step of size delta, this is order delta^2, while a nonzero cross term is generally order delta.

### Zero task coupling does not mean zero stochastic risk drift

If QTP=0, L_k vanishes identically, so

\[
 |\mu_k|\leq\omega E[s^2\mid F_k],\qquad
 v_k\leq\omega^2E[s^4\mid F_k]. \tag{7}
\]

Thus first-order task noise disappears, but second-order leakage remains. If the row-space task spectrum exceeds the normal-space task spectrum by gamma>0, (2) gives the pathwise inequality

\[
 D_k\geq\gamma s^2.
\]

Every nonzero departure hurts the task, regardless of whether the stochastic gradient is unbiased. In the especially transparent case T=lambda P with lambda>0,

\[
 D_k=\lambda s^2\quad\text{exactly}. \tag{8}
\]

For a small parameter step, let N=Q(DA[F])^T(A^dagger)^T. Then s^2=delta^2||N||_F^2+O(delta^3), uniformly on a smooth compact rank-preserving tube. Consequently

\[
 E[D_k\mid F_k]
 =\lambda\delta^2\{\|E[N\mid F_k]\|_F^2
           +\operatorname{tr}\operatorname{Cov}(\operatorname{vec}N\mid F_k)\}
 +O(\delta^3). \tag{9}
\]

The noise covariance itself contributes to expected task loss. Equation (8), which is exact, should be used whenever possible; (9) is only its small-step interpretation.

An explicit remainder in (9) is available without informal O notation. If ||D^2P||<=H_P on the parameter segment and ||F||<=M, write Delta P=delta(N+N^T)+E_P with ||E_P||_F<=H_P delta^2M^2/2. The error in s^2-delta^2||N||_F^2 has absolute value at most

\[
 \frac{H_P}{\sqrt2}\delta^3M^2\|N\|_F
       +\frac{H_P^2}{8}\delta^4M^4. \tag{10}
\]

A sufficient parameter-tube bound is H_P=6J^2/sigma^2+2H_A/sigma when ||DA||<=J, ||D^2A||<=H_A, and sigma_min(A)>=sigma>0. This follows by differentiating the pseudoinverse projector formula and composing with A(theta).

## 3. A scalar all-prefix concentration lemma

Suppose xi_k is a martingale difference, |xi_k|<=c almost surely, and sum_{k<n}E[xi_k^2|F_k]<=Vbar_n for every n<=H. The upper variance process Vbar_n can be predictable; a deterministic calibration-derived envelope is enough here.

Fix a finite grid Lambda of J positive numbers less than 3/c before the future stream, and put

\[
 x=\log(2J/\delta_{\rm mart}),\qquad
 \mathfrak B(v)=\min_{\lambda\in\Lambda}
 \left\{\frac{x}{\lambda}
 +\frac{\lambda v}{2(1-\lambda c/3)}\right\}. \tag{11}
\]

With probability at least 1-delta_mart,

\[
 |M_n|\leq\mathfrak B(\overline V_n)
 \quad\text{simultaneously for all }1\leq n\leq H. \tag{12}
\]

No union penalty in H is needed for this finite grid of linear boundaries. A larger grid gives a closer approximation to a curved variance boundary, at cost log J. The grid may be selected from independent source calibration, since the proof conditions on that calibration.

**Proof.** Bounded centered increments satisfy

\[
 E[e^{\lambda\xi_k}\mid F_k]
 \leq\exp\left\{\frac{\lambda^2E[\xi_k^2\mid F_k]}
                          {2(1-\lambda c/3)}\right\}.
\]

Expand the exponential, use E[xi_k|F_k]=0 and |E[xi_k^j|F_k]|<=c^{j-2}E[xi_k^2|F_k], and use j!>=2*3^{j-2}. The resulting exponential process is a nonnegative supermartingale. Ville's maximal inequality gives a crossing probability at most exp(-x) for each lambda and each sign. A union bound over 2J choices proves (12). This is the classical Bernstein–Freedman exponential-supermartingale argument, not a new probability inequality.

For a simpler finite-H alternative, union-bound the fixed-prefix Freedman inequality. With x=log(2H/delta_mart), a conservative explicit boundary is

\[
 \mathfrak B_H(v)=\sqrt{2vx}+\frac{2cx}{3}. \tag{13}
\]

The coefficient 2/3 is deliberately conservative for direct inversion of the usual fraction-form Freedman bound. One should not silently replace it by 1/3 while using that weaker tail formula. Formula (11) avoids this inversion issue entirely.

For (13), Vbar_n must be deterministic after conditioning on the independent calibration, as it is in Sections 5 and 6. Substituting a random predictable variance into a fixed-time square-root boundary is not justified by a direct union bound; it needs a valid curved boundary, peeling, or the finite-lambda construction (11). Formula (11) itself permits a predictable random upper-variance process.

If c=0, M_n is identically zero and set the boundary to zero.

Primary probability references: [Freedman, *On Tail Probabilities for Martingales*, 1975](https://doi.org/10.1214/aop/1176996452), and [Howard, Ramdas, McAuliffe, Sekhon, *Time-uniform Chernoff bounds via nonnegative supermartingales*, 2020](https://doi.org/10.1214/18-PS321). The latter explicitly develops the supermartingale/line-crossing formulation used in (11). No verbatim text from either reference is used here.

## 4. Independent source calibration with genuine upper confidence bounds

For any fixed bounded scalar batch statistic X(zeta) in [-B,B], draw N independent **whole batches** at theta_0, separate from future updates. Record the actual statistic, including all batch centering, augmentation, inverse matrices, and any clipping that belongs to the specified algorithm.

Let mhat=N^{-1}sum X_i and m2hat=N^{-1}sum X_i^2. Hoeffding bounds give, with total failure probability at most delta_mu+delta_v,

\[
 |E[X]-\widehat m|\leq a,
 \qquad a=B\sqrt{\frac{2\log(2/\delta_\mu)}{N}}, \tag{14}
\]

\[
 E[X^2]\leq U_2:=\min\left\{B^2,
 \widehat m_2+B^2\sqrt{\frac{\log(1/\delta_v)}{2N}}\right\},
\]

\[
 \operatorname{Var}(X)\leq\overline\sigma_0^2
 :=\max\{0,U_2-(|\widehat m|-a)_+^2\}. \tag{15}
\]

These are confidence bounds, rather than substituting the empirical variance into Freedman's predictable variance. The deterministic B must come from a verified support/range argument; the largest observed |X_i| cannot replace it.

An alternative avoids estimating the mean inside the variance formula. With N=2m independent batches, set Z_i=(X_{2i}-X_{2i-1})^2/2. Then E Z_i=Var(X), 0<=Z_i<=2B^2, and

\[
 \operatorname{Var}(X)\leq
 \min\left\{B^2,\frac1m\sum_iZ_i
       +2B^2\sqrt{\frac{\log(1/\delta_v)}{2m}}\right\} \tag{16}
\]

with probability at least 1-delta_v. Mean and variance events need not be independent for the union bound. If both (15) and (16) are used and the minimum selected, assign separate failure probabilities to them.

The bounds may be conservative. Their nonemptiness and practical width must be measured. An empirical percentile of X or of |X-mhat| is a diagnostic until accompanied by a valid distribution-free tolerance bound or a justified tail model covering future draws.

## 5. A computable exact-step theorem for constant step size

This theorem calibrates actual finite parameter updates, which automatically includes the second-order noise in (2). It is useful when a tangent-only prediction misses leakage bias.

Fix delta>0 and H. On the parameter ball D={||theta-theta_0||<=rho}, assume:

* A has constant full row rank, sigma_min(A)>=sigma>0, ||DA||<=J.
* Every allowed batch satisfies ||F_r(theta,zeta)||<=M and F_r is K_F-Lipschitz in theta.
* ||grad R||<=G_R and ||Hessian R||<=H_R.
* M delta H<=rho, and the fresh-batch law is the one used in calibration.

Bounds are required on the whole tube, including segments between step endpoints, not only at observed checkpoints. Define the exact step statistic

\[
 D(\theta,\zeta)=R(\theta+\delta F_r(\theta,\zeta))-R(\theta).
\]

Here delta times the number of updates is a nominal clock, not actual parameter path length. The ratio multiplier may be D_r=diag(I,rI) or a specified fixed-sum normalization; M and K_F must correspond to that actual choice. The bound M delta H<=rho converts the nominal clock into a genuine sufficient parameter-tube condition. Fixed rate sum does not equalize realized update length.

On the reachable pre-step ball it is Lipschitz with

\[
 L_D=\delta(H_RM+G_RK_F). \tag{17}
\]

Indeed, differentiating D gives [grad R(theta+delta F)-grad R(theta)]+delta DF^T grad R(theta+delta F), whose norm is bounded by (17). The same conclusion holds for Lipschitz F by its almost-everywhere derivative and integration on line segments.

Let a_N=JM/sigma, alpha_0=||Q_0TP_0||_F, and omega=osc(T). At pre-step k,

\[
 s(\theta_k,\theta_0)\leq a_N\delta k,
 \quad \alpha_k\leq\alpha_0+\sqrt2\omega a_N\delta k,
\]

because ||QTP-Q_0TP_0||_F<=omega||P-P_0||_F after subtracting the midpoint of the spectrum of T. Thus a deterministic exact-increment range is

\[
 B_k=2(\alpha_0+\sqrt2\omega a_N\delta k)a_N\delta
                         +\omega(a_N\delta)^2. \tag{18}
\]

This follows from (3), using s_step<=a_N delta. Principal-angle distance and task-coupling bounds can be clamped at their global maxima if that sharpens the envelope.

Calibrate X_i=D(theta_0,zeta_i) independently using B_0 and (14)–(15), obtaining mhat, a, and sigma0bar. For future states,

\[
 |E[D(\theta_k,\zeta)\mid F_k]-E[D(\theta_0,\zeta)]|
                  \leq L_DM\delta k, \tag{19}
\]

\[
 \sqrt{\operatorname{Var}(D(\theta_k,\zeta)\mid F_k)}
 \leq\overline\sigma_0+L_DM\delta k. \tag{20}
\]

For (20), couple the two statistics using the same batch randomness and use the L2 contraction of centering: standard deviations differ by at most the L2 norm of their difference. A valid predictable variance envelope is consequently

\[
 \overline v_k=\min\{(\overline\sigma_0+L_DM\delta k)^2,B_k^2\},
 \quad\overline V_n=\sum_{k<n}\overline v_k,
 \quad c=2\max_{k<H}B_k. \tag{21}
\]

**Exact-step all-prefix prediction.** With probability at least

\[
 1-\delta_\mu-\delta_v-\delta_{\rm mart},
\]

simultaneously for every 1<=n<=H,

\[
 |R_n-R_0-n\widehat m|
 \leq n a+L_DM\delta\frac{n(n-1)}2
                +\mathfrak B(\overline V_n). \tag{22}
\]

The proof is (4), the independent calibration event, the drift sum in (19), and the all-prefix martingale lemma. Conditional on good independent calibration, the future crossing probability is at most delta_mart; adding calibration failure probabilities gives the unconditional statement.

This theorem includes the exact nonlinear finite-step conditional mean. It therefore does not need to assert that risk increments, finite-batch gradients, or normal projector increments are unbiased. Calibration at the source is not performed on the future update batches.

Equation (18) yields a geometric improvement when alpha_0 is zero or small: source increments and source variance are then order delta^2 and delta^4 respectively, rather than the order delta and delta^2 bounds supplied by generic risk-gradient estimates. State drift can restore first-order coupling later, as its explicit k dependence shows.

## 6. A tangent-calibrated version separating source, drift, and discretization

For a varying deterministic step schedule, set b_n=sum_{k<n}delta_k and assume M b_H<=rho. Define the actual batch directional derivative

\[
 q(\theta,\zeta)=\langle\nabla R(\theta),F_r(\theta,\zeta)\rangle
                =-2\langle QTP,N_r(\theta,\zeta)\rangle_F,
\]

where N_r=Q(DA[F_r])^T(A^dagger)^T. The sign here is for the actual velocity F_r, in contrast to a Gram action defined with the negative sign removed.

Let L_q=H_RM+G_RK_F. Then q is L_q-Lipschitz in theta. Independently calibrate q(theta_0,zeta) using its known source range Q_0=2alpha_0 a_N, obtaining qhat_0, error a_q, and standard-deviation upper bound sigmaqbar_0. For k<H set

\[
 Q_k=2a_N(\alpha_0+\sqrt2\omega a_N b_k),
\]

\[
 \overline v_k=\delta_k^2
 \min\{(\overline\sigma_{q,0}+L_qMb_k)^2,Q_k^2\},
 \quad c=2\max_{k<H}\delta_kQ_k. \tag{23}
\]

The leading martingale is sum delta_k[q(theta_k,zeta_k)-E(q(theta_k,zeta)|F_k)]. Taylor's theorem supplies the **pathwise** remainder

\[
 R_{k+1}-R_k=\delta_kq(\theta_k,\zeta_k)+e_k,
 \quad |e_k|\leq\tfrac12H_RM^2\delta_k^2.
\]

With the same total failure accounting, every prefix satisfies

\[
 |R_n-R_0-b_n\widehat q_0|\leq
 \underbrace{a_qb_n}_{\text{independent source calibration}}
 +\underbrace{\frac{L_qM}{2}(b_n^2-\sum_{k<n}\delta_k^2)}_{\text{state drift}}
 +\underbrace{\frac{H_RM^2}{2}\sum_{k<n}\delta_k^2}_{\text{finite-step curvature}}
 +\underbrace{\mathfrak B(\sum_{k<n}\overline v_k)}_{\text{task-directed random fluctuation}}. \tag{24}
\]

The drift identity uses sum delta_k b_k=(b_n^2-sum delta_k^2)/2. Risk curvature is deliberately kept outside the leading martingale; doing so is safe but can be less sharp than (22), especially at a task-stationary source. Equations (22) and (24) are alternatives, not independent evidence to multiply together.

For two blocks sharing the same batch, q_r is a fixed linear combination of their task-direction statistics. Its conditional variance contains their covariance:

\[
 \operatorname{Var}(q_r)=a_h^2\operatorname{Var}(q_h)
 +a_W^2\operatorname{Var}(q_W)+2a_ha_W\operatorname{Cov}(q_h,q_W). \tag{25}
\]

Independent source calibration should retain the paired whole-batch values. Replacing them by separately estimated block variances can discard strong cancellation or reinforcement. There is no automatic 1/batch-size variance law for the nonlinear sample-moment objective; that scaling requires its own proof.

## 7. Finite-prefix abstention and a nonempty-window condition

From either theorem define lower/upper prediction envelopes L_n,U_n for R_n-R_0 before the future stream. Fix a practical risk threshold epsilon_task>=0. At a prefix n:

* certify improvement only when U_n<-epsilon_task;
* certify deterioration only when L_n>epsilon_task;
* otherwise abstain.

On the common all-prefix coverage event, every issued statement is correct, including a prefix chosen after inspecting the stream. This is a statement about confidence coverage of a fixed source/task/ratio experiment. It does not authorize selecting among unaccounted task/rate/source candidates using the same future data.

A simple sufficient condition shows when the certified prefix set can be empty. In the tangent version, suppose constant delta, a source mean margin mu=|qhat_0|-a_q>0, and uniform task-scalar bounds Var(q|F_k)<=sigma_q^2 and |q|<=Q on the tube. Using (13), x=log(2H/delta_mart), the risk sign is certified whenever

\[
 \mu>
 \sigma_q\sqrt{2x/n}+\frac{4Qx}{3n}
 +\frac{L_qM\delta(n-1)}2+\frac{H_RM^2\delta}2. \tag{26}
\]

The random part decreases with n while source-state drift grows with n. For a conservative half-margin guarantee, it suffices that

\[
 n\geq\max\left\{\frac{128\sigma_q^2x}{\mu^2},
                       \frac{32Qx}{3\mu}\right\},
 \quad \delta(n-1)\leq\frac{\mu}{4L_qM},
 \quad \delta\leq\frac{\mu}{4H_RM^2}. \tag{27}
\]

Then the four error terms in (26) total at most mu/2, so the risk-change magnitude is at least mu*n*delta/2 with the predicted sign. Also impose n<=H and M delta n<=rho. Zero denominators are interpreted in the usual way: a vanishing error coefficient imposes no corresponding restriction.

Equation (27) is an explicit lower/upper window criterion. Near a rate-cancellation threshold, mu shrinks while stochastic covariance need not shrink; the lower horizon can exceed the drift-limited upper horizon. A generic local sign reversal then has no certifiable persistent effect. Conversely, task-directed noise cancellation in (25) can improve the lower horizon. This is a falsifiable mechanism to measure, not an explanation already established for CIFAR/DTD.

For two fixed rate ratios, calibrate both using paired independent source batches, assign simultaneous failure probabilities, and intersect their valid prefix sets. A risk split is certified only where one branch's upper envelope is negative and the other's lower envelope is positive. The future streams may be paired; independence between branches is unnecessary for a union bound.

### Exact finite-projector example: equal task coupling and displacement, different survival

The following finite-step geometry example shows why the **orientation of random variation** matters beyond the norm alpha=||QTP||_F. It is not an additional training theorem for the paper's architecture and makes no typicality claim.

Let P=e_1e_1^T in R^3 and let the next rank-one row representation be

\[
 v=e_1+\delta m e_2+\delta\sigma\varepsilon e_3,
 \quad P'=vv^T/D,\quad D=1+\delta^2(m^2+\sigma^2),
\]

with m,sigma>0 and epsilon taking plus/minus one with equal probability. Both possible steps have the same geometric displacement,

\[
 s^2=\delta^2(m^2+\sigma^2)/D.
\]

Consider the unit scalar tasks beta_2=(e_1+e_2)/sqrt(2) and beta_3=(e_1+e_3)/sqrt(2). Both have ||Q beta_j beta_j^T P||_F=1/2 and the same initial risk 1/2. Exact finite-projector calculation nevertheless gives

\[
 \Delta R_{\beta_2}=\frac{-\delta m+\tfrac12\delta^2\sigma^2}{D},
 \qquad\operatorname{Var}(\Delta R_{\beta_2})=0,
\]

\[
 \Delta R_{\beta_3}=\frac{-\delta\sigma\varepsilon+
                              \tfrac12\delta^2m^2}{D},
 \quad E\Delta R_{\beta_3}=\frac{\delta^2m^2}{2D},
 \quad\operatorname{Var}(\Delta R_{\beta_3})=
                  \frac{\delta^2\sigma^2}{D^2}. \tag{30}
\]

Thus beta_2 has a deterministic improvement when delta<2m/sigma^2. For beta_3, when delta<2sigma/m^2, the realized sign is equally likely to improve or deteriorate, and its mean is only a second-order deterioration. This difference occurs despite equal task-cross norm and identical geometric movement. It is captured by the task-scalar covariance contraction in (6), and is lost if one keeps only alpha and a norm bound for all normal noise. The second-order terms in (30) also demonstrate why dropping leakage can misstate the conditional mean.

## 8. Estimating the task matrix without pretending that finite OLS generalizes

If T is unknown, use an independent labeled calibration sample to estimate C=E[uY^T], and certify a matrix error before gradient calibration. A simple bounded-coordinate example is enough to make this concrete. If |u_iY_j|<=L almost surely, d feature coordinates, p label coordinates, and N_T independent labeled examples, then a coordinatewise Hoeffding union bound gives

\[
 \|\widehat C-C\|_F\leq
 e_C:=L\sqrt{\frac{2dp\log(2dp/\delta_T)}{N_T}}
\]

with probability at least 1-delta_T. For Ttilde=Chat Chat^T,

\[
 \|\widetilde T-T\|_2\leq e_T:=e_C(2\|\widehat C\|_2+e_C). \tag{28}
\]

Run source batch calibration conditional on this Ttilde, using new independent calibration batches. The theorem predicts its associated risk change. True task-risk changes differ by at most

\[
 |\operatorname{tr}[(\widetilde T-T)(P_n-P_0)]|
 \leq2e_T\sum_i\sin\theta_i
 \leq2e_T\sqrt{r_*}\,s_{0n},\qquad r_*=\min(k,d-k). \tag{29}
\]

Use the prospective bound s_0n<=a_N b_n and add (29) to the error envelope; add delta_T to the total failure probability. An available Frobenius error bound also gives sqrt(2)||Ttilde-T||_F*s_0n. This controls the particular predictable linear task component. It is not a theorem about finite-sample OLS decoder estimation, covariance whitening estimation, test accuracy, or validation MSE under a different distribution. Estimated whitening needs its own separately justified error bound, or the risk must be explicitly defined for a fixed calibration feature metric.

## 9. Support bounds, unbounded augmentations, and honest stopping

The strict statements above require their stated range/tube/variance conditions. Three cases must be distinguished.

1. **Known bounded support.** Finite stored images with deterministically bounded augmentations, a compact parameter tube, and the GUPA resolvent bound can yield M,K_F,J,sigma and scalar ranges. Evaluating a few batches does not prove these uniform bounds, but elementary image/parameter norm bounds or interval calculations can.
2. **A proved conditional moment-generating-function bound.** A valid conditional sub-gamma or Bernstein moment condition can replace bounded increments in Section 3. Its scale and variance proxy must themselves be known or upper-confident. A Gaussian input does not automatically make a polynomial batch gradient sub-Gaussian or give it a finite positive exponential moment.
3. **A proved finite-horizon tail event.** If an analytic bound ensures P(||F||>M | past state in tube)<=p uniformly, a counterfactual process using a bounded version of F can be coupled to the actual process. They agree at every prefix before the first exceedance, and the probability of any exceedance by H is at most Hp. Calibrate and prove the bounded counterfactual theorem, then add Hp to the failure probability for the actual trajectory. This does not change the executed training; it does change the mean used in the proof and must be declared. A bound based only on an observed source percentile cannot justify Hp.

If a verified condition becomes unavailable, abstain from subsequent claims. An observed departure from the certified tube is not a reason to keep using the original source envelope. It is also not legitimate to condition silently on the future event that all batches were small: that conditioning can alter their means. The unconditional tail-event accounting or a properly stopped/coupled argument is required.

This theoretical protocol should remain a small direct calculation: source batches, scalar means/variance upper bounds, deterministic tube constants, and prefix envelopes. It requires no hashes, script registries, new training jobs, or experiment-management machinery.

## 10. What would constitute progress on existing trajectories

The immediately useful quantities are alpha_0, the source signed task mean, the paired block covariance, the task-scalar variance upper bound, the normal step budget, and the predicted prefix interval. The exact-step method additionally detects positive second-order leakage when the source tangent is zero.

A credible check fixes a source, task, ratio, calibration-batch law, and H; constructs all envelopes from independent source calibration and justified bounds; then evaluates them on an untouched future update stream. Reporting coverage and non-abstention are separate: a vacuous interval can cover every future value. Failure due to calibration uncertainty, variance, drift, step curvature, rank/tube loss, or insufficient risk scale should remain distinguishable.

Saved stochastic trajectories may lack independent source batches, exact batch-gradient records, or valid support constants. They can then test the geometry or diagnose empirical variance, but cannot retrospectively instantiate a strict all-prefix confidence theorem merely by fitting quantiles to the same stream. No claim is made here that the natural-image windows are nonempty.
