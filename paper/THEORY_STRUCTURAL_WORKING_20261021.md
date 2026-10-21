# Structural theory working note — 2026-10-21

This note contains proved algebraic statements and one exactly solvable training family. It does not claim a literature novelty audit, natural-image predictive validation, or finite-sample OLS generalization. No training is needed for any calculation below.

The most substantive result is Section 6: an explicit, finite-amplitude rebound family in a genuine encoder bottleneck, with a three-dimensional task cone, common comparison sections, explicit layer-rate and augmentation ranges, and no numerical ODE certificate. It also provides a symmetry condition that **excludes** learning-rate competition. Sections 1–5 connect that result to general local second moments.

## 1. Exact local-moment representation and the clean/augmentation support caveat

Let patches have dimension m and let svec be the isometry from real symmetric m by m matrices to R^D, D=m(m+1)/2: diagonal entries are unchanged and off-diagonal entries are multiplied by sqrt(2). Thus svec(S)^T svec(H)=tr(SH).

For a square convolution followed by global average pooling, define

\[
 M(x)=|U|^{-1}\sum_{u\in U}p_u(x)p_u(x)^T,\quad
 \phi(x)=\operatorname{svec}(M(x)-\mathbb E M(x)).
\]

Exactly, for every image and every real filter h_c,

\[
 r_c(x)-\mathbb E r_c(x)=\operatorname{svec}(h_ch_c^T)^T\phi(x).
\]

No Gaussian or independence assumption is involved. Finite second moments of phi, equivalently the required fourth patch moments, suffice for the covariance calculations below.

Let the positive eigendecomposition of the clean covariance be Sigma=U Lambda U^T, with U having d orthonormal columns and Lambda positive diagonal. Define u=Lambda^{-1/2} U^T phi. Then Cov(u)=I_d and phi=U Lambda^{1/2}u almost surely under the clean distribution. The latter follows since every direction in ker(Sigma) has zero variance and phi is centered. Write K=U Lambda^{1/2} and

\[
 b_c=K^T\operatorname{svec}(h_ch_c^T),\qquad
 B=\begin{pmatrix}b_1^T\\\vdots\\b_C^T\end{pmatrix},\qquad A=WB.
\]

Clean centered encoder and projection features are Bu and Au.

**Necessary qualification.** Whitening on the clean support does not establish that the augmented objective is a function of A alone. Augmented moment features can have components in ker(Sigma). Statements below using an objective L(A) assume that both augmented views lie in the clean covariance support after their appropriate centering, or, more generally, that clean covariance is positive definite on the dictionary supporting clean and augmented features. If that condition fails, retain the full moment dictionary and its clean singular covariance. Gradients in augmented-only coordinates can affect the clean representation through shared parameters; they are an additional forcing term and cannot be silently discarded. The clean least-squares formula itself remains valid without this extra support assumption.

For a centered vector label Y with finite covariance, let C=E[uY^T] and T=CC^T. The optimal affine population square risk is

\[
 R(A)=\operatorname{tr}\operatorname{Cov}(Y)-\operatorname{tr}(P_AT),\qquad
 P_A=A^\dagger A.
\]

This follows by minimizing E||Y-DAu||^2 over D. Labels need not be linear functions of u. T only records their linearly predictable component. This is a standard least-squares identity, a bridge rather than a contribution. It does not identify validation MSE after finite-sample OLS with population R(A).

## 2. Parameterization mobility and a measurable normal decomposition

Let G=grad_A L and J_c=D_{h_c}b_c, a d by m matrix. Set T_c=J_cJ_c^T, which is positive semidefinite. To avoid confusion, T is the task matrix and T_c is a channel mobility.

Actual Euclidean parameter gradient flow gives

\[
 \dot W=-\eta_W GB^T,\qquad
 \dot b_c=-\eta_hT_c(W^TG)_{c:}^T,
\]

and consequently

\[
 \dot A=-\eta_W\mathcal M_W[G]-\eta_h\mathcal M_h[G],
\]

\[
 \mathcal M_W[G]=GB^TB,\qquad
 \mathcal M_h[G]=\sum_c w_cw_c^TG T_c,
\]

where w_c is column c of W. Both are positive semidefinite self-adjoint operators on the effective-map Frobenius space, since they are parameter-Jacobian Gram operators. The last display follows by multiplying channel updates by W, without treating constrained second-moment coefficients as free optimization variables.

On a constant-rank interval, for P=P_A, Q=I-P, define

\[
 Z=QTP,\quad N_j=Q\mathcal M_j[G]^T(A^\dagger)^T,\quad j\in\{h,W\}.
\]

Then N_j=QN_jP and

\[
 \dot P=-\sum_j\eta_j(N_j+N_j^T),\qquad
 \dot R=2\sum_j\eta_j\langle Z,N_j\rangle_F.
\]

This identifies the **off-diagonal task block** and the **normal mobility response**, not total gradient size. Neither a large loss decrease, feature norm growth, nor a large tangential gradient supplies a lower bound on task-risk change.

For the general moment GUPA objective, write S for the within-view covariance and D for the view-difference second moment. With Q_v=ASA^T, set

\[
 F(Q_v)=(I+Q_v)^{-1}+(I+Q_v)^{-2},\qquad
 G=AD-F(Q_v)AS.
\]

For positive semidefinite Q_v, ||F(Q_v)||_2<=2. Choosing S=sI+Delta_S and D=d_0I+Delta_D gives the exact decomposition

\[
 G=HA+E,\quad H=d_0I-sF(Q_v),\quad
 E=A\Delta_D-F(Q_v)A\Delta_S,
\]

\[
 \|E\|_F\leq\|A\|_F(\|\Delta_D\|_2+2\|\Delta_S\|_2).
\]

The full residual, its normal projection, and the mobility responses can all be measured at a calibration state without using future labels. The task matrix still requires task information.

### Linear near-conservation

For A=WV with ordinary linear parameters, dot V=-eta_V W^T(HA+E), hence

\[
 \dot V(I-P_V)=-\eta_VW^TE(I-P_V).
\]

On a constant-rank interval, let nu_V=||V^dagger dot V(I-P_V)||_F. Then

\[
 \nu_V\leq\eta_V\|V^\dagger\|_2\|W\|_2
 \|E(I-P_V)\|_F.
\]

The task-risk and finite-displacement bounds in Section 4 turn the integral of this quantity into a near-conservation bound. Under E=0 the row space is exactly constant. Ordinary discrete GD also has V_new=(I-eta_V W^THW)V; if the rank is preserved, so is the row space. Step size alone does not generate linear drift in this exact isotropic model.

### Separating objective anisotropy from parameterization anisotropy

For the square encoder, choose arbitrary scalars tau_c and write T_c=tau_c I+D_c. Let P_B be its current row projector and Q_B=I-P_B. With G=HA+E,

\[
 \dot b_c^TQ_B=-\eta_h\left[(W^THWB)_{c:}D_c
 +(W^TE)_{c:}T_c\right]Q_B.
\]

The scalar mobility term disappears because every row of W^THWB belongs to row(B). Thus the first term requires parameterization anisotropy; the second is data/objective forcing filtered through the parameterization. Choosing tau_c halfway between the extreme eigenvalues gives ||D_c||_2=(lambda_max(T_c)-lambda_min(T_c))/2. Direct off-diagonal blocks relative to row(B) can be much sharper than this global spectral spread.

Likewise, for projection normal motion under E=0,

\[
 \mathcal M_h[HA](I-P_A)
 =\sum_cw_cw_c^THA\,P_AT_c(I-P_A).
\]

If all these mobility off-diagonal blocks vanish, the front block cannot move the projection row space at that state. Nonlinearity alone does not establish motion.

## 3. Competition and exclusion conditions

For a fixed task, define c_j=2<Z,N_j>_F. Positive learning rates can reverse the instantaneous risk sign exactly when c_h c_W<0. The useful additional content is the explicit structural calculation of N_j above.

Some verifiable exclusions are stronger than this scalar statement:

1. If [P,T]=0, then Z=0 and every parameter block has zero instantaneous risk derivative. Finite change can still be quadratic, as Section 4 quantifies.
2. If N_h=0, changing its positive rate cannot reverse the head's nonzero instantaneous sign. This covers the exact isotropic ordinary linear front layer.
3. If N_h=aN_W with a>=0, no task can have opposing nonzero block derivatives.
4. If ||N_h-aN_W||_F<=epsilon, a>0, and |<Z,N_W>|>||Z||_F epsilon/a, the two block signs agree for that task. This is a quantitative exclusion cone.

There is also an exact statement about the richness of possible tasks: for nonzero N_h,N_W, some positive semidefinite task matrix produces opposing block signs **if and only if** the two normal responses are not positive multiples. For sufficiency, choose an off-diagonal matrix Z_0 with <Z_0,N_h> and <Z_0,N_W> of opposite signs, possible for any two vectors not on the same positive ray. Then T=tI+Z_0+Z_0^T is positive semidefinite for sufficiently large t and QTP=Z_0. Such T is realizable by the linear multivariate label Y=T^{1/2}u. Necessity follows immediately from positive proportionality. This is a task-existence statement, not a prediction that a supplied classification task will compete.

## 4. Finite risk amplitude from task coupling and normal movement budget

Let P_0 and P_1 be rank-k orthogonal projectors in R^d. Put Q_0=I-P_0,

\[
 s^2=\frac12\|P_1-P_0\|_F^2=\sum_i\sin^2\theta_i,\quad
 \alpha_0=\|Q_0TP_0\|_F,\quad
 \omega=\lambda_{\max}(T)-\lambda_{\min}(T).
\]

Then

\[
 |R(P_1)-R(P_0)|\leq2\alpha_0s+\omega s^2. \tag{1}
\]

**Proof.** In P_0/Q_0 blocks, X=P_0-P_0P_1P_0 and Y=Q_0P_1Q_0 are positive semidefinite and tr X=tr Y=s^2. The cross block C_1=Q_0P_1P_0 has squared Frobenius norm sum_i sin^2(theta_i)cos^2(theta_i)<=s^2. The risk difference is tr(T_PP X)-tr(T_QQ Y)-2<T_QP,C_1>. Subtracting lambda_min(T)I leaves the difference unchanged because ranks agree. The two diagonal traces have difference at most omega s^2 in absolute value, and Cauchy–Schwarz bounds the cross term by 2alpha_0s.

A sharper directional version is

\[
 (m_P-M_Q)s^2-2\alpha_0s
 \leq R(P_1)-R(P_0)
 \leq(M_P-m_Q)s^2+2\alpha_0s, \tag{2}
\]

where m_P,M_P are extreme eigenvalues of T restricted to row(P_0), and m_Q,M_Q those on its orthogonal complement. If r=min(k,d-k)>0, replace s in the cross term by s sqrt(1-s^2/r), since sum sin^4(theta_i)>=s^4/r. These bounds require no dynamical smallness assumption.

If [P_0,T]=0 and m_P>M_Q, all departures from the initial subspace worsen risk by at least (m_P-M_Q)s^2. Thus a task-optimal eigenspace cannot exhibit initial improvement, irrespective of the optimizer. If T=tI, risk is constant at fixed rank for every path. If alpha_0 is small, substantial risk movement requires a substantial normal movement budget; feature energy alone cannot supply it.

For a differentiable constant-rank A(t), its Grassmann path length is

\[
 L_N(t)=\int_0^t\|A^\dagger\dot A(I-P_A)\|_F\,ds
       =\frac1{\sqrt2}\int_0^t\|\dot P_A\|_F\,ds.
\]

The projector chord satisfies s(t)<=L_N(t). Combining with (1) gives the finite-budget bound

\[
 |R(t)-R(0)|\leq2\alpha_0L_N(t)+\omega L_N(t)^2. \tag{3}
\]

For parameter blocks, L_N<=sum_j integral eta_j||A^dagger M_j[G](I-P_A)||_F. The integral is an effective normal update budget. A bound measured retrospectively along the whole path is a verification, not a calibration-state forecast. To use (3) prospectively, one must bound that integral on a specified neighborhood or future update stream without inspecting its future task risks. Likewise, a raw learning-rate sum is not a normal movement budget unless a Jacobian and singular-value bound converts it.

## 5. Gaussian patch model: the Jordan-product mechanism

The following submodel makes the general mobility interpretable and gives a true K<C<d example. Let x~N(0,I_m), m>=2, use a single patch M(x)=xx^T, and set

\[
 u=\frac{\operatorname{svec}(xx^T-I)}{\sqrt2}.
\]

Gaussian fourth moments give Cov(u)=I_D. Let H_c=h_ch_c^T, so b_c=sqrt(2)svec(H_c). Identifying a vector with its symmetric matrix under svec, the channel mobility is exactly

\[
 T_c[S]=4(H_cS+SH_c). \tag{4}
\]

Indeed, J_cv=sqrt(2)svec(h_cv^T+vh_c^T), J_c^*S=2sqrt(2)Sh_c, and their composition gives (4). The mobility has eigenvalues 8||h_c||^2 along H_c, 4||h_c||^2 on mixed h_c/orthogonal directions, and zero on symmetric matrices supported on h_c^perp. Even perfectly isotropic whitened data therefore induce a highly anisotropic parameter mobility.

Suppose G=HA with H symmetric and set K_0=W^THW. Then

\[
 \dot b_c=-4\sqrt2\eta_h\operatorname{svec}
 \left(\sum_j(K_0)_{cj}(H_cH_j+H_jH_c)\right). \tag{5}
\]

Since

\[
 H_cH_j+H_jH_c=(h_c^Th_j)(h_ch_j^T+h_jh_c^T),
\]

new second-order directions require both filter coherence and objective/head coupling. Pairwise orthogonal or collinear groups are a closure case: their Jordan products remain in span{H_c}, so the encoder row space is conserved under this isotropic objective. This is an explicit nonlinear exclusion condition.

### A two-channel necessary and sufficient criterion

Take C=2 and linearly independent h_1,h_2. Let rho=h_1^Th_2, k_12=(W^THW)_12, S_12=h_1h_2^T+h_2h_1^T, n=(I-P_B)svec(S_12), and p=B^dagger(1,1)^T. Both n and p are nonzero. For n, write h_1,h_2 as independent coordinate vectors: S_12 cannot be a linear combination of their two squares, since an off-diagonal product of two independent linear forms is not a combination of their squares. For p, B has full row rank and BB^dagger=I_2.

Removing the self-Jordan terms in (5) gives the exact rank-one normal update

\[
 \dot B(I-P_B)=-4\sqrt2\eta_h k_{12}\rho\,(1,1)^Tn^T. \tag{6}
\]

Hence the encoder moves if and only if k_12 rho is nonzero (for eta_h>0). Its projector velocity is

\[
 \dot P_B=v n^T+n v^T,\quad
 v=-4\sqrt2\eta_h k_{12}\rho\,p,
\]

and ||dot P_B||_F=sqrt(2)||v||||n||. This holds for K=1,C=2,D>=3, a genuine encoder bottleneck.

For unit scalar tasks beta_±=(v/||v|| ± n/||n||)/sqrt(2), the optimal encoder-risk derivatives are respectively minus/plus ||v||||n||. By continuity, each sign holds on an open cone of task directions with a nonzero margin. A task exactly along n has zero first derivative, because it is initially orthogonal to the encoder; it is not a first-order information gain example. This distinction is essential.

Equation (6) isolates an interaction rather than asserting that all nonlinear encoders drift. For continuous random independent filters and a head with nonzero cross coupling, rho=0 is a measure-zero condition, but a probability claim for a particular initialization law still needs that law and its coupling distribution to be specified.

## 6. An explicit finite-amplitude rebound family with a genuine encoder bottleneck

This family strengthens the local criterion: the full parameter flow is solvable up to a positive scalar time change. It establishes a **structural rebound mechanism**, while exhibiting an explicit setting in which layer-rate competition is absent.

Take m=2 (thus D=3), C=2, K=1. Let ||h_1(0)||=||h_2(0)||, let the initial scalar head entries both equal w_0>0, and assume 0<rho_0=h_1(0)^Th_2(0)<||h_1(0)||^2. Consider any radial objective L(a)=Phi(||a||^2) for which kappa=2Phi'(V)<0 on the trajectory interval. The actual parameter rates eta_h>0 and eta_W>=0 are constant. Equal head entries and equal filter norms are invariant under this flow.

Define fixed orthonormal directions

\[
 e_+=\frac{h_1(0)+h_2(0)}{\|h_1(0)+h_2(0)\|},\qquad
 e_-=\frac{h_1(0)-h_2(0)}{\|h_1(0)-h_2(0)\|},
\]

and E_+=e_+e_+^T, E_-=e_-e_-^T, E_cross=(e_+e_-^T+e_-e_+^T)/sqrt(2). These form an orthonormal basis of Sym_2. Put lambda_±=||h_1||^2 ± h_1^Th_2, and use the increasing clock

\[
 \tau(t)=-8\eta_h\int_0^t\kappa(s)w(s)^2\,ds.
\]

### Exact solution

The parameter equations reduce to

\[
 \dot h_1=-4\eta_h\kappa w^2(\|h_1\|^2h_1+\rho h_2),\quad
 \dot h_2=-4\eta_h\kappa w^2(\rho h_1+\|h_1\|^2h_2),
\]

\[
 \dot\lambda_\pm=-8\eta_h\kappa w^2\lambda_\pm^2,
 \qquad \dot w=-\eta_W\kappa w(\lambda_+^2+\lambda_-^2).
\]

Consequently, for 0<=tau<1/lambda_+(0),

\[
 \lambda_\pm(\tau)=\frac{\lambda_\pm(0)}{1-\tau\lambda_\pm(0)}, \tag{7}
\]

\[
 w(\tau)^2=w_0^2+\frac{\eta_W}{4\eta_h}
 [\lambda_+(\tau)+\lambda_-(\tau)-\lambda_+(0)-\lambda_-(0)]. \tag{8}
\]

The filter sum and difference keep directions e_+ and e_-, respectively. In the E basis,

\[
 \operatorname{row}(B(\tau))=
 \operatorname{span}\{E_{\rm cross},\lambda_+(\tau)E_++\lambda_-(\tau)E_-\},
\]

\[
 \operatorname{row}(a(\tau))=
 \operatorname{span}\{\lambda_+(\tau)E_++\lambda_-(\tau)E_-\}.
\]

Write theta=atan(lambda_-/lambda_+). It decreases strictly from theta_0 in (0,pi/4) toward zero. If z_0=lambda_-(0)/lambda_+(0), the unique time for 0<theta<theta_0 is

\[
 \tau(\theta)=\frac{z_0-\tan\theta}
 {z_0\lambda_+(0)(1-\tan\theta)}
 <\frac1{\lambda_+(0)}. \tag{9}
\]

### Task cone and common sections

Take the unit task direction

\[
 \beta_{\psi,z}=\sqrt{1-z^2}(\cos\psi\,E_++\sin\psi\,E_-)
                 +zE_{\rm cross},
\]

with psi in [2theta_0/3,5theta_0/6] and |z|<=1/2. This set has nonempty interior on the unit task sphere; it is not restricted to the diagonal task plane. Let Y=<beta,u> with no label noise, or add independent noise and subtract its constant floor. Then

\[
 R_{\rm enc}(\tau)=(1-z^2)\sin^2(\theta(\tau)-\psi),\qquad
 R_{\rm proj}(\tau)=z^2+(1-z^2)\sin^2(\theta(\tau)-\psi). \tag{10}
\]

Thus the projection omits a fixed z^2 component, while both encoder and projection have the **same training-induced risk changes**. Their ranks remain two and one, respectively.

All tasks in this cone share comparison sections theta_1=3theta_0/4 and theta_2=theta_0/6. For either encoder or projection risk,

\[
 R(\tau_1)\leq R(0)-\Delta_1,\qquad
 R(\tau_2)\geq R(0)+\Delta_2, \tag{11}
\]

where the explicit uniform gaps are

\[
 \Delta_1=\frac34\sin(\theta_0/4)\sin(\theta_0/12)>0,
 \quad
 \Delta_2=\frac34\sin(5\theta_0/6)\sin(\theta_0/6)>0. \tag{12}
\]

To verify the constants, use sin^2 a-sin^2 b=sin(a+b)sin(a-b). The smallest initial-to-first-section improvement occurs at psi=5theta_0/6 and the smallest second-section excess at psi=2theta_0/3. Finally multiply by 1-z^2>=3/4. The comparison times are independent of the task within this cone.

This mechanism is geometric passage through a task-aligned direction, followed by continued movement toward a dominant quadratic mode. It changes encoder information in a true bottleneck; it does not rely only on head selection within a fixed full encoder.

### Explicit nonempty GUPA parameter range

Choose

\[
 \lambda_+(0)=2\varepsilon,\quad\lambda_-(0)=\varepsilon,
 \quad0<\varepsilon\leq1/100,\quad0<w_0\leq1/10,
 \quad0\leq r=\eta_W/\eta_h\leq10.
\]

For example, h_1(0)=sqrt(epsilon)e_+ +sqrt(epsilon/2)e_- and h_2(0)=sqrt(epsilon)e_+ -sqrt(epsilon/2)e_- realize these values. The orientation of e_+,e_- is arbitrary. Here theta_0=atan(1/2), so

\[
 \Delta_1=0.0033505421007594457\ldots,
 \qquad\Delta_2=0.021817886558302697\ldots .
\]

These are analytic uniform gap constants, not fitted or sampled means. Without the cross component restriction's worst-case factor 3/4, the two constants are 0.004467389467679261 and 0.029090515411070265.

Take GUPA augmentations x_j=x+epsilon_j, where epsilon_j~N(0,eta I_2) independently with 0<=eta<=1/5. On the quadratic coordinates, S=sI, D=2(s-1)I, s=(1+eta)^2 in [1,36/25]. Center each view at its own population mean. The clean and augmented covariance supports are the full Sym_2 dictionary, so the support issue of Section 1 is absent. Its scalar gradient is

\[
 \kappa=-2\Gamma_s(sV),\quad
 \Gamma_s(Q)=s\frac{Q+2}{2(Q+1)^2}-(s-1),
 \quad V=2w^2(\lambda_+^2+\lambda_-^2).
\]

The explicit rate range stays in the strictly descending regime through theta_2. Here is a rational bound, independent of numerical integration. Since atan(1/2)>1/2-(1/2)^3/3=11/24, tan(theta_2)>theta_2>3/40. At theta_2, formula (9) implies

\[
 \lambda_+(\tau_2)=2\varepsilon\frac{1-\tan\theta_2}{\tan\theta_2}
 <\frac{74}{3}\varepsilon\leq\frac14,
 \quad
 \lambda_-(\tau_2)=2\varepsilon(1-\tan\theta_2)<2\varepsilon\leq\frac1{50}.
\]

Moreover lambda_++lambda_--3epsilon<71epsilon/3, and hence (8) gives w^2<361/600<61/100. All these quantities increase in tau, so throughout [0,tau_2],

\[
 V<2\frac{61}{100}\left(\frac1{16}+\frac1{2500}\right)
   =0.076738<\frac2{25},
 \quad sV<\frac{72}{625}<\frac18.
\]

Since gamma(1/8)=68/81, the gain obeys

\[
 \Gamma_s(sV)\geq1-\frac{36}{25}\left(1-\frac{68}{81}\right)
 =\frac{173}{225}>\frac34.
\]

Thus kappa<-3/2, tau_dot>=12eta_h w_0^2, and both comparison sections are reached in finite physical time. In particular

\[
 t_2\leq\frac{\tau_2}{12\eta_hw_0^2}
 <\frac1{24\eta_hw_0^2\varepsilon}.
\]

Parameters remain finite before tau_2 by (7)–(8). The gradient is nonzero, so loss strictly decreases throughout this interval. This closes the potential circularity in assuming kappa<0: the exact invariant expressions and the displayed strict energy barrier keep the first-exit trajectory inside the descending regime.

For the upper-end example epsilon=0.01,w_0=0.1,r=10, the closed formulas give tau_2=45.80364204927046, lambda_+=0.2383018826661699, lambda_-=0.018451424372632384, w^2=0.5768832675970057, V=0.06591265423052223, and maximum-noise Q=0.09491422209195201. These decimals illustrate the formulas; the proof uses the rational inequalities above.

### What this family does and does not establish

It supplies a continuum of filter scales, orientations, task directions, augmentation strengths, and learning-rate ratios with common nonzero risk gaps. Its symmetry conditions are explicit checkable sufficient initialization conditions. They are not a high-probability initialization theorem: equal norms and equal head entries define an invariant manifold, not an open neighborhood in the full parameter space.

Within this family, theta as a function of tau is independent of the layer-rate ratio. Changing r changes w(t), energy, and physical time, but it does not select a different subspace path. Therefore this family **excludes rate-induced directional competition**, rather than proving it. The general normal-response criterion in Section 3 diagnoses competition away from the symmetry manifold. A finite-size stability theorem for that symmetry breaking would still be required to establish robust rate bifurcation.

The analytically open task cone and finite parameter ranges are substantive improvements over a single task and narrow numerical box. The structured anisotropy family below supplies a nonempty covariance range, but not an arbitrary off-manifold perturbation radius. A general perturbation radius must use a bounded vector field/Jacobian on a specified compact parameter tube and subtract the resulting risk error from (12); a bare continuity assertion would add little. Likewise, the Gaussian-patch construction is not evidence that the natural-image failures are explained by these constants.

### A structured five-percent anisotropy range with the same risk gaps

The same construction admits an explicit nonisotropic clean-input covariance interval. Let

\[
 x\sim N(0,\Sigma_x),\qquad
 \Sigma_x=\nu_+E_++\nu_-E_-,\qquad
 \frac{19}{20}\leq\nu_\pm\leq\frac{21}{20}.
\]

Write y=Sigma_x^{-1/2}x and define u=svec(yy^T-I)/sqrt(2). Let q_c=Sigma_x^{1/2}h_c be the whitened filters, and initialize

\[
 q_1(0)=\sqrt\varepsilon e_++\sqrt{\varepsilon/2}e_-,\quad
 q_2(0)=\sqrt\varepsilon e_+-\sqrt{\varepsilon/2}e_-,\quad
 w_1(0)=w_2(0)=w_0.
\]

Thus lambda_+(0)=2epsilon, lambda_-(0)=epsilon, exactly as before, where lambda and the feature coordinates are now defined using q rather than h. Training is still Euclidean in the physical h parameters, not q. Consequently the q update acquires the metric Sigma_x:

\[
 \dot q_c=-4\eta_h\kappa w_c\Sigma_x
              \left(\sum_jw_jq_jq_j^T\right)q_c.
\]

Reflection symmetry and equal head entries are invariant. With the same clock tau_dot=-8eta_h kappa w^2, the exact solution is

\[
 \lambda_\pm(\tau)=\frac{\lambda_\pm(0)}
      {1-\nu_\pm\lambda_\pm(0)\tau}, \tag{13}
\]

\[
 w^2=w_0^2+\frac r4
 \left[\frac{\lambda_+-2\varepsilon}{\nu_+}
       +\frac{\lambda_--\varepsilon}{\nu_-}\right]. \tag{14}
\]

Since 2nu_+>nu_-, theta=atan(lambda_-/lambda_+) still decreases strictly from the same theta_0=atan(1/2) to zero. The task cone, the two common sections, the risk identities (10), and both uniform gaps (12) are unchanged. Its section clock is

\[
 \tau(\theta)=\frac{1/2-\tan\theta}
 {\varepsilon(\nu_+-\nu_-\tan\theta)}
 <\frac1{2\varepsilon\nu_+}. \tag{15}
\]

Use augmentations x_j=x+epsilon_j with independent epsilon_j~N(0,eta Sigma_x), 0<=eta<=1/5. The whitened quadratic view moments remain S=sI,D=2(s-1)I, so GUPA is radial, but the actual parameter metric and its induced lambda equations are nonisotropic. The statement is therefore about a concrete physical-data anisotropy family and Euclidean training, not an artificial change to a free effective-map optimizer.

Keep 0<epsilon<=1/100, 0<w_0<=1/10, and 0<=r<=10. At the second section, let t=tan(theta_0/6)>3/40. Solving (13) at lambda_-/lambda_+=t gives

\[
 \lambda_+=\varepsilon\frac{\nu_+-\nu_-t}
                 {t(\nu_+-\nu_-/2)}
 <\frac{560}{17}\varepsilon<33\varepsilon,
\]

\[
 \lambda_-=\varepsilon\frac{\nu_+-\nu_-t}
                 {\nu_+-\nu_-/2}
 <\frac{42}{17}\varepsilon<\frac52\varepsilon.
\]

Here nu_+-nu_-/2>=17/40; all displayed denominators are separated from zero. Formula (14) then yields

\[
 w^2<\frac1{100}+\frac{10}{4}\frac{20}{19}
                \frac{65}{2}\frac1{100}
 =\frac1{100}+\frac{65}{76}<\frac{87}{100}.
\]

These bounds hold throughout the interval because lambda_± and w are increasing in tau. Therefore

\[
 V<2\frac{87}{100}
       \left[\left(\frac{33}{100}\right)^2
             +\left(\frac1{40}\right)^2\right]
 =0.1905735<\frac15,
 \qquad sV<\frac{36}{125}<\frac13.
\]

As gamma(1/3)=21/32,

\[
 \Gamma_s(sV)\geq1-\frac{36}{25}\frac{11}{32}
 =\frac{101}{200}>\frac12,
\]

so kappa<-1 and tau_dot>8eta_h w_0^2. Since tau_2<1/(2epsilon nu_+)<=10/(19epsilon),

\[
 t_2<\frac5{76\eta_hw_0^2\varepsilon}. \tag{16}
\]

This is an explicit nonempty five-percent covariance range with nonzero, unchanged risk gaps. Its restrictions must be retained: covariance eigenvectors align with the filter sum/difference axes, the initialization is balanced in whitened q coordinates, and augmentation noise is relative to Sigma_x. It is a structured anisotropy stability family, not stability under every five-percent perturbation of the input/augmentation moments or every nearby physical initialization. In particular, changing Sigma_x here adjusts physical h(0)=Sigma_x^{-1/2}q(0); it does not keep the same physical filters while perturbing the data.

## 7. Practical interpretation for the feasibility sprint

The generic local-moment representation and least-squares formula are connecting identities. The most defensible structural statements are the two-forcing near-conservation decomposition, the task-coupling versus finite-normal-budget inequality, the Jordan closure/motion criterion, and the exact bottleneck rebound family with explicit gaps.

Existing trajectory checks should ask separately whether (i) normal movement is small, (ii) task coupling is small, (iii) parameter normal responses oppose each other, and (iv) a calibration-only bound remains below or above a useful risk scale on future checkpoints. A retrospective integral can confirm a bound but cannot validate a claim of advance prediction. Natural-image validation additionally needs a clean treatment of population moments versus finite-sample OLS risk.
