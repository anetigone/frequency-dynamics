# 核心定理复核（2026-10-21）

结论：本次从 Gaussian 四阶矩和实际参数目标逐步重推，未发现使当前两个核心定理失效的数学缺口。主文、证明和两份 asymmetric 检查脚本的固定增强条件现均包含 **`eta=0.1`**。非对称定理只放松六个初始化坐标，比较的是共同角度首次截面；没有证明共同物理时刻、相同更新预算或非对称状态下的学习率分叉。

复核对象：`bottleneck_family.tex`、`asymmetric_stability.tex`、`asymmetric_stability_proof.tex`、`task_geometry_proofs.tex`，以及 `asymmetric_geometry_check.py`、`asymmetric_stability_constants.py`。未修改这些数学文件或脚本；本记录不新增定理或优化扰动半径。

| 项目 | 结论 | 独立推导或直接证据 |
|---|---|---|
| 输入矩、增强与归一化 | 通过 | 对 `u=svec(xx^T-I)/sqrt(2)`，Wick 公式给 `Cov(u)=I_3`。中心化增强视图满足 `S=(1+eta)^2 I`、跨视图协方差 `I`，所以差分矩 `D=2((1+eta)^2-1)I`。固定 `eta=.1` 对应 `s_v=1.21`、`D=.42I`。 |
| 实际 Euclidean 梯度 | 通过 | 令 `F=sum_i w_i q_i q_i^T`，则有效输出 `A=sqrt(2)svec(F)^T`、`V=2||F||_F^2`。直接求导得 `grad_qi V=8w_i Fq_i`、`grad_wi V=4q_i^T Fq_i`；故 `qdot_i=-4 kappa w_i Fq_i`、`wdot_i=-2 kappa q_i^T Fq_i`，`kappa=2Phi'(V)=-2Gamma_(1.21)(1.21V)`。无丢失系数或隐含归一化。 |
| 非各向同性对称族的物理度量 | 通过 | `q=Sigma_x^(1/2)h` 给 `qdot=-eta_h Sigma_x grad_q L`，而非直接在 `q` 上做 Euclidean flow。附录的 `nu_±` 因子因此正确。反射对称和相等 head 形成不变族。 |
| 对称族闭式解与风险间隔 | 通过 | 由 `lambda_±=2A_±^2` 重推 `lambda_dot_±=-8eta_h kappa w^2 nu_± lambda_±^2`、`wdot=-eta_W kappa w(lambda_+^2+lambda_-^2)`。改变时钟得到现稿闭式解。编码器保留交叉统计量及一个对角方向，投影只保留后者，故两风险相差固定 `z^2`；正弦差公式在整个任务区间的最小值正是所列 `Delta_1,Delta_2`。 |
| 对称族有限时间屏障 | 通过 | `lambda_+<33epsilon`、`lambda_-<2.5epsilon`、`w^2<.87` 给 `V<.1905735<.2`。`s<=1.44` 进一步给 `Gamma_s(sV)>=101/200`，排除径向增益先消失。进度上界和时间下界组合得到 `t_2<5/(76eta_h w_0^2 epsilon)`。 |
| 非对称角度方程 | 通过 | 从 `q_i=sqrt(v_i)dir(alpha_i)` 重推 `alpha_1'=-w_1w_2v_2 sinD cosD`、`alpha_2'=w_1w_2v_1 sinD cosD`，其中撇号为 `ds/dt=-4kappa` 的时钟。故 `d'=-u^2m(1-z_w^2)sinD cosD`、`a'/(-d')=g`，进度 `sigma=.5log(tanD_init/tanD)` 的速度系数正确。 |
| `g/z_w` 精确方程及不变量 | 通过 | 独立展开 `v_i'=2w_i q_i^T Fq_i`、`w_i'=.5q_i^T Fq_i`，得到现稿两条有理方程；每通道 `v_i-2w_i^2` 确实守恒。保留 `z_w` 方程的负阻尼项后，`D^+N<=4.002N` 成立，实际粗系数小于 `4.000801`。 |
| 六坐标盒的源状态界 | 通过 | `e=8e-6`、`L=.1+1/sqrt(200)` 给 `v_i in [.015-2Le+2e^2,.015+2Le+2e^2]`，从而 `|g_0|<23e`、`|z_w0|<=10e`。角度扰动界 `arcsin(sqrt(2)e/sqrt(.015))<12e` 覆盖平均角和半角；`I_i in (-.005006,-.004994)`，严格为负。 |
| 截面进度、横向放大和平均转角 | 通过 | 两参考进度为约 `.2019211613/.7725906218`；源半角误差使其分别仍小于 `.203/.774`。`N_0<33e`、放大 `<2.254/22.144` 给 `N<.005847<.01`，闭合 bootstrap。积分 `|a_sigma|<=N/2`，而非把最大误差乘整个长物理时间，得到两截面 `|a|/e<17.2/99.2`。 |
| Encoder projector 算子界 | 通过 | 行空间法向量是归一化的 `R_a diag(sin^2d,-cos^2d)R_a^T`；旋转生成元在对称矩阵空间的范数不超过 `2`，故算子误差 `<=2|a|`。未误用投影头的 `sqrt(2)` 常数。 |
| Projection projector 算子界 | 通过 | 混合系数确为 `(g+z_w)/(1+gz_w)`。其非对角扰动与基准对角矩阵 Frobenius 正交，相对范数不超过混合系数绝对值。基准方向的旋转速度不超过 `sqrt(2)`，故算子误差 `<=sqrt(2)|a|+|delta_m|`。 |
| 初态风险比较及固定间隔 | 通过 | 源状态同时计入平均角、半角和混合误差，得到 encoder `<48e`、projection `<75e`。共同截面没有半角误差，两时刻 projection 界为 `<99e/<872e`，encoder 更小。比较的是风险增量，因此必须再加源误差：总界为 `174e=.001392`、`947e=.007576`。保守剩余间隔 `.001958/.014234` 严格大于主文 `.001675/.010905`。 |
| First exit 与正增益 | 通过 | 论证在首次可能的横向越界或增益消失前成立。负不变量控制 `m/u^2`，`d logm/dsigma<4.002` 给 `m<.333,u^2<.17`；`V<=8u^2m^2(1+gz_w)^2<.15084<.151`。在固定 `s_v=1.21` 下得到 `Gamma>.7340497>29/40`，排除正增益消失；参数不爆炸且进度分母不为零，能够延拓至目标截面。没有用未来已到达的事实反证其可达性。 |
| 秩、严格损失下降与时间 | 通过 | `0<D_2<=D<pi/2`、正长度和正 head 保证 encoder 秩二、projection 秩一。`q_i^T Fq_i>0` 且 `kappa<0`，head 梯度非零，故真实 Euclidean flow 严格降损失。当前非零增强给 `-4kappa>5.8`，不是旧版的 `>6`；正文已正确更新为 `sigma_dot>.000869`、`t_2<900`。 |

## 实际执行的检查与证明边界

使用指定解释器实际重跑：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/theory/asymmetric_geometry_check.py
& H:/anaconda3/envs/py312/python.exe -B experiments/code/theory/asymmetric_stability_constants.py
```

两者均退出成功。第一个脚本检查 81 个非对称闭式状态、162 组物理梯度（增强功率 `0` 与 `.1`），并检查 64 个初始化盒角点。最大误差：物理梯度 `2.08e-17`、polar 方程 `6.83e-18`、projector 公式 `5.55e-16`。这些是实现与系数核对，不证明连续初始化盒，更不证明盒内初始化出现概率。

第二个脚本的断言使用 `Fraction` 区间端点和外向有理数舍入；本次另行审读了 atan 交错余项、sin/cos Taylor 阶数余项、正参数 log 尾和及 exp 几何尾，均形成有效包含。浮点仅用于展示。它验证常数算术，不能替代微分方程和 projector 推导。当前输出的能量上界为 `0.1508392033160904`、增益下界为 `0.7340497359330658`、物理时间上界为 `890.0488602939125`，与主文保守时间 `900` 一致。未下载数据、训练或积分新轨迹。

## 未解决范围及文字备注

- 无本轮发现的实质证明缺口；这不等于已经确立文献新颖性或自然图像预测价值。
- 当前稳定性结论固定 `Sigma=I`、增强功率 `.1` 和两学习率均为一；不能将对称族的协方差区间与非对称初始化盒直接相乘为一个联合鲁棒性定理。
- 共同截面不是共同物理时间，也不是共同能量或参数更新预算；当前文本已作区分。
- 仅有两处轻微记号可进一步清楚：附录可显式写 `D_0=2d_0`；two-filter motion criterion 中的 “independent filters” 指线性无关。它们不改变上述结论。
