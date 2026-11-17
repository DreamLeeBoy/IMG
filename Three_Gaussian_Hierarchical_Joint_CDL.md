# Three-Gaussian Hierarchical Joint Complementary Learning (3G-HJCDL)

## Scope

This implementation adds a joint complementary loss to the original IMG temporal localization heads. It does not add a pairwise prediction head, change IMGNet.py/layers.py, or change inference. The existing Start and End logits are combined only inside the auxiliary training loss.

The branch is feature/three-gaussian-hjcdl.

## 1. Joint candidate space


For each sample with valid length \(L\), let the ground-truth interval be inclusive indices \((s^*,e^*)\). The legal interval set is

$$
\Omega=\mathrm{Set}\left((i,j):0\le i\le j<L\right)
$$

The exact GT interval is excluded from the target support:

$$
\Omega^-=\Omega-\{(s^*,e^*)\}.
$$

It remains in the prediction softmax denominator. Padding and lower-triangular pairs are excluded from both target and prediction distributions.

Candidate IoU is computed using inclusive temporal indices:

$$
IoU(i,j)=
\frac{|[i,j]\cap[s^*,e^*]|}
{|[i,j]\cup[s^*,e^*]|}
$$

## 2. IoU Gaussian: dynamic layer mass

Four half-open IoU bins are used:

$$
B_1=[0,0.25),\quad B_2=[0.25,0.5),\quad
B_3=[0.5,0.75),\quad B_4=[0.75,1).
$$
The GT interval is removed before counting. Let n_m=|B_m intersect Omega^-| and let the bin centers be

$$
c=(0.125,0.375,0.625,0.875),\qquad \mu_I=0.
$$
The IoU Gaussian is

$$
g_m=\exp\left(-\frac{c_m^2}{2\sigma_I^2}\right).
$$
The dynamic layer mass is

$$
\pi_m=
\text{Softmax}_{m:n_m>0}
\left(\log n_m-\frac{c_m^2}{2\sigma_I^2}\right).
$$
Empty bins are omitted. This is equivalent to pi_m=n_m*g_m/sum_k(n_k*g_k). Consequently,

$$
\frac{\pi_m}{n_m}=\frac{g_m}{\sum_{k} n_k g_k},
$$
so the mean per-candidate mass strictly decreases with the IoU-bin center, independently of candidate counts. Individual candidates in different bins are not required to be strictly ordered.

## 3. Asymmetric Start/End Gaussians

Distances are normalized by the sample's valid temporal span L-1. The default normalized bandwidths are:

| Parameter | Default | Meaning |
|---|---:|---|
| sigma_start_left | 0.1 | Steeper side before GT Start |
| sigma_start_right | 0.2 | Smoother side after GT Start |
| sigma_end_left | 0.2 | Smoother side before GT End |
| sigma_end_right | 0.1 | Steeper side after GT End |

The Gaussian similarities are represented through their negative log energies:

$$
E_s(i)=\frac12\left(\frac{(i-s^*)/(L-1)}
{\sigma_s^{\,\mathrm{side}(i)}}\right)^2,
$$

$$
E_e(j)=\frac12\left(\frac{(j-e^*)/(L-1)}
{\sigma_e^{\,\text{side}(j)}}\right)^2,
\qquad E_G(i,j)=E_s(i)+E_e(j).
$$

The Start Gaussian's right branch terminates at GT End: for i>e*, its factor is neutral G_s(i)=1, equivalently E_s(i)=0. The End Gaussian's left branch starts at GT Start: for j<s*, G_e(j)=1, equivalently E_e(j)=0. This neutral truncation keeps all legal non-GT intervals eligible without creating infinite energy.

Equivalently, E_G(i,j)=-log(G_s(i)*G_e(j)).

Within each non-empty IoU bin:

$$
r(i,j\,|\,m)=
\frac{\exp(\beta E_G(i,j))}
{\sum_{(a,b)\in B_m\cap\Omega^-}\exp(\beta E_G(a,b))}.
$$
The final target is

$$
Q(i,j)=\pi_m r(i,j\,|\,m),\quad (i,j)\in B_m\cap\Omega^-
$$
and zero elsewhere. Each non-empty layer sums to pi_m, and the full target sums to one for samples with at least two valid positions.

### Optional energy standardization

The option joint_energy_normalization=zscore standardizes E_G across all non-target legal intervals in one sample before the within-bin softmax. The default is none; this reproduces the numerical reference below. Treat z-score normalization as a separate ablation because it changes the meaning of beta.

## 4. Numerical reference: T=5, GT=[2,4]

With one-based display indices, sigma_I=.4, beta=.2, and the default boundary bandwidths, the non-target layer counts are [2,5,5,2]. The IoU Gaussian values are approximately

$$
[0.952345,0.644389,0.295023,0.091394].
$$
The resulting reference target matrix is:

$$
Q=
\left[
\begin{array}{ccccc}
.140370&.137169&.046572&.013471&.074422\\
0&.073421&.024928&0&.013471\\
0&0&.053716&.024928&.046572\\
0&0&0&.073421&.137169\\
0&0&0&0&.140370
\end{array}
\right].
$$
The GT cell (2,4) is zero. Rows below the diagonal are illegal. The total mass is one. The layer mean single-candidate masses are approximately 14.0370%, 9.4979%, 4.3485%, and 1.3471%.

## 5. Joint reverse prediction and loss

For original IMG logits z_s,z_e, define

$$
P^{-}(i,j)=
\frac{\exp[-z_s(i)-z_e(j)]}
{\sum_{(a,b)\in\Omega}\exp[-z_s(a)-z_e(b)]}.
$$
The normalizer includes the GT interval. The loss is

$$
\mathcal L_{JCDL}=-\sum_{(i,j)\in\Omega}Q(i,j)\log P^{-}(i,j).
$$
Its logit gradients are the marginals of the joint residual:

$$
\frac{\partial L}{\partial z_s(k)}
=Q_s(k)-P^{-}_s(k),\qquad
\frac{\partial L}{\partial z_e(k)}
=Q_e(k)-P^{-}_e(k).
$$
The full joint target is not generally factorable into independent Start and End marginals. The loss nevertheless trains the existing factorized IMG prediction mechanism without changing model architecture. This expressivity limitation is intentional for the first experiment.

## 6. IMG integration

- The GT-derived target Q is built once per batch and shared by AV, Visual, and Audio; each branch evaluates its own Start/End logits against that same Q.
- The AV and Visual per-sample losses are averaged.
- Audio keeps the original boola importance-aware weighting.
- The original localization CE, saliency loss, KD loss, and Important-Aware loss are unchanged.
- w_cdl controls the weight. cdl_type defaults to uniform, so legacy runs remain unchanged.

## 7. Parameters

~~~bash
--cdl_type three_gaussian
--w_cdl 1.0
--joint_iou_sigma 0.4
--joint_beta 0.2
--joint_sigma_start_left 0.1
--joint_sigma_start_right 0.2
--joint_sigma_end_left 0.2
--joint_sigma_end_right 0.1
--joint_energy_normalization none
~~~

All bandwidths are positive fixed hyperparameters; beta is non-negative. They are not learned.

## 8. Tests

Run from the repository root:

~~~bash
python -m unittest discover -s tests -p 'test_*cdl*.py' -v
python -m py_compile main.py model/losses.py
~~~

The joint tests cover the numerical reference matrix, probability normalization, GT exclusion, triangular masking, layer mean monotonicity, beta=0 behavior, padding invariance, zero padding gradients, analytical marginal gradients, one-position sequences, invalid inputs, and optional energy normalization.

These are loss-level and integration-code checks. They do not replace training on Charades/ActivityNet or statistical evaluation over multiple seeds.
