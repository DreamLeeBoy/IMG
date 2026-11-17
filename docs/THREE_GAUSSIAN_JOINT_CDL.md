# Three-Gaussian Hierarchical Joint CDL

This branch adds a loss-only extension to original IMG. It does not add a
T x T prediction head and does not change inference.

## Mathematical definition

For a valid video of length L, the candidate set is
\[
\Omega=\{(i,j):0\le i\le j<L\}\setminus\{(s^*,e^*)\}.
\]
Discrete temporal IoU uses inclusive feature-bin intervals. Four half-open
IoU bins are [0,.25), [.25,.5), [.5,.75), [.75,1). The exact GT pair is
excluded from Q, but remains in the reverse-prediction softmax denominator.

For bin midpoint c_m and count n_m:
\[
g_m=\exp(-c_m^2/(2\sigma_I^2)),\quad
\pi_m=\operatorname{softmax}_{m:n_m>0}(\log n_m-c_m^2/(2\sigma_I^2)).
\]
Therefore the mean mass per candidate is
\[
\bar q_m=\pi_m/n_m=g_m/\sum_k n_kg_k,
\]
strictly decreasing with IoU-bin midpoint among non-empty bins.

Start/End Gaussian energy:
\[
E_G(i,j)=-\log(G_s(i)G_e(j)).
\]
Default asymmetric normalized bandwidths: Start-left=0.1, Start-right=0.2,
End-left=0.2, End-right=0.1. Start support is truncated at GT End; End support
is truncated at GT Start. Within each IoU bin:
\[
r(i,j\mid m)=\operatorname{softmax}_{(i,j)\in B_m}(\beta E_G(i,j)),
\quad q(i,j)=\pi_m r(i,j\mid m).
\]

energy_normalization=none is the default and reproduces the derived T=5,
GT=[2,4], sigma_I=0.4, beta=0.2 reference matrix. zscore is an optional
ablation standardizing Gaussian energy over all non-GT candidates. Record
this setting in every experiment.

The reverse prediction distribution is
\[
P^-(i,j)=\operatorname{softmax}_{(i,j)\in\Omega_{valid}}
[-z_s(i)-z_e(j)].
\]
The softmax denominator includes the GT pair; Q assigns the GT pair zero mass.
\[
L_{JCDL}=-\sum_{i\le j}q(i,j)\log P^-(i,j).
\]

## Training

Keep original IMG Start CE + End CE, saliency, KD, and Important-Aware losses.
Add --w_cdl * L_JCDL. AV and Visual losses are averaged; Audio uses the
original boola importance weighting.

Example:
\`\`\`bash
python main.py --task charades --mode train --gpu_idx 0 --seed 515 \
  --cdl_type three_gaussian --w_cdl 1.0 \
  --joint_iou_sigma 0.4 --joint_beta 0.2 \
  --joint_sigma_start_left 0.1 --joint_sigma_start_right 0.2 \
  --joint_sigma_end_left 0.2 --joint_sigma_end_right 0.1 \
  --joint_energy_normalization none
\`\`\`

## Tests

\`\`\`bash
python -m unittest tests.test_three_gaussian_cdl -v
\`\`\`
