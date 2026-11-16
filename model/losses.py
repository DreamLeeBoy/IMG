"""Loss functions used by IMG."""

import math

import torch
import torch.nn.functional as F


def uniform_cdl_loss(logits, targets, valid_mask):
    """Compute per-sample Uniform Complementary Dissimilarity Loss.

    Args:
        logits: Temporal classification logits with shape [batch_size, seq_len].
        targets: Ground-truth temporal indices with shape [batch_size].
        valid_mask: Binary mask with shape [batch_size, seq_len]. True (or 1)
            marks a real temporal position; padded positions are excluded.

    Returns:
        A tensor with shape [batch_size]. For each sample this is the mean
        negative log-probability, under softmax(-logits), of all valid classes
        except the ground-truth class. Keeping the per-sample values allows IMG
        to apply its existing importance-aware weighting to the audio branch.

    The original UniformCDL assumes every sample has the same number of
    classes. IMG batches variable-length videos, so the uniform complementary
    target is normalized over each sample's valid temporal positions only.
    """
    if logits.dim() != 2:
        raise ValueError("logits must have shape [batch_size, seq_len]")
    if targets.dim() != 1 or targets.size(0) != logits.size(0):
        raise ValueError("targets must have shape [batch_size]")
    if valid_mask.dim() != 2 or valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must have the same [batch_size, seq_len] shape as logits")

    targets = targets.to(device=logits.device, dtype=torch.long)
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)

    target_is_valid = valid_mask.gather(1, targets.unsqueeze(1)).squeeze(1)
    if not torch.all(target_is_valid):
        raise ValueError("Every ground-truth index must point to a valid temporal position")

    # One-cold target: all valid non-target positions are equally weighted.
    complement_mask = valid_mask.clone()
    complement_mask.scatter_(1, targets.unsqueeze(1), False)
    complement_count = complement_mask.sum(dim=1)

    # CDL applies softmax to the negative logits. Mask padding *after* negation:
    # IMG's predictor sets padded logits to a large negative value, which would
    # otherwise become a large positive value after negation.
    negative_logits = -logits.float()
    negative_logits = negative_logits.masked_fill(
        ~valid_mask, torch.finfo(negative_logits.dtype).min
    )
    log_complement_probs = F.log_softmax(negative_logits, dim=1)
    log_complement_probs = log_complement_probs.masked_fill(~complement_mask, 0.0)

    per_sample_loss = -log_complement_probs.sum(dim=1) / complement_count.clamp_min(1).to(
        dtype=log_complement_probs.dtype
    )
    # A one-position sequence has no complementary class and therefore zero CDL.
    return per_sample_loss.masked_fill(complement_count.eq(0), 0.0)



def _validate_bandwidth(bandwidth):
    """Return a finite positive Python float for the fixed Gaussian bandwidth."""
    if isinstance(bandwidth, torch.Tensor):
        if bandwidth.numel() != 1 or bandwidth.requires_grad:
            raise ValueError("bandwidth must be a fixed scalar and must not require gradients")
        bandwidth = bandwidth.detach().item()
    try:
        bandwidth = float(bandwidth)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bandwidth must be a finite positive scalar") from exc
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and greater than zero")
    return bandwidth


def gaussian_complementary_targets(targets, valid_mask, bandwidth):
    """Build per-sample Gaussian temporal complementary target distributions.

    The temporal distance is normalized by each sample's own valid sequence
    length: |t - target| / (valid_length - 1). The target class and padded
    positions receive zero mass. All other valid positions receive normalized
    weights proportional to 1 - exp(-d^2 / (2 * bandwidth^2)).

    Args:
        targets: Ground-truth temporal indices, shape [batch_size].
        valid_mask: Binary valid-position mask, shape [batch_size, seq_len].
            IMG supplies prefix masks generated from each video's valid length.
        bandwidth: One fixed, globally shared positive scalar. It is not a
            learnable parameter and is not part of the autograd graph.

    Returns:
        Float32 complementary target distributions with shape [batch_size,
        seq_len]. Rows with only one valid position are all zeros because no
        complementary class exists.
    """
    bandwidth = _validate_bandwidth(bandwidth)

    if not torch.is_tensor(valid_mask) or valid_mask.dim() != 2:
        raise ValueError("valid_mask must have shape [batch_size, seq_len]")
    if not torch.is_tensor(targets) or targets.dim() != 1:
        raise ValueError("targets must have shape [batch_size]")

    batch_size, seq_len = valid_mask.shape
    if seq_len == 0:
        raise ValueError("valid_mask must contain at least one temporal position")
    if targets.size(0) != batch_size:
        raise ValueError("targets must have shape [batch_size]")

    device = valid_mask.device
    targets = targets.to(device=device, dtype=torch.long)
    valid_mask = valid_mask.to(dtype=torch.bool)

    if torch.any((targets < 0) | (targets >= seq_len)):
        raise ValueError("Every ground-truth index must be within [0, seq_len)")

    target_is_valid = valid_mask.gather(1, targets.unsqueeze(1)).squeeze(1)
    if not torch.all(target_is_valid):
        raise ValueError("Every ground-truth index must point to a valid temporal position")

    valid_lengths = valid_mask.sum(dim=1)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    normalized_denominator = (valid_lengths - 1).clamp_min(1).to(torch.float32).unsqueeze(1)
    distances = (positions - targets.to(torch.float32).unsqueeze(1)).abs()
    distances = distances / normalized_denominator

    complement_mask = valid_mask.clone()
    complement_mask.scatter_(1, targets.unsqueeze(1), False)

    # -expm1(-x) evaluates 1 - exp(-x) more accurately when x is small.
    scaled_distances = distances / bandwidth
    weights = -torch.expm1(-0.5 * scaled_distances.square())
    weights = weights.masked_fill(~complement_mask, 0.0)

    normalizer = weights.sum(dim=1, keepdim=True)
    has_complement = complement_mask.any(dim=1, keepdim=True)
    has_positive_mass = normalizer > 0

    # Normally normalize the exact Gaussian-complement weights. In the extreme
    # large-bandwidth regime, float32 can underflow every weight to zero; the
    # normalized d^2 limit is then the numerically stable equivalent.
    safe_normalizer = torch.where(
        has_positive_mass, normalizer, torch.ones_like(normalizer)
    )
    gaussian_distribution = weights / safe_normalizer

    squared_distance_weights = distances.square().masked_fill(~complement_mask, 0.0)
    squared_distance_normalizer = squared_distance_weights.sum(dim=1, keepdim=True)
    safe_squared_normalizer = torch.where(
        squared_distance_normalizer > 0,
        squared_distance_normalizer,
        torch.ones_like(squared_distance_normalizer),
    )
    squared_distance_distribution = squared_distance_weights / safe_squared_normalizer

    targets_distribution = torch.where(
        has_positive_mass,
        gaussian_distribution,
        squared_distance_distribution,
    )
    targets_distribution = targets_distribution.masked_fill(~has_complement, 0.0)

    # The target distribution is constructed only from labels and masks.
    return targets_distribution.detach()


def gaussian_cdl_loss(logits, targets, valid_mask, bandwidth):
    """Compute per-sample Gaussian temporal Complementary Dissimilarity Loss.

    The complementary target is generated from normalized temporal distance
    using a Gaussian similarity complement. The prediction distribution keeps
    the original DCL mechanism, softmax(-logits), over valid positions only.

    Args:
        logits: Temporal classification logits, shape [batch_size, seq_len].
        targets: Ground-truth temporal indices, shape [batch_size].
        valid_mask: Binary valid-position mask, shape [batch_size, seq_len].
        bandwidth: Fixed positive scalar shared by all samples, branches, and
            Start/End predictors.

    Returns:
        Per-sample loss tensor with shape [batch_size]. A sample with only one
        valid position returns zero because it has no complementary class.
    """
    if not torch.is_tensor(logits) or logits.dim() != 2:
        raise ValueError("logits must have shape [batch_size, seq_len]")
    if valid_mask.dim() != 2 or valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must have the same [batch_size, seq_len] shape as logits")

    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    targets_distribution = gaussian_complementary_targets(
        targets=targets,
        valid_mask=valid_mask,
        bandwidth=bandwidth,
    )

    # DCL predicts with softmax(-logits). Mask after negation because IMG's
    # predictor writes a very negative value into padded logits.
    negative_logits = -logits.float()
    negative_logits = negative_logits.masked_fill(
        ~valid_mask, torch.finfo(negative_logits.dtype).min
    )
    log_complement_probs = F.log_softmax(negative_logits, dim=1)
    log_complement_probs = log_complement_probs.masked_fill(~valid_mask, 0.0)

    return -(targets_distribution * log_complement_probs).sum(dim=1)

def _joint_positive_finite_scalar(name, value):
    """Validate a fixed positive scalar hyperparameter."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _joint_nonnegative_finite_scalar(name, value):
    """Validate a fixed non-negative scalar hyperparameter."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and greater than or equal to zero")
    return value


def _validate_joint_cdl_inputs(start_targets, end_targets, valid_mask):
    if not torch.is_tensor(valid_mask) or valid_mask.dim() != 2:
        raise ValueError("valid_mask must have shape [batch_size, seq_len]")
    if not torch.is_tensor(start_targets) or start_targets.dim() != 1:
        raise ValueError("start_targets must have shape [batch_size]")
    if not torch.is_tensor(end_targets) or end_targets.dim() != 1:
        raise ValueError("end_targets must have shape [batch_size]")

    batch_size, seq_len = valid_mask.shape
    if seq_len < 1:
        raise ValueError("valid_mask must contain at least one temporal position")
    if start_targets.numel() != batch_size or end_targets.numel() != batch_size:
        raise ValueError("start_targets and end_targets must match the batch size")

    device = valid_mask.device
    valid_mask = valid_mask.to(dtype=torch.bool)
    start_targets = start_targets.to(device=device, dtype=torch.long)
    end_targets = end_targets.to(device=device, dtype=torch.long)

    if torch.any((start_targets < 0) | (start_targets >= seq_len)):
        raise ValueError("start_targets must be within [0, seq_len)")
    if torch.any((end_targets < 0) | (end_targets >= seq_len)):
        raise ValueError("end_targets must be within [0, seq_len)")
    if torch.any(start_targets > end_targets):
        raise ValueError("Every ground-truth interval must satisfy start <= end")

    lengths = valid_mask.sum(dim=1)
    if torch.any(lengths < 1):
        raise ValueError("Every sample must contain at least one valid temporal position")
    expected_prefix_mask = (
        torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    )
    if not torch.equal(valid_mask, expected_prefix_mask):
        raise ValueError("valid_mask must be a prefix mask (valid positions followed by padding)")

    gt_start_is_valid = valid_mask.gather(1, start_targets.unsqueeze(1)).squeeze(1)
    gt_end_is_valid = valid_mask.gather(1, end_targets.unsqueeze(1)).squeeze(1)
    if not torch.all(gt_start_is_valid & gt_end_is_valid):
        raise ValueError("Both ground-truth boundaries must point to valid temporal positions")

    return start_targets, end_targets, valid_mask, lengths


def three_gaussian_joint_targets(
    start_targets,
    end_targets,
    valid_mask,
    iou_sigma=0.4,
    beta=0.2,
    sigma_start_left=0.1,
    sigma_start_right=0.2,
    sigma_end_left=0.2,
    sigma_end_right=0.1,
    energy_normalization="none",
):
    """Build the three-Gaussian hierarchical joint complementary target Q.

    Legal candidates are all valid upper-triangular intervals i <= j, except
    the exact ground-truth interval. The GT remains in the prediction
    distribution's normalizer in three_gaussian_joint_dcl_loss.

    IoU bins are [0,.25), [.25,.5), [.5,.75), [.75,1). For each non-empty
    bin m, its total mass is proportional to n_m * g_m, where n_m is its
    number of non-target candidates and g_m=exp(-c_m^2/(2*iou_sigma^2)).
    Therefore the mean per-candidate mass is g_m/sum_k(n_k*g_k), strictly
    decreasing across IoU-bin midpoints, regardless of candidate counts.

    Within a bin, candidates are normalized by exp(beta*E_G), with
    E_G=-log(G_start(i)*G_end(j)). Distances are normalized by L-1.
    Start uses a steeper left side and smoother right side; End is mirrored.
    The Start Gaussian is truncated at GT End (neutral G=1 beyond it), and
    the End Gaussian is truncated at GT Start (neutral G=1 before it).

    energy_normalization='zscore' optionally standardizes E_G over all
    non-target candidates in a sample before the within-bin softmax. It is
    disabled by default to reproduce the previously agreed T=5 reference
    matrix exactly.

    Returns a detached float32 tensor [B,T,T]. Each sample with at least two
    valid positions sums to one. The GT, lower triangle, and padding receive
    zero target mass. A one-position sample has no complement and returns 0.
    """
    iou_sigma = _joint_positive_finite_scalar("iou_sigma", iou_sigma)
    beta = _joint_nonnegative_finite_scalar("beta", beta)
    sigma_start_left = _joint_positive_finite_scalar("sigma_start_left", sigma_start_left)
    sigma_start_right = _joint_positive_finite_scalar("sigma_start_right", sigma_start_right)
    sigma_end_left = _joint_positive_finite_scalar("sigma_end_left", sigma_end_left)
    sigma_end_right = _joint_positive_finite_scalar("sigma_end_right", sigma_end_right)
    if energy_normalization not in ("none", "zscore"):
        raise ValueError("energy_normalization must be 'none' or 'zscore'")

    start_targets, end_targets, valid_mask, lengths = _validate_joint_cdl_inputs(
        start_targets, end_targets, valid_mask
    )
    batch_size, seq_len = valid_mask.shape
    device = valid_mask.device
    target_q = torch.zeros((batch_size, seq_len, seq_len), device=device, dtype=torch.float32)

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    start_grid = positions[:, None].expand(seq_len, seq_len)
    end_grid = positions[None, :].expand(seq_len, seq_len)
    legal_triangle = start_grid <= end_grid
    bin_centers = torch.tensor(
        [0.125, 0.375, 0.625, 0.875], device=device, dtype=torch.float32
    )

    for b in range(batch_size):
        length = int(lengths[b].item())
        gt_start = int(start_targets[b].item())
        gt_end = int(end_targets[b].item())

        candidate_mask = legal_triangle.clone()
        candidate_mask &= valid_mask[b, :, None]
        candidate_mask &= valid_mask[b, None, :]
        candidate_mask &= ~((start_grid == gt_start) & (end_grid == gt_end))
        if not torch.any(candidate_mask):
            continue

        intersection = (
            torch.minimum(end_grid, torch.tensor(float(gt_end), device=device))
            - torch.maximum(start_grid, torch.tensor(float(gt_start), device=device))
            + 1.0
        ).clamp_min(0.0)
        candidate_length = end_grid - start_grid + 1.0
        gt_length = float(gt_end - gt_start + 1)
        union = candidate_length + gt_length - intersection
        iou = intersection / union.clamp_min(1.0)

        # Exact .25/.50/.75 boundaries enter the higher-IoU bin, matching
        # the half-open intervals [lower, upper).
        layer_index = (
            (iou >= 0.25).to(torch.long)
            + (iou >= 0.50).to(torch.long)
            + (iou >= 0.75).to(torch.long)
        )
        layer_counts = torch.stack([
            (candidate_mask & (layer_index == m)).sum() for m in range(4)
        ]).to(torch.float32)
        nonempty_layers = layer_counts > 0

        # pi_m = softmax(log(n_m) - c_m^2/(2*sigma_I^2)).
        layer_logits = (
            torch.log(layer_counts.clamp_min(1.0))
            - bin_centers.square() / (2.0 * iou_sigma * iou_sigma)
        )
        layer_logits = layer_logits.masked_fill(~nonempty_layers, float("-inf"))
        layer_masses = torch.softmax(layer_logits, dim=0)

        distance_denominator = float(max(length - 1, 1))
        start_distance = (start_grid - float(gt_start)) / distance_denominator
        end_distance = (end_grid - float(gt_end)) / distance_denominator

        start_sigma = torch.where(
            start_grid < gt_start,
            torch.tensor(sigma_start_left, device=device),
            torch.tensor(sigma_start_right, device=device),
        )
        end_sigma = torch.where(
            end_grid < gt_end,
            torch.tensor(sigma_end_left, device=device),
            torch.tensor(sigma_end_right, device=device),
        )

        # -log(G) for a Gaussian similarity exp(-d^2/(2*sigma^2)).
        start_energy = 0.5 * (start_distance / start_sigma).square()
        end_energy = 0.5 * (end_distance / end_sigma).square()

        # The Start Gaussian's right branch ends at GT End; the End Gaussian's
        # left branch starts at GT Start. Outside those supports, G is neutral 1.
        start_energy = torch.where(start_grid <= gt_end, start_energy, 0.0)
        end_energy = torch.where(end_grid >= gt_start, end_energy, 0.0)
        gaussian_energy = start_energy + end_energy

        if energy_normalization == "zscore":
            candidate_energies = gaussian_energy[candidate_mask]
            energy_std = candidate_energies.std(unbiased=False)
            gaussian_energy = (
                gaussian_energy - candidate_energies.mean()
            ) / energy_std.clamp_min(1e-6)

        for m in range(4):
            in_layer = candidate_mask & (layer_index == m)
            if not torch.any(in_layer):
                continue
            within_layer_probs = torch.softmax(beta * gaussian_energy[in_layer], dim=0)
            target_q[b][in_layer] = layer_masses[m] * within_layer_probs

    return target_q.detach()


def three_gaussian_joint_dcl_loss(
    start_logits,
    end_logits,
    start_targets,
    end_targets,
    valid_mask,
    iou_sigma=0.4,
    beta=0.2,
    sigma_start_left=0.1,
    sigma_start_right=0.2,
    sigma_end_left=0.2,
    sigma_end_right=0.1,
    energy_normalization="none",
    target_q=None,
):
    """Compute per-sample Joint Gaussian-DCL on IMG's legal interval space.

    P^-(i,j) = softmax_{i<=j, valid}(-start_logits[i]-end_logits[j]).
    The prediction normalizer includes the exact GT interval; Q assigns the
    GT interval zero target mass. Returns one loss value per sample [B].
    """
    if not torch.is_tensor(start_logits) or start_logits.dim() != 2:
        raise ValueError("start_logits must have shape [batch_size, seq_len]")
    if not torch.is_tensor(end_logits) or end_logits.shape != start_logits.shape:
        raise ValueError("end_logits must have the same shape as start_logits")
    if not torch.is_tensor(valid_mask) or valid_mask.shape != start_logits.shape:
        raise ValueError("valid_mask must have the same shape as start_logits")

    start_targets, end_targets, valid_mask, _ = _validate_joint_cdl_inputs(
        start_targets, end_targets, valid_mask.to(device=start_logits.device)
    )
    if target_q is None:
        target_q = three_gaussian_joint_targets(
            start_targets=start_targets,
            end_targets=end_targets,
            valid_mask=valid_mask,
            iou_sigma=iou_sigma,
            beta=beta,
            sigma_start_left=sigma_start_left,
            sigma_start_right=sigma_start_right,
            sigma_end_left=sigma_end_left,
            sigma_end_right=sigma_end_right,
            energy_normalization=energy_normalization,
        )
    else:
        if not torch.is_tensor(target_q) or target_q.shape != (
            start_logits.size(0), start_logits.size(1), start_logits.size(1)
        ):
            raise ValueError("target_q must have shape [batch_size, seq_len, seq_len]")
        target_q = target_q.to(device=start_logits.device, dtype=torch.float32).detach()
        if not torch.isfinite(target_q).all() or torch.any(target_q < 0.0):
            raise ValueError("target_q must contain finite non-negative probabilities")

    batch_size, seq_len = start_logits.shape
    positions = torch.arange(seq_len, device=start_logits.device)
    legal_pairs = positions[:, None] <= positions[None, :]
    pair_mask = (
        legal_pairs.unsqueeze(0)
        & valid_mask[:, :, None]
        & valid_mask[:, None, :]
    )

    # Mask after negation/combination: IMG's padding logits are very negative,
    # so their negated values must not enter the joint softmax.
    negative_joint_logits = -(
        start_logits.float().unsqueeze(2) + end_logits.float().unsqueeze(1)
    )
    negative_joint_logits = negative_joint_logits.masked_fill(
        ~pair_mask, torch.finfo(negative_joint_logits.dtype).min
    )
    log_probs = F.log_softmax(negative_joint_logits.flatten(1), dim=1).view(
        batch_size, seq_len, seq_len
    )
    log_probs = log_probs.masked_fill(~pair_mask, 0.0)

    return -(target_q * log_probs).sum(dim=(1, 2))

