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
