"""Loss functions used by IMG."""

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
