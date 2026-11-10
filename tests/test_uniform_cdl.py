import unittest

import torch
import torch.nn.functional as F

from model.losses import uniform_cdl_loss


class UniformCDLLossTest(unittest.TestCase):
    def test_matches_official_uniform_cdl_without_padding(self):
        torch.manual_seed(7)
        logits = torch.randn(4, 6, requires_grad=True)
        targets = torch.tensor([0, 1, 4, 5], dtype=torch.long)
        valid_mask = torch.ones_like(logits, dtype=torch.bool)

        actual = uniform_cdl_loss(logits, targets, valid_mask)

        log_complement_probs = F.log_softmax(-logits, dim=1)
        complement_targets = torch.ones_like(logits)
        complement_targets.scatter_(1, targets.unsqueeze(1), 0.0)
        expected = -(
            complement_targets * log_complement_probs
        ).sum(dim=1) / (logits.size(1) - 1)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_padding_is_excluded_from_loss_and_gradient(self):
        targets = torch.tensor([1], dtype=torch.long)
        valid_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        logits_a = torch.tensor([[0.2, -0.1, 0.7, -1e30, -1e30]], requires_grad=True)
        logits_b = torch.tensor([[0.2, -0.1, 0.7, 1000.0, -1000.0]])

        loss_a = uniform_cdl_loss(logits_a, targets, valid_mask)
        loss_b = uniform_cdl_loss(logits_b, targets, valid_mask)
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-6, rtol=1e-6))

        loss_a.sum().backward()
        self.assertTrue(torch.equal(logits_a.grad[:, 3:], torch.zeros_like(logits_a.grad[:, 3:])))
        self.assertTrue(torch.isfinite(logits_a.grad).all())

    def test_single_valid_position_has_zero_complementary_loss(self):
        logits = torch.tensor([[0.1, 2.0, -3.0]], requires_grad=True)
        targets = torch.tensor([0], dtype=torch.long)
        valid_mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)

        loss = uniform_cdl_loss(logits, targets, valid_mask)
        self.assertTrue(torch.equal(loss, torch.zeros_like(loss)))

    def test_ground_truth_must_be_valid(self):
        logits = torch.randn(1, 3)
        targets = torch.tensor([2], dtype=torch.long)
        valid_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

        with self.assertRaises(ValueError):
            uniform_cdl_loss(logits, targets, valid_mask)


if __name__ == "__main__":
    unittest.main()
