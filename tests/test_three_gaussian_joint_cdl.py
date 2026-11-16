import unittest

import torch
import torch.nn.functional as F

from model.losses import three_gaussian_joint_dcl_loss, three_gaussian_joint_targets


class ThreeGaussianJointCDLTest(unittest.TestCase):
    def setUp(self):
        self.mask = torch.ones(1, 5, dtype=torch.bool)
        self.start = torch.tensor([1], dtype=torch.long)
        self.end = torch.tensor([3], dtype=torch.long)

    def test_reproduces_t5_gt_2_4_reference_matrix(self):
        actual = three_gaussian_joint_targets(self.start, self.end, self.mask)
        expected = torch.tensor([[
            [0.140370, 0.137169, 0.046572, 0.013471, 0.074422],
            [0.0, 0.073421, 0.024928, 0.0, 0.013471],
            [0.0, 0.0, 0.053716, 0.024928, 0.046572],
            [0.0, 0.0, 0.0, 0.073421, 0.137169],
            [0.0, 0.0, 0.0, 0.0, 0.140370],
        ]], dtype=torch.float32)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_probability_mass_gt_and_illegal_intervals(self):
        q = three_gaussian_joint_targets(self.start, self.end, self.mask)
        self.assertTrue(torch.allclose(q.sum((1, 2)), torch.ones(1), atol=1e-6))
        self.assertEqual(q[0, 1, 3].item(), 0.0)
        self.assertEqual(torch.tril(q[0], diagonal=-1).abs().sum().item(), 0.0)

    def test_iou_layer_mean_single_class_probability_strictly_decreases(self):
        q = three_gaussian_joint_targets(self.start, self.end, self.mask)[0]
        layer_values = [[], [], [], []]
        for i in range(5):
            for j in range(i, 5):
                if (i, j) == (1, 3):
                    continue
                inter = max(0, min(j, 3) - max(i, 1) + 1)
                union = (j - i + 1) + 3 - inter
                iou = inter / union
                layer = 0 if iou < .25 else 1 if iou < .5 else 2 if iou < .75 else 3
                layer_values[layer].append(q[i, j].item())
        means = [sum(values) / len(values) for values in layer_values]
        self.assertEqual([len(values) for values in layer_values], [2, 5, 5, 2])
        self.assertTrue(all(means[k] > means[k + 1] for k in range(3)))

    def test_beta_zero_is_uniform_inside_each_iou_layer(self):
        q = three_gaussian_joint_targets(self.start, self.end, self.mask, beta=0.0)[0]
        values = torch.stack([q[0, 1], q[1, 1], q[2, 2], q[3, 3], q[3, 4]])
        self.assertTrue(torch.allclose(values, values[0].expand_as(values), atol=1e-7))

    def test_padding_has_zero_target_mass_and_zero_gradient(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        start, end = torch.tensor([1]), torch.tensor([3])
        q = three_gaussian_joint_targets(start, end, mask)
        self.assertEqual(q[:, 5:, :].abs().sum().item(), 0.0)
        self.assertEqual(q[:, :, 5:].abs().sum().item(), 0.0)
        start_logits = torch.tensor([[0.2, -0.1, 0.7, 0.3, -0.4, -1e30, -1e30]], requires_grad=True)
        end_logits = torch.tensor([[0.1, 0.4, -0.2, 0.6, -0.3, -1e30, -1e30]], requires_grad=True)
        three_gaussian_joint_dcl_loss(start_logits, end_logits, start, end, mask).sum().backward()
        self.assertTrue(torch.equal(start_logits.grad[:, 5:], torch.zeros_like(start_logits.grad[:, 5:])))
        self.assertTrue(torch.equal(end_logits.grad[:, 5:], torch.zeros_like(end_logits.grad[:, 5:])))
        self.assertTrue(torch.isfinite(start_logits.grad).all())
        self.assertTrue(torch.isfinite(end_logits.grad).all())

    def test_padding_logit_values_do_not_change_loss(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
        start, end = torch.tensor([1]), torch.tensor([3])
        start_a = torch.tensor([[0.2, -0.1, 0.7, 0.3, -0.4, -1e30]])
        end_a = torch.tensor([[0.1, 0.4, -0.2, 0.6, -0.3, -1e30]])
        start_b, end_b = start_a.clone(), end_a.clone()
        start_b[:, 5], end_b[:, 5] = 1e4, -1e4
        loss_a = three_gaussian_joint_dcl_loss(start_a, end_a, start, end, mask)
        loss_b = three_gaussian_joint_dcl_loss(start_b, end_b, start, end, mask)
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-6))

    def test_joint_loss_gradient_matches_target_minus_prediction_marginals(self):
        start_logits = torch.tensor([[0.2, -0.1, 0.7, 0.3, -0.4]], requires_grad=True)
        end_logits = torch.tensor([[0.1, 0.4, -0.2, 0.6, -0.3]], requires_grad=True)
        three_gaussian_joint_dcl_loss(
            start_logits, end_logits, self.start, self.end, self.mask
        ).sum().backward()
        q = three_gaussian_joint_targets(self.start, self.end, self.mask)
        scores = -(start_logits.detach().unsqueeze(2) + end_logits.detach().unsqueeze(1))
        triangle = torch.triu(torch.ones(5, 5, dtype=torch.bool))
        log_p = F.log_softmax(
            scores.masked_fill(~triangle, torch.finfo(torch.float32).min).flatten(1), dim=1
        ).view(1, 5, 5)
        p = log_p.exp().masked_fill(~triangle, 0.0)
        self.assertTrue(torch.allclose(start_logits.grad, q.sum(2) - p.sum(2), atol=1e-6))
        self.assertTrue(torch.allclose(end_logits.grad, q.sum(1) - p.sum(1), atol=1e-6))

    def test_prediction_normalizer_includes_gt_interval(self):
        start_logits, end_logits = torch.zeros(1, 5), torch.zeros(1, 5)
        loss = three_gaussian_joint_dcl_loss(start_logits, end_logits, self.start, self.end, self.mask)
        q = three_gaussian_joint_targets(self.start, self.end, self.mask)
        expected = q.sum((1, 2)) * torch.tensor(15.0).log()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_single_valid_position_returns_zero(self):
        mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)
        start, end = torch.tensor([0]), torch.tensor([0])
        self.assertEqual(three_gaussian_joint_targets(start, end, mask).sum().item(), 0.0)
        logits_s = torch.tensor([[0.1, -1e30, -1e30]], requires_grad=True)
        logits_e = torch.tensor([[0.2, -1e30, -1e30]], requires_grad=True)
        self.assertEqual(three_gaussian_joint_dcl_loss(logits_s, logits_e, start, end, mask).item(), 0.0)

    def test_zscore_energy_normalization_is_finite(self):
        q = three_gaussian_joint_targets(
            self.start, self.end, self.mask, energy_normalization="zscore"
        )
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(torch.allclose(q.sum((1, 2)), torch.ones(1), atol=1e-6))

    def test_invalid_gt_order_is_rejected(self):
        with self.assertRaises(ValueError):
            three_gaussian_joint_targets(torch.tensor([3]), torch.tensor([1]), self.mask)

    def test_gt_must_be_valid(self):
        mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        with self.assertRaises(ValueError):
            three_gaussian_joint_targets(torch.tensor([0]), torch.tensor([3]), mask)

    def test_non_prefix_mask_is_rejected(self):
        mask = torch.tensor([[1, 0, 1, 1, 0]], dtype=torch.bool)
        with self.assertRaises(ValueError):
            three_gaussian_joint_targets(torch.tensor([0]), torch.tensor([2]), mask)

    def test_invalid_hyperparameters_are_rejected(self):
        invalid_cases = (
            {"iou_sigma": 0.0}, {"beta": -0.1}, {"sigma_start_left": -1.0},
            {"sigma_end_right": float("inf")}, {"energy_normalization": "unknown"},
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    three_gaussian_joint_targets(self.start, self.end, self.mask, **kwargs)

    def test_batch_samples_have_independent_dynamic_layer_counts(self):
        masks = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.bool)
        q = three_gaussian_joint_targets(torch.tensor([1, 0]), torch.tensor([3, 2]), masks)
        self.assertTrue(torch.allclose(q[0].sum(), torch.tensor(1.0), atol=1e-6))
        self.assertTrue(torch.allclose(q[1].sum(), torch.tensor(1.0), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
