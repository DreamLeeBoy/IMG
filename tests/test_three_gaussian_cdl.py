"""Tests for the three-Gaussian hierarchical joint CDL objective."""
import unittest
import torch
from model.losses import three_gaussian_joint_dcl_loss, three_gaussian_joint_targets

class ThreeGaussianJointTargetTests(unittest.TestCase):
    def setUp(self):
        self.start = torch.tensor([1], dtype=torch.long)
        self.end = torch.tensor([3], dtype=torch.long)
        self.mask = torch.ones(1, 5, dtype=torch.bool)
        self.kwargs = dict(iou_sigma=0.4, beta=0.2,
            sigma_start_left=0.1, sigma_start_right=0.2,
            sigma_end_left=0.2, sigma_end_right=0.1,
            energy_normalization="none")

    def test_reference_t5_gt_2_4_matrix(self):
        q = three_gaussian_joint_targets(self.start, self.end, self.mask, **self.kwargs)[0]
        expected = torch.tensor([
            [0.140369965, 0.137168608, 0.046572149, 0.013470896, 0.074422083],
            [0.0, 0.073421065, 0.024928275, 0.0, 0.013470896],
            [0.0, 0.0, 0.053715999, 0.024928275, 0.046572149],
            [0.0, 0.0, 0.0, 0.073421065, 0.137168608],
            [0.0, 0.0, 0.0, 0.0, 0.140369965]])
        self.assertTrue(torch.allclose(q.cpu(), expected, atol=2e-6, rtol=2e-6))

    def test_mass_gt_and_triangle_constraints(self):
        q = three_gaussian_joint_targets(self.start, self.end, self.mask, **self.kwargs)
        self.assertTrue(torch.allclose(q.sum((1, 2)), torch.ones(1), atol=1e-6))
        self.assertEqual(float(q[0, 1, 3]), 0.0)
        self.assertEqual(float(torch.tril(q[0], diagonal=-1).sum()), 0.0)

    def test_layer_mean_probability_strictly_decreases(self):
        q = three_gaussian_joint_targets(self.start, self.end, self.mask, **self.kwargs)[0]
        groups = [[], [], [], []]
        for i in range(5):
            for j in range(i, 5):
                if (i, j) == (1, 3): continue
                inter = max(0, min(j, 3) - max(i, 1) + 1)
                union = (j - i + 1) + 3 - inter
                iou = inter / union
                layer = 3 if iou >= .75 else 2 if iou >= .50 else 1 if iou >= .25 else 0
                groups[layer].append(float(q[i, j]))
        means = [sum(v) / len(v) for v in groups]
        self.assertEqual([len(v) for v in groups], [2, 5, 5, 2])
        self.assertTrue(all(means[k] > means[k + 1] for k in range(3)))

    def test_beta_zero_is_uniform_inside_each_iou_layer(self):
        q = three_gaussian_joint_targets(self.start, self.end, self.mask,
            **dict(self.kwargs, beta=0.0))[0]
        groups = [[], [], [], []]
        for i in range(5):
            for j in range(i, 5):
                if (i, j) == (1, 3): continue
                inter = max(0, min(j, 3) - max(i, 1) + 1)
                union = (j - i + 1) + 3 - inter
                iou = inter / union
                layer = 3 if iou >= .75 else 2 if iou >= .50 else 1 if iou >= .25 else 0
                groups[layer].append(float(q[i, j]))
        for values in groups: self.assertLess(max(values) - min(values), 1e-7)

    def test_padding_is_excluded_and_valid_block_is_invariant(self):
        q5 = three_gaussian_joint_targets(self.start, self.end, self.mask, **self.kwargs)
        mask7 = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        q7 = three_gaussian_joint_targets(self.start, self.end, mask7, **self.kwargs)
        self.assertTrue(torch.allclose(q5[0], q7[0, :5, :5], atol=1e-7))
        self.assertEqual(float(q7[:, 5:, :].sum() + q7[:, :, 5:].sum()), 0.0)

    def test_zscore_energy_normalization_is_finite(self):
        q = three_gaussian_joint_targets(self.start, self.end, self.mask,
            **dict(self.kwargs, energy_normalization="zscore"))
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(torch.allclose(q.sum((1, 2)), torch.ones(1), atol=1e-6))

    def test_single_valid_position_has_no_complement(self):
        mask = torch.ones(1, 1, dtype=torch.bool)
        q = three_gaussian_joint_targets(torch.tensor([0]), torch.tensor([0]), mask, **self.kwargs)
        self.assertEqual(float(q.sum()), 0.0)

class ThreeGaussianJointLossTests(unittest.TestCase):
    def test_loss_has_finite_nonzero_gradients(self):
        torch.manual_seed(7)
        zs = torch.randn(2, 5, requires_grad=True)
        ze = torch.randn(2, 5, requires_grad=True)
        s, e = torch.tensor([1, 0]), torch.tensor([3, 4])
        mask = torch.ones(2, 5, dtype=torch.bool)
        q = three_gaussian_joint_targets(s, e, mask)
        loss = three_gaussian_joint_dcl_loss(zs, ze, s, e, mask, target_q=q)
        self.assertEqual(tuple(loss.shape), (2,))
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        for grad in (zs.grad, ze.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_reverse_softmax_normalizer_includes_gt_pair(self):
        zs = torch.tensor([[0.2, -0.1, 0.3]])
        ze = torch.tensor([[0.1, 0.4, -0.2]])
        s, e = torch.tensor([0]), torch.tensor([2])
        mask = torch.ones(1, 3, dtype=torch.bool)
        q = three_gaussian_joint_targets(s, e, mask)
        actual = three_gaussian_joint_dcl_loss(zs, ze, s, e, mask, target_q=q)
        joint = -(zs[0, :, None] + ze[0, None, :])
        legal = torch.triu(torch.ones(3, 3, dtype=torch.bool))
        expected = -(q[0][legal] * torch.log_softmax(joint[legal], dim=0)).sum().view(1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_invalid_gt_and_non_prefix_mask_raise(self):
        with self.assertRaises(ValueError):
            three_gaussian_joint_targets(torch.tensor([2]), torch.tensor([1]), torch.ones(1, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            three_gaussian_joint_targets(torch.tensor([0]), torch.tensor([2]),
                torch.tensor([[1, 0, 1]], dtype=torch.bool))

if __name__ == "__main__":
    unittest.main()
