import unittest
import torch
from torch import nn
from tcnn.layers.compound.constant_compound_min_max_plus_sum_conv1d import ConstantCompoundMinMaxPlusSumConv1d

class TestConstantCompoundMinMaxPlusSumConv1d1p(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 6
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.batch_size = 2
        self.seq_length = 10
        self.input_tensor = torch.randn(self.batch_size, self.in_channels, self.seq_length)
        self.model = ConstantCompoundMinMaxPlusSumConv1d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding
        )

    def test_forward_shape(self):
        output = self.model(self.input_tensor)
        expected_shape = (self.batch_size, self.out_channels, self.seq_length)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_computation(self):
        output = self.model(self.input_tensor)
        self.assertIsInstance(output, torch.Tensor)

    def test_no_bias(self):
        model_no_bias = ConstantCompoundMinMaxPlusSumConv1d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False
        )
        output = model_no_bias(self.input_tensor)
        expected_shape = (self.batch_size, self.out_channels, self.seq_length)
        self.assertEqual(output.shape, expected_shape)
        self.assertIsNone(model_no_bias.bias)

    def test_reset_parameters(self):
        initial_weight = self.model.weight.clone()
        self.model.reset_parameters()
        self.assertFalse(torch.equal(initial_weight, self.model.weight))

    def test_device(self):
        model = ConstantCompoundMinMaxPlusSumConv1d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, device="cuda"
        )
        self.assertEqual(model.weight.device, torch.device("cuda", index=0))

if __name__ == "__main__":
    unittest.main()