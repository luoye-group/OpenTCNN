import unittest
import torch
from tcnn.layers.parallel.constant_parallel_min_max_plus_sum_conv1d import ConstantParallelMinMaxPlusSumConv1d

class TestConstantParallelMinMaxPlusSumConv1d(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 6
        self.kernel_size = 3
        self.batch_size = 2
        self.seq_length = 10

    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.in_channels, self.seq_length)
        model = ConstantParallelMinMaxPlusSumConv1d(self.in_channels, self.out_channels, self.kernel_size)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.seq_length - self.kernel_size + 1))

    def test_with_bias(self):
        input_tensor = torch.randn(self.batch_size, self.in_channels, self.seq_length)
        model = ConstantParallelMinMaxPlusSumConv1d(self.in_channels, self.out_channels, self.kernel_size, bias=True)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.seq_length - self.kernel_size + 1))

    def test_without_bias(self):
        input_tensor = torch.randn(self.batch_size, self.in_channels, self.seq_length)
        model = ConstantParallelMinMaxPlusSumConv1d(self.in_channels, self.out_channels, self.kernel_size, bias=False)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.seq_length - self.kernel_size + 1))

    def test_padding(self):
        padding = 1
        input_tensor = torch.randn(self.batch_size, self.in_channels, self.seq_length)
        model = ConstantParallelMinMaxPlusSumConv1d(self.in_channels, self.out_channels, self.kernel_size, padding=padding)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.seq_length))

    def test_stride(self):
        stride = 2
        input_tensor = torch.randn(self.batch_size, self.in_channels, self.seq_length)
        model = ConstantParallelMinMaxPlusSumConv1d(self.in_channels, self.out_channels, self.kernel_size, stride=stride)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, (self.seq_length - self.kernel_size) // stride + 1))


    def test_groups(self):
        groups = 3
        input_tensor = torch.randn(self.batch_size, self.in_channels, self.seq_length)
        model = ConstantParallelMinMaxPlusSumConv1d(self.in_channels, self.out_channels, self.kernel_size, groups=groups)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.seq_length - self.kernel_size + 1))

if __name__ == '__main__':
    unittest.main()
