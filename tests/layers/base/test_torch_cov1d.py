import unittest

import torch
from torch.nn import Conv1d


class TestConv1d(unittest.TestCase):
    def test_forward(self):
        # Create an instance of Conv1d
        in_channels = 3
        out_channels = 5
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        groups = 1
        bias = True
        padding_mode = "zeros"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        conv = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        # Create some input data
        batch_size = 100
        length = 10
        input_data = torch.randn(batch_size, in_channels, length).to(device)

        # Perform forward pass
        output = conv.forward(input_data)

        # Assert the output shape is correct
        expected_output_length = (
            length + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        expected_output_shape = (batch_size, out_channels, expected_output_length)
        self.assertEqual(output.shape, expected_output_shape)

    def test_backward(self):
        # Create an instance of Conv1d
        in_channels = 3
        out_channels = 5
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        groups = 1
        bias = True
        padding_mode = "zeros"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        conv = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        # Create some input data
        batch_size = 7
        length = 10
        input_data = torch.randn(batch_size, in_channels, length).to(device)

        # Perform forward pass
        output = conv.forward(input_data)
        output.sum().backward()
        # Assert the kernel gradient shape is correct
        expected_weight_gradient_shape = (out_channels, in_channels, kernel_size)
        self.assertEqual(conv.weight.grad.shape, expected_weight_gradient_shape)

        # Assert the bias gradient shape is correct
        expected_bias_gradient_shape = (out_channels,)
        self.assertEqual(conv.bias.grad.shape, expected_bias_gradient_shape)


if __name__ == "__main__":
    unittest.main()
