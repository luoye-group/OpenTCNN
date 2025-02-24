import unittest

import torch
from torch.nn.functional import mse_loss

from tcnn.layers import Conv2d


class TestConv2d(unittest.TestCase):
    def test_forward(self):
        # Create an instance of Conv2d
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
        conv = Conv2d(
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
        height = 10
        width = 10
        input_data = torch.randn(batch_size, in_channels, height, width).to(device)

        # Perform forward pass
        output = conv.forward(input_data)

        # Assert the output shape is correct
        expected_output_height = (
            height + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        expected_output_width = (
            width + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        expected_output_shape = (
            batch_size,
            out_channels,
            expected_output_height,
            expected_output_width,
        )
        self.assertEqual(output.shape, expected_output_shape)

    def test_backward(self):
        # Create an instance of Conv2d
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
        conv = Conv2d(
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
        height = 10
        width = 10
        input_data = torch.randn(batch_size, in_channels, height, width).to(device)

        # Perform forward pass
        output = conv.forward(input_data)

        # Create some output data
        batch_size = 7
        output_height = output.shape[2]
        output_width = output.shape[3]
        output_data = torch.randn(
            batch_size, out_channels, output_height, output_width
        ).to(device)

        # Perform backward pass
        loss = mse_loss(output, output_data)
        loss.backward()

        # Assert the kernel gradient shape is correct
        expected_weight_gradient_shape = (
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        )
        self.assertEqual(conv.weight.grad.shape, expected_weight_gradient_shape)

        # Assert the bias gradient shape is correct
        expected_bias_gradient_shape = (out_channels,)
        self.assertEqual(conv.bias.grad.shape, expected_bias_gradient_shape)


if __name__ == "__main__":
    unittest.main()
