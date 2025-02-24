import copy
import unittest

import torch
import torch.nn.functional as F

from tcnn.layers.functional import (
    channels_max1d,
    channels_max2d,
    channels_min1d,
    channels_min2d,
    channels_sum1d,
    channels_sum2d,
    conv1d,
    conv2d,
    max_plus1d,
    max_plus2d,
    maxplus,
    min_plus1d,
    min_plus2d,
    minplus,
    preprocess1d,
    preprocess2d,
    unsqueeze,
    unfold
)


class TestFunctional(unittest.TestCase):
    def test_preprocess1d(self):
        # Create input and weight tensors
        n = 2
        c = 3
        l_in = 10
        d = 4
        k = 3
        input_tensor = torch.randn(n, c, l_in)
        weight_tensor = torch.randn(d, c, k)
        stride = 2

        # Call the preprocess1d function
        preprocessed_input, preprocessed_weight = preprocess1d(
            input_tensor, weight_tensor, stride
        )

        # Check the shape of preprocessed input tensor
        expected_input_shape = (n, 1, c, (l_in - k) // stride + 1, k)
        self.assertEqual(preprocessed_input.shape, expected_input_shape)

        # Check the shape of preprocessed weight tensor
        expected_weight_shape = (1, d, c, 1, k)
        self.assertEqual(preprocessed_weight.shape, expected_weight_shape)

    def test_conv1d(self):
        # Create input and weight tensors
        n = 2
        c = 3
        l_in = 10
        d = 4
        k = 3
        stride = 1
        input_tensor = torch.randn(n, c, l_in)
        weight_tensor = torch.randn(d, c, k)

        # Call the conv1d function
        input_tensor, weight_tensor = preprocess1d(input_tensor, weight_tensor, stride)
        output = conv1d(input_tensor, weight_tensor)

        # Check the shape of output tensor
        expected_output_shape = (n, d, c, (l_in - k) // stride + 1)
        self.assertEqual(output.shape, expected_output_shape)

    def test_max_plus1d(self):
        # Create input and weight tensors
        n = 2
        c = 3
        l_in = 10
        d = 4
        k = 3
        stride = 1
        input_tensor = torch.randn(n, c, l_in)
        weight_tensor = torch.randn(d, c, k)

        # Call the max_plus1d function
        input_tensor, weight_tensor = preprocess1d(input_tensor, weight_tensor, stride)
        output = max_plus1d(input_tensor, weight_tensor)

        # Check the shape of output tensor
        expected_output_shape = (n, d, c, (l_in - k) // stride + 1)
        self.assertEqual(output.shape, expected_output_shape)

    def test_min_plus1d(self):
        # Create input and weight tensors
        n = 2
        c = 3
        l_in = 10
        d = 4
        k = 3
        stride = 1
        input_tensor = torch.randn(n, c, l_in)
        weight_tensor = torch.randn(d, c, k)

        # Call the min_plus1d function
        input_tensor, weight_tensor = preprocess1d(input_tensor, weight_tensor, stride)
        output = min_plus1d(input_tensor, weight_tensor)

        # Check the shape of output tensor
        expected_output_shape = (n, d, c, (l_in - k) // stride + 1)
        self.assertEqual(output.shape, expected_output_shape)

    def channels_sum1d(self):
        # Create input tensor
        n = 2
        c = 3
        l_in = 10
        input_tensor = torch.randn(n, c, l_in)

        # Call the channels_sum1d function
        output = channels_sum1d(input_tensor)

        # Check the shape of output tensor
        expected_output_shape = (n, 1, l_in)
        self.assertEqual(output.shape, expected_output_shape)

    def test_channels_min1d(self):
        # Create input tensor
        n = 2
        c = 3
        l_in = 10
        d = 7
        input_tensor = torch.randn(n, d, c, l_in)

        # Call the channels_min1d function
        output = channels_min1d(input_tensor)

        # Check the shape of output tensor
        expected_output_shape = (n, d, l_in)
        self.assertEqual(output.shape, expected_output_shape)

    def test_channels_max1d(self):
        # Create input tensor
        n = 2
        c = 3
        d = 5
        l_in = 10
        input_tensor = torch.randn(n, d, c, l_in)

        # Call the channels_max1d function
        output = channels_max1d(input_tensor)

        # Check the shape of output tensor
        expected_output_shape = (n, d, l_in)
        self.assertEqual(output.shape, expected_output_shape)

    def test_preprocess2d(self):
        # Create input and weight tensors
        n = 2
        c = 3
        h_in = 10
        w_in = 10
        d = 4
        k = 3
        j = 3
        input_tensor = torch.randn(n, c, h_in, w_in)
        weight_tensor = torch.randn(d, c, k, j)
        stride = (2, 2)

        # Call the preprocess2d function
        preprocessed_input, preprocessed_weight = preprocess2d(
            input_tensor, weight_tensor, stride
        )

        # Check the shape of preprocessed input tensor
        expected_input_shape = (
            n,
            1,
            c,
            (h_in - k) // stride[0] + 1,
            (w_in - j) // stride[1] + 1,
            k,
            j,
        )
        self.assertEqual(preprocessed_input.shape, expected_input_shape)

        # Check the shape of preprocessed weight tensor
        expected_weight_shape = (1, d, c, 1, 1, k, j)
        self.assertEqual(preprocessed_weight.shape, expected_weight_shape)

    def test_conv2d(self):
        # Create input and weight tensors
        n = 2
        c = 3
        h_in = 10
        w_in = 10
        d = 4
        k = 3
        j = 3
        input_tensor = torch.randn(n, c, h_in, w_in)
        weight_tensor = torch.randn(d, c, k, j)
        stride = (1, 2)

        # Call the conv2d function
        input_tensor, weight_tensor = preprocess2d(input_tensor, weight_tensor, stride)
        output = conv2d(input_tensor, weight_tensor)

        # Check the shape of output tensor
        expected_output_shape = (
            n,
            d,
            c,
            (h_in - k) // stride[0] + 1,
            (w_in - j) // stride[1] + 1,
        )
        self.assertEqual(output.shape, expected_output_shape)

    def test_max_plus2d(self):
        # Create input and weight tensors
        n = 2
        c = 3
        h_in = 10
        w_in = 10
        d = 4
        k = 3
        j = 3
        stride = (1, 2)
        input_tensor = torch.randn(n, c, h_in, w_in)
        weight_tensor = torch.randn(d, c, k, j)

        # Call the max_plus2d function
        input_tensor, weight_tensor = preprocess2d(input_tensor, weight_tensor, stride)
        output = max_plus2d(input_tensor, weight_tensor)

        # Check the shape of output tensor
        expected_output_shape = (
            n,
            d,
            c,
            (h_in - k) // stride[0] + 1,
            (w_in - j) // stride[1] + 1,
        )
        self.assertEqual(output.shape, expected_output_shape)

    def test_min_plus2d(self):
        # Create input and weight tensors
        n = 2
        c = 3
        h_in = 10
        w_in = 10
        d = 4
        k = 3
        j = 3
        stride = (1, 2)
        input_tensor = torch.randn(n, c, h_in, w_in)
        weight_tensor = torch.randn(d, c, k, j)

        # Call the min_plus2d function
        input_tensor, weight_tensor = preprocess2d(input_tensor, weight_tensor, stride)
        output = min_plus2d(input_tensor, weight_tensor)

        # Check the shape of output tensor
        expected_output_shape = (
            n,
            d,
            c,
            (h_in - k) // stride[0] + 1,
            (w_in - j) // stride[1] + 1,
        )
        self.assertEqual(output.shape, expected_output_shape)

    def test_channels_sum2d(self):
        # Create input tensor
        n = 2
        c = 3
        d = 7
        h_in = 10
        w_in = 10
        input_tensor = torch.randn(n, d, c, h_in, w_in)

        # Call the channels_sum2d function
        output = channels_sum2d(input_tensor)

        # Check the shape of output tensor
        expected_output_shape = (n, d, h_in, w_in)
        self.assertEqual(output.shape, expected_output_shape)

    def test_channels_min2d(self):
        # Create input tensor
        n = 2
        c = 3
        d = 7
        h_in = 10
        w_in = 10
        input_tensor = torch.randn(n, d, c, h_in, w_in)

        # Call the channels_min2d function
        output = channels_min2d(input_tensor)

        # Check the shape of output tensor
        expected_output_shape = (n, d, h_in, w_in)
        self.assertEqual(output.shape, expected_output_shape)

    def test_channels_max2d(self):
        # Create input tensor
        n = 2
        c = 3
        d = 5
        h_in = 10
        w_in = 10
        input_tensor = torch.randn(n, d, c, h_in, w_in)

        # Call the channels_max2d function
        output = channels_max2d(input_tensor)

        # Check the shape of output tensor
        expected_output_shape = (n, d, h_in, w_in)
        self.assertEqual(output.shape, expected_output_shape)



if __name__ == "__main__":
    unittest.main()
