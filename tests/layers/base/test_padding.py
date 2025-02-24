import unittest

import torch
import torch.nn.functional as F


class TestPadding(unittest.TestCase):
    def test_zero_padding(self):
        padding = 1
        input_tensor = torch.arange(1 * 1 * 5 * 5).view(1, 1, 5, 5).float()
        zero_padding_tensor = F.pad(
            input_tensor, (padding, padding, padding, padding), mode="constant", value=0
        )
        expected_shape = (1, 1, 7, 7)
        self.assertEqual(zero_padding_tensor.shape, expected_shape)

    def test_reflect_padding(self):
        padding = 1
        input_tensor = torch.arange(1 * 1 * 5 * 5).view(1, 1, 5, 5).float()
        reflect_padding_tensor = F.pad(
            input_tensor, (padding, padding, padding, padding), mode="reflect"
        )
        expected_shape = (1, 1, 7, 7)
        self.assertEqual(reflect_padding_tensor.shape, expected_shape)

    def test_replicate_padding(self):
        padding = 1
        input_tensor = torch.arange(1 * 1 * 5 * 5).view(1, 1, 5, 5).float()
        replicate_padding_tensor = F.pad(
            input_tensor, (padding, padding, padding, padding), mode="replicate"
        )
        expected_shape = (1, 1, 7, 7)
        self.assertEqual(replicate_padding_tensor.shape, expected_shape)

    def test_circle_padding(self):
        padding = 1
        input_tensor = torch.arange(1 * 1 * 5 * 5).view(1, 1, 5, 5).float()
        circle_padding_tensor = F.pad(
            input_tensor, (padding, padding, padding, padding), mode="circular"
        )
        expected_shape = (1, 1, 7, 7)
        self.assertEqual(circle_padding_tensor.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
