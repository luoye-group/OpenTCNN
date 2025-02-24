import unittest
from unittest.mock import patch

from torchvision import datasets, transforms

from tcnn.utils.simulation.dataset.datasets_archive import load_data


class TestLoadData(unittest.TestCase):
    @patch("os.makedirs")
    @patch("datasets.MNIST")
    def test_load_data_mnist(self, mock_makedirs, mock_mnist):
        dataset_name = "mnist"
        cache_dir = "./data"

        traintd_mock = mock_mnist.return_value
        testtd_mock = mock_mnist.return_value

        result = load_data(dataset_name, cache_dir)

        mock_makedirs.assert_called_with(cache_dir, exist_ok=True)
        mock_mnist.assert_called_with(
            root=cache_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self.assertEqual(result, (traintd_mock, testtd_mock))

    @patch("os.makedirs")
    @patch("datasets.FashionMNIST")
    def test_load_data_fashion(self, mock_makedirs, mock_fashionmnist):
        dataset_name = "fashion"
        cache_dir = "./data"

        traintd_mock = mock_fashionmnist.return_value
        testtd_mock = mock_fashionmnist.return_value

        result = load_data(dataset_name, cache_dir)

        mock_makedirs.assert_called_with(cache_dir, exist_ok=True)
        mock_fashionmnist.assert_called_with(
            root=cache_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                ]
            ),
        )
        self.assertEqual(result, (traintd_mock, testtd_mock))

    # Add more test cases for other datasets...


if __name__ == "__main__":
    unittest.main()
