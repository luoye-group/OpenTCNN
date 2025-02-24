import unittest

import numpy as np

from tcnn.utils.simulation.dataset.ecg import ECGDataset


class TestECGDataset(unittest.TestCase):
    """
    Unit tests for the ECGDataset class.
    """

    def setUp(self):
        self.ecg_dataset = ECGDataset(save_dir="/home/limingbo/projects/data")

    def test_slide_and_cut(self):
        pass
        # X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        # Y = np.array([0, 1])
        # window_size = 3
        # stride = 2
        # out_X, out_Y = self.ecg_dataset.slide_and_cut(X, Y, window_size, stride)
        # expected_X = [[1, 2, 3], [3, 4, 5], [6, 7, 8], [8, 9, 10]]
        # expected_Y = [0, 0, 1, 1]
        # self.assertEqual(out_X.tolist(), expected_X)
        # self.assertEqual(out_Y.tolist(), expected_Y)

    def test_read_data_physionet_2_clean_federated(self):
        """
        Test case for the read_data_physionet_2_clean_federated method.

        This method tests the functionality of the read_data_physionet_2_clean_federated method
        by verifying the shape and content of the output data.

        Parameters:
        - m_clients (int): The number of clients.
        - test_ratio (float): The ratio of test data.
        - window_size (int): The size of the sliding window.
        - stride (int): The stride of the sliding window.

        Returns:
        None
        """
        m_clients = 2
        test_ratio = 0.2
        window_size = 3000
        stride = 500
        out_data = self.ecg_dataset.read_data_physionet_2_clean_federated(
            m_clients, test_ratio, window_size, stride
        )
        self.assertEqual(len(out_data), m_clients)
        for data in out_data:
            self.assertEqual(len(data), 5)
            X_train, X_test, Y_train, Y_test, pid_test = data
            self.assertEqual(X_train.shape[1:], (1, window_size))
            self.assertEqual(X_test.shape[1:], (1, window_size))
            self.assertEqual(X_train.shape[0], Y_train.shape[0])
            self.assertEqual(X_test.shape[0], Y_test.shape[0])
            self.assertEqual(X_test.shape[0], pid_test.shape[0])

    def test_read_data_physionet_2_clean(self):
        """
        Test case for the read_data_physionet_2_clean method.

        This method tests the functionality of the read_data_physionet_2_clean method by
        checking the shape of the returned arrays and ensuring that the number of samples
        in the training and testing sets match.

        Parameters:
        - self: The instance of the test class.

        Returns:
        - None
        """
        window_size = 3000
        stride = 500
        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            pid_test,
        ) = self.ecg_dataset.read_data_physionet_2_clean(window_size, stride)
        self.assertEqual(X_train.shape[1:], (1, window_size))
        self.assertEqual(X_test.shape[1:], (1, window_size))
        self.assertEqual(X_train.shape[0], Y_train.shape[0])
        self.assertEqual(X_test.shape[0], Y_test.shape[0])
        self.assertEqual(X_test.shape[0], pid_test.shape[0])


if __name__ == "__main__":
    unittest.main()
