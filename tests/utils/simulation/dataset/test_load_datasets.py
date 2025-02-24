import unittest
from unittest.mock import patch

from tcnn.utils.simulation.dataset.load_datasets import load_urban_sound_8k


class TestLoadUrbanSound8K(unittest.TestCase):
    @patch("tcnn.utils.simulation.dataset.load_datasets.UrbanSound8K")
    @patch("torchaudio.transforms.Resample")
    # @patch("random_split")
    @patch("torch.utils.data.DataLoader")
    def test_load_urban_sound_8k(
        self, mock_dataloader, mock_resample, mock_urbansound8k
    ):
        batch_size = 32
        data_dir = "./data"

        urban_sound_dataset_mock = mock_urbansound8k.return_value
        waveform_mock = urban_sound_dataset_mock.__getitem__.return_value
        sample_rate_mock = waveform_mock[1]
        labels_mock = [0, 1, 2, 3, 4]

        mock_resample.return_value = waveform_mock
        # mock_random_split.return_value = ([1, 2, 3], [4, 5, 6])

        train_dataloader_mock = mock_dataloader.return_value
        test_dataloader_mock = mock_dataloader.return_value
        eval_dataloader_mock = mock_dataloader.return_value

        result = load_urban_sound_8k(batch_size, data_dir)

        mock_urbansound8k.assert_called_with(data_dir)
        mock_resample.assert_called_with(orig_freq=sample_rate_mock, new_freq=8000)
        mock_random_split.assert_called_with(urban_sound_dataset_mock, (8, 2))
        mock_dataloader.assert_called_with(
            [1, 2, 3], batch_size=batch_size, shuffle=True, collate_fn=mock_resample
        )
        self.assertEqual(
            result,
            (
                train_dataloader_mock,
                test_dataloader_mock,
                eval_dataloader_mock,
                waveform_mock.shape[0],
                len(labels_mock),
                labels_mock,
            ),
        )


if __name__ == "__main__":
    unittest.main()
