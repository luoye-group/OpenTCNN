import unittest
from unittest.mock import patch

from tcnn.utils.simulation.dataset.download import download


class TestDownload(unittest.TestCase):
    def test_download_existing_file(self):
        # Test downloading an existing file
        url = "http://example.com/file.txt"
        save_dir = "tests"
        file_name = "file.txt"
        file_path = "tests/file.txt"

        with patch("os.path.exists", return_value=True) as mock_exists:
            result = download(url, save_dir)
            mock_exists.assert_called_with(file_path)
            self.assertEqual(result, file_path)

    @patch("requests.get")
    @patch("builtins.open", create=True)
    @patch("tqdm.tqdm")
    def test_download_successful(self, mock_tqdm, mock_open, mock_get):
        # Test successful download
        url = "http://example.com/file.txt"
        save_dir = "tests"
        file_name = "file.txt"
        file_path = "tests/file.txt"
        response_content = b"file content"
        response_headers = {"content-length": "12"}

        mock_get.return_value.ok = True
        mock_get.return_value.headers = response_headers
        mock_get.return_value.content = response_content

        result = download(url, save_dir)

        mock_get.assert_called_with(url)
        mock_open.assert_called_with(file_path, "wb")
        mock_tqdm.assert_called_with(
            desc=file_name,
            total=12,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        )
        mock_open.return_value.__enter__.return_value.write.assert_called_with(
            response_content
        )
        self.assertEqual(result, file_path)

    @patch("requests.get")
    def test_download_failed(self, mock_get):
        # Test failed download
        url = "http://example.com/file.txt"
        save_dir = "tests"
        file_name = "file.txt"

        mock_get.return_value.ok = False

        result = download(url, save_dir)

        mock_get.assert_called_with(url)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
