"""
tests/test_ingestion.py
-----------------------
Unit tests for src/ingestion.py.
We only test the logic we can control without real Kaggle credentials.
The actual download call (kaggle.api.dataset_download_files) is mocked.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion import download_data


class TestDownloadData:
    def test_raises_when_kaggle_json_missing(self, tmp_path, monkeypatch):
        """Should raise FileNotFoundError if .kaggle/kaggle.json doesn't exist."""
        # Point the module's `root` to a temp dir that has NO .kaggle/kaggle.json
        import src.ingestion as ingestion_module
        monkeypatch.setattr(ingestion_module, "root", tmp_path)

        with pytest.raises(FileNotFoundError, match="Kaggle credentials not found"):
            download_data("someuser/somedataset")

    def test_creates_output_directory(self, tmp_path, monkeypatch):
        """Should create data/raw/<dataset> directory before downloading."""
        import src.ingestion as ingestion_module
        monkeypatch.setattr(ingestion_module, "root", tmp_path)

        # Create a fake .kaggle/kaggle.json so credentials check passes
        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle.json").write_text('{"username":"x","key":"y"}')

        with patch("kaggle.api.dataset_download_files") as mock_dl:
            download_data("ealaxi/paysim1")
            expected_dir = tmp_path / "data" / "raw" / "ealaxi" / "paysim1"
            assert expected_dir.exists() and expected_dir.is_dir()

    def test_calls_kaggle_api_with_correct_args(self, tmp_path, monkeypatch):
        """Should call kaggle.api.dataset_download_files with the right dataset name."""
        import src.ingestion as ingestion_module
        monkeypatch.setattr(ingestion_module, "root", tmp_path)

        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle.json").write_text('{"username":"x","key":"y"}')

        with patch("kaggle.api.dataset_download_files") as mock_dl:
            download_data("ealaxi/paysim1")
            mock_dl.assert_called_once()
            call_kwargs = mock_dl.call_args
            # First positional arg should be the dataset name
            assert call_kwargs.args[0] == "ealaxi/paysim1"
            assert call_kwargs.kwargs.get("unzip") is True
