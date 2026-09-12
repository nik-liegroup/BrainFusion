import pytest
import numpy as np
import pandas as pd

from brainfusion.load_experiments.load_parquet import load_parquet_samples


def write_parquet(path, n=5, value_start=0.0):
    pd.DataFrame({
        "x_image": np.arange(n, dtype=float),
        "y_image": np.arange(n, dtype=float),
        "value_background_corrected": value_start + np.arange(n, dtype=float),
    }).to_parquet(path, engine="pyarrow")


def write_contour(path, points=((0, 0), (10, 0), (10, 10), (0, 10))):
    path.write_text("\n".join(f"{x},{y}" for x, y in points))


class TestLoadParquetSamples:

    def test_pairs_matched_files_by_sorted_order(self, tmp_path):
        write_parquet(tmp_path / "sample1_data.parquet", value_start=0.0)
        write_parquet(tmp_path / "sample2_data.parquet", value_start=100.0)
        write_contour(tmp_path / "sample1_outline.txt")
        write_contour(tmp_path / "sample2_outline.txt")

        samples = load_parquet_samples(str(tmp_path), data_pattern=r"_data\.parquet$",
                                       contour_pattern=r"_outline\.txt$")
        assert len(samples) == 2
        np.testing.assert_array_equal(samples[0].dataset["value_background_corrected"][:1], [0.0])
        np.testing.assert_array_equal(samples[1].dataset["value_background_corrected"][:1], [100.0])
        assert samples[0].contour.shape == (4, 2)

    def test_name_group_in_data_pattern_becomes_filename(self, tmp_path):
        write_parquet(tmp_path / "sampleA_data.parquet")
        write_contour(tmp_path / "sampleA_outline.txt")
        samples = load_parquet_samples(str(tmp_path), data_pattern=r"(?P<name>sampleA)_data\.parquet$",
                                       contour_pattern=r"_outline\.txt$")
        assert samples[0].filename == "sampleA"

    def test_no_name_group_falls_back_to_filename_stem(self, tmp_path):
        write_parquet(tmp_path / "raw_measurement_data.parquet")
        write_contour(tmp_path / "raw_measurement_outline.txt")
        samples = load_parquet_samples(str(tmp_path), data_pattern=r"_data\.parquet$",
                                       contour_pattern=r"_outline\.txt$")
        assert samples[0].filename == "raw_measurement_data"

    def test_landmarks_pattern_loads_landmarks(self, tmp_path):
        write_parquet(tmp_path / "sample1_data.parquet")
        write_contour(tmp_path / "sample1_outline.txt")
        write_contour(tmp_path / "sample1_landmarks.txt", points=((1, 1), (2, 2)))
        samples = load_parquet_samples(str(tmp_path), data_pattern=r"_data\.parquet$",
                                       contour_pattern=r"_outline\.txt$", landmarks_pattern=r"_landmarks\.txt$")
        assert samples[0].landmarks.shape == (2, 2)

    def test_mismatched_data_and_contour_counts_raises(self, tmp_path):
        write_parquet(tmp_path / "sample1_data.parquet")
        write_parquet(tmp_path / "sample2_data.parquet")
        write_contour(tmp_path / "sample1_outline.txt")
        with pytest.raises(AssertionError, match="Found 2 data file"):
            load_parquet_samples(str(tmp_path), data_pattern=r"_data\.parquet$", contour_pattern=r"_outline\.txt$")

    def test_mismatched_landmarks_count_raises(self, tmp_path):
        write_parquet(tmp_path / "sample1_data.parquet")
        write_contour(tmp_path / "sample1_outline.txt")
        with pytest.raises(AssertionError, match="landmark file"):
            load_parquet_samples(str(tmp_path), data_pattern=r"_data\.parquet$", contour_pattern=r"_outline\.txt$",
                                 landmarks_pattern=r"_landmarks\.txt$")

    def test_sampling_size_subsamples_data(self, tmp_path):
        write_parquet(tmp_path / "sample1_data.parquet", n=20)
        write_contour(tmp_path / "sample1_outline.txt")
        samples = load_parquet_samples(str(tmp_path), data_pattern=r"_data\.parquet$",
                                       contour_pattern=r"_outline\.txt$", sampling_size=5)
        assert samples[0].dataset["value_background_corrected"].shape == (5,)
        assert samples[0].grid.shape == (5, 2)

    def test_custom_dataset_key(self, tmp_path):
        write_parquet(tmp_path / "sample1_data.parquet")
        write_contour(tmp_path / "sample1_outline.txt")
        samples = load_parquet_samples(str(tmp_path), data_pattern=r"_data\.parquet$",
                                       contour_pattern=r"_outline\.txt$", dataset_key="modulus")
        assert set(samples[0].dataset.keys()) == {"modulus"}

    def test_name_pattern_attaches_metadata(self, tmp_path):
        write_parquet(tmp_path / "sample1_data.parquet")
        write_contour(tmp_path / "sample1_outline.txt")
        samples = load_parquet_samples(str(tmp_path), data_pattern=r"_data\.parquet$",
                                       contour_pattern=r"_outline\.txt$",
                                       name_pattern=r"sample(?P<animal_number>\d+)",
                                       name_converters={"animal_number": int})
        assert samples[0].metadata == {"animal_number": 1}
