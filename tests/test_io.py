import pytest
import numpy as np
import pandas as pd

from brainfusion.sample import Sample
from brainfusion.io import (parse_name, attach_metadata, check_parameters, export_analysis, import_analysis,
                            get_roi_from_txt, read_parquet_file)


class TestParseName:

    def test_extracts_named_groups_with_converters(self):
        fields = parse_name("#1_XenopusExposedBrain_Control_Stage37_20260729",
                            r"#(?P<animal_number>\d+)_.*?_(?P<condition>[A-Za-z]+)_Stage(?P<stage>\d+)",
                            converters={"animal_number": int, "stage": int})
        assert fields == {"animal_number": 1, "condition": "Control", "stage": 37}

    def test_groups_without_converter_stay_strings(self):
        fields = parse_name("sample_42", r"sample_(?P<id>\d+)")
        assert fields == {"id": "42"}

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="does not match pattern"):
            parse_name("no_numbers_here", r"(?P<id>\d+)")

    def test_matches_anywhere_in_string(self):
        fields = parse_name("prefix_sample_42_suffix", r"sample_(?P<id>\d+)")
        assert fields == {"id": "42"}


class TestAttachMetadata:

    def test_single_sample_merges_metadata(self):
        sample = Sample(contour=np.zeros((3, 2)), metadata={"existing": "value"})
        updated = attach_metadata(sample, {"animal_number": 1})
        assert updated.metadata == {"existing": "value", "animal_number": 1}
        assert sample.metadata == {"existing": "value"}  # original untouched

    def test_list_of_samples_each_get_metadata(self):
        samples = [Sample(contour=np.zeros((3, 2)), filename="a"), Sample(contour=np.zeros((3, 2)), filename="b")]
        updated = attach_metadata(samples, {"stage": 37})
        assert [s.metadata for s in updated] == [{"stage": 37}, {"stage": 37}]

    def test_new_metadata_overrides_existing_key(self):
        sample = Sample(contour=np.zeros((3, 2)), metadata={"stage": 1})
        updated = attach_metadata(sample, {"stage": 2})
        assert updated.metadata == {"stage": 2}


class TestCheckParameters:

    def test_no_output_when_parameters_match(self, capsys):
        check_parameters({"a": 1, "b": "x"}, {"a": 1, "b": "x"})
        assert capsys.readouterr().out == ""

    def test_prints_differences(self, capsys):
        check_parameters({"a": 1}, {"a": 2})
        assert "Differences found" in capsys.readouterr().out

    def test_missing_key_in_loaded_reported_as_difference(self, capsys):
        check_parameters({"a": 1, "new_param": True}, {"a": 1})
        assert "Differences found" in capsys.readouterr().out


class TestGetRoiFromTxt:

    def test_reads_comma_delimited_coordinates(self, tmp_path):
        path = tmp_path / "roi.txt"
        path.write_text("0,0\n1,0\n1,1\n")
        result = get_roi_from_txt(str(path), delimiter=',')
        np.testing.assert_array_equal(result, [[0, 0], [1, 0], [1, 1]])

    def test_skips_header_lines(self, tmp_path):
        path = tmp_path / "roi.txt"
        path.write_text("header\n0,0\n1,1\n")
        result = get_roi_from_txt(str(path), delimiter=',', skip=1)
        np.testing.assert_array_equal(result, [[0, 0], [1, 1]])

    def test_missing_file_returns_none(self, capsys):
        result = get_roi_from_txt("does_not_exist.txt")
        assert result is None
        assert "not a valid .txt file" in capsys.readouterr().out

    def test_malformed_content_returns_none(self, tmp_path, capsys):
        path = tmp_path / "roi.txt"
        path.write_text("not,numbers\n")
        result = get_roi_from_txt(str(path), delimiter=',')
        assert result is None


class TestReadParquetFile:

    def test_reads_grid_and_data_columns(self, tmp_path):
        path = tmp_path / "data.parquet"
        pd.DataFrame({"x_image": [0, 1, 2], "y_image": [0, 1, 2],
                     "value_background_corrected": [10, 20, 30]}).to_parquet(path, engine='pyarrow')
        grid, data = read_parquet_file(str(path))
        np.testing.assert_array_equal(grid, [[0, 0], [1, 1], [2, 2]])
        np.testing.assert_array_equal(data, [10, 20, 30])

    def test_reads_image_pivoted_on_xy(self, tmp_path):
        path = tmp_path / "image.parquet"
        pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 0, 1, 1], "value_orig": [1, 2, 3, 4]}).to_parquet(
            path, engine='pyarrow')
        img = read_parquet_file(str(path), image=True)
        np.testing.assert_array_equal(img, [[1, 2], [3, 4]])


class TestExportImportAnalysisRoundTrip:

    def test_round_trips_arrays_dicts_and_strings(self, tmp_path):
        path = str(tmp_path / "analysis.h5")
        analysis = {
            "grid": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "filenames": ["a", "b"],
            "dataset": {"value": np.array([1.0, 2.0])},
            "scalar": 42,
        }
        export_analysis(path, analysis, params={"clustering": "Mean"})
        loaded, params = import_analysis(path)

        np.testing.assert_array_equal(loaded["grid"], analysis["grid"])
        assert loaded["filenames"] == ["a", "b"]
        np.testing.assert_array_equal(loaded["dataset"]["value"], analysis["dataset"]["value"])
        assert loaded["scalar"] == 42
        assert params == {"clustering": "Mean"}

    def test_round_trips_none_scalar_and_none_in_lists(self, tmp_path):
        path = str(tmp_path / "analysis.h5")
        analysis = {
            "missing_value": None,
            "arrays_with_gap": [np.array([1.0, 2.0]), None],
            "numbers_with_gap": [1.0, None, 3.0],
        }
        export_analysis(path, analysis, params={})
        loaded, _ = import_analysis(path)

        assert loaded["missing_value"] is None
        assert loaded["arrays_with_gap"][1] is None
        np.testing.assert_array_equal(loaded["arrays_with_gap"][0], [1.0, 2.0])
        assert loaded["numbers_with_gap"][1] is None
        assert loaded["numbers_with_gap"][0] == 1.0

    def test_round_trips_list_of_dicts(self, tmp_path):
        path = str(tmp_path / "analysis.h5")
        analysis = {"per_sample_metadata": [{"stage": 1}, {"stage": 2}]}
        export_analysis(path, analysis, params={})
        loaded, _ = import_analysis(path)
        assert loaded["per_sample_metadata"] == [{"stage": 1}, {"stage": 2}]
