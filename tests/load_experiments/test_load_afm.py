import pytest
import numpy as np
import pandas as pd
from PIL import Image

from brainfusion.load_experiments.load_afm import load_batchforce_single, load_batchforce_all


def make_batchforce_folder(base_path, name="#1_sample", orientation="right", landmarks=False, grid_ext="csv"):
    folder = base_path / name
    (folder / "region analysis").mkdir(parents=True)
    (folder / "Pics" / "calibration").mkdir(parents=True)

    pd.DataFrame({"modulus": [1.0, 2.0, 3.0], "x_image": [2, 4, 6], "y_image": [2, 4, 6],
                 "x": [2.0, 4.0, 6.0], "y": [2.0, 4.0, 6.0]}).to_csv(folder / "region analysis" / "data.csv",
                                                                    index=False)

    if grid_ext == "csv":
        np.savetxt(folder / "GridInversionMatrix.csv", np.eye(3), delimiter=",")
    else:
        (folder / f"GridInversionMatrix.{grid_ext}").write_text("dummy")

    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(folder / "Pics" / "calibration" / "overview.tif")

    contour_text = "0\t0\n8\t0\n8\t8\n0\t8\n"
    suffix = "oriLeft" if orientation == "left" else ("oriRight" if orientation == "right" else None)
    if suffix is not None:
        (folder / "Pics" / "calibration" / f"brain_outline_{suffix}.txt").write_text(contour_text)

    if landmarks:
        (folder / "Pics" / "calibration" / "landmarks.txt").write_text("1\t1\n7\t7\n")

    return folder


def make_long_format_data(path):
    pd.DataFrame({
        "result_parameter": ["Reduced apparent elastic modulus"] * 3,
        "result": [1.0, 2.0, 3.0],
        "x_image": [2, 4, 6], "y_image": [2, 4, 6], "x": [2.0, 4.0, 6.0], "y": [2.0, 4.0, 6.0],
    }).to_csv(path, index=False)


class TestLoadBatchforceSingle:

    def test_long_format_result_parameter_export(self, tmp_path):
        # Regression test: some batchforce versions export one row per (point, quantity) with the
        # quantity's name in 'result_parameter' and its value in 'result', instead of one column per
        # quantity - `afm_variables` as a dict maps the desired dataset key to that label.
        folder = make_batchforce_folder(tmp_path, orientation="right")
        make_long_format_data(folder / "region analysis" / "data.csv")
        sample = load_batchforce_single(str(folder), afm_variables={"modulus": "Reduced apparent elastic modulus"})
        assert set(sample.dataset.keys()) == {"modulus"}
        np.testing.assert_array_equal(sample.dataset["modulus"], [1.0, 2.0, 3.0])
        assert sample.grid.shape == (3, 2)

    def test_loads_right_orientation(self, tmp_path):
        folder = make_batchforce_folder(tmp_path, orientation="right")
        sample = load_batchforce_single(str(folder), afm_variables=["modulus"])
        assert set(sample.dataset.keys()) == {"modulus"}
        assert sample.grid.shape == (3, 2)
        assert sample.contour.shape[1] == 2
        assert sample.scale.shape == (3, 3)
        assert sample.bg_image.shape == (10, 10)
        assert sample.filename == folder.name

    def test_loads_left_orientation_and_flips(self, tmp_path):
        folder = make_batchforce_folder(tmp_path, orientation="left")
        sample = load_batchforce_single(str(folder), afm_variables=["modulus"])
        assert sample.contour.shape[1] == 2
        assert sample.grid.shape == (3, 2)

    def test_landmarks_are_loaded_when_requested(self, tmp_path):
        folder = make_batchforce_folder(tmp_path, orientation="right", landmarks=True)
        sample = load_batchforce_single(str(folder), afm_variables=["modulus"], landmarks_filename="landmarks")
        assert sample.landmarks is not None
        assert sample.landmarks.shape == (2, 2)

    def test_landmarks_none_when_not_requested(self, tmp_path):
        folder = make_batchforce_folder(tmp_path, orientation="right", landmarks=True)
        sample = load_batchforce_single(str(folder), afm_variables=["modulus"])
        assert sample.landmarks is None

    def test_missing_contour_raises(self, tmp_path):
        folder = make_batchforce_folder(tmp_path, orientation=None)
        with pytest.raises(ValueError, match="No matching contour was found"):
            load_batchforce_single(str(folder), afm_variables=["modulus"])

    def test_mat_extension_not_implemented(self, tmp_path):
        folder = make_batchforce_folder(tmp_path, orientation="right")
        (folder / "region analysis" / "data.mat").write_text("dummy")
        with pytest.raises(ValueError, match="Importing .mat files is not implemented"):
            load_batchforce_single(str(folder), afm_variables=["modulus"], batchforce_filename="data.mat")

    def test_unsupported_data_extension_raises(self, tmp_path):
        folder = make_batchforce_folder(tmp_path, orientation="right")
        (folder / "region analysis" / "data.xlsx").write_text("x")
        with pytest.raises(ValueError, match="are not supported"):
            load_batchforce_single(str(folder), afm_variables=["modulus"], batchforce_filename="data.xlsx")

    def test_missing_data_file_raises(self, tmp_path):
        folder = make_batchforce_folder(tmp_path, orientation="right")
        with pytest.raises(AssertionError, match="does not point to an AFM analysis file"):
            load_batchforce_single(str(folder), afm_variables=["modulus"], batchforce_filename="missing.csv")

    def test_non_csv_grid_conversion_falls_back_to_estimate(self, tmp_path, capsys):
        folder = make_batchforce_folder(tmp_path, orientation="right", grid_ext="mat")
        sample = load_batchforce_single(str(folder), afm_variables=["modulus"],
                                        grid_conv_filename="GridInversionMatrix.mat")
        assert "Estimating transformation matrix instead now" in capsys.readouterr().out
        assert sample.scale.shape == (3, 3)


class TestLoadBatchforceAll:

    def test_loads_every_hash_folder_and_skips_others(self, tmp_path):
        make_batchforce_folder(tmp_path, name="#1_sample", orientation="right")
        make_batchforce_folder(tmp_path, name="#2_sample", orientation="right")
        (tmp_path / "not_an_experiment").mkdir()

        samples = load_batchforce_all(str(tmp_path), afm_variables=["modulus"], batchforce_filename="data.csv",
                                      grid_conv_filename="GridInversionMatrix.csv", boundary_filename="brain_outline")
        assert [s.filename for s in samples] == ["#1_sample", "#2_sample"]

    def test_name_pattern_attaches_metadata(self, tmp_path):
        make_batchforce_folder(tmp_path, name="#3_Stage37", orientation="right")
        samples = load_batchforce_all(str(tmp_path), afm_variables=["modulus"], batchforce_filename="data.csv",
                                      grid_conv_filename="GridInversionMatrix.csv", boundary_filename="brain_outline",
                                      name_pattern=r"#(?P<animal_number>\d+)_Stage(?P<stage>\d+)",
                                      name_converters={"animal_number": int, "stage": int})
        assert samples[0].metadata == {"animal_number": 3, "stage": 37}
