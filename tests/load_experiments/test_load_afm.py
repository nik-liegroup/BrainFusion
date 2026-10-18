import pytest
import numpy as np
import pandas as pd
from PIL import Image

from brainfusion.load_experiments.load_afm import load_batchforce_single, load_batchforce_all

# With the identity grid matrix the loader's only transformation is its -90 degree stage rotation, whose inverse
# maps image pixels (Matlab convention, first pixel = 1) to (x, y) = (-y, x)


def make_afm_folder(base_path, name="#1_sample", mirror_horizontal=0, mirror_vertical=0, image="overview.tif",
                    annotate=True, long_format=False):
    folder = base_path / name
    (folder / "region analysis").mkdir(parents=True)
    calibration = folder / "Pics" / "calibration"
    calibration.mkdir(parents=True)

    grid = {"x_image": [2, 4, 6], "y_image": [2, 4, 6], "x": [2.0, 4.0, 6.0], "y": [2.0, 4.0, 6.0]}
    if long_format:
        pd.DataFrame({"result_parameter": ["Reduced apparent elastic modulus"] * 3, "result": [1.0, 2.0, 3.0],
                      **grid}).to_csv(folder / "region analysis" / "data.csv", index=False)
    else:
        pd.DataFrame({"modulus": [1.0, 2.0, 3.0], **grid}).to_csv(folder / "region analysis" / "data.csv",
                                                                   index=False)
    np.savetxt(folder / "GridInversionMatrix.csv", np.eye(3), delimiter=",")
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(calibration / "overview.tif")
    if annotate:
        (calibration / "outline_brainfusion.txt").write_text(
            f"# image: {image}\n# mirror_horizontal: {mirror_horizontal}\n# mirror_vertical: {mirror_vertical}\n"
            "[outline]\n0.00\t0.00\n8.00\t0.00\n8.00\t8.00\n0.00\t8.00\n[landmarks]\n1.00\t1.00\n7.00\t7.00\n")
    return folder


class TestLoadBatchforceSingle:

    def test_loads_data_grid_outline_and_landmarks(self, tmp_path):
        sample = load_batchforce_single(str(make_afm_folder(tmp_path)), afm_variables=["modulus"])
        assert set(sample.dataset.keys()) == {"modulus"}
        np.testing.assert_array_equal(sample.dataset["modulus"], [1.0, 2.0, 3.0])
        assert sample.grid.shape == (3, 2) and sample.contour.shape == (4, 2) and sample.landmarks.shape == (2, 2)
        assert sample.scale.shape == (3, 3)
        assert sample.bg_image.shape == (10, 10)
        assert sample.filename == "#1_sample"

    def test_annotation_pixels_are_zero_based_the_data_matlab_pixels(self, tmp_path):
        sample = load_batchforce_single(str(make_afm_folder(tmp_path)), afm_variables=["modulus"])
        # outline corner (0, 0) is Matlab pixel (1, 1) -> (x, y) = (-1, 1); grid point x_image = y_image = 2 -> (-2, 2)
        np.testing.assert_allclose(sample.contour[0], [-1, 1])
        np.testing.assert_allclose(sample.grid[0], [-2, 2])
        np.testing.assert_allclose(sample.landmarks[0], [-2, 2])  # the landmark (1, 1) is the grid point at pixel 2

    def test_long_format_result_parameter_export(self, tmp_path):
        # some batchforce versions export one row per (point, quantity): `afm_variables` as a dict maps the
        # dataset key to the quantity's 'result_parameter' label
        folder = make_afm_folder(tmp_path, long_format=True)
        sample = load_batchforce_single(str(folder), afm_variables={"modulus": "Reduced apparent elastic modulus"})
        np.testing.assert_array_equal(sample.dataset["modulus"], [1.0, 2.0, 3.0])

    def test_use_landmarks_false_ignores_them(self, tmp_path):
        sample = load_batchforce_single(str(make_afm_folder(tmp_path)), afm_variables=["modulus"],
                                        use_landmarks=False)
        assert sample.landmarks is None

    def test_vertical_mirror_in_header_mirrors_grid_outline_landmarks_and_image(self, tmp_path):
        plain = load_batchforce_single(str(make_afm_folder(tmp_path / "a")), afm_variables=["modulus"])
        mirrored = load_batchforce_single(str(make_afm_folder(tmp_path / "b", mirror_vertical=1)),
                                          afm_variables=["modulus"])
        # image 10 px high: pixel y -> 9 - y, so the outline corner (0, 0) becomes (0, 9) = Matlab (1, 10)
        np.testing.assert_allclose(mirrored.contour[0], [-10, 1])
        np.testing.assert_allclose(mirrored.grid[0], [-9, 2])  # y_image 2 -> 0-based 1 -> 8 -> Matlab pixel 9
        assert not np.allclose(mirrored.grid, plain.grid)
        apply_off = load_batchforce_single(str(make_afm_folder(tmp_path / "c", mirror_vertical=1)),
                                           afm_variables=["modulus"], apply_mirror=False)
        np.testing.assert_allclose(apply_off.contour, plain.contour)

    def test_another_annotation_file_can_be_given(self, tmp_path):
        folder = make_afm_folder(tmp_path / "a", annotate=False)
        other = make_afm_folder(tmp_path / "b")
        sample = load_batchforce_single(str(folder), afm_variables=["modulus"],
                                        annotation_file=str(other / "Pics" / "calibration" / "outline_brainfusion.txt"))
        assert sample.contour.shape == (4, 2)

    def test_annotation_on_another_image_than_overview_is_refused(self, tmp_path):
        folder = make_afm_folder(tmp_path, image="overview_56x.tif")
        with pytest.raises(AssertionError, match="overview.tif"):
            load_batchforce_single(str(folder), afm_variables=["modulus"])

    def test_missing_annotation_raises(self, tmp_path):
        folder = make_afm_folder(tmp_path, annotate=False)
        with pytest.raises(AssertionError, match="No roi-annotator file"):
            load_batchforce_single(str(folder), afm_variables=["modulus"])

    def test_missing_data_file_raises(self, tmp_path):
        folder = make_afm_folder(tmp_path)
        with pytest.raises(AssertionError, match="does not point to an AFM analysis file"):
            load_batchforce_single(str(folder), afm_variables=["modulus"], batchforce_filename="missing.csv")

    def test_non_csv_data_or_grid_conversion_is_refused(self, tmp_path):
        folder = make_afm_folder(tmp_path)
        (folder / "region analysis" / "data.mat").write_text("dummy")
        with pytest.raises(AssertionError, match="must be a .csv"):
            load_batchforce_single(str(folder), afm_variables=["modulus"], batchforce_filename="data.mat")
        (folder / "GridInversionMatrix.mat").write_text("dummy")
        with pytest.raises(AssertionError, match="must be a .csv"):
            load_batchforce_single(str(folder), afm_variables=["modulus"],
                                   grid_conv_filename="GridInversionMatrix.mat")


class TestLoadBatchforceAll:

    def test_loads_every_hash_folder_and_skips_others(self, tmp_path):
        make_afm_folder(tmp_path, name="#1_sample")
        make_afm_folder(tmp_path, name="#2_sample")
        (tmp_path / "not_an_experiment").mkdir()
        samples = load_batchforce_all(str(tmp_path), afm_variables=["modulus"])
        assert [s.filename for s in samples] == ["#1_sample", "#2_sample"]

    def test_name_pattern_attaches_metadata(self, tmp_path):
        make_afm_folder(tmp_path, name="#3_Stage37")
        samples = load_batchforce_all(str(tmp_path), afm_variables=["modulus"],
                                      name_pattern=r"#(?P<animal_number>\d+)_Stage(?P<stage>\d+)",
                                      name_converters={"animal_number": int, "stage": int})
        assert samples[0].metadata == {"animal_number": 3, "stage": 37}
