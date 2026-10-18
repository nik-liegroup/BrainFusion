import pytest
import numpy as np
import pandas as pd
from PIL import Image

from brainfusion.load_experiments.load_brillouin import load_brillouin_experiment, load_brillouin_all

# Layout of an exported experiment: images with their 'TransformMatrices' folder below 'Plots/<kind>_BMrep<n>',
# the roi-annotator's outline_brainfusion.txt in 'Plots', the data in 'Export', the raw file in 'RawData'.
# The matrix is 2 px/um, so a pixel is half the stage um.


def make_brillouin_folder(base_path, name="#1_sample", grid_offset=0.0, mirror_horizontal=0, transform=True,
                          annotate=True):
    folder = base_path / name
    (folder / "Export").mkdir(parents=True)
    image_dir = folder / "Plots" / "Overview_BMrep0"
    (image_dir / "TransformMatrices").mkdir(parents=True)

    x = y = np.array([2.0, 4.0, 6.0]) + grid_offset  # stage x/y
    pd.DataFrame({"x": x, "y": y, "z": [0.1, 0.2, 0.3], "shift": [5.0, 5.5, 6.0],
                  "quality_pass": [1.0, 0.0, 1.0]}).to_csv(folder / "Export" / "scan_BMrep0_data_driftcorrected.csv", index=False)

    if transform:
        np.savetxt(image_dir / "TransformMatrices" / "ov_transform.csv", np.diag([2.0, 2.0, 1.0]), delimiter=",")
    Image.fromarray(np.arange(100, dtype=np.uint8).reshape(10, 10)).save(image_dir / "ov.png")
    if annotate:
        (folder / "Plots" / "outline_brainfusion.txt").write_text(
            "# roi-annotator 0.1.0\n# status: done\n# image: Overview_BMrep0/ov.png\n"
            f"# mirror_horizontal: {mirror_horizontal}\n# mirror_vertical: 0\n"
            "[outline]\n0.00\t0.00\n8.00\t0.00\n8.00\t8.00\n0.00\t8.00\n[landmarks]\n1.00\t1.00\n7.00\t7.00\n")
    return folder


class TestLoadBrillouinExperiment:

    def test_loads_dataset_grid_outline_landmarks_and_image(self, tmp_path):
        sample = load_brillouin_experiment(str(make_brillouin_folder(tmp_path)), brillouin_variables=["shift"])
        assert set(sample.dataset.keys()) == {"shift", "z"}
        np.testing.assert_array_equal(sample.dataset["shift"], [5.0, 5.5, 6.0])
        np.testing.assert_array_equal(sample.dataset["z"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(sample.grid, [[2, 2], [4, 4], [6, 6]])
        np.testing.assert_allclose(sample.contour, [[0, 0], [4, 0], [4, 4], [0, 4]])  # pixels / 2 -> stage um
        np.testing.assert_allclose(sample.landmarks, [[0.5, 0.5], [3.5, 3.5]])
        assert sample.bg_image.shape == (10, 10)
        assert sample.scale.shape == (3, 3)
        assert sample.filename == "#1_sample"

    def test_z_not_overwritten_if_explicitly_requested(self, tmp_path):
        sample = load_brillouin_experiment(str(make_brillouin_folder(tmp_path)), brillouin_variables=["shift", "z"])
        np.testing.assert_array_equal(sample.dataset["z"], [0.1, 0.2, 0.3])

    def test_use_landmarks_false_ignores_them(self, tmp_path):
        sample = load_brillouin_experiment(str(make_brillouin_folder(tmp_path)), brillouin_variables=["shift"],
                                           use_landmarks=False)
        assert sample.landmarks is None

    def test_quality_column_drops_failing_points(self, tmp_path):
        sample = load_brillouin_experiment(str(make_brillouin_folder(tmp_path)), brillouin_variables=["shift"],
                                           quality_column="quality_pass")
        np.testing.assert_array_equal(sample.dataset["shift"], [5.0, 6.0])
        assert len(sample.grid) == 2

    def test_mirror_flag_mirrors_grid_outline_landmarks_and_image(self, tmp_path):
        folder = make_brillouin_folder(tmp_path, mirror_horizontal=1)
        sample = load_brillouin_experiment(str(folder), brillouin_variables=["shift"])
        # image 10 px wide, matrix 2 px/um: pixel x -> 9 - x, i.e. stage x -> 4.5 - x
        np.testing.assert_allclose(sample.contour[:, 0], [4.5, 0.5, 0.5, 4.5])
        np.testing.assert_allclose(sample.grid[:, 0], [2.5, 0.5, -1.5])
        np.testing.assert_allclose(sample.grid[:, 1], [2.0, 4.0, 6.0])
        assert sample.bg_image[0, 0] == 9
        unmirrored = load_brillouin_experiment(str(folder), brillouin_variables=["shift"], apply_mirror=False)
        np.testing.assert_allclose(unmirrored.contour[:, 0], [0, 4, 4, 0])

    def test_grid_far_from_the_outline_is_refused(self, tmp_path):
        # e.g. an older export whose x/y are centred on the scan grid instead of absolute stage positions
        folder = make_brillouin_folder(tmp_path, grid_offset=-5000.0)
        with pytest.raises(ValueError, match="export again"):
            load_brillouin_experiment(str(folder), brillouin_variables=["shift"])

    def test_several_data_files_need_an_explicit_choice(self, tmp_path):
        folder = make_brillouin_folder(tmp_path)
        (folder / "Export" / "scan_BMrep1_data_driftcorrected.csv").write_text("x,y,z,shift\n2,2,0,9\n")
        with pytest.raises(AssertionError, match="Expected exactly one"):
            load_brillouin_experiment(str(folder), brillouin_variables=["shift"])
        sample = load_brillouin_experiment(str(folder), brillouin_variables=["shift"],
                                           data_filename="scan_BMrep1_data_driftcorrected.csv")
        np.testing.assert_array_equal(sample.dataset["shift"], [9.0])

    def test_uncorrected_file_is_never_substituted(self, tmp_path):
        folder = make_brillouin_folder(tmp_path)
        export = folder / "Export"
        (export / "scan_BMrep0_data_driftcorrected.csv").rename(export / "scan_BMrep0_data.csv")
        with pytest.raises(AssertionError, match="Expected exactly one"):
            load_brillouin_experiment(str(folder), brillouin_variables=["shift"])
        sample = load_brillouin_experiment(str(folder), brillouin_variables=["shift"], data_suffix="_data.csv")
        np.testing.assert_array_equal(sample.dataset["shift"], [5.0, 5.5, 6.0])

    def test_another_annotation_file_can_be_given(self, tmp_path):
        folder = make_brillouin_folder(tmp_path / "a", annotate=False)
        other = make_brillouin_folder(tmp_path / "b")
        sample = load_brillouin_experiment(str(folder), brillouin_variables=["shift"],
                                           annotation_file=str(other / "Plots" / "outline_brainfusion.txt"))
        assert sample.contour.shape == (4, 2)

    def test_missing_annotation_raises(self, tmp_path):
        folder = make_brillouin_folder(tmp_path, annotate=False)
        with pytest.raises(AssertionError, match="No roi-annotator file"):
            load_brillouin_experiment(str(folder), brillouin_variables=["shift"])

    def test_image_without_transform_matrix_raises(self, tmp_path):
        folder = make_brillouin_folder(tmp_path, transform=False)
        with pytest.raises(FileNotFoundError, match="no transform matrix"):
            load_brillouin_experiment(str(folder), brillouin_variables=["shift"])


class TestLoadBrillouinAll:

    def test_loads_every_hash_folder_with_metadata(self, tmp_path):
        make_brillouin_folder(tmp_path, name="#1_Stage37")
        make_brillouin_folder(tmp_path, name="#2_Stage40")
        (tmp_path / "not_an_experiment").mkdir()

        samples = load_brillouin_all(str(tmp_path), brillouin_variables=["shift"],
                                     name_pattern=r"#(?P<animal_number>\d+)_Stage(?P<stage>\d+)",
                                     name_converters={"animal_number": int, "stage": int})
        assert [s.filename for s in samples] == ["#1_Stage37", "#2_Stage40"]
        assert samples[0].metadata == {"animal_number": 1, "stage": 37}
        assert samples[1].metadata == {"animal_number": 2, "stage": 40}
