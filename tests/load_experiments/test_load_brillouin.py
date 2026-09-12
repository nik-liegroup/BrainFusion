import pytest
import numpy as np
import pandas as pd
from PIL import Image

from brainfusion.load_experiments.load_brillouin import load_brillouin_experiment, load_brillouin_all


def make_brillouin_folder(base_path, name="#1_sample", data_filename="scan1_data.csv",
                          transform_filename="scan1_transform.csv", image_name="scan1.png", landmarks=False):
    folder = base_path / name
    (folder / "Export").mkdir(parents=True)
    (folder / "Plots" / "TransformMatrices").mkdir(parents=True)

    pd.DataFrame({"x": [2.0, 4.0, 6.0], "y": [2.0, 4.0, 6.0], "z": [0.1, 0.2, 0.3],
                 "shift": [5.0, 5.5, 6.0]}).to_csv(folder / "Export" / data_filename, index=False)
    np.savetxt(folder / "Plots" / "TransformMatrices" / transform_filename, np.eye(3), delimiter=",")
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(folder / "Plots" / image_name)
    (folder / "Plots" / "brain_outline.txt").write_text("0\t0\n8\t0\n8\t8\n0\t8\n")
    if landmarks:
        (folder / "Plots" / "landmarks.txt").write_text("1\t1\n7\t7\n")

    return folder


class TestLoadBrillouinExperiment:

    def test_loads_dataset_grid_and_z_passthrough(self, tmp_path):
        folder = make_brillouin_folder(tmp_path)
        sample = load_brillouin_experiment(str(folder), brillouin_variables=["shift"])
        assert set(sample.dataset.keys()) == {"shift", "z"}
        np.testing.assert_array_equal(sample.dataset["shift"], [5.0, 5.5, 6.0])
        np.testing.assert_array_equal(sample.dataset["z"], [0.1, 0.2, 0.3])
        assert sample.grid.shape == (3, 2)
        assert sample.contour.shape[1] == 2
        assert sample.bg_image.shape == (10, 10)
        assert sample.filename == folder.name

    def test_z_not_overwritten_if_explicitly_requested(self, tmp_path):
        folder = make_brillouin_folder(tmp_path)
        sample = load_brillouin_experiment(str(folder), brillouin_variables=["shift", "z"])
        np.testing.assert_array_equal(sample.dataset["z"], [0.1, 0.2, 0.3])

    def test_landmarks_loaded_when_requested(self, tmp_path):
        folder = make_brillouin_folder(tmp_path, landmarks=True)
        sample = load_brillouin_experiment(str(folder), brillouin_variables=["shift"], landmarks_filename="landmarks")
        assert sample.landmarks is not None
        assert sample.landmarks.shape == (2, 2)

    def test_explicit_filenames_pick_one_of_several(self, tmp_path):
        folder = make_brillouin_folder(tmp_path, data_filename="rep1_data.csv", transform_filename="rep1_transform.csv",
                                       image_name="rep1.png")
        # A second, unrelated data/transform file in the same folders
        pd.DataFrame({"x": [1.0], "y": [1.0], "z": [0.0], "shift": [9.0]}).to_csv(
            folder / "Export" / "rep2_data.csv", index=False)
        np.savetxt(folder / "Plots" / "TransformMatrices" / "rep2_transform.csv", np.eye(3), delimiter=",")

        sample = load_brillouin_experiment(str(folder), brillouin_variables=["shift"], data_filename="rep1_data.csv",
                                           transform_filename="rep1_transform.csv")
        np.testing.assert_array_equal(sample.dataset["shift"], [5.0, 5.5, 6.0])

    def test_ambiguous_data_file_without_explicit_choice_raises(self, tmp_path):
        folder = make_brillouin_folder(tmp_path, data_filename="rep1_data.csv", transform_filename="rep1_transform.csv",
                                       image_name="rep1.png")
        pd.DataFrame({"x": [1.0], "y": [1.0], "z": [0.0], "shift": [9.0]}).to_csv(
            folder / "Export" / "rep2_data.csv", index=False)
        with pytest.raises(AssertionError, match="Expected exactly one"):
            load_brillouin_experiment(str(folder), brillouin_variables=["shift"])

    def test_missing_matching_image_raises(self, tmp_path):
        folder = make_brillouin_folder(tmp_path)
        (folder / "Plots" / "scan1.png").unlink()
        with pytest.raises(AssertionError, match="Expected exactly one image"):
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
