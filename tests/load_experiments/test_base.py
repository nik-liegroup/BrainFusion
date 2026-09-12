import numpy as np

from brainfusion.load_experiments.base import iter_experiment_folders, list_matching_files, load_template_sample


class TestIterExperimentFolders:

    def test_only_yields_folders_containing_marker(self, tmp_path):
        (tmp_path / "#1_sample").mkdir()
        (tmp_path / "#2_sample").mkdir()
        (tmp_path / "no_marker_folder").mkdir()
        (tmp_path / "#3_not_a_folder.txt").write_text("x")  # file, not a directory

        found = list(iter_experiment_folders(str(tmp_path)))
        names = [name for name, _ in found]
        assert names == ["#1_sample", "#2_sample"]

    def test_sorted_order(self, tmp_path):
        (tmp_path / "#b_sample").mkdir()
        (tmp_path / "#a_sample").mkdir()
        names = [name for name, _ in iter_experiment_folders(str(tmp_path))]
        assert names == ["#a_sample", "#b_sample"]

    def test_custom_marker(self, tmp_path):
        (tmp_path / "keep_this").mkdir()
        (tmp_path / "skip_this").mkdir()
        names = [name for name, _ in iter_experiment_folders(str(tmp_path), marker="keep")]
        assert names == ["keep_this"]


class TestListMatchingFiles:

    def test_filters_and_sorts(self, tmp_path):
        (tmp_path / "b.csv").write_text("x")
        (tmp_path / "a.csv").write_text("x")
        (tmp_path / "c.txt").write_text("x")
        result = list_matching_files(str(tmp_path), lambda f: f.endswith(".csv"))
        assert result == ["a.csv", "b.csv"]

    def test_no_matches_returns_empty_list(self, tmp_path):
        result = list_matching_files(str(tmp_path), lambda f: f.endswith(".csv"))
        assert result == []


class TestLoadTemplateSample:

    def test_loads_contour_and_derives_filename(self, tmp_path):
        contour_path = tmp_path / "atlas_outline.txt"
        contour_path.write_text("0,0\n1,0\n1,1\n")
        sample = load_template_sample(str(contour_path), delimiter=',')
        np.testing.assert_array_equal(sample.contour, [[0, 0], [1, 0], [1, 1]])
        assert sample.filename == "atlas_outline"
        assert sample.landmarks is None
        assert sample.dataset is None

    def test_explicit_filename_overrides_derived_one(self, tmp_path):
        contour_path = tmp_path / "atlas_outline.txt"
        contour_path.write_text("0,0\n1,0\n")
        sample = load_template_sample(str(contour_path), delimiter=',', filename="custom_name")
        assert sample.filename == "custom_name"

    def test_landmarks_and_dataset_are_attached(self, tmp_path):
        contour_path = tmp_path / "atlas_outline.txt"
        contour_path.write_text("0,0\n1,0\n1,1\n")
        landmarks_path = tmp_path / "atlas_landmarks.txt"
        landmarks_path.write_text("0.5,0.5\n")
        dataset = {"region": np.array([1, 2, 3])}

        sample = load_template_sample(str(contour_path), delimiter=',', dataset=dataset,
                                      landmarks_path=str(landmarks_path))
        np.testing.assert_array_equal(sample.landmarks, [[0.5, 0.5]])
        assert sample.dataset is dataset
