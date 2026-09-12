import numpy as np
import tifffile

from brainfusion.load_experiments.load_images import load_microscopy_single, load_microscopy_all


def write_tif(path, image, resolution=1.0):
    tifffile.imwrite(str(path), image, resolution=(resolution, resolution))


def write_contour(path, points):
    lines = ["x\ty"] + [f"{x}\t{y}" for x, y in points]
    path.write_text("\n".join(lines))


class TestLoadMicroscopySingle:

    def test_single_channel_image(self, tmp_path):
        write_tif(tmp_path / "sample.tif", np.arange(400, dtype=np.uint16).reshape(20, 20))
        write_contour(tmp_path / "sampleBrainBoundary.txt", [(0, 0), (10, 0), (10, 10), (0, 10)])

        sample = load_microscopy_single(str(tmp_path), "sample.tif")
        assert set(sample.dataset.keys()) == {"Channel_1"}
        assert sample.dataset["Channel_1"].shape == (400,)
        assert sample.contour.shape == (4, 2)
        assert sample.grid_shape.tolist() == [20, 20]
        assert sample.filename == "sample"

    def test_multi_channel_image(self, tmp_path):
        image = np.stack([np.zeros((20, 20), dtype=np.uint16), np.ones((20, 20), dtype=np.uint16)])
        write_tif(tmp_path / "multi.tif", image)
        write_contour(tmp_path / "multiBrainBoundary.txt", [(0, 0), (10, 0), (10, 10), (0, 10)])

        sample = load_microscopy_single(str(tmp_path), "multi.tif")
        assert set(sample.dataset.keys()) == {"Channel_1", "Channel_2"}

    def test_custom_boundary_filename(self, tmp_path):
        write_tif(tmp_path / "sample.tif", np.zeros((10, 10), dtype=np.uint16))
        write_contour(tmp_path / "sampleOutline.txt", [(0, 0), (5, 0), (5, 5)])
        sample = load_microscopy_single(str(tmp_path), "sample.tif", boundary_filename="Outline")
        assert sample.contour.shape == (3, 2)

    def test_landmarks_loaded_when_requested(self, tmp_path):
        write_tif(tmp_path / "sample.tif", np.zeros((10, 10), dtype=np.uint16))
        write_contour(tmp_path / "sampleBrainBoundary.txt", [(0, 0), (5, 0), (5, 5)])
        (tmp_path / "sampleLandmarks.txt").write_text("1\t1\n2\t2\n")
        sample = load_microscopy_single(str(tmp_path), "sample.tif", landmarks_filename="Landmarks")
        assert sample.landmarks.shape == (2, 2)

    def test_binning_reduces_grid_shape(self, tmp_path):
        write_tif(tmp_path / "sample.tif", np.zeros((20, 20), dtype=np.uint16))
        write_contour(tmp_path / "sampleBrainBoundary.txt", [(0, 0), (10, 0), (10, 10), (0, 10)])
        sample = load_microscopy_single(str(tmp_path), "sample.tif", bin_size=2)
        assert sample.grid_shape.tolist() == [10, 10]
        assert sample.dataset["Channel_1"].shape == (100,)

    def test_scale_maps_pixels_to_physical_units(self, tmp_path):
        write_tif(tmp_path / "sample.tif", np.zeros((10, 10), dtype=np.uint16), resolution=2.0)
        write_contour(tmp_path / "sampleBrainBoundary.txt", [(0, 0), (5, 0), (5, 5)])
        sample = load_microscopy_single(str(tmp_path), "sample.tif")
        # 2 px/unit resolution -> 0.5 physical units per pixel
        np.testing.assert_allclose(sample.grid[1] - sample.grid[0], [0.5, 0])

    def test_invert_flips_contour_y(self, tmp_path):
        write_tif(tmp_path / "sample.tif", np.zeros((10, 10), dtype=np.uint16))
        write_contour(tmp_path / "sampleBrainBoundary.txt", [(0, 0)])
        inverted = load_microscopy_single(str(tmp_path), "sample.tif", invert=True)
        not_inverted = load_microscopy_single(str(tmp_path), "sample.tif", invert=False)
        assert inverted.contour[0, 1] != not_inverted.contour[0, 1]


class TestLoadMicroscopyAll:

    def test_loads_every_tif_and_attaches_metadata(self, tmp_path):
        write_tif(tmp_path / "Stage37_a.tif", np.zeros((10, 10), dtype=np.uint16))
        write_contour(tmp_path / "Stage37_aBrainBoundary.txt", [(0, 0), (5, 0), (5, 5)])
        write_tif(tmp_path / "Stage40_b.tif", np.zeros((10, 10), dtype=np.uint16))
        write_contour(tmp_path / "Stage40_bBrainBoundary.txt", [(0, 0), (5, 0), (5, 5)])
        (tmp_path / "notes.txt").write_text("not a tif")

        samples = load_microscopy_all(str(tmp_path), name_pattern=r"Stage(?P<stage>\d+)",
                                      name_converters={"stage": int})
        assert sorted(s.filename for s in samples) == ["Stage37_a", "Stage40_b"]
        metadata_by_file = {s.filename: s.metadata for s in samples}
        assert metadata_by_file["Stage37_a"] == {"stage": 37}
        assert metadata_by_file["Stage40_b"] == {"stage": 40}
