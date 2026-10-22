"""Regions drawn on a fusion's template pulled back onto its samples, from what the fusion stored."""
import h5py
import numpy as np
import pytest
from matplotlib.path import Path

from brainfusion import (atlas_pullback, brain_fusion, map_regions_between_outlines, read_regions_file, region_masks,
                         save_atlas_pullback)
from brainfusion.io import export_analysis, import_analysis
from brainfusion.sample import Sample


def egg(width=150.0, height=300.0):
    t = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    return np.column_stack([width * np.cos(t) * (1 - 0.25 * np.sin(t)), height * np.sin(t)])


def sample(index, rng):
    angle = np.deg2rad(rng.uniform(-30, 30))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = rng.uniform(0.9, 1.15)
    shift = rng.uniform(-50, 50, 2)
    contour = (egg(150 * rng.uniform(0.9, 1.1), 300 * rng.uniform(0.9, 1.1)) * scale) @ rotation.T + shift
    xs, ys = np.meshgrid(np.linspace(-350, 350, 36), np.linspace(-350, 350, 36))
    grid = np.column_stack([xs.ravel(), ys.ravel()]) @ rotation.T + shift
    grid = grid[Path(contour).contains_points(grid)]
    return Sample(contour=contour, grid=grid, dataset={"value": grid[:, 0] * 0 + 1.0 + index}, filename=f"s{index}", metadata={})


@pytest.fixture(scope="module")
def fusion():
    rng = np.random.default_rng(3)
    samples = [sample(i, rng) for i in range(5)]
    return samples, brain_fusion(samples, contour_template="average", clustering="Median", contour_interp_n=100)


def region_in_template(template):
    centre = template.mean(axis=0)
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    return centre + [0.0, 0.2 * np.ptp(template[:, 1])] + 0.3 * np.ptp(template[:, 0]) * np.column_stack([np.cos(t), np.sin(t)])


def test_the_regions_file_names_the_regions_itself(tmp_path):
    (tmp_path / "regions_brainfusion.txt").write_text(
        "# roi-annotator 0.1.0\n# status: drawn\n# frame: x\n[Alpha]\n0\t0\n10\t0\n10\t10\n0\t0\n[Beta two]\n1\t1\n2\t1\n2\t2\n", encoding="utf-8")
    regions, header = read_regions_file(tmp_path / "regions_brainfusion.txt", scale=1e-6)
    assert list(regions) == ["Alpha", "Beta two"] and header["status"] == "drawn"
    np.testing.assert_allclose(regions["Alpha"], np.array([[0, 0], [10, 0], [10, 10]]) * 1e-6)   # the repeated first point dropped


def test_a_pulled_back_region_holds_the_same_points_as_in_the_template(fusion):
    samples, analysis = fusion
    template = np.asarray(analysis["template_contours"][0], dtype=float)
    region = region_in_template(template)
    pulled = atlas_pullback(analysis, {"X": region})
    assert list(pulled) == [s.filename for s in samples] and list(pulled["s0"]) == ["X"]
    for index, s in enumerate(samples):
        in_template = region_masks(analysis["measurement_trafo_grids"][index], {"X": region})["X"]   # where the sample's points went
        in_sample = region_masks(s.grid, {"X": pulled[s.filename]["X"]})["X"]                        # the pulled back outline, own frame
        assert in_template.sum() > 20
        assert (in_template == in_sample).mean() > 0.9   # only points on the region's border differ: the inverse is a second RBF fit


def test_the_atlas_and_the_sample_regions_go_into_the_h5_file(fusion, tmp_path):
    _, analysis = fusion
    template = np.asarray(analysis["template_contours"][0], dtype=float)
    atlas_pullback(analysis, {"X": region_in_template(template)})
    path = str(tmp_path / "fusion.h5")
    export_analysis(path, {k: v for k, v in analysis.items() if k not in ("atlas", "sample_regions")}, {})
    save_atlas_pullback(path, analysis)
    with h5py.File(path) as h5:
        assert "atlas" in h5 and "sample_regions" in h5
    back, _ = import_analysis(path)
    np.testing.assert_allclose(back["atlas"]["regions"]["X"], analysis["atlas"]["regions"]["X"])
    np.testing.assert_allclose(back["sample_regions"][2]["X"], analysis["sample_regions"][2]["X"])
    save_atlas_pullback(path, analysis)   # again: replaces them


def test_regions_of_another_outline_are_moved_onto_the_template(fusion):
    _, analysis = fusion
    template = np.asarray(analysis["template_contours"][0], dtype=float)
    other = template * 1.5 + [20.0, -10.0]
    region = region_in_template(template) * 1.5 + [20.0, -10.0]
    moved = map_regions_between_outlines({"X": region}, other, template)["X"]
    np.testing.assert_allclose(moved.mean(axis=0), region_in_template(template).mean(axis=0), atol=0.08 * np.ptp(template, axis=0).max())
    same = map_regions_between_outlines({"X": region}, other, other)["X"]      # the same outline: nothing moves
    np.testing.assert_allclose(same, region)
