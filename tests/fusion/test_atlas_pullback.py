"""Regions drawn on one outline of a fusion pulled back onto its samples, from what the fusion stored."""
import numpy as np
import pytest
from matplotlib.path import Path

from brainfusion import atlas_pullback, brain_fusion, read_regions_file, region_masks
from brainfusion.fusion.atlas import _preimage_outline, pull_back_regions
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
    return Sample(contour=contour, grid=grid, dataset={"value": grid[:, 0] * 0 + 1.0 + index}, filename=f"s{index}",
                  metadata={"length_unit": "um"})


@pytest.fixture(scope="module")
def fusion():
    rng = np.random.default_rng(3)
    samples = [sample(i, rng) for i in range(5)]
    return samples, brain_fusion(samples, contour_template="average", clustering="Median", contour_interp_n=100)


def region_in_template(template, dy=0.2):
    centre = template.mean(axis=0)
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    return centre + [0.0, dy * np.ptp(template[:, 1])] + 0.3 * np.ptp(template[:, 0]) * np.column_stack([np.cos(t), np.sin(t)])


def test_the_regions_file_names_the_regions_itself(tmp_path):
    (tmp_path / "regions_brainfusion.txt").write_text(
        "# roi-annotator 0.1.0\n# status: drawn\n# frame: x\n[Alpha]\n0\t0\n10\t0\n10\t10\n0\t0\n[Beta two]\n1\t1\n2\t1\n2\t2\n", encoding="utf-8")
    regions, header = read_regions_file(tmp_path / "regions_brainfusion.txt", scale=1e-6)
    assert list(regions) == ["Alpha", "Beta two"] and header["status"] == "drawn"
    np.testing.assert_allclose(regions["Alpha"], np.array([[0, 0], [10, 0], [10, 10]]) * 1e-6)   # the repeated first point dropped


def test_only_the_largest_part_of_what_lands_in_a_region_is_kept_with_its_holes_filled():
    xs = ys = np.arange(100.0)
    mask = np.zeros((100, 100), bool)
    mask[20:60, 20:60] = True
    mask[35:45, 35:45] = False      # a hole
    mask[80:85, 80:85] = True       # a second, smaller part
    outline = _preimage_outline(mask, xs, ys)
    assert Path(outline).contains_points([[40, 40], [25, 25]]).all() and not Path(outline).contains_points([[82, 82]]).any()
    assert np.ptp(outline, axis=0) == pytest.approx([40, 40], abs=2)
    assert _preimage_outline(np.zeros((10, 10), bool), np.arange(10.0), np.arange(10.0)) is None


def test_a_pulled_back_region_holds_the_same_points_as_in_the_template(fusion):
    samples, analysis = fusion
    template = np.asarray(analysis["template_contours"][0], dtype=float)
    region = region_in_template(template)
    _, pulled = pull_back_regions(analysis, {"X": region})
    assert len(pulled) == len(samples)
    for index, s in enumerate(samples):
        in_template = region_masks(analysis["measurement_trafo_grids"][index], {"X": region})["X"]   # where the sample's points went
        in_sample = region_masks(s.grid, {"X": pulled[index]["X"]})["X"]                             # the pulled back outline, own frame
        assert in_template.sum() > 20
        assert (in_template == in_sample).mean() > 0.99   # the points the fusion put into the region, found again in the sample


def test_regions_drawn_on_a_sample_go_to_the_template_through_its_own_warp(fusion):
    samples, analysis = fusion
    template = np.asarray(analysis["template_contours"][0], dtype=float)
    region = region_in_template(template)
    _, pulled = pull_back_regions(analysis, {"X": region})
    drawn_in_sample_2 = pulled[2]["X"]                                   # as if drawn on sample 2's outline
    box = np.linspace(template.min(0), template.max(0), 120)
    grid = np.array([[x, y] for x in box[:, 0] for y in box[:, 1]])
    a = region_masks(grid, {"X": region})["X"]
    for drawn_on in (2, "s2"):                                            # by position or by file name
        atlas, _ = pull_back_regions(analysis, {"X": drawn_in_sample_2}, drawn_on=drawn_on)
        b = region_masks(grid, {"X": atlas["X"]})["X"]
        assert (a & b).sum() / (a | b).sum() > 0.9                       # the same region on the template
    with pytest.raises(ValueError, match="no sample"):
        pull_back_regions(analysis, {"X": region}, drawn_on="nothing")
    with pytest.raises(ValueError, match="5 samples"):
        pull_back_regions(analysis, {"X": region}, drawn_on=7)


def write_regions(path, regions, unit="um"):
    lines = ["# status: drawn", f"# unit: {unit}"]
    for name, polygon in regions.items():
        lines += [f"[{name}]"] + [f"{x:.9g}\t{y:.9g}" for x, y in np.vstack([polygon, polygon[:1]])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_whole_pullback_from_a_fusion_file(fusion, tmp_path):
    samples, analysis = fusion
    template = np.asarray(analysis["template_contours"][0], dtype=float)
    export_analysis(str(tmp_path / "fusion.h5"), analysis, {})
    regions = {"Zeta": region_in_template(template, 0.2), "Alpha": region_in_template(template, -0.2)}
    write_regions(tmp_path / "atlas.txt", regions)   # not alphabetical on purpose

    found = atlas_pullback(tmp_path / "fusion.h5", tmp_path / "atlas.txt")
    assert (tmp_path / "Regions_average.png").stat().st_size > 0
    assert sorted(p.name for p in tmp_path.glob("Regions_s*.png")) == [f"Regions_s{i}.png" for i in range(5)]   # one per sample
    assert (tmp_path / "Regions_s0.svg").exists()

    back, _ = import_analysis(str(tmp_path / "fusion.h5"))                  # downstream: everything is in the .h5
    assert [str(n) for n in back["regions"]["names"]] == ["Zeta", "Alpha"] and str(back["regions"]["drawn_on"]) == "average"
    np.testing.assert_allclose(back["regions"]["atlas"]["Zeta"], found["regions"]["atlas"]["Zeta"])
    np.testing.assert_allclose(back["regions"]["samples"][3]["Alpha"], found["regions"]["samples"][3]["Alpha"])

    atlas_pullback(tmp_path / "fusion.h5", tmp_path / "atlas.txt", drawn_on="s1")   # again: replaces the stored regions and pictures
    assert len(list(tmp_path.glob("Regions_s*.png"))) == len(samples)
    assert str(import_analysis(str(tmp_path / "fusion.h5"))[0]["regions"]["drawn_on"]) == "s1"
    (tmp_path / "none.txt").write_text("# status: drawn\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no regions"):
        atlas_pullback(tmp_path / "fusion.h5", tmp_path / "none.txt")

