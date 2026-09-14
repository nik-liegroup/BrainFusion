import numpy as np

from brainfusion.sample import Sample
from brainfusion.fusion.core import brain_fusion
from brainfusion.io import export_analysis
from brainfusion.load_experiments.load_fused import load_fused_analysis


def circle_sample(cx, cy, r, value, n_contour=40, n_grid=8, filename="s", metadata=None):
    theta = np.linspace(0, 2 * np.pi, n_contour, endpoint=False)
    contour = np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    grid_theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    grid = np.column_stack((cx + (r / 2) * np.cos(grid_theta), cy + (r / 2) * np.sin(grid_theta)))
    dataset = {"modulus": np.full(n_grid, value)}
    return Sample(contour=contour, grid=grid, dataset=dataset, filename=filename, metadata=metadata or {})


def write_analysis(path, grouped=False):
    if grouped:
        samples = [
            circle_sample(0, 0, 1, value=1.0, filename="a1", metadata={"condition": "Control"}),
            circle_sample(0.05, 0, 1.02, value=1.0, filename="a2", metadata={"condition": "Control"}),
            circle_sample(-0.03, 0.02, 0.98, value=2.0, filename="b1", metadata={"condition": "CS"}),
        ]
    else:
        samples = [circle_sample(0, 0, 1, value=1.0, filename="a"), circle_sample(0.1, 0, 1.05, value=3.0, filename="b")]

    analysis = brain_fusion(samples, contour_template="average", contour_interp_n=40, clustering="Mean")
    export_analysis(str(path), analysis, {"contour_template": "average"})


class TestLoadFusedAnalysis:

    def test_no_group_field_returns_single_sample_with_overall_average(self, tmp_path):
        path = tmp_path / "analysis.h5"
        write_analysis(path)

        samples = load_fused_analysis(str(path), key_quant="modulus")
        assert len(samples) == 1
        np.testing.assert_allclose(samples[0].dataset["modulus"], 2.0)  # mean of 1.0 and 3.0
        assert samples[0].metadata == {}

    def test_group_field_returns_one_sample_per_group(self, tmp_path):
        path = tmp_path / "analysis.h5"
        write_analysis(path, grouped=True)

        samples = load_fused_analysis(str(path), key_quant="modulus", group_field="condition")
        assert len(samples) == 2
        by_condition = {s.metadata["condition"]: s for s in samples}
        np.testing.assert_allclose(by_condition["Control"].dataset["modulus"], 1.0)
        np.testing.assert_allclose(by_condition["CS"].dataset["modulus"], 2.0)

    def test_rename_key_renames_dataset_and_preserves_original_key_name_by_default(self, tmp_path):
        path = tmp_path / "analysis.h5"
        write_analysis(path)

        default_key = load_fused_analysis(str(path), key_quant="modulus")
        assert set(default_key[0].dataset.keys()) == {"modulus"}

        renamed = load_fused_analysis(str(path), key_quant="modulus", rename_key="value")
        assert set(renamed[0].dataset.keys()) == {"value"}
        np.testing.assert_allclose(renamed[0].dataset["value"], default_key[0].dataset["modulus"])

    def test_extra_metadata_is_merged_without_dropping_group_label(self, tmp_path):
        path = tmp_path / "analysis.h5"
        write_analysis(path, grouped=True)

        samples = load_fused_analysis(str(path), key_quant="modulus", group_field="condition",
                                      metadata={"modality": "AFM"})
        for sample in samples:
            assert sample.metadata["modality"] == "AFM"
            assert sample.metadata["condition"] in {"Control", "CS"}
