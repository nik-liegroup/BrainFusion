import numpy as np
import pytest

from brainfusion.sample import Sample
from brainfusion.fusion.core import brain_fusion
from brainfusion.fusion.grouping import (list_groups, extract_groups, group_average_on_shared_grid,
                                         extract_group_native_data, merge_keys)


def circle_sample(cx, cy, r, value, n_contour=40, n_grid=8, filename="s", metadata=None):
    theta = np.linspace(0, 2 * np.pi, n_contour, endpoint=False)
    contour = np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    grid_theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    grid = np.column_stack((cx + (r / 2) * np.cos(grid_theta), cy + (r / 2) * np.sin(grid_theta)))
    dataset = {"value": np.full(n_grid, value)}
    return Sample(contour=contour, grid=grid, dataset=dataset, filename=filename, metadata=metadata or {})


@pytest.fixture
def two_group_analysis():
    """Two groups ('A' constant value 1.0, 'B' constant value 2.0), 2 near-identical samples each. Fused
    WITHOUT group_field, so group_average_on_shared_grid must compute this post-hoc."""
    samples = [
        circle_sample(0, 0, 1, value=1.0, filename="a1", metadata={"group": "A"}),
        circle_sample(0.05, 0, 1.02, value=1.0, filename="a2", metadata={"group": "A"}),
        circle_sample(-0.03, 0.02, 0.98, value=2.0, filename="b1", metadata={"group": "B"}),
        circle_sample(0.02, -0.03, 1.01, value=2.0, filename="b2", metadata={"group": "B"}),
    ]
    return brain_fusion(samples, contour_template="average", contour_interp_n=40, clustering="Mean")


@pytest.fixture
def two_group_analysis_with_group_field():
    """Same as `two_group_analysis`, but fused WITH group_field="group", so group_datasets is baked in."""
    samples = [
        circle_sample(0, 0, 1, value=1.0, filename="a1", metadata={"group": "A"}),
        circle_sample(0.05, 0, 1.02, value=1.0, filename="a2", metadata={"group": "A"}),
        circle_sample(-0.03, 0.02, 0.98, value=2.0, filename="b1", metadata={"group": "B"}),
        circle_sample(0.02, -0.03, 1.01, value=2.0, filename="b2", metadata={"group": "B"}),
    ]
    return brain_fusion(samples, contour_template="average", contour_interp_n=40, clustering="Mean",
                        group_field="group")


class TestListGroups:

    def test_lists_unique_groups_sorted(self, two_group_analysis):
        assert list_groups(two_group_analysis, "group") == ["A", "B"]


class TestExtractGroups:

    def test_returns_groups_baked_in_at_fusion_time(self, two_group_analysis_with_group_field):
        assert set(extract_groups(two_group_analysis_with_group_field)) == {"A", "B"}

    def test_raises_when_no_group_field_was_used(self, two_group_analysis):
        with pytest.raises(ValueError, match="group_field"):
            extract_groups(two_group_analysis)


class TestGroupAverageOnSharedGrid:

    def test_splits_into_correct_per_group_constant_values(self, two_group_analysis):
        grid, contour, group_maps = group_average_on_shared_grid(two_group_analysis, "group")
        assert set(group_maps.keys()) == {"A", "B"}
        np.testing.assert_allclose(group_maps["A"]["value"], 1.0)
        np.testing.assert_allclose(group_maps["B"]["value"], 2.0)
        assert grid.shape[1] == 2
        assert contour.shape[1] == 2

    def test_raises_for_gmm_clustering(self):
        samples = [circle_sample(0, 0, 1, value=1.0, filename="a", metadata={"group": "A"}),
                  circle_sample(0.1, 0, 1, value=2.0, filename="b", metadata={"group": "B"})]
        analysis = brain_fusion(samples, clustering="GMM", contour_interp_n=40)
        with pytest.raises(ValueError, match="clustering"):
            group_average_on_shared_grid(analysis, "group")

    def test_raises_for_unknown_group(self, two_group_analysis):
        with pytest.raises(ValueError, match="No samples found"):
            group_average_on_shared_grid(two_group_analysis, "group", groups=["C"])


class TestExtractGroupNativeData:

    def test_concatenates_each_groups_own_samples(self, two_group_analysis):
        native = extract_group_native_data(two_group_analysis, "group", "value")
        assert set(native.keys()) == {"A", "B"}

        grid_a, data_a = native["A"]
        assert grid_a.shape == (16, 2)  # 2 samples x n_grid=8 points each, native (not resampled)
        np.testing.assert_allclose(data_a, 1.0)

        grid_b, data_b = native["B"]
        assert grid_b.shape == (16, 2)
        np.testing.assert_allclose(data_b, 2.0)

    def test_restricts_to_requested_groups(self, two_group_analysis):
        native = extract_group_native_data(two_group_analysis, "group", "value", groups=["A"])
        assert set(native.keys()) == {"A"}

    def test_raises_for_unknown_group(self, two_group_analysis):
        with pytest.raises(ValueError, match="No samples found"):
            extract_group_native_data(two_group_analysis, "group", "value", groups=["C"])


class TestMergeKeys:

    def test_combines_two_fields_into_one_new_key(self):
        samples = [circle_sample(0, 0, 1, value=1.0, metadata={"condition": "cond1", "modality": "AFM"}),
                  circle_sample(0, 0, 1, value=1.0, metadata={"condition": "cond2", "modality": "Brillouin"})]

        merged = merge_keys(samples, "group", ["condition", "modality"])
        assert merged[0].metadata["group"] == "cond1_AFM"
        assert merged[1].metadata["group"] == "cond2_Brillouin"

    def test_keeps_original_fields_and_leaves_input_samples_untouched(self):
        samples = [circle_sample(0, 0, 1, value=1.0, metadata={"condition": "cond1", "modality": "AFM"})]

        merged = merge_keys(samples, "group", ["condition", "modality"])
        assert merged[0].metadata == {"condition": "cond1", "modality": "AFM", "group": "cond1_AFM"}
        assert "group" not in samples[0].metadata

    def test_skips_keys_missing_on_a_given_sample_instead_of_raising(self):
        # AFM's Sample never had a 'condition' (it wasn't split by condition) - merge_keys should just use
        # whichever of keys_to_merge it does have, not raise a KeyError.
        samples = [circle_sample(0, 0, 1, value=1.0, metadata={"modality": "AFM"}),
                  circle_sample(0, 0, 1, value=1.0, metadata={"condition": "cond2", "modality": "Brillouin"})]

        merged = merge_keys(samples, "group", ["condition", "modality"])
        assert merged[0].metadata["group"] == "AFM"
        assert merged[1].metadata["group"] == "cond2_Brillouin"

    def test_custom_separator(self):
        samples = [circle_sample(0, 0, 1, value=1.0, metadata={"condition": "cond1", "modality": "AFM"})]

        merged = merge_keys(samples, "group", ["condition", "modality"], separator="-")
        assert merged[0].metadata["group"] == "cond1-AFM"
