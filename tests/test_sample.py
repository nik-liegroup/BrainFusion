import numpy as np
from brainfusion.sample import Sample, replace


class TestSample:

    def test_only_contour_is_required(self):
        sample = Sample(contour=np.zeros((3, 2)))
        assert sample.grid is None
        assert sample.dataset is None
        assert sample.filename == ""
        assert sample.metadata == {}

    def test_metadata_default_is_not_shared_between_instances(self):
        a = Sample(contour=np.zeros((3, 2)))
        b = Sample(contour=np.zeros((3, 2)))
        a.metadata["condition"] = "Control"
        assert b.metadata == {}

    def test_replace_creates_independent_copy(self):
        original = Sample(contour=np.zeros((3, 2)), filename="a")
        updated = replace(original, filename="b")
        assert original.filename == "a"
        assert updated.filename == "b"
        assert updated.contour is original.contour  # unspecified fields are shared, not copied
