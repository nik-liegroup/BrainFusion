import pytest
import brainfusion  # ✅ Import your package


def test_brainfusion_import():
    """Test if the package imports correctly."""
    assert brainfusion is not None
