from pathlib import Path
import warnings
import matplotlib
import pytest

# Use a headless backend for all plot tests
matplotlib.use("Agg")


@pytest.fixture(scope="session")
def data_dir():
    """Root folder for external test data."""
    d = Path(__file__).parent / "data"
    if not d.exists():
        pytest.skip("tests/data folder not found", allow_module_level=True)
    return d


@pytest.fixture
def require():
    """Skip the current test if the given path does not exist."""

    def _require(path, reason=None):
        if not Path(path).exists():
            pytest.skip(reason or f"Missing test data: {path}")

    return _require


@pytest.fixture(autouse=True)
def _close_figs():
    """
    Auto-close figures between tests to avoid leakage
    """
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
