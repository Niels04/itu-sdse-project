import pytest

# PURPOSE:
#   python unit tests can be added here if necessary
# Intended invocation point in pipeline:
#   can be run at any point (also before model training) since it just tests small individual units

ROOT = Path(__file__).resolve().parent.parent#---- get root of repo - regardless of where pytest was run from

def test_dummy():
    """Just a dummy test that never fails."""

    assert True, "This should not fail."
