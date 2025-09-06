import os, pytest

MAIN_PDF = "Autocorrelation_Penalty_Module.pdf"
TESTS_ZIP = "Tests.zip"

def test_artifacts_exist_or_skip():
    missing = [p for p in [MAIN_PDF, TESTS_ZIP] if not os.path.exists(p)]
    if missing:
        pytest.skip("Missing research artifacts: " + ", ".join(missing))
    assert os.path.getsize(MAIN_PDF) > 0, f"{MAIN_PDF} exists but is empty"
