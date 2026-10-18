#!/usr/bin/env python3
"""
Validate the exported serving model against known-fixture images.

The v2 replacement for the v1 model-validator's inline sklearn test.
The serving model is no longer a sklearn model.pkl - it is the MLflow
pyfunc export (SimpleLoggingWrapper) that train.yml uploads as the
`model` artifact. This script loads that export, runs the fixture
images in the fixtures directory through it, and fails unless every
prediction matches the expected label in expected.csv.

The model-validator action calls this script; the validation method
itself lives in this repository on purpose.

Usage:
    python scripts/validate_model.py \
        [--model-dir models/model] [--fixtures-dir validation/fixtures]
"""

import argparse
import base64
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = {0: "clean", 1: "defective"}


def load_fixtures(fixtures_dir: Path) -> list[tuple[Path, int]]:
    """Read expected.csv and return (image_path, expected_label) pairs."""
    expected_path = fixtures_dir / "expected.csv"
    if not expected_path.exists():
        sys.exit(f"ERROR: {expected_path} not found - no fixtures defined.")

    fixtures: list[tuple[Path, int]] = []
    missing: list[Path] = []
    with open(expected_path, newline="") as f:
        for row in csv.DictReader(f):
            image = fixtures_dir / row["filename"]
            label = int(row["label"])
            if label not in CLASS_NAMES:
                sys.exit(f"ERROR: unknown label {label} for {image.name}")
            if not image.exists():
                missing.append(image)
            fixtures.append((image, label))

    if not fixtures:
        sys.exit(f"ERROR: {expected_path} lists no fixtures.")
    if missing:
        sys.exit(
            "ERROR: fixture images not committed yet: "
            + ", ".join(str(m) for m in missing)
        )
    return fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir",
        default="models/model",
        help="Exported MLflow pyfunc model directory (default: models/model)",
    )
    parser.add_argument(
        "--fixtures-dir",
        default="validation/fixtures",
        help="Directory with expected.csv and fixture images "
        "(default: validation/fixtures)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"ERROR: model directory {model_dir} not found.")

    fixtures = load_fixtures(Path(args.fixtures_dir))

    # The pyfunc wrapper imports src.* when the model is loaded, so make
    # the repository root importable regardless of how this was invoked.
    sys.path.insert(0, str(REPO_ROOT))
    import mlflow.pyfunc

    print(f"Loading pyfunc model from {model_dir.resolve()}")
    model = mlflow.pyfunc.load_model(str(model_dir))

    # Base64-encode the fixtures - the same portable contract the serving
    # app accepts in POST /invocations, rather than local file paths.
    encoded = [
        base64.b64encode(image.read_bytes()).decode() for image, _ in fixtures
    ]

    print(f"Predicting {len(encoded)} fixture image(s)...")
    result = model.predict({"instances": encoded})

    for key in ("predictions", "confidences"):
        if key not in result:
            sys.exit(f"ERROR: model response is missing '{key}': {result}")

    predictions = result["predictions"]
    if len(predictions) != len(fixtures):
        sys.exit(
            f"ERROR: expected {len(fixtures)} predictions, "
            f"got {len(predictions)}."
        )

    failures = 0
    for (image, expected), predicted in zip(fixtures, predictions):
        ok = predicted == expected
        failures += 0 if ok else 1
        print(
            f"  {'PASS' if ok else 'FAIL'}  {image.name}: "
            f"expected {CLASS_NAMES[expected]}, "
            f"predicted {CLASS_NAMES.get(predicted, repr(predicted))}"
        )

    if failures:
        sys.exit(
            f"Model validation FAILED: {failures}/{len(fixtures)} "
            "fixtures misclassified."
        )
    print("Model validation PASSED: all fixtures classified correctly.")


if __name__ == "__main__":
    main()
