import pytest
import torch

from pathlib import Path
from vial_classifier.modeling.train import SimpleCNN, split_dataset

# PURPOSE:
#   python unit tests can be added here if necessary
# Intended invocation point in pipeline:
#   can be run at any point (also before model training) since it just tests small individual units

ROOT = Path(__file__).resolve().parent.parent#---- get root of repo - regardless of where pytest was run from

def test_split_keeps_class_balance():
    # 100 images, 80 of them are good, 20 are defective
    dataset = [(f"img_{i}.jpg", 0) for i in range(80)] # good
    dataset += [(f"img_{i}.jpg", 1) for i in range(80, 100)] # defective
    # chose these numbers because they are round and can keep all 3 data sets (training, validation, evaluation) proportionate

    train, val, test = split_dataset(dataset)

    assert len(train) + len(val) + len(test) == 100 # making sure no images got lost in the split
    for sets in (train, val, test): # for each of the  the train, val, test sets, runs:
        prop_defect = sum(label for _, label in sets) / len(sets)
        # good = 0, bad = 1, the sum will show how many bad viles, divided by the total count
        assert abs(prop_defect - 0.20) < 0.01
        # checks that set's share has no big gape from the entire sets share (decided prior is 20)


def test_simple_cnn_output_shape():
    model = SimpleCNN()
    model.eval()  # makes sure it only evaluates, doesn't train
    image = torch.randn(1, 3, 64, 64)  # 1 image, 3 colours, 64x64 pixels, making sure the input is right
    with torch.no_grad():
        output = model(image)
    assert output.shape == (1, 1) # check output has right size


def test_dummy():
    """Just a dummy test that never fails."""

    assert True, "This should not fail."
