import pytest

from nn.datasets import get_dataset


@pytest.mark.parametrize("name", ["and", "or", "xor", "blobs"])
def test_get_dataset_shapes(name):
    data = get_dataset(name)
    assert len(data["points"]) == len(data["labels"])
    assert len(data["points"]) > 0
    assert all(len(p) == 2 for p in data["points"])


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        get_dataset("not-a-dataset")
