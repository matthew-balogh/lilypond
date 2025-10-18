import pytest
import numpy as np

from minisom import MiniSom
from lilypond.basin import Basin

def __data():
    return np.random.rand(25, 2)

def __som():
    data = __data()
    som = MiniSom(3, 3, data.shape[1])
    som.random_weights_init(data)
    som.train(data, num_iteration=1, use_epochs=True)
    return som


def test_prepare():
    data, som = __data(), __som()

    basin = Basin(som, data)
    basin.prepare()

    assert hasattr(basin, "hitmap_") and basin.hitmap_ is not None
    assert hasattr(basin, "distmap_") and basin.distmap_ is not None
    assert hasattr(basin, "lattice_shape_") and basin.lattice_shape_ is not None
    assert hasattr(basin, "rows_") and basin.rows_ is not None
    assert hasattr(basin, "cols_") and basin.cols_ is not None
    assert hasattr(basin, "cmapWaterBlue_") and basin.cmapWaterBlue_ is not None

    assert basin.prepared_ is True

def test_unprepared_pond_raises_error():
    data, som = __data(), __som()

    basin = Basin(som, data)

    with pytest.raises(AssertionError, match="The Basin must be prepared before creating a pond."):
        basin.pond()

def test_unprepared_legacy_pond_raises_error():
    data, som = __data(), __som()

    basin = Basin(som, data)

    with pytest.raises(AssertionError, match="The Basin must be prepared before creating a pond."):
        basin.legacy_pond()
    