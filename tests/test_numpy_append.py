import pytest
import numpy as np
from openmm_numpy_reporters import NumpyAppendFile


def test_append_one_1D_array(tmp_path) -> None:
    '''
    Append a 1D array to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    array = np.random.rand(10)
    with NumpyAppendFile(filename) as file:
        file.append(array)

    actual = np.load(filename)
    expected = array

    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_multiple_1D_arrays_same_shape(tmp_path) -> None:
    '''
    Append multiple 1D arrays of the same shape to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    arrays.append(np.random.rand(10))
    arrays.append(np.random.rand(10))

    with NumpyAppendFile(filename) as file:
        for array in arrays:
            file.append(array)

    actual = np.load(filename)
    expected = np.concatenate(arrays)

    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_multiple_1D_arrays_different_shapes(tmp_path) -> None:
    '''
    Append multiple 1D arrays of different shapes to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    arrays.append(np.random.rand(10))
    arrays.append(np.random.rand(20))

    with NumpyAppendFile(filename) as file:
        for array in arrays:
            file.append(array)

    actual = np.load(filename)
    expected = np.concatenate(arrays)

    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_many_1D_arrays(tmp_path) -> None:
    '''
    Append many 1D arrays to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    for _ in range(100):
        num_elements = np.random.randint(low=10, high=100)
        arrays.append(np.random.rand(num_elements))

    with NumpyAppendFile(filename) as file:
        for array in arrays:
            file.append(array)

    actual = np.load(filename)
    expected = np.concatenate(arrays)

    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_one_2D_array(tmp_path) -> None:
    '''
    Append a 2D array to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    array = np.random.rand(10, 10)
    with NumpyAppendFile(filename) as file:
        file.append(array)

    actual = np.load(filename)
    expected = array

    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_multiple_2D_arrays_same_shape(tmp_path) -> None:
    '''
    Append multiple 2D arrays of the same shape to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    arrays.append(np.random.rand(10, 10))
    arrays.append(np.random.rand(10, 10))

    with NumpyAppendFile(filename) as file:
        for array in arrays:
            file.append(array)

    actual = np.load(filename)
    expected = np.concatenate(arrays)

    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_multiple_2D_arrays_different_shapes(tmp_path) -> None:
    '''
    Append multiple 2D arrays of compatibly different shapes to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    arrays.append(np.random.rand(10, 10))
    arrays.append(np.random.rand(20, 10))

    with NumpyAppendFile(filename) as file:
        for array in arrays:
            file.append(array)

    actual = np.load(filename)
    expected = np.concatenate(arrays)

    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_many_2D_arrays(tmp_path) -> None:
    '''
    Append many 2D arrays to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    num_columns = np.random.randint(low=10, high=100)
    for _ in range(100):
        num_rows = np.random.randint(low=10, high=100)
        arrays.append(np.random.rand(num_rows, num_columns))

    with NumpyAppendFile(filename) as file:
        for array in arrays:
            file.append(array)

    actual = np.load(filename)
    expected = np.concatenate(arrays)

    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_incompatible_2D_arrays(tmp_path) -> None:
    '''
    Append incompatible 2D arrays to a NumPy file.
    Incompatible arrays have at least one dimension (other than the 0th) that doesn't match.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    arrays.append(np.random.rand(10, 10))
    arrays.append(np.random.rand(10, 20))

    with pytest.raises(ValueError):
        with NumpyAppendFile(filename) as file:
            for array in arrays:
                file.append(array)


def test_append_integer_array(tmp_path) -> None:
    '''
    Append an integer array to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    array = np.random.randint(low=-1e6, high=1e6, size=(10, 10))
    with NumpyAppendFile(filename) as file:
        file.append(array)

    actual = np.load(filename)
    expected = array

    assert actual.dtype == expected.dtype == np.dtype(int)
    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_float_array(tmp_path) -> None:
    '''
    Append a float array to a NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    array = np.random.rand(10, 10)
    with NumpyAppendFile(filename) as file:
        file.append(array)

    actual = np.load(filename)
    expected = array

    assert actual.dtype == expected.dtype == np.dtype(float)
    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_to_existing_file(tmp_path) -> None:
    '''
    Append to an existing NumPy file.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    arrays.append(np.random.rand(10, 10))
    arrays.append(np.random.rand(10, 10))

    with NumpyAppendFile(filename) as file:
        file.append(arrays[0])
    with NumpyAppendFile(filename) as file:
        file.append(arrays[-1])

    actual = np.load(filename)
    expected = np.concatenate(arrays)

    assert actual.shape == expected.shape
    assert np.all(actual == expected)


def test_append_to_existing_file_incompatible_shape(tmp_path) -> None:
    '''
    Append to an existing NumPy file with an incompatible shape.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    arrays.append(np.random.rand(10, 10))
    arrays.append(np.random.rand(10, 20))

    with NumpyAppendFile(filename) as file:
        file.append(arrays[0])
    with pytest.raises(ValueError):
        with NumpyAppendFile(filename) as file:
            file.append(arrays[-1])


def test_append_to_existing_file_incompatible_dtype(tmp_path) -> None:
    '''
    Append to an existing NumPy file with an incompatible dtype.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    arrays.append(np.random.rand(10, 10))
    arrays.append(np.random.randint(low=-1e6, high=1e6, size=(10, 10)))

    with NumpyAppendFile(filename) as file:
        file.append(arrays[0])
    with pytest.raises(TypeError):
        with NumpyAppendFile(filename) as file:
            file.append(arrays[-1])


def test_append_to_existing_file_incompatible_padding(tmp_path) -> None:
    '''
    Append to an existing NumPy file with an incompatible padding.
    '''
    filename = tmp_path / 'test.npy'
    arrays = []
    arrays.append(np.random.rand(10, 10))
    arrays.append(np.random.rand(10, 10))

    np.save(filename, arrays[0])
    with pytest.raises(ValueError):
        with NumpyAppendFile(filename) as file:
            file.append(arrays[-1])
