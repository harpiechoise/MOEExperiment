import numpy as np
import pytest
from src import DatasetPartitioner

import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


# class DatasetPartitioner:
#     @staticmethod
#     def __get_num_classes(y: np.ndarray) -> int:
#         """Takes a labels vector and returns the number of classes.

#         Args:
#             y (np.ndarray): Is the labels vector.

#         Returns:
#             int: The number of classes.
#         """
#         return len(np.unique(y))

#     @staticmethod
#     def __get_partitions(X: np.ndarray, y: np.ndarray, cls) -> Tuple[np.ndarray, np.ndarray]:
#         """Takes a multiclass dataset and returns a binary dataset for a given class.

#         Args:
#             X (np.ndarray): Is the examples matrix.
#             y (np.ndarray): Is the labels vector.

#         Returns:
#             _type_: _description_
#         """
#         return X[y == cls].tolist(), y[y == cls].tolist()

#     @staticmethod
#     def __partition_by_class(X_train: np.ndarray,
#                              y_train: np.ndarray,
#                              X_test: np.ndarray = None,
#                              y_test: np.ndarray = None) -> Tuple[list, list, list, list]:
#         """Takes a multiclass dataset and returns a binary dataset for each class and yields the binary datasets."""
#         if X_test is None or y_test is None:
#             for cls in np.unique(y_train):
#                 yield DatasetPartitioner.__get_partitions(X_train, y_train, cls)
#         if X_test and y_test:
#             for cls in np.unique(y_train):
#                 X_train_cls, y_train_cls = DatasetPartitioner.__get_partitions(X_train, y_train, cls)
#                 X_test_cls, y_test_cls = DatasetPartitioner.__get_partitions(X_test, y_test, cls)
#                 yield X_train_cls, y_train_cls, X_test_cls, y_test_cls

    
#     @staticmethod
#     def get_dataset(X, y, X_test=None, y_test=None, test_size=None):
#         if not X_test and not y_test and not test_size:
#             X_train, y_train = [], []
#             for X_train_cls, y_train_cls in DatasetPartitioner.__partition_by_class(X, y):
#                 X_train.append(X_train_cls)
#                 y_train.append(y_train_cls)
#             return np.array(X_train), np.array(y_train)
#         elif X_test and y_test:
#             X_train, y_train = DatasetPartitioner.get_dataset(X, y)
#             X_test, y_test = DatasetPartitioner.get_dataset(X_test, y_test)
#             return (X_train, y_train), (X_test, y_test)
#         elif test_size:
#             X_train, y_train = DatasetPartitioner.get_dataset(X, y)
#             _, _, X_test, y_test = DatasetPartitioner.get_dataset(X, y, test_size=test_size)
#             return (X_train, y_train), (X_test, y_test)



# Setup
@pytest.fixture
def X():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

@pytest.fixture
def y():
    return np.array([1, 2, 3])

def test_get_num_classes(X, y):
    assert DatasetPartitioner._DatasetPartitioner__get_num_classes(y) == 3

def test_get_partitions(X, y):
    X_train_cls, y_train_cls = DatasetPartitioner._DatasetPartitioner__get_partitions(X, y, 1)
    assert X_train_cls == [[1, 2, 3]]
    assert y_train_cls == [1]

def test_partition_by_class(X, y):
    for X_train_cls, y_train_cls in DatasetPartitioner._DatasetPartitioner__partition_by_class(X, y):
        assert X_train_cls == [[1, 2, 3]]
        assert y_train_cls == [1]
        break

def test_get_dataset(X, y):
    X_train, y_train = DatasetPartitioner.get_dataset(X, y)
    assert (X_train == [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]).all()
    assert (y_train == [[1], [2], [3]]).all()

def test_get_dataset_shape(X, y):
    X_train, y_train = DatasetPartitioner.get_dataset(X, y)
    assert X_train.shape[0] == 3
    assert y_train.shape[0] == 3