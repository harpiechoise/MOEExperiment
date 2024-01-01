import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


class DatasetPartitioner:
    @staticmethod
    def __get_num_classes(y: np.ndarray) -> int:
        """Takes a labels vector and returns the number of classes.

        Args:
            y (np.ndarray): Is the labels vector.

        Returns:
            int: The number of classes.
        """
        return len(np.unique(y))

    @staticmethod
    def __get_partitions(X: np.ndarray, y: np.ndarray, cls) -> Tuple[np.ndarray, np.ndarray]:
        """Takes a multiclass dataset and returns a binary dataset for a given class.

        Args:
            X (np.ndarray): Is the examples matrix.
            y (np.ndarray): Is the labels vector.

        Returns:
            _type_: _description_
        """
        return X[y == cls].tolist(), y[y == cls].tolist()

    @staticmethod
    def __partition_by_class(X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_test: np.ndarray = None,
                             y_test: np.ndarray = None,
                             test_size: float = None,
                             random_state=None) -> Tuple[list, list, list, list]:
        """Takes a multiclass dataset and returns a binary dataset for each class and yields the binary datasets.
        
        Args:
            X_train (np.ndarray): Is the examples matrix for the training set.
            y_train (np.ndarray): Is the labels vector for the training set.
            X_test (np.ndarray): Is the examples matrix for the test set.
            y_test (np.ndarray): Is the labels vector for the test set.
        
        Yields:
            Tuple[list, list]: The binary datasets.
            Tuple[list, list, list, list]: The binary datasets.
        
        Raises:
            ValueError: If X_train and y_train have different number of examples.
            ValueError: If y_train is not a vector.
            ValueError: If X_test and y_test have different number of examples.
            ValueError: If y_test is not a vector.
        """
        if X_test is None or y_test is None:
            for cls in np.unique(y_train):
                yield DatasetPartitioner.__get_partitions(X_train, y_train, cls)
        elif X_test and y_test:
            for cls in np.unique(y_train):
                X_train_cls, y_train_cls = DatasetPartitioner.__get_partitions(X_train, y_train, cls)
                X_test_cls, y_test_cls = DatasetPartitioner.__get_partitions(X_test, y_test, cls)
                yield X_train_cls, y_train_cls, X_test_cls, y_test_cls
        elif test_size:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, 
                                                                random_state=random_state)
            X_train, y_train = DatasetPartitioner.__get_partitions(X_train, y_train)
            X_test, y_test = DatasetPartitioner.__get_partitions(X_test, y_test)
            return X_train, y_train, X_test, y_test
    

    @staticmethod
    def get_dataset_partitioned(X, y, X_test=None, y_test=None, test_size=None):
        """ Takes a multiclass dataset and returns a binary dataset for each class.

        Args:
            X (np.ndarray): Is the examples matrix.
            y (np.ndarray): Is the labels vector.
            X_test (np.ndarray): Is the examples matrix for the test set.
            y_test (np.ndarray): Is the labels vector for the test set.
            test_size (float): Is the proportion of the dataset to include in the test split.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The training and test sets.
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The training and test sets.
        
        Raises:
            ValueError: If X and y have different number of examples.
            ValueError: If y is not a vector.
        """
        if not X_test and not y_test and not test_size:
            X_train, y_train = [], []
            for X_train_cls, y_train_cls in DatasetPartitioner.__partition_by_class(X, y):
                X_train.append(X_train_cls)
                y_train.append(y_train_cls)
            return np.array(X_train), np.array(y_train)
        elif X_test and y_test:
            X_train, y_train = DatasetPartitioner.get_dataset(X, y)
            X_test, y_test = DatasetPartitioner.get_dataset(X_test, y_test)
            return (X_train, y_train), (X_test, y_test)
        elif test_size:
            X_train, y_train = DatasetPartitioner.get_dataset(X, y)
            _, _, X_test, y_test = DatasetPartitioner.get_dataset(X, y, test_size=test_size)
            return (X_train, y_train), (X_test, y_test)
