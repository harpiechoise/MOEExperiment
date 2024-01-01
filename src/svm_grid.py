from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from typing import Optional
import numpy as np

def make_optional_alias(cls):
    return Optional[cls]

opt_int = Optional[int]
opt_float = Optional[float]
opt_str = Optional[str]
opt_bool = Optional[bool]

class SVCEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators: opt_int = 10, model_names: Optional[list[str]] = None,
                 C: opt_float = 1.0, kernel: opt_str = 'rbf', degree: opt_int = 3, gamma: opt_str = 'auto',
                 coef0: opt_float = 0.0, shrinking: opt_bool = True, probability: opt_bool = False,
                 tol: opt_float = 1e-3, cache_size: opt_int=200, class_weight=None,
                 verbose: opt_bool = False, max_iter: opt_int =-1, decision_function_shape: opt_str ='ovr',
                 break_ties: opt_bool =False, random_state=None):
        
        """All parameters not specified are set to their defaults
        The model was taken from sklearn.svm.SVC, the only difference is that
        this model is an ensemble of SVCs.
        """
        self.n_estimators = n_estimators
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

        # Initialize the ensemble
        self.ensemble = []
        for i in range(self.n_estimators):
            self.ensemble.append(SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                                     coef0=self.coef0, shrinking=self.shrinking, probability=self.probability,
                                     tol=self.tol, cache_size=self.cache_size, class_weight=self.class_weight,
                                     verbose=self.verbose, max_iter=self.max_iter, decision_function_shape=self.decision_function_shape,
                                     break_ties=self.break_ties, random_state=self.random_state))
    
    def __validate_data_format(self, X: np.ndarray, y: np.ndarray):
        """Validates the format of the data.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of examples.")
        if len(y.shape) != 1:
            raise ValueError("y must be a vector.")
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model according to the given training data.
        """
        self.__validate_data_format(X, y)
        for i in range(self.n_estimators):
            # extract data for the i-th estimator
            X_i = X[i::self.n_estimators]
            y_i = y[i::self.n_estimators]
            # fit the i-th estimator
            self.ensemble[i].fit(X_i, y_i)
        return self
    
    def predict(self, X: np.ndarray):
        """Predict class labels for samples in X.
        """
        predictions = []
        # predict for each estimator
        for i in range(self.n_estimators):
            predictions.append(self.ensemble[i].predict(X))
