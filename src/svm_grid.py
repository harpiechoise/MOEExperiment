from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

class SVCEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10,
                 C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False, random_state=None):
        
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
    
    def __validate_data_format(self, X, y):
        """Validates the format of the data.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of examples.")
        if len(y.shape) != 1:
            raise ValueError("y must be a vector.")