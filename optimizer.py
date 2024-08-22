from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
import sys
class ModelOptimizer:
    def __init__(self, estimator, param_grid=None, optimization_algorithm='grid_search'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.optimization_algorithm = optimization_algorithm

    def optimize(self, X, y, cv=5):
        if self.param_grid is None:
            raise ValueError("Parameter grid not provided!")

        if self.optimization_algorithm == 'grid_search':
            optimizer = GridSearchCV(self.estimator, self.param_grid, cv=cv)
        elif self.optimization_algorithm == 'randomized_search':
            optimizer = RandomizedSearchCV(self.estimator, self.param_grid, cv=cv)
        elif self.optimization_algorithm == 'bayesian_optimization':
            optimizer = BayesSearchCV(self.estimator, self.param_grid, cv=cv)
        else:
            raise ValueError("Invalid optimization algorithm!")
        try:
            optimizer.fit(X, y)
        except Exception as e:
            sys.exit(e)
        # best_params = optimizer.best_params_
        # best_model = optimizer.best_estimator_

        return optimizer

    def create_pipeline(self, steps):
        self.estimator = Pipeline(steps)
