from data_handler import bagging_sampler
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.model_list = []

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            self.base_estimator.fit(X_sample, y_sample)
            self.model_list.append(self.base_estimator)




    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        y_hat = []
        for i in range(self.n_estimator):
            y_hat.append(self.model_list[i].predict(X))
        return np.where(np.mean(y_hat, axis=0) > 0.5, 1, 0)

