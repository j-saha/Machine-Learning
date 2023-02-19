import numpy as np
import warnings
warnings.filterwarnings('ignore')


class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.epochs = params['epochs']
        self.lr = params['lr']

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0


        for i in range(self.epochs):
            z = np.matmul(X, self.weights) + self.bias
            y_hat = self.sigmoid(z)
            d_weight = (1.0/X.shape[0])*np.matmul(X.T, (y_hat - y))
            d_bias = (1.0/X.shape[0])*np.sum(y_hat - y)
            self.weights = self.weights - self.lr*d_weight
            self.bias = self.bias - self.lr*d_bias



    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        z = np.matmul(X, self.weights) + self.bias
        y_hat = self.sigmoid(z)
        return np.where(y_hat > 0.5, 1, 0)






