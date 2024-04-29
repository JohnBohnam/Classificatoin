import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)

            if self.verbose == True and i % 1000 == 0:
                loss = self.__loss(h, y)
                print(f'loss: {loss} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()


class NaiveBayes:
    def __init__(self, n_features, n_feature_values, laplace_smoothing=True):
        self.n_features = n_features
        self.n_feature_values = n_feature_values
        self.classes = [0, 1]
        self.laplace_smoothing = laplace_smoothing
        self.fi1 = np.zeros((n_features, n_feature_values))
        self.fi0 = np.zeros((n_features, n_feature_values))
        self.fiy = 0.5

    def fit(self, X, y):
        n_ones = np.sum(y == 1)
        n_zeros = np.sum(y == 0)

        if self.laplace_smoothing:
            self.fiy = (n_ones + 1) / (2 + len(y))
        else:
            self.fiy = n_ones / len(y)

        for j in range(self.n_features):
            for k in range(self.n_feature_values):
                self.fi1[j, k] = np.sum((X[:, j] == k) & (y == 1))
                if self.laplace_smoothing:
                    self.fi1[j, k] += 1
                    self.fi1[j, k] /= n_ones + self.n_feature_values
                else:
                    self.fi1[j, k] /= n_ones

                self.fi0[j, k] = np.sum((X[:, j] == k) & (y == 0))
                if self.laplace_smoothing:
                    self.fi0[j, k] += 1
                    self.fi0[j, k] /= n_zeros + self.n_feature_values
                else:
                    self.fi0[j, k] /= n_zeros

    def predict(self, X):
        y_pred = np.zeros(len(X))
        prior0 = 1 - self.fiy
        prior1 = self.fiy
        for i in range(len(X)):
            p0 = np.log(prior0)
            p1 = np.log(prior1)
            for j in range(self.n_features):
                p0 += np.log(self.fi0[j, X[i, j]])
                p1 += np.log(self.fi1[j, X[i, j]])
            y_pred[i] = 1 if p1 > p0 else 0
        return y_pred

    def predict_prob(self, X):
        y_pred = np.zeros(len(X))
        prior0 = 1 - self.fiy
        prior1 = self.fiy
        for i in range(len(X)):
            p0 = np.log(prior0)
            p1 = np.log(prior1)
            for j in range(self.n_features):
                p0 += np.log(self.fi0[j, X[i, j]])
                p1 += np.log(self.fi1[j, X[i, j]])
            y_pred[i] = p1 / (p0 + p1)
        return 1 - y_pred
