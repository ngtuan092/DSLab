import numpy as np


class RidgeRegression:
    NUM_FOLDS = 5

    def __init__(self):
        return

    def fit(self, X_train, Y_train, LAMBDA):
        assert X_train.shape[0] == Y_train.shape[0] and len(X_train.shape) == 2
        w = np.linalg.inv(X_train.transpose().dot(X_train) + LAMBDA *
                          np.identity(X_train.shape[1])).dot(X_train.transpose()).dot(Y_train)
        return w

    def predict(self, W, X_test):
        X_test = np.array(X_test)
        Y_new = X_test.dot(W)
        return Y_new

    def compute_RSS(self, Y_predict, Y_test):
        return 1. / Y_test.shape[0] * (Y_predict - Y_test).transpose().dot(Y_predict - Y_test)

    def get_the_best_LAMBDA(self, X_train, Y_train):
        def range_scan(bestLAMBDA, minimum_RSS, LAMBDA_values):
            for currentLAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(self.NUM_FOLDS, currentLAMBDA)
                if aver_RSS < minimum_RSS:
                    minimum_RSS = aver_RSS
                    bestLAMBDA = currentLAMBDA
            return bestLAMBDA, minimum_RSS

        def cross_validation(num_folds, LAMBDA):
            # split out X_train in to num_folds folds
            n = X_train.shape[0]
            row_ids = np.array(range(n))
            valid_ids = np.split(row_ids[: n - n % num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1],
                                      row_ids[n - n % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]]
                         for i in range(num_folds)]
            aver_RSS = 0
            for i in range(num_folds):
                train_part = {
                    'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                valid_part = {
                    'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
                w = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                Y_predicted = self.predict(w, valid_part['X'])
                aver_RSS += self.compute_RSS(Y_predicted, valid_part['Y'])

            return aver_RSS / num_folds
        bestLAMBDA, minimum_RSS = range_scan(
            bestLAMBDA=0, minimum_RSS=10000 ** 2, LAMBDA_values=range(50))
        LAMBDA_values = [
            k * 1. / 1000 for k in range(max(1, (bestLAMBDA + 1) * 1000))]
        bestLAMBDA, minimum_RSS = range_scan(
            bestLAMBDA, minimum_RSS, LAMBDA_values)
        return bestLAMBDA


def get_data(path):
    X = []
    Y = []
    with open(path, "r") as data:
        lines = data.read().split("\n")
        for line in lines:
            dataPoint = list(
                map(float, list(filter(lambda str: str != '', line.split(' ')))))
            X.append(dataPoint[1:-1])
            Y.append(dataPoint[-1])
    return X, Y


def normalize_and_add_ones(X):
    X = np.array(X)
    X_max = np.array([[np.amax(X[:, col])
                       for col in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    X_min = np.array([[np.amin(X[:, col])
                       for col in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    X_normalized = (X - X_min) / (X_max - X_min)
    ones = [[1] for _ in range(X.shape[0])]
    return np.column_stack((ones, X_normalized))


if __name__ == '__main__':
    X, Y = map(np.array, get_data(path="DeathRate.txt"))
    X = normalize_and_add_ones(X)
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]
    rg = RidgeRegression()
    bestLambda = rg.get_the_best_LAMBDA(X_train, Y_train)
    print(f'Best lambda: {bestLambda}')
    W_learned = rg.fit(X_train, Y_train, bestLambda)
    Y_predicted = rg.predict(W_learned, X_test)
    print(rg.compute_RSS(Y_test, Y_predicted))
