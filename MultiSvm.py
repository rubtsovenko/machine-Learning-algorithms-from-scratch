import numpy as np
from DualSVM import DualSVM


class MultiSVM(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.clfs = []

    def train(self, X, y, gamma=0.6, C=10):
        for k1 in np.arange(self.n_classes):
            for k2 in np.arange(k1 + 1, self.n_classes):
                print('k1 = {}, k2 = {}', k2.format(k1, k2))
                data_k = self.data_one_vs_one(k1, k2, X, y)
                y_k = data_k[0]
                X_k = data_k[1]

                clf = DualSVM(kernel='poly', gamma=gamma, C=C, degree=2)
                clf.train(X_k, y_k)
                self.clfs.append([clf, k1, k2])

    def data_one_vs_one(self, k1, k2, X_train, y_train):
        indexes_k1 = (y_train == k1)
        indexes_k2 = (y_train == k2)
        y_train_k = np.concatenate((y_train[indexes_k1], y_train[indexes_k2]))
        y_train_k = self.one_vs_one_transformed_labels(k1, k2, y_train_k)
        X_train_k = np.vstack((X_train[indexes_k1], X_train[indexes_k2]))
        return y_train_k, X_train_k

    @staticmethod
    def one_vs_one_transformed_labels(k1, k2, y_train_k):
        y = np.zeros(y_train_k.shape[0])
        for i in np.arange(y_train_k.shape[0]):
            if y_train_k[i] == k1:
                y[i] = 1
            else:
                y[i] = -1
        return y

    def predict(self, X):
        predictions = []
        size = X.shape[0]

        for j in np.arange(size):
            x = X[j, :]
            scores = np.zeros(self.n_classes)
            for i in np.arange(len(self.clfs)):
                temp = self.clfs[i]
                clf = temp[0]
                k1 = temp[1]
                k2 = temp[2]
                pred = clf.predict(x)
                if pred == 1:
                    scores[k1] += 1
                else:
                    scores[k2] += 1
            predictions.append(np.random.choice(np.where(scores == max(scores))[0]))

            if j % 100 == 0:
                print(j)

        return np.array(predictions)