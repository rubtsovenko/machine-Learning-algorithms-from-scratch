import numpy as np
import random


class KMeans:
    def __init__(self, k=3, n_init=5, max_iter=5):
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter

    def train(self, X):
        all_J = []
        all_centroids = []

        for _ in np.arange(self.n_init):
            r_nk = np.zeros((X.shape[0], self.k))
            centroids = X[random.sample(np.arange(1, X.shape[0]), self.k)]

            for iter in np.arange(self.max_iter):
                for n in np.arange(X.shape[0]):
                    distances = np.linalg.norm(X[n, :] - centroids, ord=2, axis=1)
                    r_nk[n, np.argmin(distances)] = 1
                for i in np.arange(self.k):
                    centroids[i, :] = (np.sum(X * r_nk[:, i].reshape(X.shape[0], 1), axis=0)) / float(sum(r_nk[:, i]))
            J = 0
            for i in np.arange(X.shape[0]):
                for j in np.arange(self.k):
                    J += r_nk[i, j] * np.linalg.norm(X[i, :] - centroids[j, :], ord=2)

            all_J.append(J)
            all_centroids.append(centroids)

        all_J = np.array(all_J)
        all_centroids = np.array(all_centroids)

        index = np.argmin(all_J)

        self.centroids = all_centroids[index]
        self.J = all_J[index]

    def predict(self, X):
        n = X.shape[0]
        labels = []

        for i in np.arange(n):
            distances = np.linalg.norm(X[i, :] - self.centroids, ord=2, axis=1)
            labels.append(np.argmin(distances))

        return labels

    def predict_soft(self, X):
        n = X.shape[0]
        labels = []

        for i in np.arange(n):
            z = np.linalg.norm(X[i, :] - self.centroids, ord=2, axis=1)
            mu = np.mean(z)
            labels.append(np.maximum(mu - z, 0))

        return labels