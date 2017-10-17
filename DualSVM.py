import numpy as np
import cvxopt


# kernel = 'linear', 'rdf', 'poly'
# labels_y have to be -1 or +1
class DualSVM(object):
    def __init__(self, C=1.0, kernel='linear', gamma=0.5, coef_pol=0.0, degree=2):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef_pol = coef_pol
        self.degree = degree

    def train(self, X, y):
        K = self.gram_matrix(X)
        n = y.shape[0]

        # Matrices for quadratic program
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n))

        # equality constarins
        G1 = np.diag(-np.ones(n))
        h1 = np.zeros((n, 1))

        G2 = np.diag(np.ones(n))
        h2 = self.C * np.ones((n, 1))

        G = cvxopt.matrix(np.vstack((G1, G2)))
        h = cvxopt.matrix(np.vstack((h1, h2)))

        # inequality constrains (only one inequality)
        A = cvxopt.matrix(y, (1, n), tc='d')
        b = cvxopt.matrix(0.0)

        lambdas = cvxopt.solvers.qp(P, q, G, h, A, b)
        lambdas = np.squeeze(np.round(np.array(lambdas['x']), decimals=15))

        if self.kernel == 'poly':
            self.support_indexes = lambdas > 10 ** -12
        else:
            self.support_indexes = lambdas > 10 ** -4
        self.lambdas = lambdas[self.support_indexes]
        self.support_vectors = X[self.support_indexes, :]
        self.sup_vec_labels = y[self.support_indexes]
        self.coef = self.get_weights()
        self.bias = self.compute_bias()

    def compute_bias(self):
        if self.C >= 1:
            indexes = (self.lambdas > 0.1) & (self.lambdas < 0.8 * self.C)
        else:
            indexes = (self.lambdas > 0.1 * self.C) & (self.lambdas < 0.8 * self.C)

        if np.all(self.lambdas <= 0.1 * self.C):
            indexes = self.lambdas < 0.1 * self.C

        max_index = np.argmax(self.lambdas[indexes])

        sup = self.support_vectors[indexes][max_index]
        label = self.sup_vec_labels[indexes][max_index]

        total = -label
        for i in np.arange(self.lambdas.shape[0]):
            total += self.lambdas[i] * self.sup_vec_labels[i] * self.f_kernel(sup, self.support_vectors[i])

        return -total

    def gram_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in np.arange(n):
            for j in np.arange(n):
                K[i, j] = self.f_kernel(X[i, :], X[j, :])

        return K

    def f_kernel(self, a, b):

        if self.kernel == 'linear':
            return np.dot(a, b)
        elif self.kernel == 'rbf':
            return np.exp(-np.linalg.norm(a - b) ** 2 * self.gamma)
        elif self.kernel == 'poly':
            return (np.dot(a, b) + self.coef_pol) ** self.degree
        else:
            return -0.013

    def predict(self, x):
        res = self.bias
        for lambda_i, y_i, x_i in zip(self.lambdas, self.sup_vec_labels, self.support_vectors):
            res += lambda_i * y_i * self.f_kernel(x_i, x)

        return np.sign(res)

    def get_weights(self):
        n = self.support_vectors.shape[0]
        weights = np.zeros(self.support_vectors.shape[1])
        for i in np.arange(n):
            weights += self.lambdas[i] * self.sup_vec_labels[i] * self.support_vectors[i, :]

        return weights

    def pred_margin(self, x):
        res = self.bias
        for lambda_i, y_i, x_i in zip(self.lambdas, self.sup_vec_labels, self.support_vectors):
            res += lambda_i * y_i * self.f_kernel(x_i, x)
        return (res, np.sign(res))