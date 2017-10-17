import numpy as np
import copy
from tqdm import tqdm
import time


# labels y have to be -1 or +1
class LinearSvm(object):
    def __init__(self, C=1.0, mini_batch_size=None, max_iter=1000, learning_rate=0.001, epsilon=0.00001,
                 check_stopping_condition=False, random_seed=None):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.C = C
        self.mini_batch_size = mini_batch_size
        self.epsilon = epsilon
        self.check_stopping_conditions = check_stopping_condition
        self.random_seed = random_seed

    # I use (sub) gradient decent method to learn parameters of the linear model (linear SVM) if mini_batch_size is None
    # or stochastic (sub) gradient decent otherwise
    def train(self, X, y):
        start_time = time.clock()
        self._init(X, y)

        for i in tqdm(range(self.max_iter)):
            weights_prev = np.copy(self.weights)
            bias_prev = np.copy(self.bias)

            if self.mini_batch_size == X.shape[0]:
                self._weights_bias_update(X, y)
            else:
                X_batch, y_batch = self._sample_mini_batch()
                self._weights_bias_update(X_batch, y_batch)
            self.loss.append(self._compute_loss(X, y))

            if self.loss[-1] < self.loss_min:
                self.loss_min = self.loss[-1]
                self.weights_min = copy.copy(self.weights)
                self.bias_min = copy.copy(self.bias)
                self.iteration_min = i+1

            if self.check_stopping_conditions:
                if self._stopping_conditions(self.loss[-1], self.loss[-2], np.hstack((self.weights, self.bias)),
                                             np.hstack((weights_prev, bias_prev))):
                    break
        self._set_weights_with_min_loss()
        finish_time = time.clock()
        print('Training time:', finish_time - start_time)

    # we want a mini_batch to have equal number of elements with labels -1 and +1
    # TODO: add errors handling
    def _set_weights_with_min_loss(self):
        self.weights = self.weights_min
        self.bias = self.bias_min

    def predict(self, X):
        return np.sign(np.matmul(X, self.weights) + self.bias)

    def _stopping_conditions(self, loss_cur, loss_prev, parameters_cur, parameters_prev):
        if np.abs(loss_cur - loss_prev) < self.epsilon:
            return True
        if np.linalg.norm(parameters_cur - parameters_prev) < self.epsilon:
            return True

        return False

    def _init(self, X, y):
        if self.mini_batch_size is None:
            self.mini_batch_size = X.shape[0]
        else:
            self.y_neg = y[y == -1]
            self.X_neg = X[y == -1]
            self.y_pos = y[y == 1]
            self.X_pos = X[y == 1]

            if len(self.y_neg) < len(self.y_pos):
                if len(self.y_neg) < self.mini_batch_size // 2:
                    print('Error: half size of a mini batch is greater then number of negative samples in the dataset')
            else:
                if len(self.y_pos) < self.mini_batch_size // 2:
                    print('Error: half size of a mini batch is greater then number of positive samples in the dataset')

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn(1)
        self.loss = []
        self.loss.append(self._compute_loss(X, y))
        self.loss_min = self.loss[-1]
        self.weights_min = self.weights
        self.bias_min = self.bias
        self.iteration_min = 0

    def _sample_mini_batch(self):
        inds_neg = np.random.choice(self.X_neg.shape[0], self.mini_batch_size // 2, replace=False)
        inds_pos = np.random.choice(self.X_pos.shape[0], self.mini_batch_size // 2, replace=False)

        X_batch = np.vstack((self.X_neg[inds_neg], self.X_pos[inds_pos]))
        y_batch = np.concatenate((self.y_neg[inds_neg], self.y_pos[inds_pos]))

        return X_batch, y_batch

    def _compute_loss(self, X, y):
        margins = self.get_margins(X,y)
        temp_sum = 0
        for margin in margins:
            temp_sum += self._hinge_loss(margin)
        return np.linalg.norm(self.weights) + self.C * temp_sum

    def _weights_bias_update(self, X, y):
        margins = (np.matmul(X, self.weights) + self.bias) * y
        for index in range(len(self.weights)):
            self.weights[index] -= self.learning_rate * self._compute_partial_derivative_weight(index, X, y, margins)
            self.bias -= self.learning_rate * self._compute_partial_derivative_bias(X, y, margins)

    def _compute_partial_derivative_weight(self, derivative_index, X, y, margins):
        temp_sum = 0
        for i in range(X.shape[0]):
            temp_sum += self._subgrad_hinge_loss(margins[i]) * X[i, derivative_index] * y[i]
        partial_derivative = 2 * self.weights[derivative_index] + self.C * temp_sum

        return partial_derivative

    def _compute_partial_derivative_bias(self, X, y, margins):
        temp_sum = 0
        for i in range(X.shape[0]):
            temp_sum += self._subgrad_hinge_loss(margins[i]) * y[i]
        return temp_sum

    def _hinge_loss(self, x):
        if x < 1:
            return 1 - x
        else:
            return 0

    def _subgrad_hinge_loss(self, x):
        if x < 1:
            return -1
        if x > 1:
            return 0
        if x == 1:
            return -1

    def get_margins(self, X, y):
        return (np.matmul(X, self.weights) + self.bias) * y

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_loss_history(self):
        return self.loss