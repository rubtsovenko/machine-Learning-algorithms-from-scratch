from LinearSVM import LinearSvm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


def plot_dataset(X, y):
    print('Dataset:')
    x_lim_lower = np.min(X[:, 0]) - 1
    x_lim_upper = np.max(X[:, 0]) + 1
    y_lim_lower = np.min(X[:, 1]) - 1
    y_lim_upper = np.max(X[:, 1]) + 1
    plt.xlim([x_lim_lower, x_lim_upper])
    plt.ylim([y_lim_lower, y_lim_upper])
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='b')
    plt.show()
    return [x_lim_lower, x_lim_upper, y_lim_lower, y_lim_upper]


def print_model_summary(y, preds, weights, bias, brief=False):
    if brief:
        print('accuracy:', sum(preds == y) / len(y))
    else:
        print('predictions:', preds)
        print('accuracy:', sum(preds == y) / len(y))
        print('weights:', weights)
        print('bias:', bias)


def linearSVC_model(X, y, C=1.0, random_state=None, brief=False):
    print('LinearSVC model:')
    model = LinearSVC(C=C, loss='hinge', random_state=random_state)
    model.fit(X, y)
    print_model_summary(y, model.predict(X), model.coef_[0], model.intercept_, brief=brief)
    return model


def SVC_model(X, y, C=1.0, brief=False):
    print('SVC model:')
    model = SVC(C=C, kernel='linear')
    model.fit(X, y)
    print_model_summary(y, model.predict(X), model.coef_[0], model.intercept_, brief=brief)
    return model


def my_model(X, y, C=1.0, learning_rate=0.001, max_iter=10000, random_seed=10, mini_batch_size=None, brief=False):
    print('My model:')
    model = LinearSvm(C=C, learning_rate=learning_rate, max_iter=max_iter,
                      random_seed=random_seed, mini_batch_size=mini_batch_size)
    model.train(X, y)
    print_model_summary(y, model.predict(X), model.get_weights(), model.get_bias(), brief=brief)
    return model


def plot_hyperplane(X, y, model, library, xy_lims):
    x_lim_lower, x_lim_upper, y_lim_lower, y_lim_upper = xy_lims[0], xy_lims[1], xy_lims[2], xy_lims[3]
    if library == 'sklearn':
        w1, w2 = model.coef_[0]
        bias = model.intercept_
    if library == 'my':
        w1, w2 = model.get_weights()
        bias = model.get_bias()
    x_plot = np.linspace(x_lim_lower, x_lim_upper, 1000)
    y_plot = (-x_plot * w1 - bias) / w2
    plt.xlim([x_lim_lower, x_lim_upper])
    plt.ylim([y_lim_lower, y_lim_upper])
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='b')
    plt.plot(x_plot, y_plot)
    plt.show()
    return x_plot, y_plot


def plot_hplanes_comparison(X, y, x_lin, y_lin, x_svc, y_svc, x_my, y_my, xy_lims):
    print('Models Comparison:')
    x_lim_lower, x_lim_upper, y_lim_lower, y_lim_upper = xy_lims[0], xy_lims[1], xy_lims[2], xy_lims[3]
    plt.xlim([x_lim_lower, x_lim_upper])
    plt.ylim([y_lim_lower, y_lim_upper])
    plt.plot(x_my, y_my, color='black', label='my_model')
    plt.plot(x_svc, y_svc, color='orange', label='SVC')
    plt.plot(x_lin, y_lin, color='green', label='linearSVC')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='b')
    plt.show()


def svm_test_2d(X, y, C=1.0, learning_rate=0.001, max_iter=10000, random_seed=10,
                random_state=None, mini_batch_size=None):
    xy_lims = plot_dataset(X, y)

    model_linSVC = linearSVC_model(X, y, C=C, random_state=random_state)
    x_plot_lin, y_plot_lin = plot_hyperplane(X, y, model_linSVC, 'sklearn', xy_lims)

    model_SVC = SVC_model(X, y, C=C)
    x_plot_svc, y_plot_svc = plot_hyperplane(X, y, model_SVC, 'sklearn', xy_lims)

    model_my = my_model(X, y, C=C, learning_rate=learning_rate, max_iter=max_iter,
                        random_seed=random_seed, mini_batch_size=mini_batch_size)
    x_plot_my, y_plot_my = plot_hyperplane(X, y, model_my, 'my', xy_lims)

    plot_hplanes_comparison(X, y, x_plot_lin, y_plot_lin, x_plot_svc, y_plot_svc, x_plot_my, y_plot_my, xy_lims)

    return model_linSVC, model_SVC, model_my


def svm_test(X, y, C=1.0, learning_rate=0.001, max_iter=10000, random_seed=10,
             random_state=None, mini_batch_size=None, brief=False):
    model_linSVC = linearSVC_model(X, y, C=C, random_state=random_state, brief=brief)
    model_SVC = SVC_model(X, y, C=C, brief=brief)
    model_my = my_model(X, y, C=C, learning_rate=learning_rate, max_iter=max_iter,
                        random_seed=random_seed, mini_batch_size=mini_batch_size, brief=brief)
    return model_linSVC, model_SVC, model_my
