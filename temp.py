import svm
import numpy as np


X = np.array([[1,1], [-1,-1], [2, 2], [-5, -10]])
y = np.array([1, -1, 1, -1])


model_svm = svm.Svm(num_iters=100000, learning_rate=0.001)
model_svm.train(X, y)
print(model_svm.weights)
print('margins:', model_svm.get_margins(X,y))
print('loss', model_svm.get_loss_history())
print(np.linalg.norm(model_svm.get_weights()))