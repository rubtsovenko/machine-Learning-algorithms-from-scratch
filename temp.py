import svm
import numpy as np
import time

X = np.array([[1,1], [-1,-1], [2, 2], [-5, -10]])
y = np.array([1, -1, 1, -1])

start_time = time.clock()
model_svm = svm.Svm(max_iter=100000, learning_rate=0.001)
model_svm.train(X, y)
end_time = time.clock()
print("Training without checking the stopping conditions:", end_time - start_time)

start_time = time.clock()
model_svm = svm.Svm(max_iter=100000, learning_rate=0.001, check_stopping_condition=True)
model_svm.train(X, y)
end_time = time.clock()
print("Training with checking the stopping conditions:", end_time - start_time)



print(model_svm.weights)
print('margins:', model_svm.get_margins(X,y))
print('loss', model_svm.get_loss_history()[-5:])
print(np.linalg.norm(model_svm.get_weights()))