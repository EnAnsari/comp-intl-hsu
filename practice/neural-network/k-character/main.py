import numpy as np
from sklearn.neural_network import MLPClassifier

# define the training data and labels
X_train = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])
y_train = np.array([1])

# define the MLP classifier and fit the data
mlp = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=1000)
mlp.fit(X_train.reshape(1, -1), y_train)

# define the test data
X_test = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])

# predict whether the input matrix contains the letter 'K'
y_pred = mlp.predict(X_test.reshape(1, -1))
if y_pred == 1:
    print("Letter K found in the input matrix")
else:
    print("Letter K not found in the input matrix")
