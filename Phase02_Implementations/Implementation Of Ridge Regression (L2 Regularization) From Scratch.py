"""
np.linalg.inv(A) @ B and np.linalg.solve(A, B)
Both give the same answer.
But solve() is:
1)Faster
2)Uses less memory
3)More numerically stable
Avoids explicitly computing A inverse
Therefore in scientific computing, solve() is preferred.

"""




import numpy as np

class RidgeRegression:

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        # Add bias column
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        n_features = X.shape[1]

        # Identity matrix
        I = np.eye(n_features)

        # Don't regularize bias
        I[0, 0] = 0

        self.beta = np.linalg.solve(
            X.T @ X + self.alpha * I
        , X.T @ y)

    def predict(self, X):

        X = np.array(X)

        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        return X @ self.beta




X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

y = np.array([
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21
])

indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices] #SUFFLING THE INDICES

split_index = int(0.8 * len(X))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]



model = RidgeRegression(alpha=1.0)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Actual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)

mse = np.mean((y_test - y_pred) ** 2) #CALCULATING TEST ERROR
print("MSE =", mse)

ss_res = np.sum((y_test - y_pred) ** 2)

ss_tot = np.sum(
    (y_test - np.mean(y_test)) ** 2
)

r2 = 1 - (ss_res / ss_tot)

print("R² =", r2) #CALCULATING RESIDUAL ERROR