import numpy as np

class MultipleLinearRegressionNormal:
    
    def fit(self, X, y):
        
        X = np.array(X)
        y = np.array(y)

        # Add column of ones for intercept term
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        # Normal Equation
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        
        X = np.array(X)

        # Add column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        return X @ self.beta

    def score(self, X, y):
        
        y = np.array(y)
        y_pred = self.predict(X)

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        r2 = 1 - (ss_residual / ss_total)

        return r2


X = np.array([
    [2, 3],
    [4, 8],
    [6, 5],
    [8, 12],
    [10, 7]
])

y = np.array([
    10,
    18,
    26,
    34,
    42
])


model = MultipleLinearRegressionNormal()

model.fit(X, y)



print("Intercept and Coefficients:")
print(model.beta)



predictions = model.predict(X)

print("\nPredictions:")
print(predictions)


r2 = model.score(X, y)

print("\nR² Score:")
print(r2)



new_data = np.array([
    [12, 13],
    [14, 15]
])

new_predictions = model.predict(new_data)

print("\nNew Predictions:")
print(new_predictions)


