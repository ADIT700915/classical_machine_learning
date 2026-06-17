# RIDGE REGRESSION USING GRADIENT DESCENT

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv(
    r"C:\PYHTON\classical_machine_learning\Phase02_Implementations\diabetes.csv"
)

print(df.shape)
print(df.head())
print(df.info())

from sklearn.metrics import r2_score

# Features and Target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, # 20% data to testing
    random_state=4
)

# Convert to numpy arrays
y_train = y_train.values
y_test = y_test.values


from sklearn.linear_model import SGDRegressor

reg = SGDRegressor(
    penalty='l2', # penalty is the ridge loss term (that is calculated using L2  norm)
    max_iter=5000, # refers to epochs
    eta0=0.01, # refers to learning rate
    learning_rate='constant',
    alpha=0.001, #(lambda value)
    random_state=42
)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("\nSGD REGRESSOR RESULTS")
print("R2 score:", r2_score(y_test, y_pred))
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)


from sklearn.linear_model import Ridge

reg = Ridge(
    alpha=0.001,
    max_iter=500,
    solver='sparse_cg'
)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("\nRIDGE REGRESSION (SKLEARN) RESULTS")
print("R2 score:", r2_score(y_test, y_pred))
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)


class RidgeGD:

    def __init__(self, epochs, learning_rate, alpha):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.alpha = alpha

        self.coef_ = None
        self.intercept_ = None
        self.losses = []

    def fit(self, X_train, y_train):

        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]

        # Initialize weights and bias
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Add bias column
        X_train_bias = np.insert(X_train, 0, 1, axis=1)

        # theta = [bias, weights]
        theta = np.insert(self.coef_, 0, self.intercept_)

        for i in range(self.epochs):

            # Predictions
            y_pred_train = np.dot(X_train_bias, theta)

            # Gradient of MSE
            gradient = (
                (-2 / n_samples)
                * np.dot(X_train_bias.T, (y_train - y_pred_train))
            )

            # Ridge penalty which penalises larger coefficients 
            gradient[1:] = gradient[1:] + (
                2 * self.alpha * theta[1:]
            )

            # Update parameters
            theta = theta - self.learning_rate * gradient

            # Ridge Loss
            loss = (
                np.mean((y_train - y_pred_train) ** 2)
                + self.alpha * np.sum(theta[1:] ** 2)
            )

            self.losses.append(loss)

        
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X_test):

        return np.dot(X_test, self.coef_) + self.intercept_



reg = RidgeGD(
    epochs=5000,
    learning_rate=0.001,
    alpha=0.001
)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("\nCUSTOM RIDGE GD RESULTS")
print("R2 score:", r2_score(y_test, y_pred))
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)


plt.figure(figsize=(8, 5))

plt.plot(reg.losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")

plt.grid(True)

plt.show()


plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

min_val = min(np.min(y_test), np.min(y_pred))
max_val = max(np.max(y_test), np.max(y_pred))

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    'r--',
    linewidth=2
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")

plt.grid(True)

plt.show()