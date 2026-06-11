
"""
When we write:
 PolynomialFeatures(degree=2)
 LinearRegression()

Scikit-Leran internally performs following operations:
1)Create x² column
2)Create design matrix
3)Apply Normal Equation
4)Generate predictions
 
So,in this code we have used numpy only to understand how the algorithm works internally.

Feature Expansion means creating new features like xsq and xcube from existing features.
Curve Fitting means finding the mathematical curve that best matches the data by minimizing prediction error.
Polynomial Regression works by expanding features first and then fitting a curve using ordinary linear regression on those expanded features.

Also ,There is no such thing as a “perfect feature” in a general sense.
In machine learning, what we actually do is that we search for a feature set that gives the best generalization performance on unseen data.
Thus, polynomial Regression is about Feature Selection via Validation.
In simpler terms: 
1. Try multiple feature sets
2. Train model
3. Evaluate on validation set
4. Pick best score

"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36])  # perfect quadratic


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,  # 30% data to testing 
    random_state=42
)

def create_poly_features(X, degree):
    n = X.shape[0]
    
    # start with bias term (1)( intercept = bias )
    X_poly = np.ones((n, 1))
    
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))
    
    return X_poly

def train_model(X_poly, y):
    beta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    return beta


best_degree = None
#currently assume the worst possible error, and keep improving it.
best_error = float("inf")

for degree in range(1, 6):

    # feature expansion
    X_train_poly = create_poly_features(X_train, degree)
    X_test_poly = create_poly_features(X_test, degree)

    # train model
    beta = train_model(X_train_poly, y_train)

    # predictions
    y_pred = X_test_poly @ beta

    # error
    error = mean_squared_error(y_test, y_pred)

    print(f"Degree {degree} and MSE: {error}")

    # keep best
    if error < best_error:
        best_error = error
        best_degree = degree

print("Best Degree is:", best_degree)
print("Best Error is:", best_error)