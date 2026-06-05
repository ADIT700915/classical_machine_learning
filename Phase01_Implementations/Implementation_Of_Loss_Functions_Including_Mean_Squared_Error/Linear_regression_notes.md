# Linear Regression

Linear Regression is one of the simplest and most widely used machine learning algorithms. Its goal is to model the relationship between one or more **input variables (features)** and an **output variable (target)**.

The algorithm works by finding the **best-fit straight line** that minimizes the difference between the actual target values and the predicted target values.

A linear relationship can be represented as:

$$
y = mx + b
$$

where:

- $y$ = predicted target value
- $x$ = input feature
- $m$ = slope of the line
- $b$ = intercept

## Understanding the Parameters

### Slope ($m$)

The slope determines how much the target variable changes when the feature value changes by one unit.

- Positive slope → target increases with feature.
- Negative slope → target decreases with feature.
- Larger magnitude → steeper line.

### Intercept ($b$)

The intercept represents the predicted value of the target when the feature value is zero.

It acts as an offset and ensures that predictions can still be made even when:

$$
x = 0
$$

---

# Methods to Find the Best-Fit Line

For **Simple Linear Regression**, there are multiple approaches to calculate the values of $m$ and $b$:

1. Closed-Form Solution (Ordinary Least Squares)
2. Gradient Descent Optimization

In this document, we focus on the **Closed-Form Solution**, also known as **Ordinary Least Squares (OLS)**.

---

# Ordinary Least Squares (OLS)

OLS finds the values of the slope and intercept that minimize the overall prediction error.

The idea is simple:

1. Predict values using a line.
2. Measure how far predictions are from actual values.
3. Square the errors.
4. Sum all squared errors.
5. Find the values of $m$ and $b$ that make this sum as small as possible.

---

# Derivation of OLS Estimators

## Step 1: Define the Linear Regression Model

Assume a linear relationship between input $x$ and output $y$:

$$
\hat{y}_i = mx_i + b
$$

where:

- $\hat{y}_i$ = predicted value
- $x_i$ = feature value
- $m$ = slope
- $b$ = intercept

### Residual (Prediction Error)

The residual is the difference between the actual value and predicted value:

$$
e_i = y_i - \hat{y}_i
$$

Substituting the regression equation:

$$
e_i = y_i - (mx_i + b)
$$

or

$$
e_i = y_i - mx_i - b
$$

---

## Step 2: Define the Cost Function

OLS minimizes the **Sum of Squared Errors (SSE)**.

$$
SSE = \sum_{i=1}^{n} e_i^2
$$

Substituting the residual expression:

$$
SSE = \sum_{i=1}^{n} (y_i - mx_i - b)^2
$$

The squared term ensures:

- Positive and negative errors do not cancel out.
- Larger errors are penalized more heavily.

---

## Step 3: Differentiate SSE with Respect to $b$

To find the minimum SSE, take the partial derivative with respect to $b$ and set it equal to zero.

### Derivative

```math
\frac{\partial SSE}{\partial b}
=
\frac{\partial}{\partial b}
\sum_{i=1}^{n}
(y_i - mx_i - b)^2
```


Applying the chain rule:

```math
\frac{\partial SSE}{\partial b}
=
-2
\sum_{i=1}^{n}
(y_i - mx_i - b)

```

Setting the derivative equal to zero:

```math
-2
\sum_{i=1}^{n}
(y_i - mx_i - b)
=
0
```

Removing the constant:

```math
\sum_{i=1}^{n}
(y_i - mx_i - b)
=
0
```

Expanding:

```math
\sum y_i
-
m \sum x_i
-
nb
=
0
```

Solving for $b$:

```math
b
=
\frac{\sum y_i - m \sum x_i}{n}
```

Using the definitions:

```math
\bar{x}
=
\frac{\sum x_i}{n}
```

```math
\bar{y}
=
\frac{\sum y_i}{n}
```

we obtain:

```math
b
=
\bar{y}
-
m\bar{x}
```

---

## Step 4: Differentiate SSE with Respect to $m$

Starting from:

```math
SSE
=
\sum_{i=1}^{n}
(y_i - mx_i - b)^2
```

Differentiate with respect to $m$:

```math
\frac{\partial SSE}{\partial m}
=
-2
\sum_{i=1}^{n}
x_i(y_i - mx_i - b)
```

Setting the derivative equal to zero:

```math
-2
\sum_{i=1}^{n}
x_i(y_i - mx_i - b)
=
0
```

Removing the constant:

```math
\sum_{i=1}^{n}
x_i(y_i - mx_i - b)
=
0
```

Expanding:

```math
\sum x_i y_i
-
m \sum x_i^2
-
b \sum x_i
=
0
```

Substitute:

```math
b = \bar{y} - m\bar{x}
```

into the equation:

```math
\sum x_i y_i
-
m \sum x_i^2
-
(\bar{y}-m\bar{x})\sum x_i
=
0
```

Since:

```math
\sum x_i = n\bar{x}
```

we get:

```math
\sum x_i y_i
-
m \sum x_i^2
-
n\bar{x}\bar{y}
+
mn\bar{x}^2
=
0
```

Rearranging:

```math
m
\left(
\sum x_i^2
-
n\bar{x}^2
\right)
=
\sum x_i y_i
-
n\bar{x}\bar{y}
```

Therefore:

```math
m
=
\frac{
\sum x_i y_i
-
n\bar{x}\bar{y}
}{
\sum x_i^2
-
n\bar{x}^2
}
```

---

## Step 5: Convert to Covariance-Variance Form

Using:

```math
\sum (x_i-\bar{x})(y_i-\bar{y})
=
\sum x_i y_i
-
n\bar{x}\bar{y}
```

and

```math
\sum (x_i-\bar{x})^2
=
\sum x_i^2
-
n\bar{x}^2
```

the slope becomes:

```math
m
=
\frac{
\sum (x_i-\bar{x})(y_i-\bar{y})
}{
\sum (x_i-\bar{x})^2
}
```

---

# Final OLS Estimators

## Slope

```math
m
=
\frac{
\sum (x_i-\bar{x})(y_i-\bar{y})
}{
\sum (x_i-\bar{x})^2
}
```

---

## Intercept

```math
b
=
\bar{y}
-
m\bar{x}
```

---

# Interpretation

The slope can also be written as:

```math
m
=
\frac{\text{Cov}(X,Y)}
{\text{Var}(X)}
```

This means:

- Covariance measures how $X$ and $Y$ change together.
- Variance measures how spread out $X$ is.
- The slope represents the amount of change in $Y$ per unit change in $X$.

---

# Python Implementation of OLS From Scratch

```python
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate slope
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

m = numerator / denominator

# Calculate intercept
b = y_mean - m * x_mean

print("Slope (m):", m)
print("Intercept (b):", b)
```

---

# Making Predictions

Once the values of $m$ and $b$ are known, predictions can be made using:

```math
\hat{y}
=
mx + b
```

Example:

```python
x_new = 6

y_pred = m * x_new + b

print("Predicted value:", y_pred)
```

---

# Advantages of OLS

- Simple and easy to understand.
- Computationally efficient.
- Provides an exact analytical solution.
- Works well when the relationship is approximately linear.

# Limitations of OLS

- Sensitive to outliers.
- Assumes a linear relationship.
- Performance degrades when assumptions are violated.
- Not suitable for highly non-linear data.

---

# Summary

Linear Regression attempts to find the best-fit straight line through a dataset.

The Ordinary Least Squares method achieves this by minimizing the Sum of Squared Errors:

```math
SSE
=
\sum_{i=1}^{n}
(y_i - mx_i - b)^2
```

The resulting estimators are:

```math
m
=
\frac{
\sum (x_i-\bar{x})(y_i-\bar{y})
}{
\sum (x_i-\bar{x})^2
}
```

```math
b
=
\bar{y}
-
m\bar{x}
```

These values define the regression line that best explains the relationship between the feature and the target according to the least-squares criterion.