#***LINEAR REGRESSION :-***

Linear Regression is one of the simplest machine learning algorithms. Its goal is to find the relationship between input variables( features) and output variables( target ).

It does so by plotting all the data points on a graph and drawing the best fit line through all the points.

When training, the model learns   
1) slope (m)  (determines upto what extent does target depend upon feature value)
 2) intercept (b) (act as an offset i.e.ensures that target can be obtained even when feature is set to 0)     

These two values define the equation of the best fit line as y = mx + b.

For Simple Linear Regression there are several ways to calculate the value of m and b:-

1) Closed-Form Solution  
It is also called OLS (Ordinary Least Squares) method.   It uses direct formula to find slope and intercept values to predict the target value.


**Derivation of OLS Estimators for Linear Regression**

Step 1: Define the Linear Regression Model

We assume a linear relationship between the input variable x and output variable y:

$$
\hat{y}_i = mx_i + b
$$

where:

- m = slope
- b = intercept
- $\hat{y}_i $ = predicted value

The prediction error (residual) for the $i^{th}$ observation is:

$$
e_i = y_i - \hat{y}_i
$$

Substituting the regression equation:

$$
e_i = y_i - (mx_i + b)
$$

---

Step 2: Define the Cost Function

The Ordinary Least Squares (OLS) method minimizes the Sum of Squared Errors (SSE):

$$
SSE = \sum_{i=1}^{n} e_i^2
$$

Substituting the residual expression:

$$
SSE = \sum_{i=1}^{n} (y_i - mx_i - b)^2
$$

---

Step 3: Differentiate SSE with Respect to b

To find the minimum value of SSE, set its partial derivatives equal to zero.

$$
\frac{\partial SSE}{\partial b}

\frac{\partial}{\partial b}
\sum_{i=1}^{n}(y_i-mx_i-b)^2
$$

Applying the chain rule:

$$
\frac{\partial SSE}{\partial b}

-2\sum_{i=1}^{n}(y_i-mx_i-b)
$$

Setting the derivative equal to zero:

$$
-2\sum_{i=1}^{n}(y_i-mx_i-b)=0
$$

$$
\sum_{i=1}^{n}(y_i-mx_i-b)=0
$$

Expanding:

$$
\sum y_i - m\sum x_i - nb = 0
$$

Solving for b:

$$
b = \frac{\sum y_i - m\sum x_i}{n}
$$

Using:

$$
\bar{x} = \frac{\sum x_i}{n}
\qquad
\bar{y} = \frac{\sum y_i}{n}
$$

we obtain:

$$
b = \bar{y} - m\bar{x}
$$

---

Step 4: Differentiate SSE with Respect to m

Starting from:

$$
SSE = \sum_{i=1}^{n}(y_i-mx_i-b)^2
$$

Differentiate with respect to m:

$$
\frac{\partial SSE}{\partial m}

-2\sum_{i=1}^{n}x_i(y_i-mx_i-b)
$$

Setting the derivative equal to zero:

$$
\sum_{i=1}^{n}x_i(y_i-mx_i-b)=0
$$

Expanding:

$$
\sum x_i y_i

m\sum x_i^2

b\sum x_i

0
$$

Substitute:

$$
b=\bar{y}-m\bar{x}
$$

into the equation:

$$
\sum x_i y_i

m\sum x_i^2

(\bar{y}-m\bar{x})\sum x_i

0
$$

Since:

$$
\sum x_i = n\bar{x}
$$

we get:

$$
\sum x_i y_i

m\sum x_i^2

n\bar{x}\bar{y}
+
mn\bar{x}^2

0
$$

Grouping terms containing m:

$$
m
\left(
\sum x_i^2 - n\bar{x}^2
\right)

\sum x_i y_i - n\bar{x}\bar{y}
$$

Therefore:

$$
m

\frac
{
\sum x_i y_i - n\bar{x}\bar{y}
}
{
\sum x_i^2 - n\bar{x}^2
}
$$

---

Step 5: Convert to Covariance-Variance Form

Using the identities:

$$
\sum (x_i-\bar{x})(y_i-\bar{y})

\sum x_i y_i

n\bar{x}\bar{y}
$$

and

$$
\sum (x_i-\bar{x})^2

\sum x_i^2

n\bar{x}^2
$$

the slope becomes:

$$
m

\frac
{
\sum (x_i-\bar{x})(y_i-\bar{y})
}
{
\sum (x_i-\bar{x})^2
}
$$

---

Final OLS Estimators

Slope 

$$
m =

\frac
{
\sum (x_i-\bar{x})(y_i-\bar{y})
}
{
\sum (x_i-\bar{x})^2
}
$$

Intercept

$$
b = 

\bar{y}  - 

m\bar{x}
$$

These values of m and b minimize the Sum of Squared Errors and define the best-fit regression line according to the Ordinary Least Squares criterion. 

