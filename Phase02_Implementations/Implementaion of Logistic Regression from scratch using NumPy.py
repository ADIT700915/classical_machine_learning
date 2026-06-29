import numpy as np
import matplotlib.pyplot as plt

# Feature matrix (Height, Weight)
X = np.array([
    [170, 65],
    [180, 80],
    [150, 45],
    [160, 50]
])

# Target labels (Class)
y = np.array([
    [1],
    [1],
    [0],
    [0]
])

losses = []

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# X -> (m,n) here, m=4 and n =2
# y -> (m,1)

m, n = X.shape

w = np.zeros((n,1))
b = 0 

''' weights are initialed to 0 because logistic reg has only one layer, so each 
 weight receives a different gradient from different input features i.e. there is no 
symmetry problem '''

learning_rate = 0.0001 # Learning rate earlier = 0.01 ,0.001 
epochs = 1000

for epoch in range(epochs):

    z = np.dot(X, w) + b
    y_hat = sigmoid(z)

    # Numerical stability
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

    '''
    * epsilon = 1e-15: Defines a tiny lower bound for probabilities.

    * np.clip(y_hat, epsilon, 1-epsilon): restricts all values in y_hat to
     [1e-15, 1 - 1e-15 ],value > (1-1e-15) =(1-1e-15) and value <1e-15 = 1e-15 
     This prevents taking the logarithm of exactly 0 or 1
     which would then produce not defined value.

    '''

    # Loss
    loss = -(1/m) * np.sum(
        y*np.log(y_hat) +
        (1-y)*np.log(1-y_hat)
    )
    losses.append(loss)

    # Gradients
    dw = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)

    # Update
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.4f}")

# Predictions
probabilities = sigmoid(np.dot(X, w) + b)
predictions = (probabilities >= 0.5).astype(int)

'''
* (proba > 0.5): This compares the model's output probabilities
 (values between 0 and 1) against a threshold of 0.5.  It returns a 
 boolean array (e.g., [True, False, True]) where True indicates the 
 probability exceeds the threshold. 

* .astype(int): This is a NumPy method (not a Python built-in function) 
that converts the boolean array into an integer array.  It transforms 
True to 1 and False to 0, creating the final binary class label

*astype() is used for arrays (NumPy/Pandas), whereas int() is a Python 
built-in that only works on scalar values. In simpler terms, former is used for 
batch conversion allowing us to change the dtype of entire np array/ pd series/ df
because it is built into these libraries while the latter converts an individual element. 

'''

accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)

# The Learning Curve (Loss vs Epochs)
# to improve learning rate
plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")

plt.grid(True)

plt.show()