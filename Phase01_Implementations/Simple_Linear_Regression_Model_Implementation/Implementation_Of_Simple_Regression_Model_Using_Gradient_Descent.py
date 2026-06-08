import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X=np.random.uniform(-3,3,size=(100,1))
slope =35
intercept =10
noise =np.random.normal(0,15,size=(100,1))
y= (slope*X) + intercept + noise

def train_test_split_numpy(X, y, test_size=0.2, random_state=None):
    """
    Split data into train and test sets using only NumPy.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get total number of samples
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test


class GradientDescentRegressor:
    def __init__(self,learning_rate,epochs):
        self.m=1 #arbitrary values 
        self.b= 0
        self.lr=learning_rate
        self.epochs=epochs
        self.loss_history=[]

    def fit(self,X,y):
        for i in range(self.epochs):
            errors =y.ravel() - self.m*X.ravel() - self.b
        #MSE
            mse_loss =np.mean(errors**2)
            self.loss_history.append(mse_loss)    
        #Gradients
            loss_slope_b = -2*np.sum(errors)
            loss_slope_m = -2*np.sum(errors*X.ravel())
        #Update
            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)
          # print(loss_slope_b," and ",self.b)
           # print(loss_slope_m," and ",self.m)

        print(self.m," ",self.b)

    def predict(self,X):
        return(self.m*X + self.b) 
       
X_train,X_test,y_train,y_test = train_test_split_numpy(X,y,test_size=0.2,random_state=42)

print("Train shape:",X_train.shape,y_train.shape)
print("Test shape:",X_test.shape,y_test.shape)

gd=GradientDescentRegressor(0.001,100)
# the values at lr=0.001 and epochs =100 are m: 33.850566932192415 and b: 9.778143159778477
# the values at lr=0.005 and epochs =100 are m: -5.670120112056249e+74 and b: 4.609783378215123e+73

gd.fit(X,y) 
y_pred = gd.predict(X_test)

mse = np.mean((y_test - y_pred)**2)
print(f"Test MSE: {mse}")

#Plot Epoch vs Loss
plt.figure(figsize=(8,5))
plt.plot(range(gd.epochs),gd.loss_history,color="red",linewidth=2)
plt.title("Epoch vs Mean Squared Error(Loss)")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.grid(True)
#plt.savefig("loss_chrt.png")
plt.show()

#Best Fit Line Graph
plt.scatter(X, y, color="blue", label="Actual Data")
y_final_pred = gd.m * X + gd.b
plt.plot(X, y_final_pred, color="orange", label="Regression Line", linewidth=3)
plt.title("Final Line Fit")
plt.legend()
plt.show()