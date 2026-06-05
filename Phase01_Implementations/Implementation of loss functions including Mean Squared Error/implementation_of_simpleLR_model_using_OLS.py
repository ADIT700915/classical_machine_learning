class linear_regression:
    def __init__(self):
        self.m = None
        self.b = None
    def fit(self,X_train,y_train):

        num =0
        den =0
        for i in range (X_train.shape[0]):

            num = num + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))
        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)
        
    def predict(self,X_test):

        print(X_test)

        return self.m * X_test + self.b
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r"C:\Users\Asmita\Downloads\Study Hour(Linear Regression).csv")
print(df.head())

X = df.iloc[:,0].values
y = df.iloc[:,1].values
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=2)

print(X_train.shape)

print(X_train[0])

print(X_train.mean())

print(X_test[0])

lr= linear_regression()
lr.fit(X_train,y_train)



print(lr.predict(X_test[0]))

plt.scatter(df['Hours'],df['Scores'])
plt.plot(X_test,lr.predict(X_test),color='red')
plt.xlabel('HOURS')
plt.ylabel('SCORES (OUT OF 100)')
plt.text(3,-8,'SCORES (OUT OF 100) vs HOURS OF STUDY')
plt.show()

