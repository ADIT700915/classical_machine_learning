import numpy as np
#print(np.__version__)

a = np.arange(20) # create an array containing no.s 0 to 19
b= np.arange(0,20,4) # creates an array from 0 to 19 with gap size =4
c = np.arange(20,0,-1) # creates a reverse array from 20 to 1
print(a)
print(b)
print(c)

a= np.arange(0.1,0.3,0.12) # the no. of elements obatined is not predictable so linspacefuction is used where we specify the no. of elements we want
print (a)

b=np.linspace(2.0, 3.0, num=5) # total 5 no.s between 2.0 and 3.0

c=np.linspace(2.0, 3.0, num=5, endpoint=False)
# endpoint parameter is optional by default it is true but when set to false it excludes the stop value ultimately to get same no. of elements the step size adjusts .

d=np.linspace(2.0, 3.0, num=5, retstep=True)
# restep parameter is optional ,by default it is false but when set to true returns (samples, step size)
print(d)
 
rg = np.random.default_rng(1)#creates a random no. generator object
#here 1 is called the "seed",a seed determines the starting state of the random number generator,
#Using the same seed always produces the same sequence of "random" numbers

A = rg.random((2,3)) # creates a 2 row 3 column matrix of random values from 0 to 1
print(a)

for x in np.nditer(a): #traverses through each elemnt in matrix a
    print(x)
    
B= A[:,1]
print(B)
B[0] =5
print(A) #slicing creates a view due to which changes done in sliced arr reflect in original array
#to make a copy use b= a[:,1].copy()

B = np.exp(A * 1j) # each element multiplied by 1j then converted to complex by Eulers formula.
print (B)

print(B.dtype.name)



