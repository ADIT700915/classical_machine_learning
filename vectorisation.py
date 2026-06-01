# Vectorized Computation
# It refers to performing operations on entire arrays (vectors, matrices, tensors) at once instead of using explicit Python loops.
import numpy as np
a = np.array([1,2,3,4]) 

# traditional loop method
result1 = []

for x in a:
    result1.append(int(x) * 2)

print(result1)

#vectorised method
result2 = a * 2 #much faster and much shorter
print(result2)

#Why is Vectorization Faster?
"""

Python loops have overhead.
When Python executes:
for x in a:
it repeatedly:
1)Reads next element
2)Creates Python objects
3)Executes Python bytecode
4)Stores results millions of times.
NumPy does:
a * 2
using highly optimized C code.The loop still exists—but inside compiled code, not Python 

"""
# Vectorised Computation and Broadcasting
A = np.array([
    [1,2,3],
    [4,5,6]]) # A.shape()=(2,3)
B = np.array([10,20,30]) #B.shape()=(3,)
print(A+B) #vectorised addition

"""
Two concepts work together:
1) Vectorisation : addition performed on entire arrays
2) Broadcasting : smaller array automatically expanded

In simpler words vectorisation means:
Give the entire array to NumPy ,it performs operation in optimised C code and returns the result

"""
# Some Important Vectorised Functions

# Exponential Functions

print(np.exp(a)) 
print(np.exp2(a))

# Logarithmic Functions

print(np.log(a))
print(np.log2(a))
print(np.log10(a))
print(np.log1p(a)) # equivalent to np.log(1 + a)

# Root Functions

print(np.sqrt(a))
print(np.cbrt(a))

# Trigonometric Functions

print("\nSin:")
print(np.sin(a))

print("\nCos:")
print(np.cos(a))

print("\nTan:")
print(np.tan(a))

print("\nArcSin:")
print(np.arcsin(np.sin(a)))

print("\nArcCos:")
print(np.arccos(np.cos(a)))

print("\nArcTan:")
print(np.arctan(np.tan(a)))

print("\nSinh:")
print(np.sinh(a))

print("\nCosh:")
print(np.cosh(a))

print("\nTanh:")
print(np.tanh(a))

# Statistics
print(np.sum(a))
print(np.mean(a))
print(np.min(a))
print(np.max(a))
print(np.std(a))
print(np.var(a))

#
print(a.sum())# sum of all elements
print(a.sum(axis=0))# column wise sum
print(a.sum(axis=1)) #row wise sum 
print(a.min())
print(a.max())
print(a.dtype.name)
print(a.cumsum(axis=1))#cumulative sum along each row

# Linear Algebra

# "linalg" stands for Linear Algebra which is a module containg functions such as discussed below:
print(a.T) #creates transpose of matrix i.e. interchnges row and column elements
print(np.linalg.inv(a)) #calculates inverse of matrix a ;
print(np.linalg.det(a))# determinant of a

# Vectorised Matrix Operations {matrix is a 2D tensor}

A = np.array([[2, 3],
              [0, 6]])
B = np.array([[7, 0],
              [8, 2]])

# Matrix Multiplication

print(A * B)     # elementwise product
print(A @ B)     # matrix product
print(A.dot(B))  # another matrix product 

# Matrix Addition

print(A+B) # if different data types are present type promotion happens i.e. float+int =float

#Both A+B nad np.add(A,B) produce the same output but the reason np.add() exists is to provide extra capabilities
#(out, where, reduce, etc.)

#output is stored in result ,no extra temporary matrix is created as in case of A+B; this matters when matrices are huge.
result =np.empty((2,2))
print(np.add(A, B, out=result)) 

#performs addition only at indices where mask is true,this kind of selective computation isn't available through plain A + B
mask = [[True, False],
        [False, True]]
print(np.add(A, B, where=mask)) 

# Matrix Substraction
print(A-B)

# Matrix Division
print(A/B)

# Transpose
print(A.T)


