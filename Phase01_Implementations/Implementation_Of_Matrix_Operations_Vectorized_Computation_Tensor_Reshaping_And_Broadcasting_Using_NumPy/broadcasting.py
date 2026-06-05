# BROADCASTING
"""
Broadcasting Alogorithm:-

Given two shapes:
1. Compare dimensions from the right.
2. Dimensions are compatible if:
     equal, or
     one is 1.
3. Missing dimensions are treated as 1.
4. Result dimension is the larger one.

But some operations can create very large intermediate broadcasting is ineffective.

"""
import numpy as np

A = np.array([1, 2, 3])

print(A + 10) #numpy treats 10(scalar or 0D tensor) as [10, 10, 10](arr or 1D tensor) and then performs addition
B = np.array([
    [1,2,3],
    [4,5,6]])
C = np.array([10,20,30])

print(B + C )
#on moving from rightmost dimension towards left if dimensions of both the arrays either same or one of them equal to 1 then numpy 
#then the arrays are broadcastable that is  the smaller array is “broadcast” across the larger array so that they have compatible shapes.
# here  C internally is converted to C= [[10 20 30],,[10 20 30]] and the addition occurs element wise
D =np.array([5,6,7,8]) #D.shape()=(4,) while B.shape=(2,3) i.e. no operation can be performed. 
print(C+D) #throws error due to incompatible dimensions, missing dimensions are treated as 1