# Tensor Reshaping 
#"Tensor" is another name for ndarray
#Reshaping means changing the structure (dimensions= no. of axes) without changing the data.
import numpy as np

arr = np.arange(12) # creating 1D tensor = an array
print(arr)

arr.shape # shape is determined by no. of dimensions

arr2 = arr.reshape(3,4) # axes1 = row and axes2 = column,make sure that size of matrix(2D tensor) = size of arr(1D tensor)
print(arr2)

# use -1 in place of unknown dimension 
arr.reshape(3,-1) # no. of columns = (size)/ no. of rows

print(arr2.flatten()) #reverse operation for  reshape, 2D to 1D

arr3 =arr[:, np.newaxis] #adds column,takes all elements from 1D tensor and adds new dimension to it thereby producing a 2D tensor
print(arr3)

arr4 =arr[np.newaxis,:]#adds row 
print(arr4)
