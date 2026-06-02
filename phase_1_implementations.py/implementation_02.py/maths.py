
import numpy as np
import math

# PROBABILITY OF AN EVENT
#It is a measure of how likely an event is to occur.
#mathematically, it is defined as the ratio of no. of favorable outcomes to total no. of outcomes.
def probability(favorable, total):
    return favorable / total

print(probability(1, 6))

# CONDITIONAL PROBABILITY
# P(B∣A)=P(A∩B) / P(A) ;that is,probability of occurence of an event B provided that event has 
# occured is equal to prob.of occurence of both the events divided by prob of occ. of A only

def conditional_probability(intersection, event_a):# intersection parameter stores prob. of occ. of both 
    return intersection / event_a


print(conditional_probability(0.2, 0.5))

# MEAN : mean is the sum of all values divided by the total no. of values
def mean(arr):
    return sum(arr) / len(arr)

# VARIANCE

def variance(arr):
    mu = mean(arr)

    total = 0

    for x in arr:
        total += (x - mu) ** 2

    return total / len(arr)

# SATNDARD DEVIATION
#determines how values vary around mean
def std(arr):
    return math.sqrt(variance(arr))

# COVARIANCE
#tells how, two random variables vary with each other; can be +ve,-ve or 0
#+ve cov means that the variation around mean is same for  both the vars. i.e. both increase or decrease around 
#their respective mean values ,if the two random vars.are independent then cov is 0 however, the converse is not true
def covariance(x, y):

    mean_x = mean(x)
    mean_y = mean(y)

    total = 0

    for i in range(len(x)):
        total += (x[i] - mean_x) * (y[i] - mean_y)

    return total / len(x)

# CORRELATION
# correlation is used to express the extent to which two variable say, x and y are lineraly dependent
# scale doesnot affect correlation 
def correlation(x, y):

    cov = covariance(x, y)

    return cov / (std(x) * std(y))


data = [10, 20, 30, 40, 50]

print(mean(data))
print(variance(data))
print(std(data))

#for covariance :
x = [1,2,3,4,5]
y = [2,4,6,8,10]

print(covariance(x,y))

print(correlation(x,y))
