# -*- coding: utf-8 -*-
"""numpy

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fkn3QX3N27Y2HWWEbwnGkXaXuZsxgsDn
"""

import numpy as np
#1 Creates a NumPy array from a Python list or tuple.
arr=np.array([1,2,3,4])
print(arr)
#2 Returns evenly spaced values within a given range.
arr=np.arange(0,10,2)
print(arr)
#3 Creates an array filled with zeros.
arr=np.zeros((3*3))
print(arr)
#4 Creates an array filled with ones.
arr=np.ones((2*2))
print(arr)
#5 Returns evenly spaced numbers over a specified range.
arr=np.linspace(0,1,5)
print(arr)
#6 Generates random numbers in a specified shape from a uniform distribution.
arr=np.random.rand(2*2)
print(arr)
#7
arr=np.array([1,2,3,4,5,6,7,8])
total=np.sum(arr)
print(total)
#8
mean_val=np.mean(arr)
print(mean_val)
#9
min_val=np.min(arr)
print(min_val)
#10
max_val=np.max(arr)
print(max_val)
#11
var_val=np.var(arr)
print(var_val)
#12
std_val=np.std(arr)
print(std_val)
#13
reshaped_arr = arr.reshape(2,4)
print(reshaped_arr)
#14
arr1=np.array([1,2,3,4])
arr2=np.array([2,3,4,5])
dot_product=np.dot(arr1,arr2)
print(dot_product)
#15
transpose_matrix=np.transpose(reshaped_arr)
print(transpose_matrix)
#16
median_val=np.median(arr)
print(median_val)