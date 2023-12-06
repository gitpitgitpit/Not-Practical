# -*- coding: utf-8 -*-
"""tensorflow

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11jBrzG86ulZpYoAggNxoJVyo3edbSr4L
"""

import numpy as np
tensor_1d=np.array([1,2,3,4,5])
print(tensor_1d)
print(tensor_1d[0])
print(tensor_1d[2])

tensor_2d=np.array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
print(tensor_2d)

import tensorflow as tf
import numpy as np
matrix1=np.array([(2,2,2),(2,2,2),(2,2,2)],dtype='int32')
matrix2=np.array([(1,1,1),(1,1,1),(1,1,1)],dtype='int32')
print(matrix1)
print(matrix2)

matrix1=tf.constant(matrix1)
matrix2=tf.constant(matrix2)
matrix_product=tf.matmul(matrix1,matrix2)
print("matrix Multiplication:",matrix_product)

matrix_sum=tf.add(matrix1,matrix2)
print("Sum:",matrix_sum)

matrix3=np.array([(1,2,3),(2,3,4),(4,5,6)],dtype='float32')
print(matrix3)

# Tensor Creation and Manipulation
tensor_constant = tf.constant([[1, 2], [3, 4]])
print(tensor_constant)

tensor_variable = tf.Variable([[5, 6], [7, 8]])
print(tensor_variable)

tensor_zeros = tf.zeros((2, 3))
print(tensor_zeros)

tensor_ones = tf.ones((3, 2))
print(tensor_ones)

import tensorflow as tf
import numpy as np

# Tensor Creation and Manipulation
tensor_constant = tf.constant([[1, 2], [3, 4]])
print("tensor_constant",tensor_constant)

tensor_variable = tf.Variable([[5, 6], [7, 8]])
print("tensor_variable ",tensor_variable )

tensor_zeros = tf.zeros((2, 3))
print("tensor_zeros",tensor_zeros)
tensor_ones = tf.ones((3, 2))
print("tensor_ones",tensor_ones)
tensor_fill = tf.fill((2, 2), 9)
print("tensor_fill",tensor_fill)

# Math Operations
tensor_add = tf.add(tensor_constant, tensor_variable)
print("Add:",tensor_add )
tensor_multiply = tf.multiply(tensor_constant, 2)
print("Product:",tensor_multiply )
tensor_matmul = tf.matmul(tensor_constant, tf.transpose(tensor_variable))
print("Multiplication:",tensor_matmul)

# Activation Functions
tensor_relu = tf.nn.relu(tensor_add)
print("Activation function:",tensor_relu)

# Loss Functions
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
print("Loass function:",loss_function)
# Optimizers
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
print("optimizer:",optimizer)
# Neural Network Layers
dense_layer = tf.keras.layers.Dense(units=10, activation='relu')