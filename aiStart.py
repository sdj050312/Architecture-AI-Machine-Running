import tensorflow as tf
import numpy as np  
import random
import matplotlib.pyplot as plt

X_train = np.array([[1, 2], [2, 1], [4, 5], [5, 4], [2, 2], [4, 4], [1, 1], [5, 5]])
print("TensorFlow version:", tf.__version__)

a = tf.constant(2)
b= tf.constant(3)
c = a + b 
print("a + b =", c.numpy())