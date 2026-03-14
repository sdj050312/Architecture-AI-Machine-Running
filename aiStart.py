import tensorflow as tf
import numpy as np  
import random

print("TensorFlow version:", tf.__version__)

a = tf.constant(2)
b= tf.constant(3)
c = a + b 
print("a + b =", c.numpy())