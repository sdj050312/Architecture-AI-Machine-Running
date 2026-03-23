import numpy as np   
import matplotlib.pylot as plt 

def kmeans_with_random_init(X, k, max_iters=200):

    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()

    centroids = np.zeros((k, 2))