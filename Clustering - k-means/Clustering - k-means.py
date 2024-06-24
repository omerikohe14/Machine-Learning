import numpy as np
numOfIterations = 0

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    indices = np.random.choice(X.shape[0], size=k , replace=False)
    centroids = X[indices]
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float64) 

def calcDistance(X, centroid, p):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    A single centroid (1, 3)
    The distance parameter p

    output: numpy array of shape `(num_pixels, 1)` thats holds the distances of 
    all points in RGB space from the centroid
    '''
    distances = []
    distances = np.absolute(X - centroid)
    distances = distances ** p
    sigma = np.sum(distances, axis=1)
    distances = sigma ** (1/p)
    return distances

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    for i in range(k):
        distances.append(np.array(calcDistance(X, centroids[i], p )))
    return np.array(distances)

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    global numOfIterations 
    numOfIterations = 0
    centroids = get_random_centroids(X, k)
    return optimize_Centroids(X, centroids, p, max_iter)

def optimize_Centroids(X, centroids, p , max_iter=100):
    global numOfIterations 
    for _ in range(max_iter):
        numOfIterations += 1
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        newCentoirds = [(np.mean(X[classes == j], axis=0)) for j in range(len(centroids))]
        if np.array_equal(centroids, newCentoirds):
            break
        centroids = np.array(newCentoirds)    
    return centroids, classes , 

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    global numOfIterations 
    numOfIterations = 0
    centroid_index = np.random.choice(X.shape[0], 1)
    centroids = X[centroid_index]
    for _ in range(k-1):
        distances = lp_distance(X, centroids, p)
        distances = np.min(distances, axis=0)
        distances = distances ** 2
        distances = distances / np.sum(distances)
        centroids = np.vstack((centroids, X[np.random.choice(X.shape[0], p=distances)]))
    return optimize_Centroids(X, centroids, p , max_iter)

def inertia(X, centroids, p):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - centroids: a numpy array of shape (k, 3) containing the centroid values.
    - p: the parameter governing the distance measure.
    
    Outputs:
    - The value of the inertia criterion for the given data and centroids.
    """
    distances = lp_distance(X, centroids, p)
    distances = np.min(distances, axis=0)
    distances = distances ** 2
    return np.sum(distances)

def kMaens_kMeansPP_Comparison(X):
    global numOfIterations
    kMeans = []
    kMeans_pp = []
    kMeans_Inertia = []
    kMeans_pp_Inertia = []
    kMeans_NumOfIterations = []
    kMeans_pp_NumOfIterations = []
    for k in range(1, 14):
        # Apply regular k-means
        kMeans.append(kmeans(X, k, 1, max_iter=100)[0])
        kMeans_NumOfIterations.append(numOfIterations)
        # Apply k-means++
        kMeans_pp.append(kmeans_pp(X, k, 1, max_iter=100)[0])
        kMeans_pp_NumOfIterations.append(numOfIterations)
        kMeans_Inertia.append(np.mean([inertia(X , kMeans[i] , 2) for i in range(len(kMeans))]))
        kMeans_pp_Inertia.append(np.mean([inertia(X , kMeans_pp[i] , 2) for i in range(len(kMeans_pp))]))
        print("k = " + str(k) + " is done")
        kMeans = []
        kMeans_pp = []
    return kMeans_Inertia, kMeans_pp_Inertia , kMeans_NumOfIterations, kMeans_pp_NumOfIterations
