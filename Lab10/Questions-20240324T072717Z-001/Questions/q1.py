import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) np array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    # TODO
    X = X.reshape(X.shape[0], -1)
    X= X- np.mean(X,axis = 0)
    covariance_matrix = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    sorted_eigen_vectors = eigen_vectors[:, np.argsort(eigen_values)[::-1]]
    principle_comp = sorted_eigen_vectors[:,:k]
    return principle_comp/np.linalg.norm(principle_comp, axis = 0)
    #END TODO
    

def projection(X: np.array, basis: np.array):
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (n,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    # TODO
    X = X.reshape(X.shape[0], -1)
    return X@basis

    # END TODO
