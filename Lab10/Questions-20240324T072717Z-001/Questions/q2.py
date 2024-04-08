import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        # TODO

        # self.linear_discriminants=np.zeros((len(X[0]*len(X[0])),self.n_components))
        unique_labels = np.unique(y)
        num_features = X.shape[1]*X.shape[2]
        X = X.reshape(X.shape[0], -1)
        Sw = np.zeros((X.shape[1], X.shape[1]))
        Sb = np.zeros((X.shape[1], X.shape[1]))
        mean_total = np.mean(X, axis=0)
        for label in unique_labels:
            X_class = X[y == label]
            mean_class = np.mean(X_class, axis=0)
            Sw += (X_class - mean_class).T.dot((X_class - mean_class))
            mean_difference = (mean_class - mean_total).reshape(-1, 1)
            Sb += X_class.shape[0] * mean_difference.dot(mean_difference.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))

        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eig_pairs = sorted(eig_pairs, key=lambda x: x[0], reverse=True)

        self.linear_discriminants = np.hstack([eig_pairs[i][1].reshape(num_features, 1) for i in range(0, self.n_components)])
        return self.linear_discriminants     
        #END TODO 
    
    def transform(self, X, w):
        """
        w:Linear Discriminant array of size (d*d,1)
        return: np-array of the projected features of size (n,k)
        """
        # TODO
        X = X.reshape(X.shape[0], -1)
        projected = X @ w
        return projected                  
        # END TODO
