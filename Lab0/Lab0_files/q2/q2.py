import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA
    init_array = init_array - np.mean(init_array,axis=1)
    cov_mat=init_array.cov()
    sorted_eigenvalues, eigen_vectors = np.linalg.eig(cov_mat)
    idx=np.flip(np.argsort(sorted_eigenvalues))
    sorted_eigenvalues=np.round(sorted_eigenvalues[idx],4)
    print(sorted_eigenvalues)
    print(eigen_vectors)
    eigen_vectors=eigen_vectors[:,idx]
    eigen_vectors=eigen_vectors[:,range(dimensions)]
    final_data=np.matmul(init_array.to_numpy(),eigen_vectors)
    # END TODO
    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png
    plt.scatter(np.array(final_data[:,0]),np.array(final_data[:,1]))
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.savefig("out.png")
    # END TODO
