"""
Lab week 1: Question 1: Linear programs

Implement solvers to solve linear programs of the form:

max c^{T}x
subject to:
Ax <= b
x >= 0

(a) Firstly, implement simplex method covered in class from scratch to solve the LP

simplex reference:
https://www.youtube.com/watch?v=t0NkCDigq88
"""
import numpy
import pulp
import pandas as pd
import argparse


def parse_commandline_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDirectory', type=str, required=True, help='Directory of the test case files')
    arguments = parser.parse_args()
    return arguments


def simplex_solver(A_matrix: numpy.array, c: numpy.array, b: numpy.array) -> list:
    """
    Implement LP solver using simplex method.

    :param A_matrix: Matrix A from the standard form of LP
    :param c: Vector c from the standard form of LP
    :param b: Vector b from the standard form of LP
    :return: list of pivot values simplex method encountered in the same order
    """
    pivot_value_list = []
    ################################################################
    # %% Student Code Start
    # Implement here
    identity_matrix=numpy.eye(A_matrix.shape[0])
    augmented_matrix= numpy.c_[A_matrix,identity_matrix]
    augmented_matrix=numpy.c_[augmented_matrix, b]
    new_row=numpy.zeros(A_matrix.shape[0]+1)
    new_row=numpy.append(c*(-1),new_row).reshape((1,-1))
    augmented_matrix=numpy.append(augmented_matrix,new_row, axis=0)
    flag=False
    if not numpy.all(augmented_matrix[-1]>=0):
        flag=True
    while flag:
        flag=False
        ind1=numpy.argmin(augmented_matrix[-1][:-1])
        relative_freq=augmented_matrix[:,-1]/augmented_matrix[:,ind1]
        ind2=numpy.argmin(abs(relative_freq[:-1]))
        pivot=augmented_matrix[ind2][ind1]
        pivot_value_list.append(pivot)
        augmented_matrix[ind2]=augmented_matrix[ind2]/pivot
        for i in range(augmented_matrix.shape[0]):
            if i != ind2:
                augmented_matrix[i]=augmented_matrix[i]-augmented_matrix[i][ind1]*augmented_matrix[ind2]
        if not numpy.all(augmented_matrix[-1]>=0):
            flag=True

    # %% Student Code End
    ################################################################

    # Transfer your pivot values to pivot_value_list variable and return
    return pivot_value_list


if __name__ == "__main__":
    # get command line args
    args = parse_commandline_args()
    if args.testDirectory is None:
        raise ValueError("No file provided")
    # Read the inputs A, b, c and run solvers
    # There are 2 test cases provided to test your code, provide appropriate command line args to test different cases.
    matrix_A = pd.read_csv("{}/A.csv".format(args.testDirectory), header=None, dtype=float).to_numpy()
    vector_c = pd.read_csv("{}/c.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()
    vector_b = pd.read_csv("{}/b.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()

    simplex_pivot_values = simplex_solver(matrix_A, vector_c, vector_b)
    for val in simplex_pivot_values:
        print(val)
