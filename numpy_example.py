"""
https://medium.com/@zachary.bedell/writing-beautiful-code-with-numpy-505f3b353174
"""
import functools
import numpy as np
import random as r
import time


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))

    return newfunc


""" Python lists """
my_list = ["One", "Two", 3]

# individual indexing
print(my_list[0])

# iteration
for el in enumerate(my_list):
    print(el, ', ')

for el in enumerate(my_list):
    el *= 2  # gets expensive once we start working with large data sets

""" NumPy """

# create a numpy array from original python list
my_int_list = range(100000)
my_numpy_arr = np.array(my_int_list)


# multiply each element by 2
@timeit
def numpy_mult():
    my_numpy_arr * 2


@timeit
def py_mult():
    for _el in enumerate(my_list):
        _el *= 2  # gets expensive once we start working with large data sets


numpy_mult()
py_mult()


""" Multiply Matrices """
@timeit
def py_matrix_mult():
    # create a 300x200 matrix  of 60,000 random integers
    my_list_1 = []
    for row_index in range(300):
        new_row = []
        for col_index in range(200):
            new_row.append(r.randint(0, 20))
        my_list_1.append(new_row)

    # create a 200x400 matrix of 80,000 random integers
    my_list_2 = []
    for row_index in range(200):
        new_row = []
        for col_index in range(400):
            new_row.append(r.randint(0, 20))
        my_list_2.append(new_row)

    # create a 300x400 array to hold results
    my_results_arr = []
    for row_index in range(300):
        new_row = []
        for col_index in range(400):
            new_row.append(0)
        my_results_arr.append(new_row)

    # iterate through row of my_list1_1
    for i in range(len(my_list_1)):
        # iterate through columns of my_list_2
        for j in range(len(my_list_2[0])):
            # iterate through rows of my_list_2
            for k in range(len(my_list_2)):
                my_results_arr[i][j] += my_list_1[i][k] * my_list_2[k][j]

@timeit
def numpy_matrix_mult():
    np_arr_1 = np.arange(0, 60000).reshape(300, 200)
    np_arr_2 = np.arange(0, 80000).reshape(200, 400)
    my_result_arr = np.matmul(np_arr_1, np_arr_2)


py_matrix_mult()
numpy_matrix_mult()
