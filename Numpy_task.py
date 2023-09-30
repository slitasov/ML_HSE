#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np


# ## Задание 1

# In[18]:


def result_mark(weights: np.array, marks: np.array) -> int:
    result = np.dot(weights, marks)
    return round(result)


# In[34]:


weights = np.array([0.3, 0.4, 0.2, 0.1])
marks = np.array([7, 10, 8, 6])

assert result_mark(weights, marks) == 8


# In[35]:


weights = np.array([0.3, 0.4, 0.2, 0.1])
marks = np.array([7, 0, 8, 6])

assert result_mark(weights, marks) == 4


# ## Задание 2

# In[22]:


def change_array(array: np.array, number: int) -> np.array:
    array[::3] = number
    return array


# In[30]:


array = np.array([3, 5, 1, 0, -3, 22, 213436])
number = -111

assert np.allclose(change_array(array, number), np.array([-111, 5, 1, -111, -3, 22, -111]))


# In[31]:


array = np.array([3, 14, 15, 92, 6])
number = 8

assert np.allclose(change_array(array, number), np.array([8, 14, 15, 8, 6]))


# ## Задание 3

# In[87]:


def find_close(array1: np.array, array2: np.array, precision: float) -> np.array:
    result = np.where(abs(array1-array2) < precision)[0]
    return result


# In[91]:


array1 = np.array([1.5, 0.5, 2, -4.1, -3, 6, -1])
array2 = np.array([1.2, 0.5, 1, -4.0,  3, 0, -1.2])
precision = 0.5
res = find_close(array1, array2, precision)

assert res.ndim == 1
assert np.allclose(res, np.array([0, 1, 3, 6]))


# In[92]:


array1 = np.array([3.1415, 2.7182, 1.6180, 6.6261])
array2 = np.array([6.6730, 1.3807, -1,     6.0222])
precision = 1.7
res = find_close(array1, array2, precision)

assert res.ndim == 1
assert np.allclose(res, np.array([1, 3]))


# ## Задание 4

# In[95]:


def block_matrix(block: np.array) -> np.array:
    pre_final = np.hstack((block, block))
    final = np.vstack((pre_final, pre_final))
    return final


# In[96]:


block = np.array([[1, 3, 3], [7, 0, 0]])

assert np.allclose(
    block_matrix(block),
    np.array([[1, 3, 3, 1, 3, 3],
              [7, 0, 0, 7, 0, 0],
              [1, 3, 3, 1, 3, 3],
              [7, 0, 0, 7, 0, 0]])
)


# ## Задания 5

# In[132]:


def diag_prod(matrix: np.array) -> int:
    result = np.prod(np.diag(matrix)[np.diag(matrix) !=0])
    return result


# In[133]:


matrix = np.array([[0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11],
                   [12, 13, 14, 15]])
diag_prod(matrix)


# ## Задание 6

# In[198]:


def normalize(matrix: np.array) -> np.array:
    import math
    m_e = np.mean(matrix, axis = 0)
    sd_ot = np.std(matrix, axis = 0)
    sd_ot[sd_ot == 0 ] = 1
    result = (matrix - m_e)/(sd_ot)
    return result


# In[201]:


matrix = np.array([[1, 4, 4200], [0, 10, 5000], [1, 2, 1000]])

assert np.allclose(
    normalize(matrix),
    np.array([[ 0.7071, -0.39223,  0.46291],
              [-1.4142,  1.37281,  0.92582],
              [ 0.7071, -0.98058, -1.38873]])
)


# In[196]:


matrix = np.array([[-7, 2, 42], [2, 10, 50], [5, 4, 10]])

assert np.allclose(
    normalize(matrix),
    np.array([[-1.37281, -0.98058,  0.46291],
              [ 0.39223,  1.37281,  0.92582],
              [ 0.98058, -0.39223, -1.38873]])
)


# ## Задание 7

# In[310]:


def prevZeroMax(matrix: np.array) -> int:
    ind = np.where(matrix == 0)[0]
    ind1 = ind[np.where(ind !=(len(matrix) -1))[0]] + 1
    return max(np.take(matrix, ind1))


# In[312]:


coefs = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])

assert  prevZeroMax(coefs) == 5


# In[313]:


coefs = np.array([1, 0, 1, 0, 4, 2, 0])

assert prevZeroMax(coefs) == 4


# ## Задание 8 !

# In[412]:


def make_symmetric(matrix: np.array) -> np.array:
    answer = matrix + np.triu(matrix, k = 1).T
    return answer


# In[414]:


matrix = np.array([[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 8, 9], [0, 0, 0, 10]])

assert np.allclose(
    make_symmetric(matrix),
    np.array([[ 1,  2,  3,  4],
              [ 2,  5,  6,  7],
              [ 3,  6,  8,  9],
              [ 4,  7,  9, 10]])
)


# In[415]:


matrix = np.array([[10, 21, 32, 49], [0, 53, 62, 78], [0, 0, 82, 92], [0, 0, 0, 10]])

assert np.allclose(
    make_symmetric(matrix),
    np.array([[10, 21, 32, 49],
              [21, 53, 62, 78],
              [32, 62, 82, 92],
              [49, 78, 92, 10]])
)


# ## Задание 9

# In[391]:


def construct_matrix(m: int, a: int, b: int) -> np.array:
    array = np.arange(a,b+1).reshape(1,-1).repeat(m, axis = 0)
    return array


# In[393]:


m = 5
a = 3
b = 10

assert np.allclose(
    construct_matrix(m, a, b),
    np.array([[ 3,  4,  5,  6,  7,  8,  9, 10],
              [ 3,  4,  5,  6,  7,  8,  9, 10],
              [ 3,  4,  5,  6,  7,  8,  9, 10],
              [ 3,  4,  5,  6,  7,  8,  9, 10],
              [ 3,  4,  5,  6,  7,  8,  9, 10]])
)


# In[394]:


m = 3
a = 2
b = 6

assert np.allclose(
    construct_matrix(m, a, b),
    np.array([[2, 3, 4, 5, 6],
              [2, 3, 4, 5, 6],
              [2, 3, 4, 5, 6]])
)


# ## Задание 10

# In[397]:


def cosine_similarity(vec1: np.array, vec2: np.array) -> float:
    skalar = np.dot(vec1, vec2)
    eukl = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    answer = skalar/eukl
    return answer


# In[398]:


vec1 = np.array([-2, 1, 0, -5, 4, 3, -3])
vec2 = np.array([0, 2, -2, 10, 6, 0, 0])
cosine_similarity(vec1, vec2)

