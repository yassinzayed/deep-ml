import numpy as np

def cross_product(a, b):
    elem_1 = a[1]*b[2]-a[2]*b[1]
    elem_2 = a[2]*b[0]-a[0]*b[2]
    elem_3 = a[0]*b[1]-a[1]*b[0]
    return [elem_1, elem_2, elem_3]