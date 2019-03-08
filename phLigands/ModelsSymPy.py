import numpy as np
from scipy.optimize import fsolve
from sympy.solvers import solve
from sympy import solve_poly_system
from sympy import Symbol

from phLigands.Model import Model

class test(Model):
    def setDefParams(self):
        defparams = dict()
        defparams["A_b"] = {"min" : 150, "def" : 191.8, "max" : 210}
        defparams["k"] = {"min" : 0.0001, "def" : 0.02, "max" : 1}
        defparams["n"] = {"min" : 0.001, "def" : 3.6, "max" : 7}
        return defparams
    
    def function(self, x, A_b1, A_b2, N, k, c=10):
#         A = Symbol('A')
#         c_b1 = Symbol('c_b1')
#         c_b2 = Symbol('c_b2')
#         if A_b1 == 0 or A_b2 == 0 or n == 0 or k == 0:
#             return np.nan
        def func(A):
            opt = A - c_b1 / c * A_b1 - c_b2 / c * A_b2
            feq = c_b1 - k * (c - c_b1 - c_b2) * (2 * N - c_b1) * (1 - c_b2/N)
            seq = c_b2 - k * (c - c_b1 - c_b2) * (N - c_b2) * (1 - c_b2/(2 * N)) ** 2
            return opt, feq, seq
        return fsolve(func, A)
    
    def derivative(self, x, A_b, n, k):
        if A_b == 0 or n == 0 or k == 0:
            return np.nan
        tmp = 1 / (k * A_b) + n + x
        return A_b * (1 - (2 * tmp - 4 * n) / (2 * np.sqrt(tmp**2 - 4 * n * x)))/ (2 * n)
