import numpy as np
from scipy.optimize import root_scalar


class Model:    
    def _function(self, x, k, b):
        return x * k + b
    
    def _derivative(self, x, k, b):
        return k
        
    def _eval_function(self, function, params, dataX, **kwargs):
        ### Fix this isinstance
        if isinstance(dataX, np.float):
            return function(dataX, **params)
        dataX = np.asarray(dataX)
        dataY = np.empty(dataX.shape[0])
        dataY[:] = np.nan
        for i, pointX in np.ndenumerate(dataX):
            a = function(pointX, **params)
            dataY[i] = a
        return dataY   
    
    def eval(self, params, dataX, **kwargs):
        return self._eval_function(self._function, params, dataX)
    
    def diff(self, params, dataX, **kwargs):
        return self._eval_function(self._derivative, params, dataX)

    
class LinReg(Model):
    pass


class ScatchardN(Model):    
    def _function(self, x, Ab, n, k):
        if Ab == 0 or n == 0 or k == 0:
            return np.nan
        tmp = 1 / (k * Ab) + n + x
        return Ab * (tmp - np.sqrt(tmp**2 - 4 * n * x)) / (2 * n)
    
    def _derivative(self, x, Ab, n, k):
        if Ab == 0 or n == 0 or k == 0:
            return np.nan
        tmp = 1 / (k * Ab) + n + x
        return Ab * (1 - (2 * tmp - 4 * n) / (2 * np.sqrt(tmp**2 - 4 * n * x)))/ (2 * n)
    
    
class MGvH(Model):    
    def _function(self, PD, c, Af, Ab, k, n):
        N = c * PD
        mgvh = lambda cb: cb - k * (c - cb) * (N - n * cb) * ((N - n * cb)/(N - (n-1) * cb))**(n-1)
        cb = root_scalar(mgvh, bracket=(0, min([c, N/n])), method='brentq').root
        return (c - cb)/c * Af + cb/c * Ab
    
    def _derivative(self, PD, c, Af, Ab, k, n):
        step = 0.001
        return (self._function(PD + step/2, c, Af, Ab, k, n) - self._function(PD - step / 2, c, Af, Ab, k, n)) / step
    
## I stole it
class MGvH1900(Model):    
    def _function(self, PD, c, Af, Ab, k, n):
        precision = 1e-4
        N = c * PD
        mgvh = lambda cb: k*(c-cb)*(N-cb*n)*((N-cb*n)/(N-cb*(n-1)))**(n-1)-cb;
        if (n<=0)or(k<=0): return 0
        if n<=1:
            return 0.5*(1/k+c+N/n-np.sqrt((1/k+c+N/n)**2-4*c*N/n))
        _n = 0;
        _r = min(c,N/n);
        cb = (_n + _r)/2;
        while (_r - _n > precision) and (abs(mgvh(cb)) > precision ):
            if(mgvh(cb) * mgvh(_n) <= 0):
                _r = cb
            else :
                _n = cb
            cb = (_n + _r) / 2;
        
        return (c - cb)/c * Af + cb/c * Ab
    
    def _derivative(self, PD, c, Af, Ab, k, n):
        step = 0.001
        return (self._function(PD + step/2, c, Af, Ab, k, n) - self._function(PD - step / 2, c, Af, Ab, k, n)) / step
