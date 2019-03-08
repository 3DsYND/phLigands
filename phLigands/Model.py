import numpy as np

class Model:
    params = None
    defparams = None
    
    def __init__(self, *args, **kwargs):
        self.defparams = self.setDefParams()
        self.params = self.getDefParams()
    
    def getField(self, param, range_param=None, dlt=0.01):
        minv = range_param[0] if range_param else self.defparams[param]["min"]
        maxv = range_param[1] if range_param else self.defparams[param]["max"]
        return np.arange(minv, maxv, dlt)
                      
    def getDefField(self, dlt=0.01):
        field = dict() 
        for param in self.defparams.keys():
            field[param] = self.getField(param, dlt=dlt)
        return field
    
    def getDefParams(self):
        start = dict()
        for param in self.defparams.keys():
            start[param] = self.defparams[param]["def"]
        return start
            
    def setDefParams(self):
        defparams = dict()
        defparams["k"] = {"min" : -10, "def" : 5, "max" : 10}
        defparams["b"] = {"min" : -10, "def" : 0, "max" : 10}
        return defparams
    
    def function(self, x, k, b):
        return x * k + b
    
    def derivative(self, x, k, b):
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
            print(a)
            dataY[i] = a
        return dataY   
    
    def eval(self, params, dataX, **kwargs):
        return self._eval_function(self.function, params, dataX)
    
    def diff(self, params, dataX, **kwargs):
        return self._eval_function(self.derivative, params, dataX)

    
class LinReg(Model):
    pass


class ScatchardN(Model):
    def setDefParams(self):
        defparams = dict()
        defparams["A_b"] = {"min" : 150, "def" : 191.8, "max" : 210}
        defparams["k"] = {"min" : 0.0001, "def" : 0.02, "max" : 1}
        defparams["n"] = {"min" : 0.001, "def" : 3.6, "max" : 7}
        return defparams
    
    def function(self, x, A_b, n, k):
        if A_b == 0 or n == 0 or k == 0:
            return np.nan
        tmp = 1 / (k * A_b) + n + x
        return A_b * (tmp - np.sqrt(tmp**2 - 4 * n * x)) / (2 * n)
    
    def derivative(self, x, A_b, n, k):
        if A_b == 0 or n == 0 or k == 0:
            return np.nan
        tmp = 1 / (k * A_b) + n + x
        return A_b * (1 - (2 * tmp - 4 * n) / (2 * np.sqrt(tmp**2 - 4 * n * x)))/ (2 * n)
