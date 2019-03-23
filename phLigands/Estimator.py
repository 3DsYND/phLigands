import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Estimator:
    dataX = None
    dataY = None
    model = None
    metric = None
    extremizer = None
    
    def __init__(self, extremizer=None, metric=None, model=None):
        self.extremizer = extremizer
        self.metric = metric
        self.model = model
    
    def fit(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY
        
    def bestParams(self, def_params, var_params):
        self.params = self.extremizer.extremum(self.metric, self.model, self.dataX, self.dataY, def_params, var_params)
        return self.params
    
    def score(self, params=None, is_array=False):
        if not params:
            params = self.params.copy()
        return self.metric.score(self.model, params, self.dataX, self.dataY, is_array)

    def desmos(self):
        print(f"A(x, {self.params['Ab']}, {self.params['n']}, {self.params['k']})")
    
    def plotAB(self, A, B, rangeA=None, rangeB=None, dlt=0.5, log=None, **kwargs):
        linA, linB = self.model.getField(A, rangeA, dlt=dlt), self.model.getField(B, rangeB, dlt=dlt)
        meshA, meshB = np.meshgrid(linA, linB)
        res = np.array(meshA)
        res[:] = np.nan
        
        for index, x in np.ndenumerate(res):
            params = self.params.copy()
            params[A], params[B] = meshA[index], meshB[index]
            res[index] = self.metric.score(self.model, params, self.dataX, self.dataY)
        
        lognorm = LogNorm() if log else None
        plt.figure(figsize=(10,10))
        plt.imshow(res, extent=[linA[0], linA[-1], linB[-1], linB[0]], norm=lognorm, **kwargs)
        plt.scatter(meshA[np.where(res == res.min())], meshB[np.where(res == res.min())], color="red")
        plt.show()
        return res
    
    def plot(self, params=None, iter=20):
        if not params: params = self.params.copy()
        plt.figure(figsize=(10,5))
        plt.scatter(self.dataX, self.dataY)
        xgr = np.linspace(self.dataX.min(), self.dataX.max(), iter)
        print("ASD", params)
        ygr = self.model.eval(params, xgr)
        plt.plot(xgr, ygr)
        plt.show()
