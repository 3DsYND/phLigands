import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Estimator:
    params = None
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
    
    def plotAB(self, var_params, params=None, iters=100, log=None, **kwargs):
        if not params: params = self.params
        if len(var_params.keys()) != 2:
            self.logger.warning("plotAB must have 2 variable parametrs")
            return None
        A, B = var_params.keys()
        
        lin = [np.linspace(interval[0], interval[1], iters) for param, interval in var_params.items()]
        meshA, meshB = np.meshgrid(lin[0], lin[1])
        res = np.array(meshA)
        res[:] = np.nan
        
        for index, x in np.ndenumerate(res):
            params_field = params.copy()
            params_field[A], params_field[B] = meshA[index], meshB[index]
            res[index] = self.metric.score(self.model, params_field, self.dataX, self.dataY)
        
        minA, minB = meshA[np.where(res == res.min())], meshB[np.where(res == res.min())]
        lognorm = LogNorm() if log else None
        plt.figure(figsize=(10,10))
        if not "aspect" in kwargs: kwargs["aspect"] = (var_params[A][1] - var_params[A][0]) / (var_params[B][1] - var_params[B][0]) / 3
        plt.imshow(res, extent=[lin[0][0], lin[0][-1], lin[1][-1], lin[1][0]], norm=lognorm, **kwargs)
        plt.scatter(minA, minB, color="red")
        plt.show()
        
        params_field = params.copy()
        params_field[A], params_field[B] = minA, minB
        print(f"Real minimum: {{\"{A}\": {round(minA[0], 3)}, \"{B}\": {round(minB[0], 3)}}}, Score: {round(self.metric.score(self.model, params_field, self.dataX, self.dataY), 2)}")
        
        return res
    
    def plot(self, params=None, iter=20):
        if not params: params = self.params.copy()
        plt.figure(figsize=(10,5))
        plt.scatter(self.dataX, self.dataY)
        xgr = np.linspace(self.dataX.min(), self.dataX.max(), iter)
        ygr = self.model.eval(params, xgr)
        plt.plot(xgr, ygr)
        plt.show()
