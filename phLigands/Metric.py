import numpy as np

class Metric:
    extremum = min
    
    def dist2(self, data, data_true, **kwargs):
        return (data - data_true)**2
    
    def aim_point(self, model, dataX, dataY, **kwargs):
        return dataX
    
    def score(self, model, params, dataX, dataY, return_array=False, **kwargs):
        aim_points = self.aim_point(model, dataX, dataY)
        dists = self.dist2(model.eval(params, aim_points), dataY, **kwargs)
        if return_array:
            return dists
        return sum(dists)

    
class Y2(Metric):
    pass


class R2(Metric):
    extremum = max
    
    def score(self, model, dataX, dataY, params, return_array=False, **kwargs):
        sres = super().score(model, dataX, dataY, params)
        stot = sum(self.dist2(dataY, dataY.mean()))
        if sres == 0:
            return 1.0
        if stot == 0:
            return 0.0
        return 1 - sres/stot

    
class Dist2(Metric):
    def evaldist2(self, model, params, dataX, dataX_true, dataY_true, weightsX=None, weightsY=None):
        if weightsX and weightsY:
            return self.dist2(dataX, dataX_true) * weightsX + self.dist2(model.eval(params, dataX), dataY_true) * weightsY
        if weightsX:
            return self.dist2(dataX, dataX_true) * weightsX + self.dist2(model.eval(params, dataX), dataY_true)
        if weightsY:
            return self.dist2(dataX, dataX_true) + self.dist2(model.eval(params, dataX), dataY_true) * weightsY
        return self.dist2(dataX, dataX_true) + self.dist2(model.eval(params, dataX), dataY_true)
    
    def aim_point(self, model, params, dataX, dataY, delta=0.001, return_dist=False, **kwargs):
        dist2 = list()
        aim_points = list()
        for index, row in np.ndenumerate(dataX):
            xp = dataX[index] 
            yp = dataY[index]
            r_0 = self.evaldist2(model, params, xp, xp, yp)
            r_left = self.evaldist2(model, params, xp-delta, xp, yp)
            r_right = self.evaldist2(model, params, xp+delta, xp, yp)

            arrow = 1
            if r_right < r_0:
                arrow = 1
            elif r_left < r_0:
                arrow = -1
            else:
                aim_points.append(xp)
                if return_dist: dist2.append(0)
                continue
            
            xf = xp
            r_dlt = 1.
            r = 0
            while r_dlt > 0:
                xf = xf+arrow*delta
                r_new = self.evaldist2(model, params, xf, xp, yp)
                r, r_dlt = r_new, r-r_new 
            aim_points.append(xf)
            if return_dist: dist2.append(r)
        if return_dist: 
            return dist2, aim_points
        return aim_points

    def score(self, model, params, dataX, dataY, return_array=False, **kwargs):
        dists, aim_points = self.aim_point(model, params, dataX, dataY, return_dist=True)
        if return_array:
            return dists, aim_points
        return sum(dists)
    
    
class Diff2(Dist2):
    def evaldist2(self, model, params, dataX, dataX_true, dataY_true, weightsX=None, weightsY=None):
        if weightsX and weightsY:
            return self.dist2(dataX, dataX_true) * weightsX + self.dist2(model.eval(params, dataX), dataY_true) / (model.diff(params, dataX))**2 * weightsY
        if weightsX:
            return self.dist2(dataX, dataX_true) * weightsX + self.dist2(model.eval(params, dataX), dataY_true) / (model.diff(params, dataX))**2
        if weightsY:
            return self.dist2(dataX, dataX_true) + self.dist2(model.eval(params, dataX), dataY_true) / (model.diff(params, dataX))**2 * weightsY
#         print("diff2", model.diff(dataX))
        return (self.dist2(dataX, dataX_true) + self.dist2(model.eval(params, dataX), dataY_true)) / (model.diff(params, dataX))**2
    
