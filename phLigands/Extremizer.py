import numpy as np

class Extremizer():
    def extremum(self, metric, model, start, field, dataX, dataY):
        prev_params = start
        while(True):
            iter_params = prev_params.copy()
            for param in iter_params.keys():
                errors = np.empty(field[param].shape[0])
                errors[:] = np.inf
                for i, param_val in np.ndenumerate(field[param]):
                    iter_params[param] = param_val
                    errors[i] = metric.score(model, iter_params, dataX, dataY)
#                     print(iter_params, "score: ", errors[i])
#                 print(np.isnan(errors[0]))
                best_val = metric.extremum(filter(lambda x: not np.isnan(x), errors) )
#                 print("B",param, iter_params[param], "score: ", errors)
                iter_params[param] = field[param][np.where(errors == best_val)[0]][0]
            print("AAAAAA", iter_params)
            if iter_params == prev_params:
                return iter_params
            prev_params = iter_params
            
class CoordEx(Extremizer):
    pass


class FakeGrad(Extremizer):
    def diff(self, func, x, deltadiff):
        return (func(x-deltadiff/2) - func(x+deltadiff/2))/deltadiff
    
    def extremum(self, metric, model, start, field, dataX, dataY, alpha=0.01, delta=0.1, deltadiff=0.001):
        x_prev = start
        x = x_prev.copy()
        while(True):
            for param in start.keys():
                x[key] = x[key] - alpha * diff(metric.score(model, dataX, dataY))
            if np.abs(x_prev - x) < delta:
                return x
            x_prev = x
